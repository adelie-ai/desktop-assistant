//! On-demand knowledge-maintenance service (the "dream cycle" controls).
//!
//! Implements [`KnowledgeMaintenanceService`] by driving the same storage scans
//! the daemon's periodic timers use — extraction, holistic consolidation, and a
//! force embedding recompute — so a button press and a timer tick share one
//! implementation, one configured LLM per pass, and one per-op mutual-exclusion
//! guard. The handler spawns each call as a tracked background task and hands in
//! the task's `CancellationToken`; this service builds **cancellation-aware**
//! LLM/embedding closures (a token-aware streaming callback + a per-call
//! timeout) so a run stops promptly via the existing task-cancel command and
//! can't wedge on a hung endpoint.
//!
//! An `on_change` callback, wired to the background-task registry's per-user
//! broadcast, fires as each user's entries land so connected knowledge panels
//! refetch live ("live as entries change").

use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role};
use desktop_assistant_core::ports::auth::UserId;
use desktop_assistant_core::ports::embedding::EmbeddingClient;
use desktop_assistant_core::ports::inbound::KnowledgeMaintenanceService;
use desktop_assistant_core::ports::llm::{LlmClient, ReasoningConfig, with_cancellation_token};
use desktop_assistant_storage::PgPool;
use desktop_assistant_storage::dreaming::{
    BackfillEmbedFn, DreamingLlmFn, KnowledgeChangeFn, run_consolidation_scan, run_dreaming_scan,
};
use desktop_assistant_storage::embedding_backfill::{
    backfill_knowledge_embeddings, invalidate_all_knowledge_embeddings,
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Per-call timeout for the dreaming/consolidation LLM and embedding calls, so a
/// hung or unreachable endpoint fails the pass instead of wedging it. Generous
/// because a holistic-consolidation prompt can be large and slow.
const MAINTENANCE_CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Daemon-side [`KnowledgeMaintenanceService`]. Holds the resolved LLM clients
/// (extraction and consolidation may use different purposes/models), the
/// embedding client, and the per-user change broadcaster.
pub struct DaemonKnowledgeMaintenanceService {
    pool: PgPool,
    dreaming_llm: Arc<dyn LlmClient>,
    dreaming_reasoning: ReasoningConfig,
    consolidation_llm: Arc<dyn LlmClient>,
    consolidation_reasoning: ReasoningConfig,
    embed_client: Arc<dyn EmbeddingClient>,
    embedding_model: String,
    archive_after_days: u32,
    on_change: KnowledgeChangeFn,
    // Per-op mutual exclusion. A manual trigger that collides with an already
    // running pass of the same op (timer- or manually-driven) is rejected rather
    // than run a second concurrent scan — the watermark/op-buffer logic and a
    // full re-embed are not safe to run twice at once.
    extraction_lock: Mutex<()>,
    consolidation_lock: Mutex<()>,
    embeddings_lock: Mutex<()>,
}

impl DaemonKnowledgeMaintenanceService {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pool: PgPool,
        dreaming_llm: Arc<dyn LlmClient>,
        dreaming_reasoning: ReasoningConfig,
        consolidation_llm: Arc<dyn LlmClient>,
        consolidation_reasoning: ReasoningConfig,
        embed_client: Arc<dyn EmbeddingClient>,
        embedding_model: String,
        archive_after_days: u32,
        on_change: KnowledgeChangeFn,
    ) -> Self {
        Self {
            pool,
            dreaming_llm,
            dreaming_reasoning,
            consolidation_llm,
            consolidation_reasoning,
            embed_client,
            embedding_model,
            archive_after_days,
            on_change,
            extraction_lock: Mutex::new(()),
            consolidation_lock: Mutex::new(()),
            embeddings_lock: Mutex::new(()),
        }
    }

    /// Build a cancellation-aware `DreamingLlmFn` for one pass. The returned
    /// closure: (a) installs the task's token via [`with_cancellation_token`] so
    /// the connector observes it during connect, (b) uses a token-aware
    /// streaming callback that returns `false` to stop the stream the moment the
    /// task is cancelled, and (c) bounds the whole call with a timeout.
    fn build_llm_fn(
        llm: Arc<dyn LlmClient>,
        reasoning: ReasoningConfig,
        token: CancellationToken,
    ) -> DreamingLlmFn {
        Box::new(move |system_prompt, user_prompt| {
            let llm = Arc::clone(&llm);
            let token = token.clone();
            Box::pin(async move {
                let messages = vec![
                    Message::new(Role::System, system_prompt),
                    Message::new(Role::User, user_prompt),
                ];
                let cb_token = token.clone();
                let call = with_cancellation_token(token, async move {
                    llm.stream_completion(
                        messages,
                        &[],
                        reasoning,
                        Box::new(move |_chunk| !cb_token.is_cancelled()),
                    )
                    .await
                });
                match tokio::time::timeout(MAINTENANCE_CALL_TIMEOUT, call).await {
                    Ok(Ok(resp)) => Ok(resp.text),
                    Ok(Err(e)) => Err(e.to_string()),
                    Err(_) => Err("maintenance LLM call timed out".to_string()),
                }
            })
        })
    }

    /// Build a timeout-bounded `BackfillEmbedFn`. Cancellation between batches is
    /// handled by the backfill loop itself (it checks the token); the timeout
    /// guards a single hung embed call.
    fn build_embed_fn(client: Arc<dyn EmbeddingClient>) -> BackfillEmbedFn {
        Box::new(move |texts| {
            let client = Arc::clone(&client);
            Box::pin(async move {
                match tokio::time::timeout(MAINTENANCE_CALL_TIMEOUT, client.embed(texts)).await {
                    Ok(Ok(v)) => Ok(v),
                    Ok(Err(e)) => Err(e.to_string()),
                    Err(_) => Err("maintenance embedding call timed out".to_string()),
                }
            })
        })
    }
}

#[async_trait::async_trait]
impl KnowledgeMaintenanceService for DaemonKnowledgeMaintenanceService {
    async fn run_extraction(&self, cancellation: CancellationToken) -> Result<usize, CoreError> {
        let _guard = self
            .extraction_lock
            .try_lock()
            .map_err(|_| CoreError::Storage("extraction is already running".to_string()))?;
        let llm_fn = Self::build_llm_fn(
            Arc::clone(&self.dreaming_llm),
            self.dreaming_reasoning,
            cancellation.clone(),
        );
        let embed_fn = Self::build_embed_fn(Arc::clone(&self.embed_client));
        run_dreaming_scan(
            &self.pool,
            &llm_fn,
            &embed_fn,
            &self.embedding_model,
            self.archive_after_days,
            &cancellation,
            Some(&self.on_change),
        )
        .await
    }

    async fn run_consolidation(&self, cancellation: CancellationToken) -> Result<usize, CoreError> {
        let _guard = self
            .consolidation_lock
            .try_lock()
            .map_err(|_| CoreError::Storage("consolidation is already running".to_string()))?;
        let llm_fn = Self::build_llm_fn(
            Arc::clone(&self.consolidation_llm),
            self.consolidation_reasoning,
            cancellation.clone(),
        );
        let stats =
            run_consolidation_scan(&self.pool, &llm_fn, &cancellation, Some(&self.on_change))
                .await?;
        // Collapse the per-op-kind stats into a single "changes" count for the
        // task log; the live panel refresh is driven by `on_change` per user.
        Ok(stats.updated + stats.merged_clusters + stats.soft_deleted + stats.scope_added)
    }

    async fn recalculate_embeddings(
        &self,
        cancellation: CancellationToken,
    ) -> Result<usize, CoreError> {
        let _guard = self.embeddings_lock.try_lock().map_err(|_| {
            CoreError::Storage("embedding recompute is already running".to_string())
        })?;
        // Force path: NULL out every active row's vector (catches out-of-band
        // edits the model-stamp comparison would miss), then drive the existing
        // batched backfill to re-embed them. No `on_change` — embeddings don't
        // change displayed content, and the task's progress/completion events
        // already inform the UI.
        let invalidated = invalidate_all_knowledge_embeddings(&self.pool)
            .await
            .map_err(CoreError::Storage)?;
        tracing::info!(
            "recalculate embeddings: invalidated {invalidated} row(s); re-embedding all"
        );
        let embed_fn = Self::build_embed_fn(Arc::clone(&self.embed_client));
        backfill_knowledge_embeddings(&self.pool, &embed_fn, &self.embedding_model, &cancellation)
            .await
            .map_err(CoreError::Storage)
    }
}

/// Build the registry-backed `on_change` callback: each invocation broadcasts a
/// `KnowledgeChanged` event to the given user's subscribed connections, so their
/// open knowledge panels refetch as a pass writes entries.
pub fn knowledge_change_notifier(
    registry: Arc<desktop_assistant_application::background_tasks::BackgroundTaskRegistry>,
) -> KnowledgeChangeFn {
    Arc::new(move |user_id: &UserId| {
        registry.notify_knowledge_changed(user_id);
    })
}
