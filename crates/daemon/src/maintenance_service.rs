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
use desktop_assistant_core::ports::inbound::{ConsolidationOutcome, KnowledgeMaintenanceService};
use desktop_assistant_core::ports::llm::{LlmClient, ReasoningConfig, with_cancellation_token};
use desktop_assistant_storage::PgPool;
use desktop_assistant_storage::dreaming::{
    BackfillEmbedFn, ConsolidationStats, DreamingLlmFn, KnowledgeChangeFn, run_consolidation_scan,
    run_dreaming_scan,
};
use desktop_assistant_storage::embedding_backfill::{
    backfill_knowledge_embeddings, invalidate_all_knowledge_embeddings,
};
use desktop_assistant_storage::knowledge_delete::KnowledgeDeletePolicy;
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
    /// What one consolidation run may destroy and rewrite, and the retention
    /// its opportunistic trash reap applies. A manual run and a timed one read
    /// the same configured answer the periodic sweep does.
    delete_policy: KnowledgeDeletePolicy,
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
        delete_policy: KnowledgeDeletePolicy,
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
            delete_policy,
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

    async fn run_consolidation(
        &self,
        cancellation: CancellationToken,
    ) -> Result<ConsolidationOutcome, CoreError> {
        let _guard = self
            .consolidation_lock
            .try_lock()
            .map_err(|_| CoreError::Storage("consolidation is already running".to_string()))?;
        let llm_fn = Self::build_llm_fn(
            Arc::clone(&self.consolidation_llm),
            self.consolidation_reasoning,
            cancellation.clone(),
        );
        let stats = run_consolidation_scan(
            &self.pool,
            &llm_fn,
            self.delete_policy,
            &cancellation,
            Some(&self.on_change),
        )
        .await?;
        // The live panel refresh is driven by `on_change` per user; what
        // travels back here is what the task log states about the pass.
        Ok(consolidation_outcome(&stats))
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

/// Read one run's per-op-kind counters as the outcome the maintenance task
/// reports.
///
/// Pure, and separate from the pass itself, because the distinction it draws
/// is the whole point of it: a run that proposed nothing and a run whose every
/// proposal was refused both apply zero changes, so a task log fed only the
/// change count reads the same for both. The refusal count and its
/// description travel with the change count so the two runs are told apart.
fn consolidation_outcome(stats: &ConsolidationStats) -> ConsolidationOutcome {
    let refusals = stats.refusal_count();
    ConsolidationOutcome {
        changes: stats.applied_count(),
        refusals,
        refusal_detail: (refusals > 0).then(|| stats.describe_refusals()),
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

#[cfg(test)]
mod tests {
    //! Timeout-wrapper coverage (issue #438): a wedged provider must fail the
    //! maintenance pass instead of hanging extraction/consolidation forever.
    //!
    //! Both tests run on a paused clock (`start_paused = true`), so tokio
    //! auto-advances virtual time to the [`MAINTENANCE_CALL_TIMEOUT`] deadline
    //! the moment the (never-resolving) provider future parks — the wall-clock
    //! test runs in microseconds while still exercising the real timeout branch.
    use super::*;
    use desktop_assistant_core::domain::ToolDefinition;
    use desktop_assistant_core::ports::llm::{ChunkCallback, LlmResponse};

    /// An `LlmClient` whose `stream_completion` never returns — models a hung /
    /// unreachable provider endpoint.
    struct HangingLlm;

    #[async_trait::async_trait]
    impl LlmClient for HangingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            std::future::pending::<()>().await;
            unreachable!("hung provider never resolves")
        }
    }

    /// An `EmbeddingClient` whose `embed` never returns — models a hung embedding
    /// endpoint.
    struct HangingEmbedder;

    #[async_trait::async_trait]
    impl EmbeddingClient for HangingEmbedder {
        async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
            std::future::pending::<()>().await;
            unreachable!("hung embedder never resolves")
        }

        async fn model_identifier(&self) -> Result<String, CoreError> {
            Ok("hanging".to_string())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn maintenance_llm_call_times_out() {
        let llm: Arc<dyn LlmClient> = Arc::new(HangingLlm);
        let llm_fn = DaemonKnowledgeMaintenanceService::build_llm_fn(
            llm,
            ReasoningConfig::default(),
            CancellationToken::new(),
        );

        let result = llm_fn("system".to_string(), "user".to_string()).await;

        let err = result.expect_err("a hung LLM provider must time out, not hang");
        assert!(
            err.contains("timed out"),
            "expected a timeout error, got: {err}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn maintenance_embed_call_times_out() {
        let client: Arc<dyn EmbeddingClient> = Arc::new(HangingEmbedder);
        let embed_fn = DaemonKnowledgeMaintenanceService::build_embed_fn(client);

        let result = embed_fn(vec!["text to embed".to_string()]).await;

        let err = result.expect_err("a hung embedder must time out, not hang");
        assert!(
            err.contains("timed out"),
            "expected a timeout error, got: {err}"
        );
    }

    /// Acceptance (#712 item 1): the outcome the maintenance task reports
    /// tells a run that proposed nothing from a run whose every proposal was
    /// refused. Both applied zero changes, so the change count on its own
    /// cannot be what separates them.
    #[test]
    fn the_maintenance_task_distinguishes_nothing_proposed_from_everything_refused() {
        let quiet = consolidation_outcome(&ConsolidationStats::default());
        let refused = consolidation_outcome(&ConsolidationStats {
            reviewed: 4,
            explicit_guard_refusals: 3,
            ..ConsolidationStats::default()
        });

        assert_eq!(
            quiet.changes, refused.changes,
            "both runs changed nothing, so the count alone cannot tell them apart"
        );
        assert_ne!(
            quiet, refused,
            "the two runs must not be reported the same way"
        );
        assert_eq!(refused.refusals, 3, "the refusals are carried, not dropped");
        let detail = refused
            .refusal_detail
            .expect("a run that refused something says what it refused");
        assert!(
            detail.contains("3 explicit-entry"),
            "the counts must be named, not just their presence: {detail}"
        );
        assert!(
            quiet.refusal_detail.is_none(),
            "a run with nothing to report adds nothing to its count"
        );
    }

    /// The reported change count is every kind of applied change, so a run
    /// that only added scope, or only merged, is not reported as a quiet one.
    #[test]
    fn the_reported_change_count_carries_every_kind_of_applied_change() {
        let each: [fn(&mut ConsolidationStats); 4] = [
            |s| s.updated = 1,
            |s| s.merged_clusters = 1,
            |s| s.dispositioned = 1,
            |s| s.scope_added = 1,
        ];
        for set in each {
            let mut stats = ConsolidationStats::default();
            set(&mut stats);
            let outcome = consolidation_outcome(&stats);
            assert_eq!(
                outcome.changes, 1,
                "stats {stats:?} applied a change and must be counted as one"
            );
            assert_eq!(outcome.refusals, 0);
        }
    }
}
