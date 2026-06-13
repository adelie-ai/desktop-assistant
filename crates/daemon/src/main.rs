//! Assistant daemon binary: wires the core services to storage, LLM connectors, and transports.

use std::sync::Arc;

use anyhow::Result;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role};
use desktop_assistant_core::ports::embedding::{EmbedFn, EmbeddingClient};
use desktop_assistant_core::ports::llm::{LlmClient, ReasoningConfig, RetryingLlmClient};
use desktop_assistant_core::ports::llm_profiling::MaybeProfiled;
use tracing_subscriber::EnvFilter;

mod api_surface;
mod app;
mod backend_reasoning;
mod classifying_llm;
mod config;
mod connections;
mod knowledge_service;
mod model_defaults;
mod purposes;
mod registry;
mod routing_llm;
mod settings_service;
mod store;
mod tls;
mod transports;

use crate::app::Assistant;
use crate::registry::{ConnectionHealth, build_llm_client, build_registry};
use desktop_assistant_application::DefaultAssistantApiHandler;
use desktop_assistant_core::service::ConversationHandler;
use desktop_assistant_dbus::conversation::DbusConversationAdapter;
use desktop_assistant_dbus::reload::DbusReloadAdapter;
use desktop_assistant_dbus::settings::DbusSettingsAdapter;
use desktop_assistant_mcp_client::config as mcp_config;
use desktop_assistant_mcp_client::executor::{BuiltinToolService, McpToolExecutor};
use desktop_assistant_uds as uds;
use desktop_assistant_ws as ws;
use settings_service::DaemonSettingsService;
use store::PersistentConversationStore;
use transports::{
    OidcAwareAuth, WsAsUdsAuth, WsAuthDiscoveryProvider, WsBasicLogin, WsLoginMode, WsSettingsAuth,
    daemon_host_label, env_bool, resolve_uds_socket_path, resolve_ws_login_mode,
};

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("failed to install Ctrl+C handler: {e}");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                let _ = stream.recv().await;
            }
            Err(e) => {
                tracing::error!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

/// Consume reload pings (from the D-Bus `Reload` method and the config-file
/// watcher) and apply the new config to the running daemon (#222).
///
/// Coalesces a burst: it drains any pings that arrived while the last apply ran
/// so an editor's write/rename/chmod storm collapses into one
/// validate-classify-swap. `apply_reload` is state-preserving — it swaps the
/// connection registry under its lock so new turns pick up the new clients
/// while in-flight turns keep theirs alive by refcount — and never panics on a
/// bad config: it refuses the apply, logs the cause, and keeps the last-good
/// config running.
fn spawn_reload_consumer(
    registry: Arc<api_surface::RegistryHandle>,
    mut reload_rx: tokio::sync::mpsc::Receiver<()>,
) {
    tokio::spawn(async move {
        while reload_rx.recv().await.is_some() {
            // Drain any pings queued behind this one — a single apply observes
            // the latest on-disk config, so coalescing is correct.
            while reload_rx.try_recv().is_ok() {}
            tracing::info!("config reload requested; re-reading daemon.toml");
            match registry.apply_reload() {
                Ok(plan) if plan.is_empty() => {}
                Ok(_) => tracing::info!("config reload applied"),
                Err(e) => {
                    tracing::error!("config reload refused; keeping the last-good config: {e:#}")
                }
            }
        }
        tracing::debug!("config reload consumer exiting (channel closed)");
    });
}

/// Watch `daemon.toml` for edits and ping the reload consumer (#222),
/// debounced so an editor's write/rename/chmod burst collapses into one reload.
///
/// Mirrors the voice daemon's `spawn_config_watcher`: watch the *parent
/// directory* (many editors replace the file via a temp-file rename, which
/// breaks a watch bound to the original inode), bridge `notify`'s callback
/// thread to a std channel, then debounce on a dedicated blocking thread that
/// forwards a single async ping per quiet window. The KCM gets instant reload
/// via the D-Bus `Reload` method; this covers hand-edits and other tools.
fn spawn_config_watcher(config_path: std::path::PathBuf, reload_tx: tokio::sync::mpsc::Sender<()>) {
    use notify::{RecursiveMode, Watcher};

    let dir = match config_path.parent() {
        Some(d) => d.to_path_buf(),
        None => {
            tracing::warn!("config watcher: config path has no parent dir, not watching");
            return;
        }
    };
    let file_name = config_path.file_name().map(std::ffi::OsString::from);

    // notify's callback runs on its own (non-async) thread; bridge to a std
    // mpsc, then debounce on a dedicated blocking thread that forwards into the
    // async reload channel. A blocking thread (not a Tokio task) is used because
    // the std `recv()` would otherwise park a runtime worker.
    let (raw_tx, raw_rx) = std::sync::mpsc::channel::<()>();
    let watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            // Only react to events touching our config file (the dir may hold
            // other files). Match by file name; rename targets count too.
            let touches_config = file_name.as_ref().is_none_or(|name| {
                event
                    .paths
                    .iter()
                    .any(|p| p.file_name() == Some(name.as_os_str()))
            });
            let relevant = matches!(
                event.kind,
                notify::EventKind::Modify(_)
                    | notify::EventKind::Create(_)
                    | notify::EventKind::Remove(_)
            );
            if touches_config && relevant {
                let _ = raw_tx.send(());
            }
        }
    });
    let mut watcher = match watcher {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(
                "config watcher: failed to create watcher, live reload on file edits disabled: {e}"
            );
            return;
        }
    };
    if let Err(e) = watcher.watch(&dir, RecursiveMode::NonRecursive) {
        tracing::warn!(
            dir = %dir.display(),
            "config watcher: failed to watch dir, live reload on file edits disabled: {e}"
        );
        return;
    }

    std::thread::spawn(move || {
        // Keep the watcher alive for the life of the thread (dropping it stops
        // watching).
        let _watcher = watcher;
        // Block until the first raw event, then wait out a short quiet window
        // and drain any burst — collapsing an editor's write/rename/chmod storm
        // into one reload.
        while raw_rx.recv().is_ok() {
            std::thread::sleep(std::time::Duration::from_millis(300));
            while raw_rx.try_recv().is_ok() {}
            tracing::info!("daemon.toml changed on disk; requesting reload");
            if reload_tx.blocking_send(()).is_err() {
                break; // consumer gone
            }
        }
    });
}

/// Enum wrapper to dispatch between conversation store backends at runtime.
enum AnyConversationStore {
    Json(PersistentConversationStore),
    Postgres(desktop_assistant_storage::PgConversationStore),
}

impl desktop_assistant_core::ports::store::ConversationStore for AnyConversationStore {
    async fn create(
        &self,
        conv: desktop_assistant_core::domain::Conversation,
    ) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.create(conv).await,
            Self::Postgres(s) => s.create(conv).await,
        }
    }

    async fn get(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<desktop_assistant_core::domain::Conversation, CoreError> {
        match self {
            Self::Json(s) => s.get(id).await,
            Self::Postgres(s) => s.get(id).await,
        }
    }

    async fn list(
        &self,
    ) -> Result<Vec<desktop_assistant_core::domain::ConversationSummary>, CoreError> {
        match self {
            Self::Json(s) => s.list().await,
            Self::Postgres(s) => s.list().await,
        }
    }

    async fn update(
        &self,
        conv: desktop_assistant_core::domain::Conversation,
    ) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.update(conv).await,
            Self::Postgres(s) => s.update(conv).await,
        }
    }

    async fn delete(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.delete(id).await,
            Self::Postgres(s) => s.delete(id).await,
        }
    }

    async fn archive(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.archive(id).await,
            Self::Postgres(s) => s.archive(id).await,
        }
    }

    async fn unarchive(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.unarchive(id).await,
            Self::Postgres(s) => s.unarchive(id).await,
        }
    }

    async fn create_summary(
        &self,
        conversation_id: &desktop_assistant_core::domain::ConversationId,
        summary: String,
        start_ordinal: usize,
        end_ordinal: usize,
    ) -> Result<String, CoreError> {
        match self {
            Self::Json(s) => {
                s.create_summary(conversation_id, summary, start_ordinal, end_ordinal)
                    .await
            }
            Self::Postgres(s) => {
                s.create_summary(conversation_id, summary, start_ordinal, end_ordinal)
                    .await
            }
        }
    }

    async fn expand_summary(&self, summary_id: &str) -> Result<(), CoreError> {
        match self {
            Self::Json(s) => s.expand_summary(summary_id).await,
            Self::Postgres(s) => s.expand_summary(summary_id).await,
        }
    }
}

/// Shareable store wrapper. The daemon owns the concrete
/// `AnyConversationStore` once (behind an `Arc`) and hands out cloned
/// `SharedConversationStore` handles to each consumer
/// (`ConversationHandler`, `RoutingConversationHandler`, the
/// selection-store layer). A newtype is required so the `ConversationStore`
/// impl doesn't hit the orphan rule that would bite a direct
/// `impl ... for Arc<AnyConversationStore>`.
#[derive(Clone)]
struct SharedConversationStore(Arc<AnyConversationStore>);

impl desktop_assistant_core::ports::store::ConversationStore for SharedConversationStore {
    async fn create(
        &self,
        conv: desktop_assistant_core::domain::Conversation,
    ) -> Result<(), CoreError> {
        self.0.create(conv).await
    }

    async fn get(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<desktop_assistant_core::domain::Conversation, CoreError> {
        self.0.get(id).await
    }

    async fn list(
        &self,
    ) -> Result<Vec<desktop_assistant_core::domain::ConversationSummary>, CoreError> {
        self.0.list().await
    }

    async fn update(
        &self,
        conv: desktop_assistant_core::domain::Conversation,
    ) -> Result<(), CoreError> {
        self.0.update(conv).await
    }

    async fn delete(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<(), CoreError> {
        self.0.delete(id).await
    }

    async fn archive(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<(), CoreError> {
        self.0.archive(id).await
    }

    async fn unarchive(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<(), CoreError> {
        self.0.unarchive(id).await
    }

    async fn create_summary(
        &self,
        conversation_id: &desktop_assistant_core::domain::ConversationId,
        summary: String,
        start_ordinal: usize,
        end_ordinal: usize,
    ) -> Result<String, CoreError> {
        self.0
            .create_summary(conversation_id, summary, start_ordinal, end_ordinal)
            .await
    }

    async fn expand_summary(&self, summary_id: &str) -> Result<(), CoreError> {
        self.0.expand_summary(summary_id).await
    }
}

impl api_surface::ConversationSelectionStore for SharedConversationStore {
    async fn get_selection(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<Option<desktop_assistant_core::ports::inbound::ConversationModelSelection>, CoreError>
    {
        <AnyConversationStore as api_surface::ConversationSelectionStore>::get_selection(
            &self.0, id,
        )
        .await
    }

    async fn set_selection(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
        selection: Option<&desktop_assistant_core::ports::inbound::ConversationModelSelection>,
    ) -> Result<(), CoreError> {
        <AnyConversationStore as api_surface::ConversationSelectionStore>::set_selection(
            &self.0, id, selection,
        )
        .await
    }

    async fn get_personality(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<Option<desktop_assistant_core::prompts::PersonalityOverride>, CoreError> {
        <AnyConversationStore as api_surface::ConversationSelectionStore>::get_personality(
            &self.0, id,
        )
        .await
    }

    async fn set_personality(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
        personality: Option<&desktop_assistant_core::prompts::PersonalityOverride>,
    ) -> Result<(), CoreError> {
        <AnyConversationStore as api_surface::ConversationSelectionStore>::set_personality(
            &self.0,
            id,
            personality,
        )
        .await
    }
}

// Per-conversation model selection. Only the Postgres backend persists
// selections across restarts; the JSON backend keeps them in-memory and
// drops them on shutdown (the same shape as installs without a database).
impl api_surface::ConversationSelectionStore for AnyConversationStore {
    async fn get_selection(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<Option<desktop_assistant_core::ports::inbound::ConversationModelSelection>, CoreError>
    {
        match self {
            Self::Postgres(s) => s.get_conversation_model_selection(id).await,
            Self::Json(_) => {
                // No durable storage — treat as "no stored selection". The
                // JSON fallback is deprecated; Postgres is the supported
                // backend going forward.
                Ok(None)
            }
        }
    }

    async fn set_selection(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
        selection: Option<&desktop_assistant_core::ports::inbound::ConversationModelSelection>,
    ) -> Result<(), CoreError> {
        match self {
            Self::Postgres(s) => s.set_conversation_model_selection(id, selection).await,
            Self::Json(_) => {
                // No-op on the JSON backend — see comment on `get_selection`.
                let _ = selection;
                Ok(())
            }
        }
    }

    // Per-conversation personality override (#227). Same backend split as the
    // model selection above: only Postgres persists; the JSON fallback drops it.
    async fn get_personality(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
    ) -> Result<Option<desktop_assistant_core::prompts::PersonalityOverride>, CoreError> {
        match self {
            Self::Postgres(s) => s.get_conversation_personality(id).await,
            Self::Json(_) => Ok(None),
        }
    }

    async fn set_personality(
        &self,
        id: &desktop_assistant_core::domain::ConversationId,
        personality: Option<&desktop_assistant_core::prompts::PersonalityOverride>,
    ) -> Result<(), CoreError> {
        match self {
            Self::Postgres(s) => s.set_conversation_personality(id, personality).await,
            Self::Json(_) => {
                let _ = personality;
                Ok(())
            }
        }
    }
}

// #205/#206 history: the interactive send-prompt task once spawned a multi-MB
// future — `ConversationService` used RPITIT, so the whole nested
// handler/LLM/tool stack monomorphized into one inlined state machine that
// overflowed the default 2 MB tokio worker stack at `tokio::spawn`. #206 worked
// around it by bumping `thread_stack_size` to 16 MB. #207 fixed the root cause:
// `ConversationService` is now `#[async_trait]` and the LLM decorator stack is
// erased to `Arc<dyn LlmClient>`, so the spawned future is a thin boxed
// `Pin<Box<dyn Future>>` (guarded by the `spawned_send_prompt_future_stays_small`
// tests). The workaround is therefore removed — we run on the default runtime.
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // DT-3 (#269): operator escape hatch. `desktop-assistant --revoke-token
    // <jwt>` adds the token's `jti` to the WS JWT revocation deny-list and
    // exits without starting the daemon, so a leaked short-lived token can be
    // killed before its 1h expiry. The token must still be a valid,
    // unexpired, daemon-signed JWT (we read its jti/exp); garbage is rejected.
    {
        let args: Vec<String> = std::env::args().collect();
        if let Some(pos) = args.iter().position(|a| a == "--revoke-token") {
            let token = args
                .get(pos + 1)
                .ok_or_else(|| anyhow::anyhow!("--revoke-token requires a JWT argument"))?;
            config::revoke_ws_jwt(token)
                .map_err(|e| anyhow::anyhow!("failed to revoke token: {e}"))?;
            tracing::info!("ws jwt revoked");
            println!("token revoked");
            return Ok(());
        }
    }

    tracing::info!("desktop-assistant starting");

    // Install the rustls crypto provider for TLS support. Returns Err if a
    // provider is already installed — fine on fresh start, but assert success
    // on the first install path so we don't silently run with an unexpected
    // provider (e.g. one pulled in by a transitive dep).
    if rustls::crypto::CryptoProvider::get_default().is_none()
        && rustls::crypto::aws_lc_rs::default_provider()
            .install_default()
            .is_err()
    {
        return Err(anyhow::anyhow!(
            "failed to install rustls aws_lc_rs crypto provider"
        ));
    }

    // Register the system Secret Service as keyring-core's default credential
    // store so the `keyring`/`libsecret` secret backend can read and write API
    // keys in-process (replacing the old `secret-tool` subprocess). Connecting
    // touches D-Bus via a blocking client, so do it off the async worker. When
    // there's no Secret Service (e.g. headless), log and continue — the
    // file/systemd/KWallet backends still work without it.
    match tokio::task::spawn_blocking(zbus_secret_service_keyring_store::Store::new).await {
        Ok(Ok(store)) => {
            keyring_core::set_default_store(store);
            tracing::info!("registered Secret Service credential store");
        }
        Ok(Err(error)) => {
            tracing::warn!("Secret Service unavailable; keyring secret backend disabled: {error}");
        }
        Err(error) => {
            tracing::warn!("Secret Service store init task failed: {error}");
        }
    }

    // DT-3 (#269): drop WS JWT revocation entries whose own `exp` has already
    // passed (they're rejected by the exp check anyway), so the deny-list file
    // can't grow without bound across restarts.
    config::prune_revocations();

    // Build the LLM client from daemon.toml + KWallet (fallback to env)
    let config_path = config::default_daemon_config_path();
    let daemon_config = match config::load_daemon_config(&config_path) {
        Ok(config) => config,
        Err(error) => {
            tracing::warn!(
                "failed to load daemon config at {}: {error}",
                config_path.display()
            );
            None
        }
    };

    // Transport enable/bind config (#279 item 3): the `[transports]` table is
    // the baseline; the matching `DESKTOP_ASSISTANT_*` env var overrides each
    // field when set. Absent table => historical defaults.
    let transports_config = daemon_config
        .as_ref()
        .map(|c| c.transports.clone())
        .unwrap_or_default();

    let profiling = daemon_config
        .as_ref()
        .map(|c| c.profiling.clone())
        .unwrap_or_default();

    // Build the per-connection client registry from the [connections] map
    // (#9). Purpose-based dispatch (#10 + #11) picks the right client per
    // request via `registry.get(&purpose_resolved.connection_id)`;
    // the legacy "active connection" fast-path from #9 is no longer the
    // primary dispatch. Connections that fail to build are logged and
    // marked unavailable — the daemon still starts.
    let connection_registry = match daemon_config.as_ref() {
        Some(config) => build_registry(config),
        None => registry::ConnectionRegistry::empty(),
    };
    // Kick off `/api/show` lookups for any Ollama connections so the per-
    // model context window is cached before the user fires the first
    // turn. Detached: the daemon must still start when Ollama is down.
    connection_registry.spawn_warmups();
    for status in connection_registry.status() {
        match &status.health {
            ConnectionHealth::Ok => {
                tracing::info!("connection {} ({}) ready", status.id, status.connector_type)
            }
            ConnectionHealth::Unavailable { reason } => tracing::warn!(
                "connection {} ({}) unavailable: {reason}",
                status.id,
                status.connector_type
            ),
        }
    }

    // Resolve the embedding-friendly fields from the old `[llm]` block. The
    // embedding client path still reads `[llm]` directly; #10 will move it
    // over to a purpose-based lookup. Keeping this read here means an install
    // with only a `[connections]` table still gets embedding defaults.
    let resolved_llm = config::resolve_llm_config(daemon_config.as_ref());
    tracing::info!(
        "primary LLM resolved: connector={}, model={}, base_url={}",
        resolved_llm.connector,
        resolved_llm.model,
        resolved_llm.base_url
    );
    let llm_connector = resolved_llm.connector.clone();

    // Resolve the `interactive` purpose and grab its client from the
    // registry. This is the primary dispatch target for `send_prompt`
    // (without an override) and for the conversation handler's built-in
    // fallback path. `registry.get(&id)` gives us a borrow rather than
    // moving the client out of the map, which is what #11 needs so other
    // connections (for cross-connection send overrides) stay available.
    //
    // Rather than teaching `ConversationHandler` to borrow from the
    // registry (which would require a lifetime on the handler that
    // propagates into every adapter), we build a single primary client by
    // re-resolving the interactive purpose and calling `build_llm_client`
    // a second time. It's a duplicate client but the cost is one extra
    // HTTP client allocation — the registry clients stay live for the
    // connection-listing and model-listing APIs.
    // Build the primary llm via the shared `resolve_purpose_llm_config`
    // helper so the interactive purpose's `model` actually lands on the
    // resolved config — connector clients have no per-call model knob,
    // so a dispatch via the registry's per-connection client would
    // otherwise silently use the connection's construction-time model
    // and ignore the user's choice. Using the same helper for primary
    // and background-task purposes keeps the model-override logic in
    // one place.
    let primary_resolved = config::resolve_purpose_llm_config(
        daemon_config.as_ref(),
        purposes::PurposeKind::Interactive,
    )
    .and_then(|resolved| {
        let id = daemon_config
            .as_ref()
            .and_then(|c| c.purposes.get(purposes::PurposeKind::Interactive))
            .and_then(|p| match &p.connection {
                purposes::ConnectionRef::Named(id) => Some(id.clone()),
                purposes::ConnectionRef::Primary => None,
            })?;
        Some((id, resolved))
    });

    let (active_id, llm) = match primary_resolved {
        Some((id, resolved)) => {
            tracing::info!("primary dispatch via interactive purpose → connection {id}");
            (Some(id), build_llm_client(resolved))
        }
        None => {
            // No `[purposes.interactive]` configured — fall back to the
            // legacy `[llm]` block so the daemon still comes up. Users on
            // fresh installs land here until they finish purpose migration.
            tracing::warn!(
                "no interactive purpose configured; falling back to legacy [llm] client"
            );
            (None, build_llm_client(resolved_llm.clone()))
        }
    };
    if let Some(id) = &active_id {
        tracing::info!("using {} LLM backend via connection {}", llm_connector, id);
    } else {
        tracing::info!("using {} LLM backend (legacy fallback)", llm_connector);
    }

    // Build the embedding client from resolved config
    let resolved_emb = config::resolve_embeddings_config(daemon_config.as_ref());
    tracing::info!(
        "Embeddings connector={}, model={}, base_url={}, available={}, is_default={}",
        resolved_emb.connector,
        resolved_emb.model,
        resolved_emb.base_url,
        resolved_emb.available,
        resolved_emb.is_default
    );

    let embedding_client: Option<Arc<dyn EmbeddingClient>> = if !resolved_emb.available {
        tracing::info!(
            "embeddings unavailable (connector={})",
            resolved_emb.connector
        );
        None
    } else {
        Some(match resolved_emb.connector.as_str() {
            "ollama" => {
                tracing::info!("using Ollama embedding backend");
                Arc::new(desktop_assistant_llm_ollama::OllamaClient::new(
                    resolved_emb.base_url.clone(),
                    resolved_emb.model.clone(),
                ))
            }
            "bedrock" | "aws-bedrock" => {
                tracing::info!("using Bedrock embedding backend");
                Arc::new(
                    desktop_assistant_llm_bedrock::BedrockClient::new(String::new())
                        .with_model(resolved_emb.model.clone())
                        .with_base_url(resolved_emb.base_url.clone()),
                )
            }
            _ => {
                tracing::info!("using OpenAI-compatible embedding backend");
                // `resolved_emb.api_key` is now resolved by
                // `resolve_embeddings_config` itself (purpose path uses the
                // purpose's connection's secret/env; legacy path reuses the
                // shared LLM key when connectors match, else falls back to
                // `<CONNECTOR>_API_KEY`).
                Arc::new(
                    desktop_assistant_llm_openai::OpenAiClient::new(resolved_emb.api_key.clone())
                        .with_model(resolved_emb.model.clone())
                        .with_base_url(resolved_emb.base_url.clone()),
                )
            }
        })
    };

    // Resolve model identifier once at startup (includes digest for Ollama).
    let embedding_model_id: String = if let Some(client) = &embedding_client {
        match client.model_identifier().await {
            Ok(id) => {
                tracing::info!("resolved embedding model identifier: {id}");
                id
            }
            Err(e) => {
                tracing::warn!(
                    "failed to resolve embedding model identifier, falling back to configured name: {e}"
                );
                resolved_emb.model.clone()
            }
        }
    } else {
        resolved_emb.model.clone()
    };

    let embedding_fn: Option<EmbedFn> = embedding_client.as_ref().map(|client| {
        let client = Arc::clone(client);
        Arc::new(move |texts: Vec<String>| {
            let client = Arc::clone(&client);
            Box::pin(async move { client.embed(texts).await })
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, CoreError>> + Send>,
                >
        }) as EmbedFn
    });

    // --- Database (optional) ---
    let (db_url, db_max_conns) = config::resolve_database_config(daemon_config.as_ref());
    let pg_pool = if let Some(url) = db_url {
        tracing::info!(
            "connecting to PostgreSQL (max_connections={})",
            db_max_conns
        );
        match desktop_assistant_storage::create_pool(&url, db_max_conns).await {
            Ok(pool) => {
                if let Err(e) = desktop_assistant_storage::run_migrations(&pool).await {
                    tracing::error!("failed to run database migrations: {e}");
                    return Err(e.into());
                }
                tracing::info!("database migrations applied successfully");

                // One-time JSON → Postgres migration (runs if JSON files exist)
                let conv_json = store::default_conversation_store_path();
                let data_home = conv_json.parent().unwrap_or(std::path::Path::new("."));
                let prefs_json = data_home.join("preferences.json");
                let memory_json = data_home.join("factual_memory.json");

                if conv_json.exists() || prefs_json.exists() || memory_json.exists() {
                    // Only migrate if tables are empty (first startup with Postgres)
                    if conv_json.exists()
                        && desktop_assistant_storage::is_conversations_table_empty(&pool).await
                    {
                        match desktop_assistant_storage::migrate_conversations(&conv_json, &pool)
                            .await
                        {
                            Ok(n) => tracing::info!("migrated {n} conversations from JSON"),
                            Err(e) => tracing::warn!("conversation migration failed: {e}"),
                        }
                    }
                    if (prefs_json.exists() || memory_json.exists())
                        && desktop_assistant_storage::is_knowledge_base_table_empty(&pool).await
                    {
                        match desktop_assistant_storage::migrate_knowledge(
                            &prefs_json,
                            &memory_json,
                            &pool,
                        )
                        .await
                        {
                            Ok(n) => tracing::info!("migrated {n} knowledge entries from JSON"),
                            Err(e) => tracing::warn!("knowledge migration failed: {e}"),
                        }
                    }
                }

                Some(pool)
            }
            Err(e) => {
                tracing::error!("failed to connect to PostgreSQL: {e}");
                return Err(e.into());
            }
        }
    } else {
        tracing::info!("no database URL configured; Postgres features disabled");
        None
    };

    // Invalidate embeddings from a different model so that vector-dimension
    // mismatches cannot cause search errors while the backfill is running.
    if let Some(pool) = &pg_pool
        && embedding_client.is_some()
    {
        match desktop_assistant_storage::embedding_backfill::invalidate_stale_embeddings(
            pool,
            &embedding_model_id,
        )
        .await
        {
            Ok((kb, tools)) if kb > 0 || tools > 0 => {
                tracing::warn!(
                    "embedding model changed to {}: invalidated {kb} knowledge + {tools} tool embeddings (will re-embed in background)",
                    embedding_model_id
                );
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("failed to invalidate stale embeddings: {e}");
            }
        }
    }

    // --- Knowledge base & tool registry stores ---
    let kb_store = pg_pool.as_ref().map(|pool| {
        Arc::new(desktop_assistant_storage::PgKnowledgeBaseStore::new(
            pool.clone(),
        ))
    });

    let tool_registry_store = pg_pool.as_ref().map(|pool| {
        Arc::new(desktop_assistant_storage::PgToolRegistryStore::new(
            pool.clone(),
        ))
    });

    // Load MCP server configuration and secrets
    let mcp_config_path = mcp_config::default_config_path();
    let mcp_configs = mcp_config::load_mcp_configs(&mcp_config_path).unwrap_or_else(|e| {
        tracing::warn!("failed to load MCP config: {e}");
        Vec::new()
    });
    let secrets_path = mcp_config::default_secrets_path();
    let mcp_secrets = mcp_config::load_secrets(&secrets_path).unwrap_or_else(|e| {
        tracing::warn!("failed to load secrets: {e}");
        std::collections::HashMap::new()
    });

    // Build the MCP tool executor with builtin tools
    let mut builtin_tools = BuiltinToolService::new();
    // Hold an extra clone for the knowledge management service (#73) so
    // both the LLM-tool path and the client-facing service embed via
    // the same closure.
    let embedding_fn_for_kb_service: Option<EmbedFn> = embedding_fn.clone();
    if let Some(embed_fn) = embedding_fn {
        tracing::info!(
            "enabling built-in vector search with model={}",
            embedding_model_id
        );
        builtin_tools = builtin_tools.with_embedding(embed_fn);
    } else {
        tracing::info!("built-in vector search disabled (no embedding backend available)");
    }

    if let Some(kb) = &kb_store {
        tracing::info!("wiring knowledge base store into builtin tools");
        let kb_w = Arc::clone(kb);
        let kb_s = Arc::clone(kb);
        let kb_d = Arc::clone(kb);
        let kb_emb_model = embedding_model_id.clone();
        use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
        builtin_tools = builtin_tools.with_knowledge_base(
            Arc::new(move |entry, embedding: Option<Vec<Vec<f32>>>| {
                let store = Arc::clone(&kb_w);
                let model = if embedding.is_some() {
                    Some(kb_emb_model.clone())
                } else {
                    None
                };
                Box::pin(async move { store.write(entry, embedding, model).await })
            }),
            Arc::new(move |query, embedding, tags, limit| {
                let store = Arc::clone(&kb_s);
                Box::pin(async move { store.search(&query, embedding, tags, limit).await })
            }),
            Arc::new(move |id| {
                let store = Arc::clone(&kb_d);
                Box::pin(async move { store.delete(&id).await })
            }),
        );
    }

    // Background-task registry (#111/#115). Constructed BEFORE the scratchpad
    // wiring below so the scratchpad mutating closures can notify subscribed
    // connections via `Event::ScratchpadChanged` (#190). Attaches a Postgres-
    // backed store when a pool is available, then runs the cold-restart sweep
    // (marks non-terminal rows `Failed` so the user observes them via `list`).
    let background_task_registry = {
        let registry =
            desktop_assistant_application::background_tasks::BackgroundTaskRegistry::new();
        if let Some(pool) = &pg_pool {
            let store: std::sync::Arc<
                dyn desktop_assistant_core::ports::store::BackgroundTaskStore,
            > = std::sync::Arc::new(desktop_assistant_storage::PgBackgroundTaskStore::new(
                pool.clone(),
            ));
            let registry = registry.with_store(store);
            if let Err(e) = registry.sweep_non_terminal_on_startup().await {
                tracing::warn!(
                    error = %e,
                    "cold-restart sweep of background tasks failed; continuing",
                );
            }
            Arc::new(registry)
        } else {
            Arc::new(registry)
        }
    };

    // Reader for the reserved scratchpad `goal` note, wired into the
    // conversation handler below so the evolving goal is surfaced as the task
    // anchor each turn. Populated only when a Postgres pool is available.
    let mut scratchpad_goal_fn: Option<
        desktop_assistant_core::ports::scratchpad::ScratchpadGetManyFn,
    > = None;

    // Writer + lister handed to the conversation handler for the step-planning +
    // context-compaction tools (#240): the writer records plan todos / distilled
    // outcomes (the same emit-wrapped closure, so changes reach clients via
    // ScratchpadChanged), the lister surfaces the open plan each round. Populated
    // alongside the builtin-tool wiring below; None without a Postgres pool.
    let mut scratchpad_write_fn: Option<
        desktop_assistant_core::ports::scratchpad::ScratchpadWriteFn,
    > = None;
    let mut scratchpad_list_fn: Option<
        desktop_assistant_core::ports::scratchpad::ScratchpadListFn,
    > = None;

    // Per-conversation scratchpad command closures (#190) for the API handler,
    // populated alongside the builtin-tool wiring below. The same emit-wrapped
    // closures the builtin tools get, so a mutation via either path notifies.
    type SpHandlerFns = (
        desktop_assistant_core::ports::scratchpad::ScratchpadWriteFn,
        desktop_assistant_core::ports::scratchpad::ScratchpadGetManyFn,
        desktop_assistant_core::ports::scratchpad::ScratchpadListFn,
        desktop_assistant_core::ports::scratchpad::ScratchpadDeleteManyFn,
        desktop_assistant_core::ports::scratchpad::ScratchpadClearFn,
    );
    let mut scratchpad_handler_fns: Option<SpHandlerFns> = None;

    if let Some(pool) = &pg_pool {
        tracing::info!("wiring database query into builtin tools");
        let pool_for_db = pool.clone();
        builtin_tools = builtin_tools.with_database(Arc::new(move |sql, limit| {
            let pool = pool_for_db.clone();
            Box::pin(async move {
                desktop_assistant_storage::execute_database_query(&pool, &sql, limit).await
            })
        }));

        // Issue #71: wire conversation full-text search.
        let cs_store = Arc::new(desktop_assistant_storage::PgConversationSearchStore::new(
            pool.clone(),
        ));
        tracing::info!("wiring conversation search into builtin tools");
        use desktop_assistant_core::ports::conversation_search::ConversationSearchStore;
        builtin_tools =
            builtin_tools.with_conversation_search(Arc::new(move |query, limit, role_filter| {
                let store = Arc::clone(&cs_store);
                Box::pin(async move { store.search_messages(&query, limit, role_filter).await })
            }));

        // Issue #184: wire the per-conversation scratchpad store.
        use desktop_assistant_core::ports::scratchpad::ScratchpadStore;
        let sp_store = Arc::new(desktop_assistant_storage::PgScratchpadStore::new(
            pool.clone(),
        ));
        tracing::info!("wiring scratchpad store into builtin tools");
        use desktop_assistant_core::ports::auth::current_user_id;
        use desktop_assistant_core::ports::scratchpad::{
            ScratchpadClearFn, ScratchpadDeleteManyFn, ScratchpadGetManyFn, ScratchpadListFn,
            ScratchpadSearchFn, ScratchpadWriteFn,
        };

        // The mutating closures (write / delete_many / clear) notify subscribed
        // connections after a successful change via `Event::ScratchpadChanged`,
        // reading the per-turn / per-command `current_user_id()`. The SAME
        // closures back both the builtin tools (Adele's writes) and the API
        // handler (client writes), so a change from either path emits once.
        let sp_w = Arc::clone(&sp_store);
        let reg_w = Arc::clone(&background_task_registry);
        let write_fn: ScratchpadWriteFn = Arc::new(move |conv: String, notes| {
            let store = Arc::clone(&sp_w);
            let reg = Arc::clone(&reg_w);
            Box::pin(async move {
                let saved = store.write(&conv, &notes).await?;
                reg.notify_scratchpad_changed(&current_user_id(), conv);
                Ok(saved)
            })
        });

        let sp_g = Arc::clone(&sp_store);
        let get_many_fn: ScratchpadGetManyFn = Arc::new(move |conv, keys, limit| {
            let store = Arc::clone(&sp_g);
            Box::pin(async move { store.get_many(&conv, &keys, limit).await })
        });

        let sp_l = Arc::clone(&sp_store);
        let list_fn: ScratchpadListFn = Arc::new(move |conv, note_type: Option<String>, limit| {
            let store = Arc::clone(&sp_l);
            Box::pin(async move { store.list(&conv, note_type.as_deref(), limit).await })
        });

        let sp_s = Arc::clone(&sp_store);
        let search_fn: ScratchpadSearchFn =
            Arc::new(move |conv, query, note_type: Option<String>, limit| {
                let store = Arc::clone(&sp_s);
                Box::pin(async move {
                    store
                        .search(&conv, &query, note_type.as_deref(), limit)
                        .await
                })
            });

        let sp_d = Arc::clone(&sp_store);
        let reg_d = Arc::clone(&background_task_registry);
        let delete_many_fn: ScratchpadDeleteManyFn = Arc::new(move |conv: String, keys| {
            let store = Arc::clone(&sp_d);
            let reg = Arc::clone(&reg_d);
            Box::pin(async move {
                let deleted = store.delete_many(&conv, &keys).await?;
                reg.notify_scratchpad_changed(&current_user_id(), conv);
                Ok(deleted)
            })
        });

        let sp_c = Arc::clone(&sp_store);
        let reg_c = Arc::clone(&background_task_registry);
        let clear_fn: ScratchpadClearFn = Arc::new(move |conv: String| {
            let store = Arc::clone(&sp_c);
            let reg = Arc::clone(&reg_c);
            Box::pin(async move {
                let deleted = store.clear(&conv).await?;
                reg.notify_scratchpad_changed(&current_user_id(), conv);
                Ok(deleted)
            })
        });

        builtin_tools = builtin_tools.with_scratchpad(
            Arc::clone(&write_fn),
            Arc::clone(&get_many_fn),
            Arc::clone(&list_fn),
            search_fn,
            Arc::clone(&delete_many_fn),
            Arc::clone(&clear_fn),
        );

        // Capture the same event-emitting write + list closures for the
        // conversation handler's planning/compaction tools (#240) before they
        // are moved into the API-handler tuple below.
        scratchpad_write_fn = Some(Arc::clone(&write_fn));
        scratchpad_list_fn = Some(Arc::clone(&list_fn));

        // Hand the same (emit-wrapped) closures to the API handler so clients
        // can read/write/delete the scratchpad over the command channel (#190).
        scratchpad_handler_fns = Some((
            write_fn,
            Arc::clone(&get_many_fn),
            list_fn,
            delete_many_fn,
            clear_fn,
        ));

        // Reader for the reserved goal note (a bounded single-key fetch),
        // consumed by `ConversationHandler::with_scratchpad_goal` below.
        scratchpad_goal_fn = Some(get_many_fn);
    }

    if let Some(tr) = &tool_registry_store {
        tracing::info!("wiring tool registry store into builtin tools");
        let tr_s = Arc::clone(tr);
        let tr_d = Arc::clone(tr);
        use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
        builtin_tools = builtin_tools.with_tool_registry(
            Arc::new(move |query, embedding, limit| {
                let store = Arc::clone(&tr_s);
                Box::pin(async move { store.search_tools(&query, embedding, limit).await })
            }),
            Arc::new(move |name| {
                let store = Arc::clone(&tr_d);
                Box::pin(async move { store.tool_definition(&name).await })
            }),
        );
    }

    let mut tool_executor = McpToolExecutor::with_builtin_tools_and_config_path(
        mcp_configs,
        builtin_tools,
        mcp_config_path,
        mcp_secrets,
    );
    let mcp_handle = tool_executor.control_handle();
    tool_executor
        .builtin_tools_mut()
        .set_mcp_control(mcp_handle.clone());
    if let Err(e) = tool_executor.start().await {
        tracing::warn!("failed to start MCP servers: {e}");
    }

    // Register discovered MCP tools in the tool registry (with embeddings)
    let registered_tools: Vec<(String, String)> = tool_executor.tools_by_service().await;
    if registered_tools.is_empty() {
        tracing::info!("MCP startup complete: no tools registered");
    } else {
        tracing::info!(
            "MCP startup complete: {} tool(s) registered",
            registered_tools.len()
        );
        for (i, (service, tool)) in registered_tools.iter().enumerate() {
            tracing::info!("  {}. [{}] {}", i + 1, service, tool);
        }
    }

    if let Some(tr) = &tool_registry_store {
        use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
        use desktop_assistant_core::ports::tools::ToolExecutor;

        // Register builtin tools as core (always sent to LLM)
        let builtin_defs: Vec<_> = tool_executor
            .core_tools()
            .await
            .into_iter()
            .filter(|t| t.name.starts_with("builtin_"))
            .collect();
        let builtin_embeddings = vec![None; builtin_defs.len()];
        if let Err(e) = tr
            .register_tools(builtin_defs, "builtin", true, builtin_embeddings, None)
            .await
        {
            tracing::warn!("failed to register builtin tools in registry: {e}");
        }

        // Register MCP tools as non-core (discoverable via tool_search)
        let mcp_defs: Vec<_> = tool_executor.all_mcp_tools().await;
        let mcp_embeddings = vec![None; mcp_defs.len()];
        if !mcp_defs.is_empty()
            && let Err(e) = tr
                .register_tools(mcp_defs, "mcp", false, mcp_embeddings, None)
                .await
        {
            tracing::warn!("failed to register MCP tools in registry: {e}");
        }
    }

    // Spawn background embedding backfill task
    let (backfill_shutdown_tx, backfill_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let backfill_task = if let (Some(pool), Some(client)) = (&pg_pool, &embedding_client) {
        let pool = pool.clone();
        let client = Arc::clone(client);
        let model = embedding_model_id.clone();
        Some(tokio::spawn(async move {
            // Let tool registration and MCP connections settle.
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
                _ = backfill_shutdown_rx => {
                    tracing::info!("embedding backfill cancelled before start");
                    return;
                }
            }

            tracing::info!("starting embedding backfill (model={model})");

            let embed_fn: desktop_assistant_storage::embedding_backfill::BackfillEmbedFn =
                Box::new(move |texts| {
                    let client = Arc::clone(&client);
                    Box::pin(async move { client.embed(texts).await.map_err(|e| e.to_string()) })
                });

            match desktop_assistant_storage::embedding_backfill::backfill_tool_embeddings(
                &pool, &embed_fn, &model,
            )
            .await
            {
                Ok(n) if n > 0 => tracing::info!("backfilled {n} tool embedding(s)"),
                Ok(_) => tracing::debug!("no tool embeddings to backfill"),
                Err(e) => tracing::warn!("tool embedding backfill failed: {e}"),
            }

            match desktop_assistant_storage::embedding_backfill::backfill_knowledge_embeddings(
                &pool, &embed_fn, &model,
            )
            .await
            {
                Ok(n) if n > 0 => tracing::info!("backfilled {n} knowledge embedding(s)"),
                Ok(_) => tracing::debug!("no knowledge embeddings to backfill"),
                Err(e) => tracing::warn!("knowledge embedding backfill failed: {e}"),
            }
        }))
    } else {
        drop(backfill_shutdown_rx);
        None
    };

    // Spawn background dreaming (periodic fact extraction) task
    let dreaming_enabled = daemon_config
        .as_ref()
        .map(|c| c.backend_tasks.dreaming_enabled)
        .unwrap_or(false);
    let dreaming_interval_secs = daemon_config
        .as_ref()
        .map(|c| c.backend_tasks.dreaming_interval_secs)
        .unwrap_or(3600);
    let archive_after_days = daemon_config
        .as_ref()
        .map(|c| c.backend_tasks.archive_after_days)
        .unwrap_or(7);

    let (dreaming_shutdown_tx, dreaming_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let dreaming_task = if dreaming_enabled {
        if let (Some(pool), Some(emb_client)) = (&pg_pool, &embedding_client) {
            // Prefer `[purposes.dreaming]` when configured; fall back to
            // the legacy `[backend_tasks.llm]` block otherwise so installs
            // that haven't migrated still work. Effort threading is
            // computed once at startup and copied into the closure — the
            // resolved purpose is fixed for this daemon run, and
            // `ReasoningConfig` is `Copy`.
            let (resolved_dreaming, dreaming_reasoning, source) =
                match api_surface::resolve_purpose_dispatch(
                    daemon_config.as_ref(),
                    purposes::PurposeKind::Dreaming,
                ) {
                    Some((r, c)) => (r, c, "purposes.dreaming"),
                    None => (
                        config::resolve_backend_tasks_llm_config(daemon_config.as_ref()),
                        Default::default(),
                        "backend_tasks.llm",
                    ),
                };
            tracing::info!(
                "dreaming LLM connector={}, model={}, source={}",
                resolved_dreaming.connector,
                resolved_dreaming.model,
                source
            );

            let dreaming_llm = build_llm_client(resolved_dreaming);
            let dreaming_llm = RetryingLlmClient::new(dreaming_llm, 3);
            let dreaming_llm = MaybeProfiled::from_config(
                dreaming_llm,
                profiling.enabled,
                profiling.log_path.as_deref(),
                profiling.full_content,
            );
            let dreaming_llm = Arc::new(dreaming_llm);

            let pool = pool.clone();
            let emb_client = Arc::clone(emb_client);
            let emb_model = embedding_model_id.clone();

            Some(tokio::spawn(async move {
                let mut shutdown_rx = dreaming_shutdown_rx;

                // Initial delay — let startup settle.
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {}
                    _ = &mut shutdown_rx => {
                        tracing::info!("dreaming cancelled before first scan");
                        return;
                    }
                }

                let llm_fn: desktop_assistant_storage::dreaming::DreamingLlmFn =
                    Box::new(move |system_prompt, user_prompt| {
                        let llm = Arc::clone(&dreaming_llm);
                        let reasoning = dreaming_reasoning;
                        Box::pin(async move {
                            let messages = vec![
                                Message::new(Role::System, system_prompt),
                                Message::new(Role::User, user_prompt),
                            ];
                            let response = llm
                                .stream_completion(messages, &[], reasoning, Box::new(|_| true))
                                .await
                                .map_err(|e| e.to_string())?;
                            Ok(response.text)
                        })
                    });

                let embed_fn: desktop_assistant_storage::dreaming::BackfillEmbedFn =
                    Box::new(move |texts| {
                        let client = Arc::clone(&emb_client);
                        Box::pin(
                            async move { client.embed(texts).await.map_err(|e| e.to_string()) },
                        )
                    });

                loop {
                    tracing::info!("dreaming: starting scan cycle");
                    let cycle_start = std::time::Instant::now();
                    let result = desktop_assistant_storage::dreaming::run_dreaming_scan(
                        &pool,
                        &llm_fn,
                        &embed_fn,
                        &emb_model,
                        archive_after_days,
                    )
                    .await;
                    let elapsed = cycle_start.elapsed();
                    match result {
                        Ok(n) => tracing::info!(
                            "dreaming: scan cycle finished in {elapsed:.2?}, wrote {n} new fact(s)"
                        ),
                        Err(e) => {
                            tracing::warn!("dreaming: scan cycle failed after {elapsed:.2?}: {e}")
                        }
                    }

                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs(dreaming_interval_secs)) => {}
                        _ = &mut shutdown_rx => {
                            tracing::info!("dreaming: shutdown signal received");
                            break;
                        }
                    }
                }
            }))
        } else {
            if pg_pool.is_none() {
                tracing::warn!("dreaming enabled but no database configured; dreaming disabled");
            } else {
                tracing::warn!("dreaming enabled but embeddings unavailable; dreaming disabled");
            }
            drop(dreaming_shutdown_rx);
            None
        }
    } else {
        tracing::debug!("dreaming disabled");
        drop(dreaming_shutdown_rx);
        None
    };

    // Spawn the interactive-model keep-warm task. When the interactive
    // purpose resolves to an Ollama connection with `keep_warm = true`, a
    // light background loop re-loads that one model into Ollama's memory on a
    // cadence shorter than Ollama's idle-unload window, so a chat reply isn't
    // preceded by a cold (CPU-bound, possibly minutes-long) model load. Only
    // the interactive model is kept warm; background purposes are left to
    // unload when idle. No-op for non-Ollama interactive connections.
    let (keep_warm_shutdown_tx, mut keep_warm_shutdown_rx) =
        tokio::sync::oneshot::channel::<()>();
    let keep_warm_task = {
        let resolved_interactive = api_surface::resolve_purpose_dispatch(
            daemon_config.as_ref(),
            purposes::PurposeKind::Interactive,
        )
        .map(|(r, _)| r);
        match resolved_interactive {
            Some(r) if r.connector == "ollama" && r.keep_warm => {
                let base_url = r.base_url.clone();
                let model = r.model.clone();
                // Re-load well within Ollama's default 5-minute idle-unload
                // window (an interactive turn resets keep_alive to that
                // default), so the model never lapses between turns.
                const KEEP_WARM_INTERVAL_SECS: u64 = 240;
                tracing::info!(
                    model = %model,
                    base_url = %base_url,
                    interval_secs = KEEP_WARM_INTERVAL_SECS,
                    "ollama keep-warm enabled for interactive model"
                );
                Some(tokio::spawn(async move {
                    let client = desktop_assistant_llm_ollama::OllamaClient::new(
                        base_url,
                        model.clone(),
                    );
                    loop {
                        client.warm_model(&model).await;
                        tokio::select! {
                            _ = tokio::time::sleep(std::time::Duration::from_secs(
                                KEEP_WARM_INTERVAL_SECS,
                            )) => {}
                            _ = &mut keep_warm_shutdown_rx => {
                                tracing::info!("ollama keep-warm: shutdown signal received");
                                break;
                            }
                        }
                    }
                }))
            }
            _ => {
                drop(keep_warm_shutdown_rx);
                None
            }
        }
    };

    // Build the conversation service with tool support. The store is shared
    // between the core `ConversationHandler` (for CRUD + append) and the
    // `RoutingConversationHandler` wrapper (for the per-conversation model
    // selection column, #11) so we wrap it in
    // `Arc<SharedConversationStore>` (a local newtype that lets us impl
    // `ConversationStore` for the Arc despite the orphan rule).
    let inner_store: AnyConversationStore = if let Some(pool) = &pg_pool {
        tracing::info!("using PostgreSQL conversation store");
        AnyConversationStore::Postgres(desktop_assistant_storage::PgConversationStore::new(
            pool.clone(),
        ))
    } else {
        let store = PersistentConversationStore::from_default_path().map_err(|e| {
            anyhow::anyhow!("failed to initialize persistent conversation store: {e}")
        })?;
        tracing::info!(
            "using JSON conversation store at {}",
            store::default_conversation_store_path().display()
        );
        AnyConversationStore::Json(store)
    };
    let conversation_store = SharedConversationStore(Arc::new(inner_store));

    // Wrap the interactive-purpose client in a `RoutingLlmClient`. The
    // routing wrapper (`api_surface::RoutingConversationHandler`) installs
    // a task-local per turn; when present, dispatch picks the registry's
    // client for the resolved connection id. When absent (backend tasks,
    // legacy callers without an override), the routing client falls back
    // to this interactive-purpose client.
    let fallback_client = Arc::new(llm);
    let llm = routing_llm::RoutingLlmClient::new(Arc::clone(&fallback_client));
    // Wrap the primary in a transparent `FixedReasoningLlmClient` whose
    // override is `default()`. The interactive dispatch path goes through
    // the per-turn task-local installed by `RoutingConversationHandler`,
    // which calls `stream_completion` with its mapped `ReasoningConfig` —
    // we must not stomp on that, hence the passthrough configuration.
    // The wrapper exists here only so the primary and backend handlers
    // share the same `L` type (backend tasks need a non-default override,
    // and `with_backend_llm(L)` requires both stacks to match).
    let llm = backend_reasoning::FixedReasoningLlmClient::new(llm, ReasoningConfig::default());
    let llm = RetryingLlmClient::new(llm, 3);
    // Erase the decorator stack to `Arc<dyn LlmClient>` (#207). The inner
    // type — `MaybeProfiled<Retrying<FixedReasoning<RoutingLlmClient>>>` —
    // was previously carried by value as the `L` of `ConversationHandler`,
    // so it monomorphized into the per-turn future and was a large part of
    // the multi-MB frame that overflowed the worker stack (#205/#206).
    // Behind a trait object the handler holds a thin pointer instead, and
    // the primary and backend slots share `L = Arc<dyn LlmClient>` for free
    // (the `FixedReasoning` passthrough above is no longer needed to make
    // the two slots' types match, but is left as a harmless no-op).
    let llm: Arc<dyn LlmClient> = Arc::new(MaybeProfiled::from_config(
        llm,
        profiling.enabled,
        profiling.log_path.as_deref(),
        profiling.full_content,
    ));
    let mut handler = ConversationHandler::with_tools(
        conversation_store.clone(),
        llm,
        tool_executor,
        Box::new(|| uuid::Uuid::now_v7().to_string()),
    )
    // Server-side tool localities (#243) are labelled with the daemon's host
    // identity (the hostname — a human-readable display label for the tool
    // note). Co-location itself is decided by the per-machine system-id
    // handshake (#248, wired into the WS/UDS frontends below), not this label.
    .with_host(daemon_host_label());

    // Build the shared registry handle (#11): wraps the in-memory
    // `ConnectionRegistry` plus the loaded `DaemonConfig` behind a single
    // `RwLock` so the connections-management API can mutate config + rebuild
    // the registry atomically. Constructed before the backend-task wiring
    // (#68) so the dynamic-purpose `RoutingLlmClient` can read live config
    // on every call.
    let registry_handle = Arc::new(
        api_surface::RegistryHandle::new(
            daemon_config.clone().unwrap_or_default(),
            connection_registry,
        )
        .with_config_path(config_path.clone()),
    );

    // State-preserving config hot-reload (#222). The D-Bus `Reload` method and
    // the config-file watcher both ping this bounded channel; one consumer task
    // coalesces a burst and calls `RegistryHandle::apply_reload`, which
    // validates the new config and swaps the registry under its lock — new turns
    // get the new clients while in-flight turns keep theirs alive by refcount. A
    // bad config is refused and the last-good config keeps running. The channel
    // is bounded (depth 4) so a flood of edits can't grow it without limit; the
    // consumer drains the queue before each apply.
    let (reload_tx, reload_rx) = tokio::sync::mpsc::channel::<()>(4);
    spawn_reload_consumer(Arc::clone(&registry_handle), reload_rx);
    spawn_config_watcher(config_path.clone(), reload_tx.clone());

    // Wire the learned (tier 2) and LLM (tier 3) tiers of the backend-error
    // classifier (#178). The deterministic tier-1 matchers already run inside
    // every `ClassifyingLlmClient` built by `build_llm_client`; here we install
    // the process-wide store and a cheap classifier LLM (the titling-purpose
    // routing client) so genuinely novel errors are learned. Requires a
    // database — tier 1 keeps working without one. The reentrancy guard keeps
    // the (itself-wrapped) classifier from recursing.
    if let Some(pool) = &pg_pool {
        let store: Arc<dyn desktop_assistant_core::ports::store::ErrorClassificationStore> =
            Arc::new(desktop_assistant_storage::PgErrorClassificationStore::new(
                pool.clone(),
            ));
        let classifier: Arc<dyn LlmClient> =
            Arc::new(routing_llm::RoutingLlmClient::new_dynamic_purpose(
                Arc::clone(&registry_handle),
                purposes::PurposeKind::Titling,
            ));
        // #343: the learned context-window cache (DOWN-only safety net). Shares
        // the same pool; absent without a database (window learning off, tier-1
        // classification still works).
        let window_store: Arc<dyn desktop_assistant_core::ports::store::LearnedWindowStore> =
            Arc::new(desktop_assistant_storage::PgLearnedWindowStore::new(
                pool.clone(),
            ));
        classifying_llm::install_classification_deps(classifying_llm::ClassificationDeps {
            store,
            classifier: Some(classifier),
            window_store: Some(window_store),
        });
    }

    // Build a separate LLM for backend tasks (title generation, context summary).
    //
    // Resolution order:
    //   1. `[purposes.titling]` — if set, install a dynamic-purpose client
    //      that resolves the connection/model/effort from the live config
    //      on every call. Control-panel edits take effect on the next
    //      backend dispatch with no daemon restart.
    //   2. `[backend_tasks.llm]` legacy block — install a static client
    //      only if it differs from the primary, so unmigrated installs
    //      that haven't authored a `[purposes]` table still work. The
    //      legacy path stays static; authors are expected to move to
    //      `[purposes.titling]`.
    let resolved_primary = config::resolve_llm_config(daemon_config.as_ref());
    let titling_configured = daemon_config
        .as_ref()
        .and_then(|c| c.purposes.get(purposes::PurposeKind::Titling))
        .is_some();
    if titling_configured {
        tracing::info!("backend-tasks LLM source=purposes.titling (dynamic resolution per call)");
        let bt_llm = routing_llm::RoutingLlmClient::new_dynamic_purpose(
            Arc::clone(&registry_handle),
            purposes::PurposeKind::Titling,
        );
        // Wrap in `FixedReasoningLlmClient(default)` purely so the
        // backend slot's `L` matches the primary slot's `L` —
        // `with_backend_llm(L)` requires both to be the same type. The
        // dynamic-purpose dispatch path overrides reasoning internally,
        // so the wrapper is a transparent passthrough here.
        let bt_llm =
            backend_reasoning::FixedReasoningLlmClient::new(bt_llm, ReasoningConfig::default());
        let bt_llm = RetryingLlmClient::new(bt_llm, 3);
        let bt_llm: Arc<dyn LlmClient> = Arc::new(MaybeProfiled::from_config(
            bt_llm,
            profiling.enabled,
            profiling.log_path.as_deref(),
            profiling.full_content,
        ));
        handler = handler.with_backend_llm(bt_llm);
    } else {
        let resolved_bt = config::resolve_backend_tasks_llm_config(daemon_config.as_ref());
        if resolved_bt.connector != resolved_primary.connector
            || resolved_bt.model != resolved_primary.model
        {
            tracing::info!(
                "backend-tasks LLM connector={}, model={}, source=backend_tasks.llm",
                resolved_bt.connector,
                resolved_bt.model
            );
            let bt_llm = build_llm_client(resolved_bt);
            let bt_fallback = Arc::new(bt_llm);
            let bt_llm = routing_llm::RoutingLlmClient::new(bt_fallback);
            let bt_llm =
                backend_reasoning::FixedReasoningLlmClient::new(bt_llm, ReasoningConfig::default());
            let bt_llm = RetryingLlmClient::new(bt_llm, 3);
            let bt_llm: Arc<dyn LlmClient> = Arc::new(MaybeProfiled::from_config(
                bt_llm,
                profiling.enabled,
                profiling.log_path.as_deref(),
                profiling.full_content,
            ));
            handler = handler.with_backend_llm(bt_llm);
        }
    }

    // Surface the evolving scratchpad `goal` note as the per-turn task anchor
    // (#184). No-op when no Postgres pool is available.
    if let Some(goal_fn) = scratchpad_goal_fn {
        handler = handler.with_scratchpad_goal(goal_fn);
    }

    // Enable step-planning + context compaction (#240): the writer records plan
    // todos and distilled step outcomes (emitting ScratchpadChanged); the lister
    // surfaces the open plan each round. No-op without a Postgres pool.
    if let Some(write_fn) = scratchpad_write_fn {
        handler = handler.with_scratchpad_write(write_fn);
    }
    if let Some(list_fn) = scratchpad_list_fn {
        handler = handler.with_scratchpad_list(list_fn);
    }

    // Wrap the core `ConversationHandler` in the routing wrapper so adapters
    // can call `send_prompt_with_override` and have the override/stored-
    // selection priority path applied.
    let inner_conv = Arc::new(handler);
    // #343: the learned context-window cache also feeds budget resolution
    // (caps DOWN to an observed-overflow ceiling). Same pool as the classifier
    // side; absent without a database (safety net off, resolution unchanged).
    let mut routing_conv = api_surface::RoutingConversationHandler::new(
        Arc::clone(&inner_conv),
        Arc::new(conversation_store),
        Arc::clone(&registry_handle),
    );
    if let Some(pool) = &pg_pool {
        let window_store: Arc<dyn desktop_assistant_core::ports::store::LearnedWindowStore> =
            Arc::new(desktop_assistant_storage::PgLearnedWindowStore::new(
                pool.clone(),
            ));
        routing_conv = routing_conv.with_window_store(window_store);
    }
    let conversation_service = Arc::new(routing_conv);

    let connections_service = Arc::new(api_surface::DaemonConnectionsService::new(Arc::clone(
        &registry_handle,
    )));

    let settings_service = Arc::new(
        DaemonSettingsService::new(config_path.clone())
            .with_mcp_control(mcp_handle)
            // Personality (#226) reads/writes through the shared registry handle
            // so settings changes hit the in-memory config the dispatch reads.
            .with_registry(Arc::clone(&registry_handle)),
    );

    // Knowledge management service (#73). When a Postgres pool is
    // configured, wire the embedding closure so client-authored entries
    // are discoverable by the LLM tool. Without a pool, every method
    // surfaces a uniform "not configured" error.
    let knowledge_service = Arc::new(match (&kb_store, embedding_fn_for_kb_service.clone()) {
        (Some(store), embed_fn) => {
            tracing::info!("knowledge management service ready");
            knowledge_service::AnyKnowledgeService::Configured(
                knowledge_service::DaemonKnowledgeService::new(
                    Arc::clone(store),
                    embed_fn,
                    Some(embedding_model_id.clone()),
                ),
            )
        }
        (None, _) => {
            tracing::info!("knowledge management service unavailable (no Postgres pool)");
            knowledge_service::AnyKnowledgeService::Unconfigured(
                knowledge_service::UnconfiguredKnowledgeService,
            )
        }
    });

    // Construct the shared API handler up-front so both the D-Bus and WS
    // adapters can share it (the multi-connection D-Bus interface dispatches
    // through this handler, mirroring the WS adapter).
    //
    // The handler is wired with the shared [`BackgroundTaskRegistry`]
    // (constructed earlier, before the scratchpad wiring) so foreground turns
    // register as `TaskKind::Conversation` tasks (#111), and — when a Postgres
    // pool is available — with the per-conversation scratchpad command closures
    // (#190) so clients can read/write/delete a conversation's notes.
    let mut api_handler_impl = DefaultAssistantApiHandler::new(
        Arc::new(Assistant),
        Arc::clone(&conversation_service),
        Arc::clone(&settings_service),
        Arc::clone(&connections_service),
        Arc::clone(&knowledge_service),
    )
    .with_registry(Arc::clone(&background_task_registry));
    if let Some((write, get_many, list, delete_many, clear)) = scratchpad_handler_fns {
        api_handler_impl =
            api_handler_impl.with_scratchpad(write, get_many, list, delete_many, clear);
    }
    // Idempotency-key dedup (#204): when a database is available, attach the
    // store so a retried `SendMessage` carrying an `idempotency_key` whose turn
    // already completed replays the stored reply instead of re-running it.
    // Without a pool the key is a harmless no-op.
    if let Some(pool) = &pg_pool {
        let idempotency_store: Arc<dyn desktop_assistant_core::ports::store::IdempotencyKeyStore> =
            Arc::new(desktop_assistant_storage::PgIdempotencyKeyStore::new(
                pool.clone(),
            ));
        api_handler_impl = api_handler_impl.with_idempotency_store(idempotency_store);
    }
    // Client-side tool execution (#107 / #234): one shared coordinator plus an
    // in-memory turn-state store, so `RegisterClientTools` / `ClientToolResult`
    // are served and the LLM can invoke client-local tools (suspending the turn
    // and resuming on the client's `ClientToolResult`). The in-memory store is
    // sufficient for the live single-process deploy; a DB-backed
    // `TurnStateStore` for crash-recovery is Phase-2 follow-up work.
    let client_tool_coordinator =
        Arc::new(desktop_assistant_application::client_tools::ClientToolCoordinator::new());
    let turn_state_store: Arc<dyn desktop_assistant_core::ports::store::TurnStateStore> =
        Arc::new(desktop_assistant_application::client_tools::InMemoryTurnStateStore::new());
    api_handler_impl = api_handler_impl.with_client_tool_coordinator(
        Arc::clone(&client_tool_coordinator),
        Arc::clone(&turn_state_store),
    );
    let api_handler: Arc<dyn desktop_assistant_application::AssistantApiHandler> =
        Arc::new(api_handler_impl);

    let dbus_service_name = std::env::var("DESKTOP_ASSISTANT_DBUS_SERVICE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| transports_config.dbus_service.clone());
    let dbus_required = env_bool(
        "DESKTOP_ASSISTANT_DBUS_REQUIRED",
        transports_config.dbus_required,
    );
    tracing::info!("D-Bus well-known name={dbus_service_name}");
    tracing::info!("D-Bus required={dbus_required}");

    // Set up D-Bus connection (required by default; optional in headless/container mode).
    let dbus_connection = match zbus::connection::Builder::session() {
        Ok(builder) => match builder
            .name(dbus_service_name.as_str())
            .and_then(|b| {
                b.serve_at(
                    "/org/desktopAssistant/Conversations",
                    DbusConversationAdapter::new(Arc::clone(&conversation_service)),
                )
            })
            .and_then(|b| {
                b.serve_at(
                    "/org/desktopAssistant/Settings",
                    DbusSettingsAdapter::new(Arc::clone(&settings_service)),
                )
            })
            .and_then(|b| {
                b.serve_at(
                    "/org/desktopAssistant/Connections",
                    desktop_assistant_dbus::connections::DbusConnectionsAdapter::new(Arc::clone(
                        &api_handler,
                    )),
                )
            })
            .and_then(|b| {
                b.serve_at(
                    "/org/desktopAssistant/Knowledge",
                    desktop_assistant_dbus::knowledge::DbusKnowledgeAdapter::new(Arc::clone(
                        &api_handler,
                    )),
                )
            })
            .and_then(|b| {
                // Generic command channel (#213): the shared `AssistantCommands`
                // surface over D-Bus, dispatching through the same handler the
                // socket transports use, so `TransportClient::as_commands` works
                // on every transport.
                b.serve_at(
                    "/org/desktopAssistant/Commands",
                    desktop_assistant_dbus::commands::DbusCommandsAdapter::new(Arc::clone(
                        &api_handler,
                    )),
                )
            })
            .and_then(|b| {
                // Hot-reload trigger (#222): the KCM calls `Reload` after
                // writing daemon.toml so changes apply without a restart.
                b.serve_at(
                    "/org/desktopAssistant/Reload",
                    DbusReloadAdapter::new(reload_tx.clone()),
                )
            }) {
            Ok(builder) => match builder.build().await {
                Ok(connection) => {
                    if let Some(unique_name) = connection.unique_name() {
                        tracing::info!("D-Bus service registered at {}", unique_name);
                    } else {
                        tracing::info!("D-Bus service registered");
                    }
                    Some(connection)
                }
                Err(error) => {
                    if dbus_required {
                        return Err(error.into());
                    }
                    tracing::warn!(
                        "D-Bus unavailable; continuing without D-Bus API (set DESKTOP_ASSISTANT_DBUS_REQUIRED=true to fail): {error}"
                    );
                    None
                }
            },
            Err(error) => {
                if dbus_required {
                    return Err(error.into());
                }
                tracing::warn!(
                    "failed to configure D-Bus interface; continuing without D-Bus API (set DESKTOP_ASSISTANT_DBUS_REQUIRED=true to fail): {error}"
                );
                None
            }
        },
        Err(error) => {
            if dbus_required {
                return Err(error.into());
            }
            tracing::warn!(
                "failed to connect to session D-Bus; continuing without D-Bus API (set DESKTOP_ASSISTANT_DBUS_REQUIRED=true to fail): {error}"
            );
            None
        }
    };

    // The daemon's own per-machine system id for the tool-locality co-location
    // handshake (#248), read once at startup. Each client reports its id in the
    // connect handshake (UDS frame field / WS upgrade header); when it equals
    // this one they're the same machine ⇒ co-located, even over WebSocket. The
    // id is a routing HINT, not a trust boundary — no privilege is gated on it
    // (auth remains the JWT). `None` ⇒ co-location falls back to the transport
    // heuristic for every connection (Phase-1, #243). Shared by both the WS and
    // UDS frontends below.
    let daemon_system_id = desktop_assistant_core::system_id::local_system_id();
    match &daemon_system_id {
        Some(id) => {
            tracing::info!(system_id = %id, "tool-locality co-location: daemon system id resolved")
        }
        None => tracing::warn!(
            "tool-locality co-location: could not resolve a daemon system id; \
             falling back to the transport heuristic"
        ),
    }

    // Auth validator: OIDC-aware if configured, otherwise local-only. Built
    // unconditionally because the UDS frontend reuses it even when the
    // WebSocket listener is disabled.
    let ws_auth_config = config::get_ws_auth_settings(&config_path).ok();
    let oidc_config = ws_auth_config
        .as_ref()
        .and_then(|c| c.oidc.clone())
        .filter(|_| {
            ws_auth_config
                .as_ref()
                .map(|c| c.methods.contains(&"oidc".to_string()))
                .unwrap_or(false)
        });

    let ws_auth: Arc<dyn ws::WsAuthValidator> = if let Some(oidc) = &oidc_config {
        match config::OidcValidator::from_config(oidc).await {
            Ok(oidc_validator) => {
                tracing::info!("OIDC JWT validation enabled (issuer={})", oidc.issuer_url);
                Arc::new(OidcAwareAuth {
                    local: WsSettingsAuth::new(Arc::clone(&settings_service)),
                    oidc_validator,
                })
            }
            Err(e) => {
                tracing::warn!(
                    "failed to initialize OIDC validator: {e}; falling back to local JWT only"
                );
                Arc::new(WsSettingsAuth::new(Arc::clone(&settings_service)))
            }
        }
    } else {
        Arc::new(WsSettingsAuth::new(Arc::clone(&settings_service)))
    };

    // WebSocket API (remote-friendly). OFF by default: the daemon is
    // local-first (D-Bus minter + UDS), so the remote WebSocket endpoint —
    // and its TLS/login/origin machinery — is opt-in via
    // DESKTOP_ASSISTANT_WS_ENABLED=true.
    let ws_enabled = env_bool("DESKTOP_ASSISTANT_WS_ENABLED", transports_config.ws_enabled);
    let (ws_shutdown_tx, ws_task) = if ws_enabled {
        let ws_bind = std::env::var("DESKTOP_ASSISTANT_WS_BIND")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| transports_config.ws_bind.clone());
        let ws_addr: std::net::SocketAddr = ws_bind
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid DESKTOP_ASSISTANT_WS_BIND '{ws_bind}': {e}"))?;

        // Build auth discovery provider
        let auth_discovery: Option<Arc<dyn ws::WsAuthDiscovery>> =
            match config::get_ws_auth_discovery(&config_path) {
                Ok(discovery) => {
                    tracing::info!("auth discovery: methods={:?}", discovery.methods);
                    Some(Arc::new(WsAuthDiscoveryProvider { discovery }))
                }
                Err(e) => {
                    tracing::warn!("failed to load auth discovery config: {e}");
                    None
                }
            };

        let ws_login_service: Option<Arc<dyn ws::WsLoginService>> =
            resolve_ws_login_mode().map(|(username, mode)| {
                match &mode {
                    WsLoginMode::StaticPassword(_) => {
                        tracing::info!(
                            "Web login enabled (env-password mode) for username={username}"
                        );
                    }
                    WsLoginMode::SystemPassword => {
                        tracing::info!(
                            "Web login enabled (local system-password mode) for username={username}"
                        );
                    }
                }

                Arc::new(WsBasicLogin::new(
                    Arc::clone(&settings_service),
                    username,
                    mode,
                )) as Arc<dyn ws::WsLoginService>
            });
        if ws_login_service.is_none() {
            tracing::warn!(
                "Web login disabled: set DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD or enable local auth via DESKTOP_ASSISTANT_WS_LOGIN_LOCAL_SYSTEM_AUTH=true"
            );
        }

        let allowed_origins = ws_auth_config
            .as_ref()
            .map(|c| c.allowed_origins.clone())
            .unwrap_or_default();
        if allowed_origins.is_empty() {
            tracing::info!(
                "WebSocket origin policy: browser clients blocked (no allowed_origins configured)"
            );
        } else {
            tracing::info!("WebSocket allowed origins: {allowed_origins:?}");
        }

        // TLS configuration
        let tls_config = daemon_config
            .as_ref()
            .map(|c| c.tls.clone())
            .unwrap_or_default();
        let tls_env_override = std::env::var("DESKTOP_ASSISTANT_WS_TLS")
            .ok()
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "false" | "0" | "no"));
        let tls_enabled = tls_env_override.unwrap_or(tls_config.enabled);

        let tls_acceptor = if tls_enabled {
            match tls::setup(
                tls_config.cert_file.as_deref(),
                tls_config.key_file.as_deref(),
            ) {
                Ok(server_config) => {
                    tracing::info!(
                        "TLS enabled; CA cert at {}",
                        tls::default_ca_cert_path().display()
                    );
                    Some(tokio_rustls::TlsAcceptor::from(server_config))
                }
                Err(e) => {
                    tracing::error!("TLS setup failed: {e:#}; falling back to plain ws://");
                    None
                }
            }
        } else {
            tracing::info!("TLS disabled; serving plain ws://");
            None
        };

        let (ws_shutdown_tx, ws_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let ws_task = {
            let api_handler = Arc::clone(&api_handler);
            let ws_auth = Arc::clone(&ws_auth);
            // The daemon system id (#248) so the WS upgrade handler can compare
            // it to the client-reported header and co-locate same-machine WS
            // connections.
            let ws_daemon_system_id = daemon_system_id.clone();
            tokio::spawn(async move {
                let shutdown = async {
                    let _ = ws_shutdown_rx.await;
                };
                let ws_config = ws::WsServeConfig::new(api_handler, ws_auth)
                    .with_login_service(ws_login_service)
                    .with_auth_discovery(auth_discovery)
                    .with_allowed_origins(allowed_origins)
                    .with_daemon_system_id(ws_daemon_system_id);
                let result = if let Some(acceptor) = tls_acceptor {
                    tracing::info!("WebSocket listening on wss://{ws_addr} (/ws, /auth/config)");
                    ws_config.serve_tls(acceptor, ws_addr, shutdown).await
                } else {
                    tracing::info!("WebSocket listening on ws://{ws_addr} (/ws, /auth/config)");
                    ws_config.serve(ws_addr, shutdown).await
                };
                if let Err(e) = result {
                    tracing::error!("WebSocket server error: {e}");
                }
            })
        };
        (Some(ws_shutdown_tx), Some(ws_task))
    } else {
        tracing::info!(
            "WebSocket frontend disabled (set DESKTOP_ASSISTANT_WS_ENABLED=true to expose the remote WebSocket API)"
        );
        (None, None)
    };

    // UDS frontend (#103). Local clients (D-Bus bridge, CLI, future
    // minter shim) connect over the same JSON wire format. On by default
    // for Unix targets; suppress via DESKTOP_ASSISTANT_UDS_ENABLED=false
    // or by setting DESKTOP_ASSISTANT_UDS_SOCKET to empty.
    let uds_enabled = env_bool(
        "DESKTOP_ASSISTANT_UDS_ENABLED",
        transports_config.uds_enabled,
    );
    let uds_path = if uds_enabled {
        resolve_uds_socket_path(transports_config.uds_socket.as_deref())
    } else {
        None
    };
    let (uds_shutdown_tx, uds_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let uds_task = match uds_path {
        Some(path) => {
            let api_handler = Arc::clone(&api_handler);
            let ws_auth_for_uds = Arc::clone(&ws_auth);
            // The daemon system id (#248) so the UDS handshake handler can
            // compare it to the client-reported field. (A UDS client is already
            // local, so this mostly refines the label; included for symmetry and
            // so a mismatch — e.g. a namespaced peer — is honoured.)
            let uds_daemon_system_id = daemon_system_id.clone();
            tracing::info!("UDS listening on {}", path.display());
            Some(tokio::spawn(async move {
                let auth: Arc<dyn uds::UdsAuthValidator> =
                    Arc::new(WsAsUdsAuth::new(ws_auth_for_uds));
                let config =
                    uds::UdsServerConfig::new(path).with_daemon_system_id(uds_daemon_system_id);
                let server = uds::UdsServer::new(api_handler, auth, config);
                let shutdown = async {
                    let _ = uds_shutdown_rx.await;
                };
                if let Err(e) = server.serve_with_shutdown(shutdown).await {
                    tracing::error!("UDS server error: {e}");
                }
            }))
        }
        None => {
            tracing::info!("UDS frontend disabled");
            drop(uds_shutdown_rx);
            None
        }
    };

    // Run until stopped.
    shutdown_signal().await;
    tracing::info!("shutdown signal received; stopping services");

    let _ = backfill_shutdown_tx.send(());
    if let Some(task) = backfill_task
        && let Err(e) = task.await
    {
        tracing::warn!("backfill task join error during shutdown: {e}");
    }

    let _ = dreaming_shutdown_tx.send(());
    if let Some(task) = dreaming_task
        && let Err(e) = task.await
    {
        tracing::warn!("dreaming task join error during shutdown: {e}");
    }

    let _ = keep_warm_shutdown_tx.send(());
    if let Some(task) = keep_warm_task
        && let Err(e) = task.await
    {
        tracing::warn!("keep-warm task join error during shutdown: {e}");
    }

    if let Some(tx) = ws_shutdown_tx {
        let _ = tx.send(());
    }
    if let Some(task) = ws_task
        && let Err(e) = task.await
    {
        tracing::warn!("WebSocket task join error during shutdown: {e}");
    }

    let _ = uds_shutdown_tx.send(());
    if let Some(task) = uds_task
        && let Err(e) = task.await
    {
        tracing::warn!("UDS task join error during shutdown: {e}");
    }

    drop(dbus_connection);

    Ok(())
}
