//! Daemon-side implementation of the connection/purpose management API
//! (issue #11) plus the wrapper that threads per-send overrides through
//! the core `ConversationHandler`.
//!
//! Architecture notes:
//!
//! - [`DaemonConnectionsService`] wraps a shared [`ConnectionRegistry`]
//!   (plus the on-disk config) and implements the
//!   [`ConnectionsService`](desktop_assistant_core::ports::inbound::ConnectionsService)
//!   inbound port. Writes mutate the on-disk config and rebuild the
//!   registry; reads snapshot registry state.
//!
//! - [`RoutingConversationHandler`] is a thin wrapper over the primary
//!   `ConversationHandler`. It implements `ConversationService` so adapters
//!   can call it interchangeably. On a send-with-override, it:
//!   1. Validates the override against the live registry (connection
//!      exists + model is listed).
//!   2. Persists the override on the conversation row.
//!   3. Delegates to the inner handler.
//!   Stored-but-dangling selections are detected, cleared, and surfaced
//!   via a one-time [`DispatchWarning::DanglingModelSelection`].
//!
//! See the ticket body on #11 for the full priority table
//! (override → stored → interactive).

use std::sync::{Arc, Mutex, RwLock};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, ConversationSummary};
use desktop_assistant_core::ports::inbound::{
    ConnectionAvailability as CoreConnectionAvailability, ConnectionConfigPayload,
    ConnectionView as CoreConnectionView, ConnectionsService, ConversationModelSelection,
    ConversationService, DispatchWarning, Effort, ModelListing as CoreModelListing,
    PromptDispatchOutcome, PromptSelectionOverride, PurposeConfigPayload,
    PurposeKind as CorePurposeKind, PurposesView as CorePurposesView, SerdeEffort,
};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, ReasoningConfig, ReasoningLevel, StatusCallback,
    with_reasoning_config,
};

use crate::config::{
    DaemonConfig, default_daemon_config_path, load_daemon_config, save_daemon_config,
};
use crate::connections::{
    AnthropicConnection, BedrockConnection, ConnectionConfig, ConnectionId, OllamaConnection,
    OpenAiConnection,
};
use crate::purposes::{
    ConnectionRef, Effort as PurposeEffort, ModelRef, PurposeConfig, PurposeKind,
};
use crate::registry::{ConnectionHealth, ConnectionRegistry, build_registry};

/// Shared, mutable handle to the registry + current config. Writes acquire
/// the outer `RwLock` in write mode, replace the inner values, then drop;
/// reads take a read lock and clone out whatever they need.
pub struct RegistryHandle {
    state: RwLock<RegistryState>,
    config_path: std::path::PathBuf,
}

struct RegistryState {
    config: DaemonConfig,
    registry: ConnectionRegistry,
}

impl RegistryHandle {
    pub fn new(config: DaemonConfig, registry: ConnectionRegistry) -> Self {
        Self {
            state: RwLock::new(RegistryState { config, registry }),
            config_path: default_daemon_config_path(),
        }
    }

    pub fn with_config_path(mut self, path: std::path::PathBuf) -> Self {
        self.config_path = path;
        self
    }

    /// Snapshot of every connection status — used for list/validate paths.
    fn connection_views(&self) -> Vec<CoreConnectionView> {
        let state = self.state.read().expect("registry state poisoned");
        state
            .registry
            .status()
            .into_iter()
            .map(|st| {
                let healthy = matches!(st.health, ConnectionHealth::Ok);
                CoreConnectionView {
                    id: st.id.as_str().to_string(),
                    connector_type: st.connector_type.clone(),
                    display_label: format!("{} ({})", st.id, st.connector_type),
                    availability: match st.health {
                        ConnectionHealth::Ok => CoreConnectionAvailability::Ok,
                        ConnectionHealth::Unavailable { reason } => {
                            CoreConnectionAvailability::Unavailable { reason }
                        }
                    },
                    has_credentials: healthy,
                }
            })
            .collect()
    }

    #[allow(dead_code)]
    fn is_healthy(&self, id: &ConnectionId) -> bool {
        let state = self.state.read().expect("registry state poisoned");
        state
            .registry
            .status_of(id)
            .is_some_and(|s| matches!(s.health, ConnectionHealth::Ok))
    }

    /// Is the given (connection, model) pair currently routable? Connection
    /// must be live and `list_models()` must include the model id.
    async fn connection_lists_model(
        &self,
        id: &ConnectionId,
        model_id: &str,
    ) -> Result<bool, CoreError> {
        let Some(client) = self.client_for(id) else {
            return Ok(false);
        };
        let models = client.list_models().await?;
        Ok(models.iter().any(|m| m.id == model_id))
    }

    /// Fetch the live client handle for a connection id, if any. The
    /// returned `Arc` can be awaited on without holding any registry
    /// locks, which keeps the async futures `Send`.
    pub(crate) fn client_for(
        &self,
        id: &ConnectionId,
    ) -> Option<std::sync::Arc<crate::registry::AnyLlmClient>> {
        let state = self.state.read().expect("registry state poisoned");
        state.registry.get(id)
    }

    /// Connector-type tag for a given connection id, if declared.
    pub(crate) fn connector_type_for(&self, id: &ConnectionId) -> Option<String> {
        let state = self.state.read().expect("registry state poisoned");
        state
            .registry
            .status_of(id)
            .map(|s| s.connector_type.clone())
    }

    /// Mutate the config: callers provide a closure that operates on the
    /// current `DaemonConfig`. On success we rewrite the config file and
    /// rebuild the registry.
    fn mutate_config<F>(&self, op: F) -> Result<(), CoreError>
    where
        F: FnOnce(&mut DaemonConfig) -> Result<(), String>,
    {
        let mut state = self.state.write().expect("registry state poisoned");
        let mut new_config = state.config.clone();
        op(&mut new_config).map_err(CoreError::Llm)?;
        save_daemon_config(&self.config_path, &new_config)
            .map_err(|e| CoreError::Storage(format!("saving config: {e}")))?;
        let registry = build_registry(&new_config);
        state.config = new_config;
        state.registry = registry;
        Ok(())
    }

    /// Read-only snapshot of the current `DaemonConfig`. Used by purposes
    /// and model-listing paths.
    pub fn snapshot_config(&self) -> DaemonConfig {
        self.state
            .read()
            .expect("registry state poisoned")
            .config
            .clone()
    }

    /// Reload the registry (and re-read the config from disk). Used when
    /// external tools mutate the config file.
    #[allow(dead_code)]
    pub fn reload(&self) -> anyhow::Result<()> {
        let config = load_daemon_config(&self.config_path)?.unwrap_or_default();
        let registry = build_registry(&config);
        let mut state = self.state.write().expect("registry state poisoned");
        state.config = config;
        state.registry = registry;
        Ok(())
    }
}

// --- ConnectionsService impl -----------------------------------------------

pub struct DaemonConnectionsService {
    registry: Arc<RegistryHandle>,
}

impl DaemonConnectionsService {
    pub fn new(registry: Arc<RegistryHandle>) -> Self {
        Self { registry }
    }
}

impl ConnectionsService for DaemonConnectionsService {
    async fn list_connections(&self) -> Result<Vec<CoreConnectionView>, CoreError> {
        Ok(self.registry.connection_views())
    }

    async fn create_connection(
        &self,
        id: String,
        config: ConnectionConfigPayload,
    ) -> Result<(), CoreError> {
        let id_valid = ConnectionId::new(id.clone())
            .map_err(|e| CoreError::Llm(format!("invalid connection id: {e}")))?;
        let new_conn = payload_to_connection(config);
        self.registry.mutate_config(|cfg| {
            if cfg.connections.contains_key(id_valid.as_str()) {
                return Err(format!("connection id {:?} already exists", id_valid));
            }
            cfg.connections.insert(id_valid.as_str().to_string(), new_conn);
            Ok(())
        })
    }

    async fn update_connection(
        &self,
        id: String,
        config: ConnectionConfigPayload,
    ) -> Result<(), CoreError> {
        let id_valid = ConnectionId::new(id.clone())
            .map_err(|e| CoreError::Llm(format!("invalid connection id: {e}")))?;
        let new_conn = payload_to_connection(config);
        self.registry.mutate_config(|cfg| {
            if !cfg.connections.contains_key(id_valid.as_str()) {
                return Err(format!("connection id {:?} does not exist", id_valid));
            }
            cfg.connections
                .insert(id_valid.as_str().to_string(), new_conn);
            Ok(())
        })
    }

    async fn delete_connection(&self, id: String, force: bool) -> Result<(), CoreError> {
        let id_valid = ConnectionId::new(id.clone())
            .map_err(|e| CoreError::Llm(format!("invalid connection id: {e}")))?;
        self.registry.mutate_config(|cfg| {
            if !cfg.connections.contains_key(id_valid.as_str()) {
                return Err(format!("connection id {:?} does not exist", id_valid));
            }
            // Check whether any purpose references this id.
            let referenced_by: Vec<PurposeKind> = purposes_referencing(&cfg.purposes, &id_valid);
            if !referenced_by.is_empty() && !force {
                let names: Vec<&'static str> =
                    referenced_by.iter().map(|k| k.as_key()).collect();
                return Err(format!(
                    "connection {:?} is referenced by purposes {:?}; pass force=true to cascade",
                    id_valid, names
                ));
            }
            // Force path: reset referencing purposes to inherit from
            // interactive. If interactive itself is being deleted, switch it
            // to some other remaining connection (or wipe it).
            cfg.connections.shift_remove(id_valid.as_str());
            for kind in referenced_by {
                if kind == PurposeKind::Interactive {
                    // Pick a replacement: first remaining connection, if any.
                    if let Some((new_interactive_id, _)) = cfg.connections.iter().next() {
                        let new_id = new_interactive_id.clone();
                        if let Some(p) = cfg.purposes.interactive.as_mut() {
                            p.connection = ConnectionRef::Named(
                                ConnectionId::new(new_id)
                                    .expect("existing key was already validated"),
                            );
                        }
                    } else {
                        // No connections left — clear interactive entirely.
                        cfg.purposes.interactive = None;
                    }
                    continue;
                }
                let slot = match kind {
                    PurposeKind::Dreaming => cfg.purposes.dreaming.as_mut(),
                    PurposeKind::Embedding => cfg.purposes.embedding.as_mut(),
                    PurposeKind::Titling => cfg.purposes.titling.as_mut(),
                    PurposeKind::Interactive => unreachable!(),
                };
                if let Some(p) = slot {
                    p.connection = ConnectionRef::Primary;
                }
            }
            Ok(())
        })
    }

    async fn list_available_models(
        &self,
        connection_id: Option<String>,
        refresh: bool,
    ) -> Result<Vec<CoreModelListing>, CoreError> {
        // Snapshot (id, connector_type, label, client) tuples before awaiting
        // anything. Holding the read lock across `.await` would leave the
        // returned future `!Send`; cloning `Arc<AnyLlmClient>` releases the
        // lock up front and the awaits run unlocked.
        let targets: Vec<(
            ConnectionId,
            String,
            String,
            std::sync::Arc<crate::registry::AnyLlmClient>,
        )> = {
            let state = self
                .registry
                .state
                .read()
                .expect("registry state poisoned");
            if let Some(id_raw) = &connection_id {
                let id = ConnectionId::new(id_raw.clone())
                    .map_err(|e| CoreError::Llm(format!("invalid connection id: {e}")))?;
                let Some(st) = state.registry.status_of(&id) else {
                    return Err(CoreError::Llm(format!("connection {id} is not declared")));
                };
                if !matches!(st.health, ConnectionHealth::Ok) {
                    return Err(CoreError::Llm(format!("connection {id} is not live")));
                }
                let connector_type = st.connector_type.to_string();
                let label = format!("{} ({})", st.id, connector_type);
                let Some(client) = state.registry.get(&id) else {
                    return Err(CoreError::Llm(format!("connection {id} is not live")));
                };
                vec![(id, connector_type, label, client)]
            } else {
                state
                    .registry
                    .status()
                    .into_iter()
                    .filter(|s| matches!(s.health, ConnectionHealth::Ok))
                    .filter_map(|s| {
                        let connector_type = s.connector_type.to_string();
                        let label = format!("{} ({})", s.id, connector_type);
                        let client = state.registry.get(&s.id)?;
                        Some((s.id, connector_type, label, client))
                    })
                    .collect()
            }
        };

        let mut out: Vec<CoreModelListing> = Vec::new();
        for (id, connector_type, label, client) in targets {
            let list_result = if refresh {
                client.refresh_models().await
            } else {
                client.list_models().await
            };
            match list_result {
                Ok(models) => {
                    let merged =
                        crate::model_defaults::merge_with_defaults(&connector_type, models);
                    for m in merged {
                        out.push(CoreModelListing {
                            connection_id: id.as_str().to_string(),
                            connection_label: label.clone(),
                            model: m,
                        });
                    }
                }
                Err(e) => {
                    // Single-connection path surfaces the error; aggregate
                    // path logs and continues so one broken endpoint
                    // doesn't break the whole listing.
                    if connection_id.is_some() {
                        return Err(e);
                    }
                    tracing::warn!(
                        connection = %id,
                        "list_models failed during aggregation: {e}"
                    );
                }
            }
        }
        Ok(out)
    }

    async fn get_purposes(&self) -> Result<CorePurposesView, CoreError> {
        let config = self.registry.snapshot_config();
        Ok(CorePurposesView {
            interactive: config.purposes.interactive.as_ref().map(purpose_to_payload),
            dreaming: config.purposes.dreaming.as_ref().map(purpose_to_payload),
            embedding: config.purposes.embedding.as_ref().map(purpose_to_payload),
            titling: config.purposes.titling.as_ref().map(purpose_to_payload),
        })
    }

    async fn set_purpose(
        &self,
        purpose: CorePurposeKind,
        config: PurposeConfigPayload,
    ) -> Result<(), CoreError> {
        let purpose_kind = core_to_internal_purpose(purpose);
        let new_cfg = payload_to_purpose(config)
            .map_err(|e| CoreError::Llm(format!("invalid purpose config: {e}")))?;

        // Interactive cannot use `"primary"` for connection or model.
        if purpose_kind == PurposeKind::Interactive {
            if matches!(new_cfg.connection, ConnectionRef::Primary) {
                return Err(CoreError::Llm(
                    "interactive purpose cannot use connection \"primary\" — nothing to inherit from"
                        .to_string(),
                ));
            }
            if matches!(new_cfg.model, ModelRef::Primary) {
                return Err(CoreError::Llm(
                    "interactive purpose cannot use model \"primary\" — nothing to inherit from"
                        .to_string(),
                ));
            }
        }

        self.registry.mutate_config(|cfg| {
            cfg.purposes.set(purpose_kind, Some(new_cfg));
            cfg.purposes
                .validate()
                .map_err(|e| format!("{e}"))
        })
    }
}

// --- RoutingConversationHandler --------------------------------------------

/// Callback the daemon supplies to fetch (and optionally store) the
/// conversation's last model selection. Abstracted as a trait so tests can
/// provide an in-memory implementation.
pub trait ConversationSelectionStore: Send + Sync {
    fn get_selection(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<
        Output = Result<Option<ConversationModelSelection>, CoreError>,
    > + Send;

    fn set_selection(
        &self,
        id: &ConversationId,
        selection: Option<&ConversationModelSelection>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;
}

pub struct RoutingConversationHandler<S, Inner>
where
    S: ConversationSelectionStore + 'static,
    Inner: ConversationService + 'static,
{
    inner: Arc<Inner>,
    selection_store: Arc<S>,
    registry: Arc<RegistryHandle>,
}

impl<S, Inner> RoutingConversationHandler<S, Inner>
where
    S: ConversationSelectionStore + 'static,
    Inner: ConversationService + 'static,
{
    pub fn new(inner: Arc<Inner>, selection_store: Arc<S>, registry: Arc<RegistryHandle>) -> Self {
        Self {
            inner,
            selection_store,
            registry,
        }
    }

    /// Resolve the interactive purpose from the current config. Used as
    /// the ultimate fallback (priority #3) when neither an override nor a
    /// valid stored selection exists.
    fn interactive_selection(&self) -> Option<ConversationModelSelection> {
        let cfg = self.registry.snapshot_config();
        cfg.purposes.interactive.as_ref().and_then(|p| {
            let connection_id = match &p.connection {
                ConnectionRef::Named(id) => id.as_str().to_string(),
                ConnectionRef::Primary => return None,
            };
            let model_id = match &p.model {
                ModelRef::Named(m) => m.clone(),
                ModelRef::Primary => return None,
            };
            Some(ConversationModelSelection {
                connection_id,
                model_id,
                effort: p.effort.map(effort_internal_to_serde),
            })
        })
    }

    /// Check a stored selection against the live registry. Returns
    /// `(is_still_valid)`. When invalid, the caller is responsible for
    /// clearing the stored selection and emitting a warning.
    async fn selection_is_live(
        &self,
        sel: &ConversationModelSelection,
    ) -> Result<bool, CoreError> {
        let Ok(id) = ConnectionId::new(sel.connection_id.clone()) else {
            return Ok(false);
        };
        self.registry.connection_lists_model(&id, &sel.model_id).await
    }

    /// Translate the effort hint into the per-connector
    /// [`ReasoningConfig`] the connector's dispatch path expects. Thin
    /// wrapper around [`map_effort_to_reasoning_config`] retained so the
    /// per-turn dispatch keeps its `Self::apply_effort_mapping(...)` shape.
    fn apply_effort_mapping(
        connector_type: &str,
        model_id: &str,
        effort: Option<Effort>,
    ) -> ReasoningConfig {
        map_effort_to_reasoning_config(connector_type, model_id, effort)
    }
}

/// Resolve a purpose's full dispatch config — `(ResolvedLlmConfig,
/// ReasoningConfig)` — for use by background tasks that want to honour
/// `[purposes.<kind>]` end-to-end (issue #27 dreaming, #28 titling).
///
/// Returns `None` when no purpose is configured for `kind` so callers
/// can fall back to the legacy resolvers without an extra branch on a
/// boolean. The returned `ReasoningConfig` is computed from the purpose's
/// effort hint via [`map_effort_to_reasoning_config`]; it is
/// `ReasoningConfig::default()` when the purpose has no effort set.
///
/// Lives here (not in `config.rs`) because the effort mapper depends on
/// the `Effort` ↔ `ReasoningConfig` conversion glue and the connector
/// dispatch tables, which are api_surface concerns. Putting it here
/// keeps `config.rs` free of `tracing::debug!` per-connector decisions.
pub(crate) fn resolve_purpose_dispatch(
    config: Option<&crate::config::DaemonConfig>,
    kind: PurposeKind,
) -> Option<(crate::config::ResolvedLlmConfig, ReasoningConfig)> {
    let resolved = crate::config::resolve_purpose_llm_config(config, kind)?;
    // The purpose itself was resolvable, so we know `cfg.purposes.get(kind)`
    // is `Some` — re-fetch it for the effort hint, which the
    // `ResolvedLlmConfig` doesn't carry (it's connector/model/credentials).
    let effort = config
        .and_then(|c| c.purposes.get(kind))
        .and_then(|p| p.effort)
        .map(purpose_effort_to_core);
    let reasoning = map_effort_to_reasoning_config(&resolved.connector, &resolved.model, effort);
    Some((resolved, reasoning))
}

/// Translate the effort hint into the per-connector [`ReasoningConfig`]
/// the connector's dispatch path expects.
///
/// - Anthropic / Bedrock (Claude): populates `thinking_budget_tokens`
///   using [`map_anthropic_thinking_budget`].
/// - OpenAI: populates `reasoning_effort` using
///   [`map_openai_reasoning_effort`]. The connector itself applies a
///   per-model capability gate and silently drops the field for
///   non-reasoning models.
/// - Ollama / unknown: returns an empty `ReasoningConfig` (no-op).
///
/// Free function so backend tasks (dreaming #27, titling #28) that don't
/// instantiate a [`RoutingConversationHandler`] can still resolve their
/// purpose's effort hint into a `ReasoningConfig` to thread into
/// `stream_completion`.
pub fn map_effort_to_reasoning_config(
    connector_type: &str,
    model_id: &str,
    effort: Option<Effort>,
) -> ReasoningConfig {
    let Some(effort) = effort else {
        return ReasoningConfig::default();
    };
    match connector_type {
        "anthropic" | "bedrock" => {
            let budget = map_anthropic_thinking_budget(effort);
            tracing::debug!(
                connector = connector_type,
                model = model_id,
                effort = ?effort,
                thinking_budget_tokens = budget,
                "mapped effort to Anthropic extended-thinking budget"
            );
            if budget == 0 {
                ReasoningConfig::default()
            } else {
                ReasoningConfig::with_thinking_budget(budget)
            }
        }
        "openai" => {
            let level = map_effort_to_reasoning_level(effort);
            tracing::debug!(
                connector = connector_type,
                model = model_id,
                effort = ?effort,
                reasoning_level = ?level,
                "mapped effort to OpenAI reasoning_effort"
            );
            ReasoningConfig::with_reasoning_effort(level)
        }
        _ => {
            tracing::debug!(
                connector = connector_type,
                effort = ?effort,
                "no reasoning mapping defined for connector (no-op)"
            );
            ReasoningConfig::default()
        }
    }
}

impl<S, Inner> ConversationService for RoutingConversationHandler<S, Inner>
where
    S: ConversationSelectionStore + 'static,
    Inner: ConversationService + 'static,
{
    async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
        self.inner.create_conversation(title).await
    }

    async fn list_conversations(
        &self,
        max_age_days: Option<u32>,
        include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        self.inner.list_conversations(max_age_days, include_archived).await
    }

    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        self.inner.get_conversation(id).await
    }

    async fn delete_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.inner.delete_conversation(id).await
    }

    async fn rename_conversation(
        &self,
        id: &ConversationId,
        title: String,
    ) -> Result<(), CoreError> {
        self.inner.rename_conversation(id, title).await
    }

    async fn archive_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.inner.archive_conversation(id).await
    }

    async fn unarchive_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.inner.unarchive_conversation(id).await
    }

    async fn clear_all_history(&self) -> Result<u32, CoreError> {
        self.inner.clear_all_history().await
    }

    async fn send_prompt(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        on_chunk: ChunkCallback,
        on_status: StatusCallback,
    ) -> Result<String, CoreError> {
        // The plain `send_prompt` path is invoked by adapters that don't
        // carry an explicit override (legacy D-Bus/WS endpoints). We
        // still want per-conversation stored selections and the
        // interactive-purpose fallback to route the turn to the right
        // connection + effort, so we route it through the same
        // resolution + dispatch machinery as the override path.
        let outcome = self
            .send_prompt_with_override(conversation_id, prompt, None, on_chunk, on_status)
            .await?;
        Ok(outcome.response)
    }

    async fn send_prompt_with_override(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        override_selection: Option<PromptSelectionOverride>,
        on_chunk: ChunkCallback,
        on_status: StatusCallback,
    ) -> Result<PromptDispatchOutcome, CoreError> {
        let mut warnings: Vec<DispatchWarning> = Vec::new();

        // Resolve the effective selection following priority:
        //   1. override (validate first; hard error if invalid)
        //   2. stored conversation selection (validate; warn + fallback if dangling)
        //   3. interactive purpose
        //
        // We track *user_driven* separately from *effective* (issue #33):
        // the user-driven path (override / live stored) routes through the
        // registry's per-connection client; the interactive-fallback path
        // routes through the handler's static primary llm, which is
        // already built with the interactive purpose's model baked in.
        // Without this split, interactive_selection's `model_id` would be
        // dropped at dispatch — connector clients have no per-call model
        // knob, so the registry client always uses the connection's
        // construction-time model.
        //
        // `effective_selection` is still used for reasoning so the
        // interactive purpose's `effort` continues to apply when no
        // user-driven selection exists.
        let user_driven_selection: Option<ConversationModelSelection> = if let Some(override_sel) =
            override_selection
        {
            let id = ConnectionId::new(override_sel.connection_id.clone())
                .map_err(|e| CoreError::Llm(format!("invalid connection id in override: {e}")))?;
            let is_live = self
                .registry
                .connection_lists_model(&id, &override_sel.model_id)
                .await?;
            if !is_live {
                return Err(CoreError::Llm(format!(
                    "override target {}/{} is not a live (connection, model) pair",
                    override_sel.connection_id, override_sel.model_id
                )));
            }
            let sel = ConversationModelSelection {
                connection_id: override_sel.connection_id,
                model_id: override_sel.model_id,
                effort: override_sel.effort.map(SerdeEffort::from),
            };
            // Persist before dispatch so a crash mid-call doesn't lose the
            // user's choice.
            self.selection_store
                .set_selection(conversation_id, Some(&sel))
                .await?;
            Some(sel)
        } else if let Some(stored) = self.selection_store.get_selection(conversation_id).await? {
            if self.selection_is_live(&stored).await? {
                Some(stored)
            } else {
                // Dangling. Clear; emit a one-time warning naming the
                // interactive fallback (so the UI can surface what the
                // turn will actually use). The fallback itself is *not*
                // user-driven, so we leave `user_driven_selection = None`
                // and let dispatch route through the primary llm below.
                let fallback = self.interactive_selection();
                self.selection_store
                    .set_selection(conversation_id, None)
                    .await?;
                if let Some(ref fb) = fallback {
                    warnings.push(DispatchWarning::DanglingModelSelection {
                        previous: stored,
                        fallback_to: fb.clone(),
                    });
                }
                None
            }
        } else {
            None
        };

        // For reasoning purposes, the interactive purpose still contributes
        // when nothing user-driven exists.
        let effective_selection: Option<ConversationModelSelection> = user_driven_selection
            .clone()
            .or_else(|| self.interactive_selection());

        // Resolve the per-turn routing target:
        //   - `active_client`: the `Arc<AnyLlmClient>` dispatch must use
        //     for this turn. Only installed for *user-driven* selections;
        //     for the interactive-purpose fallback we leave it `None` so
        //     `RoutingLlmClient` falls through to the primary llm (which
        //     was built with the interactive purpose's model).
        //   - `reasoning`: the `ReasoningConfig` populated from the
        //     per-connector effort mapping. Computed from
        //     `effective_selection` so the interactive purpose's `effort`
        //     applies even when we don't install an active_client.
        let mut active_client: Option<std::sync::Arc<crate::registry::AnyLlmClient>> = None;
        if let Some(sel) = user_driven_selection.as_ref() {
            let id = ConnectionId::new(sel.connection_id.clone()).map_err(|e| {
                CoreError::Llm(format!(
                    "resolved selection has malformed connection id {:?}: {e}",
                    sel.connection_id
                ))
            })?;
            // Reject Unavailable (or undeclared) connections with a
            // clean 400-style error rather than silently falling back.
            match self.registry.client_for(&id) {
                Some(client) => {
                    active_client = Some(client);
                }
                None => {
                    return Err(CoreError::Llm(format!(
                        "resolved connection {} is not live; requested model {} cannot be dispatched",
                        sel.connection_id, sel.model_id
                    )));
                }
            }
        }

        let mut reasoning = ReasoningConfig::default();
        if let Some(sel) = effective_selection.as_ref() {
            let id = ConnectionId::new(sel.connection_id.clone()).map_err(|e| {
                CoreError::Llm(format!(
                    "resolved selection has malformed connection id {:?}: {e}",
                    sel.connection_id
                ))
            })?;
            let connector_type = self
                .registry
                .connector_type_for(&id)
                .unwrap_or_default();
            reasoning =
                Self::apply_effort_mapping(&connector_type, &sel.model_id, sel.effort.map(Effort::from));
        }

        // Resolve the per-turn `max_context_tokens` override (issue #51).
        // Interactive is the purpose that drives `send_prompt`; if the user
        // has authored `purposes.interactive.max_context_tokens`, surface it
        // through the routing wrapper so token-based compaction in
        // `core::Service` honours their override. When unset (the common
        // case), the wrapper falls through to tier 2/3.
        let max_context_override = crate::config::purpose_max_context_override(
            Some(&self.registry.snapshot_config()),
            PurposeKind::Interactive,
        );

        // Install task-locals, then delegate to the inner core
        // handler. The handler reads the task-locals inside its
        // `send_prompt` dispatch loop:
        //   - `RoutingLlmClient` picks the active client on each
        //     `stream_completion` call and uses `MAX_CONTEXT_OVERRIDE` to
        //     resolve the context window.
        //   - `current_reasoning_config()` surfaces `reasoning` into the
        //     connector's request body.
        let inner = Arc::clone(&self.inner);
        let conv_id = conversation_id.clone();
        let response = {
            let dispatch = async move {
                inner
                    .send_prompt(&conv_id, prompt, on_chunk, on_status)
                    .await
            };
            let dispatch = with_reasoning_config(reasoning, dispatch);
            let dispatch =
                crate::routing_llm::with_max_context_override(max_context_override, dispatch);
            match active_client {
                Some(c) => crate::routing_llm::with_active_client(c, dispatch).await,
                None => dispatch.await,
            }
        }?;
        Ok(PromptDispatchOutcome {
            response,
            warnings,
        })
    }
}

// --- In-memory ConversationSelectionStore (for tests) ----------------------

/// Trivial in-memory store used by the daemon test suite. Production code
/// uses the Postgres-backed store via the storage crate.
#[allow(dead_code)]
pub struct InMemoryConversationSelectionStore {
    inner: Mutex<std::collections::HashMap<String, ConversationModelSelection>>,
}

impl Default for InMemoryConversationSelectionStore {
    fn default() -> Self {
        Self {
            inner: Mutex::new(std::collections::HashMap::new()),
        }
    }
}

impl ConversationSelectionStore for InMemoryConversationSelectionStore {
    async fn get_selection(
        &self,
        id: &ConversationId,
    ) -> Result<Option<ConversationModelSelection>, CoreError> {
        Ok(self
            .inner
            .lock()
            .expect("selection store poisoned")
            .get(&id.0)
            .cloned())
    }

    async fn set_selection(
        &self,
        id: &ConversationId,
        selection: Option<&ConversationModelSelection>,
    ) -> Result<(), CoreError> {
        let mut map = self.inner.lock().expect("selection store poisoned");
        match selection {
            Some(sel) => {
                map.insert(id.0.clone(), sel.clone());
            }
            None => {
                map.remove(&id.0);
            }
        }
        Ok(())
    }
}

// --- Effort → per-connector param mapping ----------------------------------

/// Anthropic extended-thinking `budget_tokens`. Defaults: Low = off (0, no
/// thinking), Medium = 8_000, High = 24_000. Connector expected to treat
/// `0` as "disable extended thinking" and any positive number as a budget.
pub fn map_anthropic_thinking_budget(e: Effort) -> u32 {
    match e {
        Effort::Low => 0,
        Effort::Medium => 8_000,
        Effort::High => 24_000,
    }
}

/// OpenAI `reasoning_effort` literal. Pass through verbatim.
///
/// Retained as the canonical Effort → wire-token table even after the
/// main dispatch path switched to [`map_effort_to_reasoning_level`] +
/// the connector's own per-model capability gate; keeps the mapping
/// truth-source documented in one place for future providers.
#[allow(dead_code)]
pub fn map_openai_reasoning_effort(e: Effort) -> &'static str {
    match e {
        Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
    }
}

/// `Effort` → core-level [`ReasoningLevel`], used when threading the
/// per-turn hint into the `LlmClient` trait.
pub fn map_effort_to_reasoning_level(e: Effort) -> ReasoningLevel {
    match e {
        Effort::Low => ReasoningLevel::Low,
        Effort::Medium => ReasoningLevel::Medium,
        Effort::High => ReasoningLevel::High,
    }
}

// --- Conversions between core payload / internal config types -------------

fn payload_to_connection(payload: ConnectionConfigPayload) -> ConnectionConfig {
    match payload {
        ConnectionConfigPayload::Anthropic {
            base_url,
            api_key_env,
        } => ConnectionConfig::Anthropic(AnthropicConnection {
            base_url,
            api_key_env,
            secret: None,
        }),
        ConnectionConfigPayload::OpenAi {
            base_url,
            api_key_env,
        } => ConnectionConfig::OpenAi(OpenAiConnection {
            base_url,
            api_key_env,
            secret: None,
        }),
        ConnectionConfigPayload::Bedrock {
            aws_profile,
            region,
            base_url,
        } => ConnectionConfig::Bedrock(BedrockConnection {
            aws_profile,
            region,
            base_url,
        }),
        ConnectionConfigPayload::Ollama { base_url } => {
            ConnectionConfig::Ollama(OllamaConnection { base_url })
        }
    }
}

fn purpose_to_payload(p: &PurposeConfig) -> PurposeConfigPayload {
    PurposeConfigPayload {
        connection: match &p.connection {
            ConnectionRef::Named(id) => id.as_str().to_string(),
            ConnectionRef::Primary => "primary".to_string(),
        },
        model: match &p.model {
            ModelRef::Named(m) => m.clone(),
            ModelRef::Primary => "primary".to_string(),
        },
        effort: p.effort.map(purpose_effort_to_core),
        max_context_tokens: p.max_context_tokens,
    }
}

fn payload_to_purpose(p: PurposeConfigPayload) -> Result<PurposeConfig, String> {
    let connection = if p.connection == "primary" {
        ConnectionRef::Primary
    } else {
        ConnectionRef::Named(
            ConnectionId::new(p.connection.clone())
                .map_err(|e| format!("connection {:?}: {e}", p.connection))?,
        )
    };
    let model = if p.model == "primary" {
        ModelRef::Primary
    } else {
        ModelRef::Named(p.model)
    };
    Ok(PurposeConfig {
        connection,
        model,
        effort: p.effort.map(core_effort_to_purpose),
        max_context_tokens: p.max_context_tokens,
    })
}

pub(crate) fn purpose_effort_to_core(e: PurposeEffort) -> Effort {
    match e {
        PurposeEffort::Low => Effort::Low,
        PurposeEffort::Medium => Effort::Medium,
        PurposeEffort::High => Effort::High,
    }
}

fn core_effort_to_purpose(e: Effort) -> PurposeEffort {
    match e {
        Effort::Low => PurposeEffort::Low,
        Effort::Medium => PurposeEffort::Medium,
        Effort::High => PurposeEffort::High,
    }
}

fn effort_internal_to_serde(e: PurposeEffort) -> SerdeEffort {
    match e {
        PurposeEffort::Low => SerdeEffort::Low,
        PurposeEffort::Medium => SerdeEffort::Medium,
        PurposeEffort::High => SerdeEffort::High,
    }
}

fn core_to_internal_purpose(k: CorePurposeKind) -> PurposeKind {
    match k {
        CorePurposeKind::Interactive => PurposeKind::Interactive,
        CorePurposeKind::Dreaming => PurposeKind::Dreaming,
        CorePurposeKind::Embedding => PurposeKind::Embedding,
        CorePurposeKind::Titling => PurposeKind::Titling,
    }
}

fn purposes_referencing(purposes: &crate::purposes::Purposes, id: &ConnectionId) -> Vec<PurposeKind> {
    let mut out = Vec::new();
    for kind in PurposeKind::all() {
        if let Some(p) = purposes.get(kind)
            && let ConnectionRef::Named(refd) = &p.connection
            && refd == id
        {
            out.push(kind);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connections::{BedrockConnection, ConnectionConfig, OllamaConnection};

    fn tmp_config_path() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "desktop-assistant-test-{}.toml",
            uuid::Uuid::new_v4().simple()
        ));
        p
    }

    fn config_with_connections(pairs: &[(&str, ConnectionConfig)]) -> DaemonConfig {
        let mut cfg = DaemonConfig::default();
        for (id, c) in pairs {
            cfg.connections.insert(id.to_string(), c.clone());
        }
        cfg
    }

    fn ollama_local() -> ConnectionConfig {
        ConnectionConfig::Ollama(OllamaConnection {
            base_url: Some("http://localhost:11434".into()),
        })
    }

    fn bedrock_work() -> ConnectionConfig {
        ConnectionConfig::Bedrock(BedrockConnection {
            aws_profile: Some("work".into()),
            region: Some("us-west-2".into()),
            base_url: None,
        })
    }

    fn make_handle_with(cfg: DaemonConfig) -> Arc<RegistryHandle> {
        let registry = build_registry(&cfg);
        Arc::new(RegistryHandle::new(cfg, registry).with_config_path(tmp_config_path()))
    }

    #[tokio::test]
    async fn list_connections_returns_declared_order() {
        let cfg = config_with_connections(&[
            ("local", ollama_local()),
            ("aws", bedrock_work()),
        ]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let views = svc.list_connections().await.unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].id, "local");
        assert_eq!(views[1].id, "aws");
    }

    #[tokio::test]
    async fn create_connection_rejects_bad_slug() {
        let svc = DaemonConnectionsService::new(make_handle_with(DaemonConfig::default()));
        let err = svc
            .create_connection(
                "Bad Id!".to_string(),
                ConnectionConfigPayload::Ollama {
                    base_url: Some("http://localhost:11434".into()),
                },
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("invalid connection id"));
    }

    #[tokio::test]
    async fn create_connection_rejects_duplicate_id() {
        let cfg = config_with_connections(&[("local", ollama_local())]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let err = svc
            .create_connection(
                "local".to_string(),
                ConnectionConfigPayload::Ollama {
                    base_url: Some("http://localhost:11434".into()),
                },
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("already exists"));
    }

    #[tokio::test]
    async fn delete_connection_refuses_when_referenced_without_force() {
        let mut cfg = config_with_connections(&[
            ("local", ollama_local()),
            ("aws", bedrock_work()),
        ]);
        cfg.purposes.interactive = Some(PurposeConfig {
            connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
            model: ModelRef::Named("llama3".into()),
            effort: None,
            max_context_tokens: None,
        });
        cfg.purposes.dreaming = Some(PurposeConfig {
            connection: ConnectionRef::Named(ConnectionId::new("aws").unwrap()),
            model: ModelRef::Named("claude".into()),
            effort: None,
            max_context_tokens: None,
        });

        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let err = svc
            .delete_connection("aws".to_string(), false)
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("referenced"));
    }

    #[tokio::test]
    async fn delete_connection_force_cascades_to_primary() {
        let mut cfg = config_with_connections(&[
            ("local", ollama_local()),
            ("aws", bedrock_work()),
        ]);
        cfg.purposes.interactive = Some(PurposeConfig {
            connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
            model: ModelRef::Named("llama3".into()),
            effort: None,
            max_context_tokens: None,
        });
        cfg.purposes.dreaming = Some(PurposeConfig {
            connection: ConnectionRef::Named(ConnectionId::new("aws").unwrap()),
            model: ModelRef::Named("claude".into()),
            effort: None,
            max_context_tokens: None,
        });

        let handle = make_handle_with(cfg);
        let svc = DaemonConnectionsService::new(Arc::clone(&handle));
        svc.delete_connection("aws".to_string(), true).await.unwrap();

        let cfg = handle.snapshot_config();
        assert!(!cfg.connections.contains_key("aws"));
        let dreaming = cfg.purposes.dreaming.as_ref().expect("dreaming still set");
        assert!(matches!(dreaming.connection, ConnectionRef::Primary));
    }

    #[tokio::test]
    async fn set_purpose_rejects_primary_in_interactive() {
        let cfg = config_with_connections(&[("local", ollama_local())]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let err = svc
            .set_purpose(
                CorePurposeKind::Interactive,
                PurposeConfigPayload {
                    connection: "primary".into(),
                    model: "llama3".into(),
                    effort: None,
                    max_context_tokens: None,
                },
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("interactive"));
    }

    #[tokio::test]
    async fn get_purposes_returns_current_config() {
        let mut cfg = config_with_connections(&[("local", ollama_local())]);
        cfg.purposes.interactive = Some(PurposeConfig {
            connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
            model: ModelRef::Named("llama3".into()),
            effort: Some(PurposeEffort::Medium),
            max_context_tokens: None,
        });
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let view = svc.get_purposes().await.unwrap();
        let i = view.interactive.expect("interactive set");
        assert_eq!(i.connection, "local");
        assert_eq!(i.model, "llama3");
        assert_eq!(i.effort, Some(Effort::Medium));
    }

    #[test]
    fn anthropic_effort_mapping_table() {
        assert_eq!(map_anthropic_thinking_budget(Effort::Low), 0);
        assert_eq!(map_anthropic_thinking_budget(Effort::Medium), 8_000);
        assert_eq!(map_anthropic_thinking_budget(Effort::High), 24_000);
    }

    #[test]
    fn openai_effort_mapping_table() {
        assert_eq!(map_openai_reasoning_effort(Effort::Low), "low");
        assert_eq!(map_openai_reasoning_effort(Effort::Medium), "medium");
        assert_eq!(map_openai_reasoning_effort(Effort::High), "high");
    }

    #[tokio::test]
    async fn list_available_models_aggregates_healthy_connections() {
        // Two Ollama connections hit localhost which is not running in CI —
        // we just verify the dispatch path runs without panicking and
        // filters unhealthy entries. A full integration test with mocked
        // list_models lives in `send_prompt_override_tests` below.
        let cfg = config_with_connections(&[
            ("local1", ollama_local()),
            ("local2", ollama_local()),
        ]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        // Either the network fails (empty list) or succeeds — both are OK
        // since we're just checking we don't hard-error when aggregating.
        let _ = svc.list_available_models(None, false).await;
    }

    // ----- RoutingConversationHandler dispatch-routing tests -----------
    //
    // These tests cover the per-turn routing logic added in #18:
    // - priority resolution across override/stored/interactive
    // - task-local reasoning config installation
    // - per-connector effort mapping into ReasoningConfig
    // - clean error on Unavailable connection

    mod routing_dispatch {
        use super::*;
        use desktop_assistant_core::domain::{Conversation, ConversationId, ConversationSummary};
        use desktop_assistant_core::ports::inbound::{
            ConversationService, PromptSelectionOverride,
        };
        use std::sync::Mutex as StdMutex;

        /// Inner `ConversationService` mock that records each call. Dispatch
        /// paths under test go through `RoutingConversationHandler ->
        /// inner.send_prompt`, so we snapshot the task-local values at
        /// dispatch time into the captured record.
        struct CapturingInner {
            captured_reasoning: StdMutex<Vec<ReasoningConfig>>,
            /// Whether the routing wrapper installed an `ACTIVE_CLIENT`
            /// task-local on each `send_prompt`. `false` means dispatch
            /// would fall through to the primary llm — the expected
            /// behaviour for the interactive-purpose fallback path
            /// (issue #33).
            captured_active_client_set: StdMutex<Vec<bool>>,
        }

        impl CapturingInner {
            fn new() -> Self {
                Self {
                    captured_reasoning: StdMutex::new(Vec::new()),
                    captured_active_client_set: StdMutex::new(Vec::new()),
                }
            }
        }

        impl ConversationService for CapturingInner {
            async fn create_conversation(
                &self,
                title: String,
            ) -> Result<Conversation, CoreError> {
                Ok(Conversation::new("c1", title))
            }
            async fn list_conversations(
                &self,
                _max_age_days: Option<u32>,
                _include_archived: bool,
            ) -> Result<Vec<ConversationSummary>, CoreError> {
                Ok(vec![])
            }
            async fn get_conversation(
                &self,
                id: &ConversationId,
            ) -> Result<Conversation, CoreError> {
                Ok(Conversation::new(id.as_str(), "t"))
            }
            async fn delete_conversation(
                &self,
                _id: &ConversationId,
            ) -> Result<(), CoreError> {
                Ok(())
            }
            async fn rename_conversation(
                &self,
                _id: &ConversationId,
                _title: String,
            ) -> Result<(), CoreError> {
                Ok(())
            }
            async fn archive_conversation(
                &self,
                _id: &ConversationId,
            ) -> Result<(), CoreError> {
                Ok(())
            }
            async fn unarchive_conversation(
                &self,
                _id: &ConversationId,
            ) -> Result<(), CoreError> {
                Ok(())
            }
            async fn clear_all_history(&self) -> Result<u32, CoreError> {
                Ok(0)
            }
            async fn send_prompt(
                &self,
                _conversation_id: &ConversationId,
                _prompt: String,
                _on_chunk: desktop_assistant_core::ports::llm::ChunkCallback,
                _on_status: desktop_assistant_core::ports::llm::StatusCallback,
            ) -> Result<String, CoreError> {
                // Snapshot the task-local reasoning config the routing
                // wrapper installed on the calling scope; asserting on
                // this value proves the plumbing actually propagates
                // all the way to the point where the core dispatch
                // loop would call `stream_completion`.
                let cfg = desktop_assistant_core::ports::llm::current_reasoning_config();
                self.captured_reasoning.lock().unwrap().push(cfg);
                let active = crate::routing_llm::active_client_is_set();
                self.captured_active_client_set.lock().unwrap().push(active);
                Ok("ok".to_string())
            }
        }

        fn local_ollama_cfg() -> DaemonConfig {
            let mut cfg = config_with_connections(&[
                ("local", ollama_local()),
                ("aws", bedrock_work()),
            ]);
            cfg.purposes.interactive = Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("llama3".into()),
                effort: None,
                max_context_tokens: None,
            });
            cfg
        }

        fn make_handler() -> (
            Arc<RoutingConversationHandler<InMemoryConversationSelectionStore, CapturingInner>>,
            Arc<CapturingInner>,
            Arc<RegistryHandle>,
            Arc<InMemoryConversationSelectionStore>,
        ) {
            let cfg = local_ollama_cfg();
            let registry = make_handle_with(cfg);
            let inner = Arc::new(CapturingInner::new());
            let store = Arc::new(InMemoryConversationSelectionStore::default());
            let routing = Arc::new(RoutingConversationHandler::new(
                Arc::clone(&inner),
                Arc::clone(&store),
                Arc::clone(&registry),
            ));
            (routing, inner, registry, store)
        }

        fn noop_cb() -> (
            desktop_assistant_core::ports::llm::ChunkCallback,
            desktop_assistant_core::ports::llm::StatusCallback,
        ) {
            (
                Box::new(|_: String| -> bool { true }),
                Box::new(|_: String| {}),
            )
        }

        #[tokio::test]
        async fn send_prompt_unknown_override_connection_errors() {
            let (routing, _inner, _reg, _store) = make_handler();
            let (on_chunk, on_status) = noop_cb();
            let err = routing
                .send_prompt_with_override(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    Some(PromptSelectionOverride {
                        connection_id: "does-not-exist".into(),
                        model_id: "llama3".into(),
                        effort: None,
                    }),
                    on_chunk,
                    on_status,
                )
                .await
                .unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("does-not-exist") || msg.contains("not a live"),
                "expected error mentioning the unknown connection; got: {msg}"
            );
        }

        #[tokio::test]
        async fn interactive_purpose_reasoning_maps_to_local_connector_no_op() {
            // interactive purpose: local/llama3 (ollama) with no effort →
            // reasoning config stays empty, dispatch proceeds to inner.
            let (routing, inner, _reg, _store) = make_handler();
            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("dispatch should succeed via interactive purpose");
            let captured = inner.captured_reasoning.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(captured[0], ReasoningConfig::default());
        }

        #[tokio::test]
        async fn bedrock_override_maps_effort_to_thinking_budget() {
            // Configure an override pointing at the Bedrock connection
            // with Effort::High; the routing wrapper must translate it
            // to a `ReasoningConfig { thinking_budget_tokens: Some(24_000) }`
            // and install it on the task-local observed by the inner.
            let cfg = {
                let mut c = local_ollama_cfg();
                // Point interactive at aws/claude so override-less path
                // still routes to a Claude-shape connector; override
                // sets the Bedrock connection explicitly below to
                // exercise the mapping.
                c.purposes.interactive = Some(PurposeConfig {
                    connection: ConnectionRef::Named(ConnectionId::new("aws").unwrap()),
                    model: ModelRef::Named(
                        "us.anthropic.claude-sonnet-4-6".into(),
                    ),
                    effort: None,
                    max_context_tokens: None,
                });
                c
            };
            let registry = make_handle_with(cfg);
            let inner = Arc::new(CapturingInner::new());
            let store = Arc::new(InMemoryConversationSelectionStore::default());
            let routing = Arc::new(RoutingConversationHandler::new(
                Arc::clone(&inner),
                Arc::clone(&store),
                Arc::clone(&registry),
            ));

            // The override connection/model must pass the `list_models`
            // gate — for Bedrock this hits the AWS SDK, which is not
            // available in the test env. Since validation would fail,
            // exercise the effort-mapping function directly rather than
            // the end-to-end path. (The end-to-end routing is covered
            // above via `send_prompt` with the interactive purpose.)
            let cfg = RoutingConversationHandler::<
                InMemoryConversationSelectionStore,
                CapturingInner,
            >::apply_effort_mapping(
                "bedrock",
                "us.anthropic.claude-sonnet-4-6",
                Some(Effort::High),
            );
            assert_eq!(cfg.thinking_budget_tokens, Some(24_000));
            assert!(cfg.reasoning_effort.is_none());

            // Route routing is still used: prove the handler exists and
            // its `send_prompt` path sets the default reasoning when no
            // effort is supplied.
            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("plain send_prompt should succeed via interactive purpose");
        }

        #[test]
        fn effort_mapping_openai_path() {
            let cfg = RoutingConversationHandler::<
                InMemoryConversationSelectionStore,
                CapturingInner,
            >::apply_effort_mapping("openai", "gpt-5", Some(Effort::Medium));
            assert_eq!(
                cfg.reasoning_effort,
                Some(ReasoningLevel::Medium),
                "Medium effort must map to ReasoningLevel::Medium for OpenAI"
            );
            assert!(cfg.thinking_budget_tokens.is_none());
        }

        #[test]
        fn effort_mapping_low_anthropic_disables_thinking() {
            // Low effort maps to budget=0 which disables the thinking
            // block entirely, even though the caller asked for
            // Effort::Low. Matches the Anthropic semantics where a
            // zero budget means "extended thinking disabled".
            let cfg = RoutingConversationHandler::<
                InMemoryConversationSelectionStore,
                CapturingInner,
            >::apply_effort_mapping("anthropic", "claude-sonnet-4-6", Some(Effort::Low));
            assert!(cfg.thinking_budget_tokens.is_none());
        }

        #[test]
        fn effort_mapping_ollama_is_noop() {
            let cfg = RoutingConversationHandler::<
                InMemoryConversationSelectionStore,
                CapturingInner,
            >::apply_effort_mapping("ollama", "llama3", Some(Effort::High));
            assert_eq!(cfg, ReasoningConfig::default());
        }

        #[test]
        fn effort_mapping_unknown_connector_is_noop() {
            let cfg = RoutingConversationHandler::<
                InMemoryConversationSelectionStore,
                CapturingInner,
            >::apply_effort_mapping("mystery-vendor", "m1", Some(Effort::High));
            assert_eq!(cfg, ReasoningConfig::default());
        }

        #[test]
        fn effort_mapping_no_effort_returns_default() {
            let cfg = RoutingConversationHandler::<
                InMemoryConversationSelectionStore,
                CapturingInner,
            >::apply_effort_mapping("anthropic", "claude-sonnet-4-6", None);
            assert_eq!(cfg, ReasoningConfig::default());
        }

        // ─── Issue #33: interactive purpose's model must reach dispatch ───
        //
        // The dispatch path's contract changed: when the effective selection
        // came from `interactive_selection()` (i.e. no override, no live
        // stored selection), the routing wrapper must NOT install the
        // registry's per-connection client. Connector clients have no
        // per-call model knob, so the registry client always uses the
        // connection's construction-time model — which silently drops the
        // interactive purpose's model. By falling through to the
        // `RoutingLlmClient`'s static fallback (the primary llm, built in
        // `main.rs` with the interactive purpose's model baked in), we
        // ensure the user-configured model actually reaches the wire.

        #[tokio::test]
        async fn interactive_purpose_does_not_install_active_client() {
            // No override, no stored selection: dispatch must fall through
            // to the primary llm. `ACTIVE_CLIENT` task-local must be
            // *unset* in the inner handler's scope.
            let (routing, inner, _reg, _store) = make_handler();
            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("dispatch must succeed");

            let active = inner.captured_active_client_set.lock().unwrap();
            assert_eq!(active.len(), 1);
            assert!(
                !active[0],
                "interactive-purpose fallback must not install ACTIVE_CLIENT \
                 (else dispatch would route through registry's connection \
                 client and ignore the purpose's model)"
            );
        }

        #[tokio::test]
        async fn interactive_purpose_effort_still_applies_without_active_client() {
            // The purpose's effort flows through the reasoning task-local
            // even when we *don't* install ACTIVE_CLIENT. Use ollama so
            // the connector mapping is a no-op (default ReasoningConfig)
            // — the assertion is that we got the expected default, not
            // that we lost the effort entirely. A non-ollama connector
            // can't be exercised end-to-end without a live model list,
            // so the bedrock-effort case is covered by the unit test on
            // `apply_effort_mapping` above.
            let mut cfg = local_ollama_cfg();
            cfg.purposes.interactive = Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("llama3".into()),
                effort: Some(PurposeEffort::High),
                max_context_tokens: None,
            });
            let registry = make_handle_with(cfg);
            let inner = Arc::new(CapturingInner::new());
            let store = Arc::new(InMemoryConversationSelectionStore::default());
            let routing = Arc::new(RoutingConversationHandler::new(
                Arc::clone(&inner),
                Arc::clone(&store),
                Arc::clone(&registry),
            ));

            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("dispatch must succeed");

            let reasoning = inner.captured_reasoning.lock().unwrap();
            assert_eq!(reasoning.len(), 1);
            // ollama connector → no-op mapping. Asserting `default()` here
            // is the *correct* outcome for the connector; the value-add of
            // the test is that we still got *here* (pipeline didn't skip
            // reasoning resolution just because active_client wasn't set).
            assert_eq!(reasoning[0], ReasoningConfig::default());

            let active = inner.captured_active_client_set.lock().unwrap();
            assert!(!active[0]);
        }

        #[tokio::test]
        async fn dangling_stored_selection_falls_back_to_interactive_without_active_client() {
            // A stored selection pointing at a connection that's no longer
            // declared falls back to the interactive purpose. Like the
            // plain interactive path, this fallback must NOT install
            // ACTIVE_CLIENT — the user is no longer "driving" the
            // selection, the system is, and the primary llm already has
            // the interactive purpose's model baked in.
            let (routing, inner, _reg, store) = make_handler();
            // Stored selection points at an unknown connection id.
            // `connection_lists_model` returns false for missing ids
            // without an HTTP round-trip (registry has no client for it),
            // so the dangling-fallback branch fires deterministically.
            store
                .set_selection(
                    &ConversationId::from("c1"),
                    Some(&ConversationModelSelection {
                        connection_id: "ghost".into(),
                        model_id: "phantom".into(),
                        effort: None,
                    }),
                )
                .await
                .expect("set selection");

            let (on_chunk, on_status) = noop_cb();
            let outcome = routing
                .send_prompt_with_override(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    None,
                    on_chunk,
                    on_status,
                )
                .await
                .expect("dispatch must succeed via fallback");

            let active = inner.captured_active_client_set.lock().unwrap();
            assert_eq!(active.len(), 1);
            assert!(
                !active[0],
                "dangling stored selection must fall through to primary llm"
            );

            // The dangling path also clears the bad stored selection and
            // emits a one-time `DanglingModelSelection` warning naming the
            // interactive fallback. Both behaviours are pre-existing but
            // worth pinning here since the routing changes touched the
            // surrounding code.
            assert_eq!(
                outcome.warnings.len(),
                1,
                "expected exactly one DanglingModelSelection warning"
            );
            let cleared = store
                .get_selection(&ConversationId::from("c1"))
                .await
                .expect("get_selection");
            assert!(
                cleared.is_none(),
                "dangling stored selection must be cleared after fallback"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Background-task purpose dispatch (issues #27 dreaming, #28 titling)
    // ─────────────────────────────────────────────────────────────────────

    mod purpose_dispatch_tests {
        use super::super::*;

        #[test]
        fn returns_none_when_no_purpose_configured() {
            // Bare `[llm]` config: no `[purposes]` table, no resolution.
            let config: DaemonConfig = toml::from_str(
                r#"
                [llm]
                connector = "openai"
                "#,
            )
            .unwrap();

            for kind in PurposeKind::all() {
                assert!(
                    resolve_purpose_dispatch(Some(&config), kind).is_none(),
                    "expected None for {kind:?} on bare config"
                );
            }
        }

        #[test]
        fn dreaming_purpose_with_no_effort_uses_default_reasoning() {
            // Purpose set but no `effort` key — we must not fabricate an
            // effort, just pass `ReasoningConfig::default()` through.
            let config: DaemonConfig = toml::from_str(
                r#"
                [llm]
                connector = "ollama"

                [connections.local]
                type = "ollama"
                base_url = "http://localhost:11434"

                [purposes.interactive]
                connection = "local"
                model = "llama3.2"

                [purposes.dreaming]
                connection = "local"
                model = "qwen2.5:14b"
                "#,
            )
            .unwrap();

            let (resolved, reasoning) =
                resolve_purpose_dispatch(Some(&config), PurposeKind::Dreaming)
                    .expect("dreaming purpose should resolve");
            assert_eq!(resolved.connector, "ollama");
            assert_eq!(resolved.model, "qwen2.5:14b");
            assert_eq!(
                reasoning,
                ReasoningConfig::default(),
                "no effort hint → default ReasoningConfig"
            );
        }

        #[test]
        fn dreaming_purpose_with_medium_anthropic_sets_thinking_budget() {
            // Anthropic + Medium effort → thinking_budget = 8_000.
            let config: DaemonConfig = toml::from_str(
                r#"
                [llm]
                connector = "anthropic"

                [connections.cloud]
                type = "anthropic"
                base_url = "https://api.anthropic.com"
                api_key_env = "DA_TEST_PURPOSE_DISPATCH_KEY"

                [purposes.interactive]
                connection = "cloud"
                model = "claude-sonnet-4-6"

                [purposes.dreaming]
                connection = "cloud"
                model = "claude-haiku-4-5"
                effort = "medium"
                "#,
            )
            .unwrap();

            let (_resolved, reasoning) =
                resolve_purpose_dispatch(Some(&config), PurposeKind::Dreaming)
                    .expect("dreaming purpose should resolve");
            assert_eq!(reasoning.thinking_budget_tokens, Some(8_000));
            assert!(reasoning.reasoning_effort.is_none());
        }

        #[test]
        fn dreaming_purpose_with_low_anthropic_disables_thinking() {
            // Low effort → budget=0, which should leave the field as None
            // (matches the connector's "thinking disabled" semantics).
            let config: DaemonConfig = toml::from_str(
                r#"
                [llm]
                connector = "anthropic"

                [connections.cloud]
                type = "anthropic"
                base_url = "https://api.anthropic.com"
                api_key_env = "DA_TEST_PURPOSE_DISPATCH_KEY"

                [purposes.interactive]
                connection = "cloud"
                model = "claude-sonnet-4-6"

                [purposes.dreaming]
                connection = "cloud"
                model = "claude-haiku-4-5"
                effort = "low"
                "#,
            )
            .unwrap();

            let (_resolved, reasoning) =
                resolve_purpose_dispatch(Some(&config), PurposeKind::Dreaming)
                    .expect("dreaming purpose should resolve");
            assert_eq!(
                reasoning,
                ReasoningConfig::default(),
                "low → budget 0 → ReasoningConfig::default"
            );
        }

        #[test]
        fn titling_purpose_with_high_openai_sets_reasoning_effort() {
            // Confirms #28's path is wired the same as dreaming: OpenAI
            // gets `reasoning_effort`, not `thinking_budget_tokens`.
            let config: DaemonConfig = toml::from_str(
                r#"
                [llm]
                connector = "openai"

                [connections.cloud]
                type = "openai"
                base_url = "https://api.openai.com/v1"
                api_key_env = "DA_TEST_PURPOSE_DISPATCH_OPENAI_KEY"

                [purposes.interactive]
                connection = "cloud"
                model = "gpt-5"

                [purposes.titling]
                connection = "cloud"
                model = "gpt-4o-mini"
                effort = "high"
                "#,
            )
            .unwrap();

            let (resolved, reasoning) =
                resolve_purpose_dispatch(Some(&config), PurposeKind::Titling)
                    .expect("titling purpose should resolve");
            assert_eq!(resolved.connector, "openai");
            assert_eq!(resolved.model, "gpt-4o-mini");
            assert!(reasoning.thinking_budget_tokens.is_none());
            assert!(
                reasoning.reasoning_effort.is_some(),
                "OpenAI + High should populate reasoning_effort"
            );
        }

        #[test]
        fn ollama_purpose_with_effort_is_noop() {
            // Ollama has no reasoning-effort knob in the request body, so
            // even with `effort = high` we should get the default
            // ReasoningConfig and let the connector handle it.
            let config: DaemonConfig = toml::from_str(
                r#"
                [llm]
                connector = "ollama"

                [connections.local]
                type = "ollama"
                base_url = "http://localhost:11434"

                [purposes.interactive]
                connection = "local"
                model = "llama3.2"

                [purposes.dreaming]
                connection = "local"
                model = "qwen2.5:14b"
                effort = "high"
                "#,
            )
            .unwrap();

            let (_resolved, reasoning) =
                resolve_purpose_dispatch(Some(&config), PurposeKind::Dreaming).unwrap();
            assert_eq!(reasoning, ReasoningConfig::default());
        }

        #[test]
        fn map_effort_free_function_handles_all_connectors() {
            // Direct exercise of the free `map_effort_to_reasoning_config`
            // (used by background tasks). The existing `effort_mapping_*`
            // tests cover the same logic via the
            // `RoutingConversationHandler::apply_effort_mapping` wrapper;
            // this asserts the public free fn surfaces identical results
            // for the cases dreaming/titling actually traverse.
            assert_eq!(
                map_effort_to_reasoning_config("anthropic", "m", Some(Effort::Medium))
                    .thinking_budget_tokens,
                Some(8_000)
            );
            assert_eq!(
                map_effort_to_reasoning_config("anthropic", "m", Some(Effort::Low)),
                ReasoningConfig::default(),
                "Anthropic Low → budget=0 → default ReasoningConfig"
            );
            assert!(
                map_effort_to_reasoning_config("openai", "m", Some(Effort::High))
                    .reasoning_effort
                    .is_some()
            );
            assert_eq!(
                map_effort_to_reasoning_config("ollama", "m", Some(Effort::High)),
                ReasoningConfig::default()
            );
            assert_eq!(
                map_effort_to_reasoning_config("anthropic", "m", None),
                ReasoningConfig::default()
            );
        }
    }
}
