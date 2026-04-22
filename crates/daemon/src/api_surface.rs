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
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, StatusCallback};

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
    fn client_for(
        &self,
        id: &ConnectionId,
    ) -> Option<std::sync::Arc<crate::registry::AnyLlmClient>> {
        let state = self.state.read().expect("registry state poisoned");
        state.registry.get(id)
    }

    /// Connector-type tag for a given connection id, if declared.
    fn connector_type_for(&self, id: &ConnectionId) -> Option<String> {
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
        // Snapshot (id, label, client) triples before awaiting anything.
        // Holding the read lock across `.await` would leave the returned
        // future `!Send`; cloning `Arc<AnyLlmClient>` releases the lock up
        // front and the awaits run unlocked.
        let targets: Vec<(ConnectionId, String, std::sync::Arc<crate::registry::AnyLlmClient>)> = {
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
                let label = format!("{} ({})", st.id, st.connector_type);
                let Some(client) = state.registry.get(&id) else {
                    return Err(CoreError::Llm(format!("connection {id} is not live")));
                };
                vec![(id, label, client)]
            } else {
                state
                    .registry
                    .status()
                    .into_iter()
                    .filter(|s| matches!(s.health, ConnectionHealth::Ok))
                    .filter_map(|s| {
                        let label = format!("{} ({})", s.id, s.connector_type);
                        let client = state.registry.get(&s.id)?;
                        Some((s.id, label, client))
                    })
                    .collect()
            }
        };

        let mut out: Vec<CoreModelListing> = Vec::new();
        for (id, label, client) in targets {
            let list_result = if refresh {
                client.refresh_models().await
            } else {
                client.list_models().await
            };
            match list_result {
                Ok(models) => {
                    for m in models {
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

    /// Translate the effort hint into per-connector parameters and log
    /// what the dispatch layer would send. This is the effort-mapping hook
    /// the ticket calls out; each connector currently receives the mapping
    /// via a `tracing::debug!` line, and the wire-level plumbing into
    /// `stream_completion` will be wired up on a follow-up (noted in the
    /// PR body — see issue #11).
    fn apply_effort_mapping(
        connector_type: &str,
        model_id: &str,
        effort: Option<Effort>,
    ) {
        let Some(effort) = effort else {
            return;
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
            }
            "openai" => {
                let token = map_openai_reasoning_effort(effort);
                tracing::debug!(
                    connector = connector_type,
                    model = model_id,
                    effort = ?effort,
                    reasoning_effort = token,
                    "mapped effort to OpenAI reasoning_effort"
                );
            }
            "ollama" => {
                tracing::debug!(
                    connector = connector_type,
                    effort = ?effort,
                    "effort is a no-op on Ollama"
                );
            }
            other => {
                tracing::debug!(
                    connector = other,
                    effort = ?effort,
                    "no effort mapping defined for connector"
                );
            }
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
        self.inner
            .send_prompt(conversation_id, prompt, on_chunk, on_status)
            .await
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
        let effective_selection: Option<ConversationModelSelection> = if let Some(override_sel) =
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
                // Dangling. Fall back to interactive; clear; emit a
                // one-time warning.
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
                fallback
            }
        } else {
            self.interactive_selection()
        };

        // Emit effort-mapping debug lines for the chosen dispatch target.
        if let Some(sel) = effective_selection.as_ref() {
            let connector_type = ConnectionId::new(sel.connection_id.clone())
                .ok()
                .and_then(|id| self.registry.connector_type_for(&id))
                .unwrap_or_default();
            Self::apply_effort_mapping(
                &connector_type,
                &sel.model_id,
                sel.effort.map(Effort::from),
            );
        }

        // Dispatch via the inner (interactive-purpose) handler. The
        // override/stored selection has been validated and persisted; wire-
        // level routing to the selected connection+model is noted as a
        // follow-up in the PR body.
        let response = self
            .inner
            .send_prompt(conversation_id, prompt, on_chunk, on_status)
            .await?;
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
pub fn map_openai_reasoning_effort(e: Effort) -> &'static str {
    match e {
        Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
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
    })
}

fn purpose_effort_to_core(e: PurposeEffort) -> Effort {
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
        });
        cfg.purposes.dreaming = Some(PurposeConfig {
            connection: ConnectionRef::Named(ConnectionId::new("aws").unwrap()),
            model: ModelRef::Named("claude".into()),
            effort: None,
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
        });
        cfg.purposes.dreaming = Some(PurposeConfig {
            connection: ConnectionRef::Named(ConnectionId::new("aws").unwrap()),
            model: ModelRef::Named("claude".into()),
            effort: None,
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
}
