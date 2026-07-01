//! Daemon-side implementation of the connection/purpose management API
//! plus the wrapper that threads per-send overrides through the core
//! `ConversationHandler`.
//!
//! Architecture:
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
//!
//!   Stored-but-dangling selections are detected, cleared, and surfaced
//!   via a one-time [`DispatchWarning::DanglingModelSelection`].
//!
//! Per-send model selection priority is `override → stored → interactive`:
//! the explicit override on the request wins; if none, fall back to the
//! conversation's last stored selection; if neither is usable, dispatch
//! through the interactive purpose's default.

use std::sync::{Arc, Mutex};

use parking_lot::RwLock;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, ConversationSummary};
use desktop_assistant_core::ports::inbound::{
    ConnectionAvailability as CoreConnectionAvailability, ConnectionConfigPayload,
    ConnectionView as CoreConnectionView, ConnectionsService, ConversationModelSelection,
    ConversationService, DispatchWarning, ModelListing as CoreModelListing, PromptDispatchOutcome,
    PromptSelectionOverride, PurposeConfigPayload, PurposesView as CorePurposesView,
};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, ReasoningConfig, ReasoningLevel, StatusCallback, with_context_budget,
    with_model_override, with_personality, with_reasoning_config, with_system_refinement,
};
use desktop_assistant_core::ports::store::LearnedWindowStore;
use desktop_assistant_core::prompts::{Personality, PersonalityOverride};

use crate::config::{
    DaemonConfig, default_daemon_config_path, load_daemon_config, save_daemon_config,
};
use crate::connections::{
    AnthropicConnection, BedrockConnection, ConnectionConfig, ConnectionId, OllamaConnection,
    OpenAiConnection,
};
use crate::purposes::{ConnectionRef, Effort, ModelRef, PurposeConfig, PurposeKind};
use crate::registry::{ConnectionHealth, ConnectionRegistry, build_registry};

/// Shared, mutable handle to the registry + current config.
///
/// `state` is a **non-poisoning** [`parking_lot::RwLock`] (DT-9 / #276): a
/// panic while a holder has the lock must not poison it and cascade into a
/// daemon-wide outage that systemd never sees. Reads take a read lock and
/// clone out whatever they need; the data lock is held only to read or to
/// swap in a freshly built state — never across blocking I/O.
///
/// `write_serializer` serializes *mutators* (config-file write + registry
/// rebuild). Those steps run **outside** the data lock so concurrent readers
/// never stall on disk I/O; the serializer prevents two concurrent mutators
/// from racing (read-modify-write on the config) and losing an update, while
/// still computing the new config/registry off the data lock and grabbing the
/// data write lock only for the final swap.
pub struct RegistryHandle {
    state: RwLock<RegistryState>,
    write_serializer: Mutex<()>,
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
            write_serializer: Mutex::new(()),
            config_path: default_daemon_config_path(),
        }
    }

    pub fn with_config_path(mut self, path: std::path::PathBuf) -> Self {
        self.config_path = path;
        self
    }

    /// Snapshot of every connection status — used for list/validate paths.
    fn connection_views(&self) -> Vec<CoreConnectionView> {
        let state = self.state.read();
        state
            .registry
            .status()
            .into_iter()
            .map(|st| {
                let healthy = matches!(st.health, ConnectionHealth::Ok);
                // Echo the stored non-secret config so clients can pre-fill an
                // edit dialog. `connection_to_payload` drops the keyring
                // `secret` coordinates; the payload type has no field for them.
                let config = state
                    .config
                    .connections
                    .get(st.id.as_str())
                    .map(connection_to_payload);
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
                    config,
                }
            })
            .collect()
    }

    #[allow(dead_code)]
    fn is_healthy(&self, id: &ConnectionId) -> bool {
        let state = self.state.read();
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
    ) -> Option<std::sync::Arc<dyn desktop_assistant_core::ports::llm::LlmClient>> {
        let state = self.state.read();
        state.registry.get(id)
    }

    /// Connector-type tag for a given connection id, if declared.
    pub(crate) fn connector_type_for(&self, id: &ConnectionId) -> Option<String> {
        let state = self.state.read();
        state
            .registry
            .status_of(id)
            .map(|s| s.connector_type.clone())
    }

    /// Mutate the config: callers provide a closure that operates on the
    /// current `DaemonConfig`. On success we rewrite the config file and
    /// rebuild the registry.
    ///
    /// The expensive, fallible steps — writing the config file and rebuilding
    /// the registry — run **outside** the data `RwLock` (DT-9 / #276) so they
    /// never stall concurrent readers (a turn dispatch resolving a client, the
    /// settings GET, etc.) for the duration of disk I/O. We hold the data
    /// write lock only to (a) clone the current config in and (b) swap the new
    /// config + registry in, both O(1)-ish under the lock.
    ///
    /// `write_serializer` makes the read-modify-write atomic *with respect to
    /// other mutators*: it is held for the whole clone→apply→save→rebuild→swap
    /// sequence so two concurrent mutators can't both read the same base
    /// config and clobber each other's change (lost update). Readers are never
    /// blocked by it — it guards mutators only. If a previous mutator panicked,
    /// `parking_lot::Mutex` does not poison, so recovery is automatic.
    fn mutate_config<F>(&self, op: F) -> Result<(), CoreError>
    where
        F: FnOnce(&mut DaemonConfig) -> Result<(), String>,
    {
        // Serialize mutators (not readers). parking_lot::Mutex is
        // non-poisoning, so a prior panicked mutator doesn't wedge this path.
        let _writer = self.write_serializer.lock();

        // Clone the current config out under a *brief* read lock, then drop it
        // so the closure, file write, and rebuild all run unlocked.
        let mut new_config = self.state.read().config.clone();
        op(&mut new_config).map_err(CoreError::Llm)?;

        // Blocking I/O + registry rebuild — performed with NO data lock held.
        save_daemon_config(&self.config_path, &new_config)
            .map_err(|e| CoreError::Storage(format!("saving config: {e}")))?;
        let registry = build_registry(&new_config);

        // Final swap: take the write lock only long enough to install the new
        // state. No I/O, no rebuild, no user closure under the lock.
        let mut state = self.state.write();
        state.config = new_config;
        state.registry = registry;
        Ok(())
    }

    /// Read-only snapshot of the current `DaemonConfig`. Used by purposes
    /// and model-listing paths.
    pub fn snapshot_config(&self) -> DaemonConfig {
        self.state.read().config.clone()
    }

    /// The active assistant personality (issue #226). Read from the in-memory
    /// config, which `mutate_config` (and `set_personality`) keep current, so
    /// the dispatch wrapper and the settings GET observe the same value and a
    /// `SetConfig` takes effect on the next turn without a separate reload.
    pub fn personality(&self) -> Personality {
        self.state.read().config.personality
    }

    /// Update the active assistant personality. Persists to the config file and
    /// refreshes the in-memory config (via `mutate_config`) so the next send's
    /// task-local reflects the change. Cheap — the registry rebuild it triggers
    /// only re-reads connection config, which is unchanged here.
    pub fn set_personality(&self, personality: Personality) -> Result<(), CoreError> {
        self.mutate_config(|cfg| {
            cfg.personality = personality;
            Ok(())
        })
    }

    /// Test-only: swap the in-memory `DaemonConfig` and rebuild the
    /// registry, bypassing disk persistence. Lets unit tests exercise the
    /// "config mutation visible on next dispatch" property without
    /// touching the user's real config file.
    #[cfg(test)]
    pub(crate) fn replace_config_for_test(&self, config: DaemonConfig) {
        let registry = build_registry(&config);
        let mut state = self.state.write();
        state.config = config;
        state.registry = registry;
    }

    /// Validate the on-disk config and, if it parses and the registry
    /// rebuilds, swap it in under the lock — a state-preserving hot reload
    /// (#222).
    ///
    /// Non-breaking swap: the registry stores clients as `Arc<dyn LlmClient>`,
    /// and dispatch clones the `Arc` it needs *before* awaiting (see
    /// `client_for` / `send_prompt_with_override`). Replacing `state.registry`
    /// here only drops the registry's own references; any in-flight turn that
    /// already cloned its client keeps that client alive by refcount until the
    /// turn finishes, while new turns resolve through the freshly built
    /// registry. Active connections and turns are never torn down.
    ///
    /// Validate-before-apply: a config that fails to parse/validate is
    /// refused — the method logs a clear error and returns `Err` while the
    /// last-good config and registry keep running untouched. A reload never
    /// panics or exits the daemon on a bad config. Subsystems wired once at
    /// startup (database, embeddings, TLS, …) are reported as
    /// "restart required" rather than silently dropped.
    ///
    /// Returns the [`ReloadPlan`] describing what was applied (and what still
    /// needs a restart) on success.
    pub fn apply_reload(&self) -> anyhow::Result<crate::config::ReloadPlan> {
        // 1. Parse + validate the candidate from disk. `load_daemon_config`
        //    surfaces TOML and [connections]/[purposes] validation errors. A
        //    failure here returns Err and leaves the running state untouched.
        let new_config = match load_daemon_config(&self.config_path) {
            Ok(Some(cfg)) => cfg,
            Ok(None) => {
                tracing::warn!(
                    "config reload: {} is missing or empty; keeping the running config",
                    self.config_path.display()
                );
                anyhow::bail!("config file is missing or empty");
            }
            Err(e) => {
                tracing::error!(
                    "config reload refused: {} failed to parse/validate: {e:#}; \
                     keeping the last-good running config",
                    self.config_path.display()
                );
                return Err(e);
            }
        };

        // 2. Build the candidate registry off the lock. `build_registry` is
        //    infallible (bad connections become `Unavailable` rows rather than
        //    aborting), but we refuse a config that yields *zero* usable
        //    connections when the running one had at least one — that would
        //    silently break every new turn. The running registry stays put.
        let new_registry = build_registry(&new_config);
        {
            let state = self.state.read();
            let plan = crate::config::plan_reload(&state.config, &new_config);
            if plan.is_empty() {
                tracing::info!("config reload: no effective changes; nothing to apply");
                return Ok(plan);
            }
            if state.registry.live_count() > 0 && new_registry.live_count() == 0 {
                tracing::error!(
                    "config reload refused: the new config has no usable LLM connection \
                     (every connection failed to build); keeping the last-good running config"
                );
                anyhow::bail!("new config has no usable LLM connection");
            }
        }

        // 3. Re-diff and swap under the write lock. Re-reading `state.config`
        //    here (rather than trusting the read-lock snapshot above) keeps the
        //    plan consistent if a concurrent `mutate_config` slipped in.
        let mut state = self.state.write();
        let plan = crate::config::plan_reload(&state.config, &new_config);
        state.config = new_config;
        // Swapping the registry drops only its own Arc handles; in-flight turns
        // that already cloned their client keep it alive (see method docs).
        state.registry = new_registry;
        drop(state);

        if plan.rebuild_registry {
            tracing::info!("config reload applied: connection registry rebuilt for new turns");
        }
        if plan.needs_restart() {
            tracing::warn!(
                "config reload: these changes need a daemon restart to take effect: {}",
                plan.restart_required.join(", ")
            );
        }
        Ok(plan)
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
            cfg.connections
                .insert(id_valid.as_str().to_string(), new_conn);
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
                let names: Vec<&'static str> = referenced_by.iter().map(|k| k.as_key()).collect();
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
                        if let Some(p) = cfg.purposes.get_mut(PurposeKind::Interactive) {
                            p.connection = ConnectionRef::Named(
                                ConnectionId::new(new_id)
                                    .expect("existing key was already validated"),
                            );
                        }
                    } else {
                        // No connections left — clear interactive entirely.
                        cfg.purposes.set(PurposeKind::Interactive, None);
                    }
                    continue;
                }
                if let Some(p) = cfg.purposes.get_mut(kind) {
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
        // returned future `!Send`; cloning `Arc<dyn LlmClient>` releases the
        // lock up front and the awaits run unlocked.
        let targets: Vec<(
            ConnectionId,
            String,
            String,
            std::sync::Arc<dyn desktop_assistant_core::ports::llm::LlmClient>,
        )> = {
            let state = self.registry.state.read();
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
            interactive: config
                .purposes
                .get(PurposeKind::Interactive)
                .map(purpose_to_payload),
            dreaming: config
                .purposes
                .get(PurposeKind::Dreaming)
                .map(purpose_to_payload),
            consolidation: config
                .purposes
                .get(PurposeKind::Consolidation)
                .map(purpose_to_payload),
            embedding: config
                .purposes
                .get(PurposeKind::Embedding)
                .map(purpose_to_payload),
            titling: config
                .purposes
                .get(PurposeKind::Titling)
                .map(purpose_to_payload),
        })
    }

    async fn set_purpose(
        &self,
        purpose: PurposeKind,
        config: PurposeConfigPayload,
    ) -> Result<(), CoreError> {
        let purpose_kind = purpose;
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
            cfg.purposes.validate().map_err(|e| format!("{e}"))
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
    ) -> impl std::future::Future<Output = Result<Option<ConversationModelSelection>, CoreError>> + Send;

    fn set_selection(
        &self,
        id: &ConversationId,
        selection: Option<&ConversationModelSelection>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    /// Read the conversation's stored personality override (#227), or `None`
    /// when no override is pinned. Mirrors [`Self::get_selection`].
    fn get_personality(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<Option<PersonalityOverride>, CoreError>> + Send;

    /// Set (or clear, with `None`) the conversation's personality override
    /// (#227). Mirrors [`Self::set_selection`].
    fn set_personality(
        &self,
        id: &ConversationId,
        personality: Option<&PersonalityOverride>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;
}

/// The complete per-turn LLM dispatch decision, resolved ONCE at the turn
/// boundary from the live config + the conversation's effective selection.
///
/// The point is a single source of truth: the context budget and the model that
/// actually runs are derived from the *same* resolution, so they cannot drift.
/// Historically they could: the budget read the live interactive purpose while
/// dispatch fell through to a construction-time *static primary* client, so a
/// stale primary could execute a different model than the budget was computed
/// for (logs reporting model A while model B ran). Resolving everything here,
/// once, closes that class of bug.
///
/// Populated incrementally (by design): today it carries the routing target,
/// the model override, the reasoning config, and the context budget. Other
/// per-turn decisions (e.g. personality) can move onto it in follow-ups so every
/// derived value shares this one resolution.
struct ResolvedTurn {
    /// The registry client this turn dispatches through. `None` means no
    /// concrete *live* connection was resolved — the interactive purpose defers
    /// to the `[llm]` primary (`connection`/`model = primary`), or its named
    /// connection isn't live — so dispatch falls through to the handler's static
    /// primary llm, exactly as before (#33).
    active_client: Option<Arc<dyn desktop_assistant_core::ports::llm::LlmClient>>,
    /// Model id pinned via the `MODEL_OVERRIDE` task-local. `Some` exactly when
    /// `active_client` is `Some` — the per-call knob that lets a single
    /// connection client run a chosen model without a construction-time rebuild.
    model_override: Option<String>,
    /// Per-connector reasoning/effort config for this turn.
    reasoning: ReasoningConfig,
    /// Context budget computed for the model that will actually run.
    budget: desktop_assistant_core::ports::llm::ContextBudget,
    /// `(connection_id, model_id)` actually chosen — what to log, so the budget
    /// line reports what runs rather than a separately-derived guess. `None`
    /// when deferring to the static primary.
    chosen: Option<(String, String)>,
}

pub struct RoutingConversationHandler<S, Inner>
where
    S: ConversationSelectionStore + 'static,
    Inner: ConversationService + 'static,
{
    inner: Arc<Inner>,
    selection_store: Arc<S>,
    registry: Arc<RegistryHandle>,
    /// Learned context-window cache (issue #343). When present, an
    /// observed-overflow ceiling for the resolved `(connector, model)` caps the
    /// per-turn budget DOWN (see [`crate::config::apply_learned_cap`]). `None`
    /// (tests, no database) disables the safety net; resolution is unchanged.
    window_store: Option<Arc<dyn LearnedWindowStore>>,
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
            window_store: None,
        }
    }

    /// Install the learned context-window cache (issue #343) so budget
    /// resolution applies the DOWN-only observed-overflow cap.
    pub fn with_window_store(mut self, window_store: Arc<dyn LearnedWindowStore>) -> Self {
        self.window_store = Some(window_store);
        self
    }

    /// Resolve the interactive purpose from the current config. Used as
    /// the ultimate fallback (priority #3) when neither an override nor a
    /// valid stored selection exists.
    fn interactive_selection(&self) -> Option<ConversationModelSelection> {
        let cfg = self.registry.snapshot_config();
        cfg.purposes.get(PurposeKind::Interactive).and_then(|p| {
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
                effort: p.effort,
            })
        })
    }

    /// Resolve the effective personality for a send (#227, Phase 2):
    /// conversation override (partial) → global config → built-in default.
    ///
    /// The global config already folds in the built-in default (an absent
    /// `[personality]` block resolves to `Personality::default()`), so the
    /// merge is just "per-trait override over the global". When the
    /// conversation has no stored override the global personality is returned
    /// unchanged — identical to Phase-1 behaviour. A failed lookup logs and
    /// falls back to the global so a storage hiccup never blocks a turn.
    async fn resolve_personality(&self, conversation_id: &ConversationId) -> Personality {
        let global = self.registry.personality();
        match self.selection_store.get_personality(conversation_id).await {
            Ok(Some(ovr)) => ovr.resolve(&global),
            Ok(None) => global,
            Err(e) => {
                tracing::warn!(
                    conversation_id = %conversation_id.0,
                    "failed to read conversation personality override; using global: {e}"
                );
                global
            }
        }
    }

    /// Check a stored selection against the live registry. Returns
    /// `(is_still_valid)`. When invalid, the caller is responsible for
    /// clearing the stored selection and emitting a warning.
    async fn selection_is_live(&self, sel: &ConversationModelSelection) -> Result<bool, CoreError> {
        let Ok(id) = ConnectionId::new(sel.connection_id.clone()) else {
            return Ok(false);
        };
        self.registry
            .connection_lists_model(&id, &sel.model_id)
            .await
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

    /// Resolve the whole per-turn dispatch decision once (see [`ResolvedTurn`]).
    ///
    /// `effective` is the turn's effective selection — a user-driven
    /// override/stored pick, else the interactive purpose. `user_driven` is
    /// `Some` only when the user actually chose a model this turn; it changes the
    /// not-live policy: a user-driven pick on a dead connection is a hard error
    /// (never silently route elsewhere), whereas the interactive-purpose fallback
    /// degrades to the static primary so a misconfigured purpose can't block a
    /// turn.
    ///
    /// Routing target, model override, reasoning, and budget are all derived
    /// from `effective` here, so the model the budget is computed for is exactly
    /// the model dispatched. When `effective` names a concrete, live connection
    /// we route through its registry client and pin the model per-call via the
    /// `MODEL_OVERRIDE` task-local — this covers both a user-driven selection and
    /// the interactive fallback, replacing the old behaviour where the fallback
    /// fell through to a construction-time static primary (which could be stale).
    async fn resolve_turn(
        &self,
        user_driven: Option<&ConversationModelSelection>,
        effective: Option<&ConversationModelSelection>,
    ) -> Result<ResolvedTurn, CoreError> {
        let mut active_client = None;
        let mut model_override = None;
        let mut reasoning = ReasoningConfig::default();
        let mut chosen = None;

        if let Some(sel) = effective {
            let id = ConnectionId::new(sel.connection_id.clone()).map_err(|e| {
                CoreError::Llm(format!(
                    "resolved selection has malformed connection id {:?}: {e}",
                    sel.connection_id
                ))
            })?;
            let connector_type = self.registry.connector_type_for(&id).unwrap_or_default();
            reasoning = Self::apply_effort_mapping(&connector_type, &sel.model_id, sel.effort);

            match self.registry.client_for(&id) {
                Some(client) => {
                    // Concrete, live connection: route through the registry
                    // client and pin the model per-call. Dispatch now follows the
                    // SAME live resolution the budget does — for a user-driven
                    // selection AND the interactive fallback — instead of a
                    // construction-time static primary that could be stale.
                    active_client = Some(client);
                    model_override = Some(sel.model_id.clone());
                    chosen = Some((sel.connection_id.clone(), sel.model_id.clone()));
                }
                None if user_driven.is_some() => {
                    // The user explicitly picked this connection — fail loudly
                    // rather than silently routing somewhere else.
                    return Err(CoreError::Llm(format!(
                        "resolved connection {} is not live; requested model {} cannot be dispatched",
                        sel.connection_id, sel.model_id
                    )));
                }
                None => {
                    // Interactive-purpose fallback to a non-live connection:
                    // degrade to the static primary (active_client stays None)
                    // instead of failing the turn (#33's spirit).
                    tracing::warn!(
                        connection = %sel.connection_id,
                        model = %sel.model_id,
                        "interactive purpose connection is not live; falling through to the primary llm"
                    );
                }
            }
        }

        // Context budget for the model that will ACTUALLY run. Tier 1: the
        // interactive purpose's `max_context_tokens` override. Tier 2: the
        // resolved client's curated window — the same client chosen above, so
        // budget and dispatch agree. Tier 3: the universal fallback. Then cap
        // DOWN to any learned overflow ceiling (#343). When `active_client` is
        // None (static-primary passthrough) tier 2 is unavailable and we fall to
        // the universal default, exactly as before.
        let purpose_override = crate::config::purpose_max_context_override(
            Some(&self.registry.snapshot_config()),
            PurposeKind::Interactive,
        );
        let connector_max = active_client.as_ref().and_then(|c| c.max_context_tokens());
        let mut budget = crate::config::resolve_context_budget(purpose_override, connector_max);
        if let (Some(store), Some(sel)) = (self.window_store.as_ref(), effective) {
            let connector = ConnectionId::new(sel.connection_id.clone())
                .ok()
                .map(|id| self.registry.connector_type_for(&id).unwrap_or_default())
                .unwrap_or_default();
            match store.lookup(&connector, &sel.model_id).await {
                Ok(learned) => budget = crate::config::apply_learned_cap(budget, learned),
                Err(e) => {
                    tracing::warn!(error = %e, "learned-window lookup failed; using resolved budget")
                }
            }
        }

        Ok(ResolvedTurn {
            active_client,
            model_override,
            reasoning,
            budget,
            chosen,
        })
    }
}

/// Resolve a purpose's full dispatch config — `(ResolvedLlmConfig,
/// ReasoningConfig)` — for background tasks that want to honour
/// `[purposes.<kind>]` end-to-end (dreaming, titling, etc.).
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
        .and_then(|p| p.effort);
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

#[async_trait::async_trait]
impl<S, Inner> ConversationService for RoutingConversationHandler<S, Inner>
where
    S: ConversationSelectionStore + 'static,
    Inner: ConversationService + 'static,
{
    async fn create_conversation(&self, title: String, tags: Vec<String>) -> Result<Conversation, CoreError> {
        self.inner.create_conversation(title, tags).await
    }

    async fn list_conversations(
        &self,
        max_age_days: Option<u32>,
        include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        self.inner
            .list_conversations(max_age_days, include_archived)
            .await
    }

    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        self.inner.get_conversation(id).await
    }

    async fn get_conversation_model_selection(
        &self,
        id: &ConversationId,
    ) -> Result<Option<ConversationModelSelection>, CoreError> {
        self.selection_store.get_selection(id).await
    }

    async fn get_conversation_personality(
        &self,
        id: &ConversationId,
    ) -> Result<Option<PersonalityOverride>, CoreError> {
        self.selection_store.get_personality(id).await
    }

    async fn set_conversation_personality(
        &self,
        id: &ConversationId,
        personality: PersonalityOverride,
    ) -> Result<(), CoreError> {
        // An empty (all-`None`) override means "no override" — clear the column
        // (store `None`) so a later `GetConversation` reports no override and
        // the send path falls back to global-only, rather than persisting an
        // empty object that resolves to the global anyway.
        let to_store = if personality.is_empty() {
            None
        } else {
            Some(&personality)
        };
        self.selection_store.set_personality(id, to_store).await
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
        //
        // Issue #109: pass a fresh, never-tripped `CancellationToken`
        // since this entry has no cancel knob yet. Adapters that want
        // cancellation must call `send_prompt_with_override` directly.
        let outcome = self
            .send_prompt_with_override(
                conversation_id,
                prompt,
                None,
                String::new(),
                on_chunk,
                on_status,
                tokio_util::sync::CancellationToken::new(),
            )
            .await?;
        Ok(outcome.response)
    }

    async fn send_prompt_with_override(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        override_selection: Option<PromptSelectionOverride>,
        system_refinement: String,
        on_chunk: ChunkCallback,
        on_status: StatusCallback,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<PromptDispatchOutcome, CoreError> {
        let mut warnings: Vec<DispatchWarning> = Vec::new();

        // Resolve the effective selection following priority:
        //   1. override (validate first; hard error if invalid)
        //   2. stored conversation selection (validate; warn + fallback if dangling)
        //   3. interactive purpose
        //
        // We track *user_driven* separately from *effective*: the
        // user-driven path (override / live stored) routes through the
        // registry's per-connection client, while the
        // interactive-fallback path routes through the handler's static
        // primary llm, which is already built with the interactive
        // purpose's model baked in.
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
                effort: override_sel.effort,
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

        // Resolve the whole per-turn dispatch decision ONCE (routing target,
        // model override, reasoning, context budget) from the effective
        // selection, so the model the budget is computed for is exactly the
        // model dispatched. See [`ResolvedTurn`] for why this is one resolution.
        let ResolvedTurn {
            active_client,
            model_override,
            reasoning,
            budget,
            chosen,
        } = self
            .resolve_turn(user_driven_selection.as_ref(), effective_selection.as_ref())
            .await?;

        tracing::info!(
            purpose = ?PurposeKind::Interactive,
            connection = ?chosen.as_ref().map(|(c, _)| c.as_str()),
            model = ?chosen.as_ref().map(|(_, m)| m.as_str()),
            source = ?budget.source,
            max_input_tokens = budget.max_input_tokens,
            "context budget resolved"
        );

        // Install task-locals, then delegate to the inner core
        // handler. The handler reads the task-locals inside its
        // `send_prompt` dispatch loop:
        //   - `RoutingLlmClient` picks the active client on each
        //     `stream_completion` call.
        //   - `current_context_budget()` surfaces the resolved budget for
        //     token-pressure compaction.
        //   - `current_reasoning_config()` surfaces `reasoning` into the
        //     connector's request body.
        //   - `current_model_override()` surfaces the resolved `model_id`
        //     so connectors send the user-chosen model rather than
        //     `self.model` (the connection's startup default).
        //   - `current_cancellation_token()` (issue #109) surfaces the
        //     per-turn cancellation token so the agentic loop and each
        //     LLM adapter can `tokio::select!` against it.
        // Resolve the effective personality for this send (#227, Phase 2):
        // conversation override (partial) → global config → built-in default.
        // Computed before the dispatch block so the lookup's `&self` borrow
        // doesn't outlive the `'static` dispatch future.
        let effective_personality = self.resolve_personality(conversation_id).await;

        // Capture the ambient "now" once per turn and render the line the core
        // assembler surfaces as a `[Now]` system message, giving the assistant a
        // standing sense of the current date/time. Rendered from the same
        // `NowSnapshot` logic that backs `builtin_sys_props`, so the ambient
        // block and the tool never disagree. Captured here (before the dispatch
        // future) so every assembly pass in the turn sees one stable value.
        let now_line = desktop_assistant_core::clock::NowSnapshot::now().ambient_line();

        let inner = Arc::clone(&self.inner);
        let conv_id = conversation_id.clone();
        let response = {
            let dispatch = async move {
                inner
                    .send_prompt(&conv_id, prompt, on_chunk, on_status)
                    .await
            };
            // Install the per-request system-prompt refinement so the core
            // context assembler appends it to this turn's system prompt. Empty
            // string = no refinement (unchanged prompt). It is request-scoped
            // and never persisted; see `SYSTEM_REFINEMENT`.
            let dispatch = with_system_refinement(system_refinement, dispatch);
            // Install the active personality (#226/#227). The effective value is
            // resolved above as: conversation override (partial) → global config
            // → built-in default. With no stored override this equals the global
            // personality, identical to Phase-1 behaviour; the core read side
            // (`current_personality`) is unchanged.
            let dispatch = with_personality(effective_personality, dispatch);
            // Install the ambient "now" line so the core assembler surfaces a
            // `[Now]` system message for this turn. Request-scoped, never
            // persisted; see `NOW_CONTEXT`.
            let dispatch =
                desktop_assistant_core::ports::llm::with_now_context(now_line, dispatch);
            let dispatch = with_reasoning_config(reasoning, dispatch);
            let dispatch = with_context_budget(budget, dispatch);
            let dispatch =
                desktop_assistant_core::ports::llm::with_cancellation_token(cancellation, dispatch);
            // Route through the resolved registry client + pinned model when
            // `resolve_turn` found a concrete live connection (a user-driven
            // selection OR the interactive purpose naming an explicit
            // connection+model). When it didn't — the interactive purpose defers
            // to the `[llm]` primary (`connection`/`model = primary`) or its
            // connection isn't live — both are `None` and dispatch falls through
            // to the static primary llm, preserving #33's passthrough for that
            // case. `active_client` and `model_override` are always set together.
            match (active_client, model_override) {
                (Some(c), Some(m)) => {
                    let dispatch = with_model_override(m, dispatch);
                    crate::routing_llm::with_active_client(c, dispatch).await
                }
                (Some(c), None) => crate::routing_llm::with_active_client(c, dispatch).await,
                (None, _) => dispatch.await,
            }
        }?;
        Ok(PromptDispatchOutcome { response, warnings })
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

/// OpenAI `reasoning_effort` wire literal for an effort hint.
///
/// Composed from [`map_effort_to_reasoning_level`] +
/// [`ReasoningLevel::as_openai_effort`] so the Effort → wire-token
/// mapping has exactly one source of truth and the two paths cannot
/// drift. Currently only used by tests; kept on the public surface
/// because future connectors that surface `reasoning_effort` directly
/// (vs going through `ReasoningConfig`) will want it.
#[allow(dead_code)]
pub fn map_openai_reasoning_effort(e: Effort) -> &'static str {
    map_effort_to_reasoning_level(e).as_openai_effort()
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
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfig::Anthropic(AnthropicConnection {
            base_url,
            api_key_env,
            secret: None,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        }),
        ConnectionConfigPayload::OpenAi {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfig::OpenAi(OpenAiConnection {
            base_url,
            api_key_env,
            secret: None,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        }),
        ConnectionConfigPayload::Bedrock {
            aws_profile,
            region,
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfig::Bedrock(BedrockConnection {
            aws_profile,
            region,
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        }),
        ConnectionConfigPayload::Ollama {
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            keep_warm,
            max_context_tokens,
        } => ConnectionConfig::Ollama(OllamaConnection {
            base_url,
            connect_timeout_secs,
            stream_timeout_secs,
            keep_warm,
            max_context_tokens,
        }),
    }
}

/// Inverse of [`payload_to_connection`]: project a stored [`ConnectionConfig`]
/// down to the protocol-neutral, **non-secret** [`ConnectionConfigPayload`]
/// echoed back through `ConnectionView`.
///
/// Only endpoint/profile/region fields and the credential *env-var name*
/// (`api_key_env`) cross this boundary. The keyring `secret` coordinates on
/// the Anthropic/OpenAI variants are deliberately dropped — the payload type
/// has no field for them, so a raw secret can never be reconstructed from the
/// echoed value.
fn connection_to_payload(conn: &ConnectionConfig) -> ConnectionConfigPayload {
    match conn {
        ConnectionConfig::Anthropic(c) => ConnectionConfigPayload::Anthropic {
            base_url: c.base_url.clone(),
            api_key_env: c.api_key_env.clone(),
            // `c.secret` (keyring coordinates) intentionally not echoed.
            connect_timeout_secs: c.connect_timeout_secs,
            stream_timeout_secs: c.stream_timeout_secs,
            max_context_tokens: c.max_context_tokens,
        },
        ConnectionConfig::OpenAi(c) => ConnectionConfigPayload::OpenAi {
            base_url: c.base_url.clone(),
            api_key_env: c.api_key_env.clone(),
            // `c.secret` (keyring coordinates) intentionally not echoed.
            connect_timeout_secs: c.connect_timeout_secs,
            stream_timeout_secs: c.stream_timeout_secs,
            max_context_tokens: c.max_context_tokens,
        },
        ConnectionConfig::Bedrock(c) => ConnectionConfigPayload::Bedrock {
            aws_profile: c.aws_profile.clone(),
            region: c.region.clone(),
            base_url: c.base_url.clone(),
            connect_timeout_secs: c.connect_timeout_secs,
            stream_timeout_secs: c.stream_timeout_secs,
            max_context_tokens: c.max_context_tokens,
        },
        ConnectionConfig::Ollama(c) => ConnectionConfigPayload::Ollama {
            base_url: c.base_url.clone(),
            connect_timeout_secs: c.connect_timeout_secs,
            stream_timeout_secs: c.stream_timeout_secs,
            keep_warm: c.keep_warm,
            max_context_tokens: c.max_context_tokens,
        },
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
        effort: p.effort,
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
        effort: p.effort,
        max_context_tokens: p.max_context_tokens,
    })
}

fn purposes_referencing(
    purposes: &crate::purposes::Purposes,
    id: &ConnectionId,
) -> Vec<PurposeKind> {
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
    use desktop_assistant_core::prompts::PersonalityLevel;

    use std::sync::Mutex;

    /// Trivial in-memory `ConversationSelectionStore` for the daemon test
    /// suite. Production code uses the Postgres-backed store via the
    /// storage crate.
    pub struct InMemoryConversationSelectionStore {
        inner: Mutex<std::collections::HashMap<String, ConversationModelSelection>>,
        personality: Mutex<std::collections::HashMap<String, PersonalityOverride>>,
    }

    impl Default for InMemoryConversationSelectionStore {
        fn default() -> Self {
            Self {
                inner: Mutex::new(std::collections::HashMap::new()),
                personality: Mutex::new(std::collections::HashMap::new()),
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

        async fn get_personality(
            &self,
            id: &ConversationId,
        ) -> Result<Option<PersonalityOverride>, CoreError> {
            Ok(self
                .personality
                .lock()
                .expect("selection store poisoned")
                .get(&id.0)
                .copied())
        }

        async fn set_personality(
            &self,
            id: &ConversationId,
            personality: Option<&PersonalityOverride>,
        ) -> Result<(), CoreError> {
            let mut map = self.personality.lock().expect("selection store poisoned");
            match personality {
                Some(p) => {
                    map.insert(id.0.clone(), *p);
                }
                None => {
                    map.remove(&id.0);
                }
            }
            Ok(())
        }
    }

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
            ..Default::default()
        })
    }

    fn bedrock_work() -> ConnectionConfig {
        ConnectionConfig::Bedrock(BedrockConnection {
            aws_profile: Some("work".into()),
            region: Some("us-west-2".into()),
            base_url: None,
            ..Default::default()
        })
    }

    /// Anthropic connection carrying a keyring `secret` reference alongside
    /// the non-secret `base_url` / `api_key_env`. Used to prove the echoed
    /// view drops the secret coordinates.
    fn anthropic_with_secret() -> ConnectionConfig {
        use crate::config::SecretConfig;
        ConnectionConfig::Anthropic(crate::connections::AnthropicConnection {
            base_url: Some("https://api.anthropic.com".into()),
            api_key_env: Some("ANTHROPIC_WORK_KEY".into()),
            secret: Some(SecretConfig {
                account: Some("super-secret-account".into()),
                entry: Some("super-secret-entry".into()),
                ..SecretConfig::default()
            }),
            ..Default::default()
        })
    }

    fn make_handle_with(cfg: DaemonConfig) -> Arc<RegistryHandle> {
        let registry = build_registry(&cfg);
        Arc::new(RegistryHandle::new(cfg, registry).with_config_path(tmp_config_path()))
    }

    #[tokio::test]
    async fn list_connections_returns_declared_order() {
        let cfg = config_with_connections(&[("local", ollama_local()), ("aws", bedrock_work())]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let views = svc.list_connections().await.unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].id, "local");
        assert_eq!(views[1].id, "aws");
    }

    #[tokio::test]
    async fn list_connections_echoes_non_secret_config() {
        let cfg = config_with_connections(&[("aws", bedrock_work())]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let views = svc.list_connections().await.unwrap();
        assert_eq!(views.len(), 1);

        let config = views[0]
            .config
            .as_ref()
            .expect("ConnectionView should echo the stored non-secret config");
        match config {
            ConnectionConfigPayload::Bedrock {
                aws_profile,
                region,
                base_url,
                ..
            } => {
                assert_eq!(aws_profile.as_deref(), Some("work"));
                assert_eq!(region.as_deref(), Some("us-west-2"));
                assert_eq!(base_url.as_deref(), None);
            }
            other => panic!("expected echoed Bedrock config, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn list_connections_echoes_config_without_leaking_secret() {
        let cfg = config_with_connections(&[("work", anthropic_with_secret())]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let views = svc.list_connections().await.unwrap();
        assert_eq!(views.len(), 1);

        let config = views[0]
            .config
            .as_ref()
            .expect("ConnectionView should echo the stored non-secret config");
        match config {
            ConnectionConfigPayload::Anthropic {
                base_url,
                api_key_env,
                ..
            } => {
                assert_eq!(base_url.as_deref(), Some("https://api.anthropic.com"));
                assert_eq!(api_key_env.as_deref(), Some("ANTHROPIC_WORK_KEY"));
            }
            other => panic!("expected echoed Anthropic config, got {other:?}"),
        }

        // The keyring `secret` coordinates (account/entry/etc.) must never
        // surface in the echoed view. The payload type has no field for them,
        // so prove it via a full debug-string scan of every view.
        let dump = format!("{views:?}");
        assert!(
            !dump.contains("super-secret-account") && !dump.contains("super-secret-entry"),
            "echoed ConnectionView leaked secret coordinates: {dump}"
        );
    }

    #[tokio::test]
    async fn create_connection_rejects_bad_slug() {
        let svc = DaemonConnectionsService::new(make_handle_with(DaemonConfig::default()));
        let err = svc
            .create_connection(
                "Bad Id!".to_string(),
                ConnectionConfigPayload::Ollama {
                    base_url: Some("http://localhost:11434".into()),
                    connect_timeout_secs: None,
                    stream_timeout_secs: None,
                    keep_warm: None,
                    max_context_tokens: None,
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
                    connect_timeout_secs: None,
                    stream_timeout_secs: None,
                    keep_warm: None,
                    max_context_tokens: None,
                },
            )
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("already exists"));
    }

    #[tokio::test]
    async fn delete_connection_refuses_when_referenced_without_force() {
        let mut cfg =
            config_with_connections(&[("local", ollama_local()), ("aws", bedrock_work())]);
        cfg.purposes.set(
            PurposeKind::Interactive,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("llama3".into()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        cfg.purposes.set(
            PurposeKind::Dreaming,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("aws").unwrap()),
                model: ModelRef::Named("claude".into()),
                effort: None,
                max_context_tokens: None,
            }),
        );

        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let err = svc
            .delete_connection("aws".to_string(), false)
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("referenced"));
    }

    #[tokio::test]
    async fn delete_connection_force_cascades_to_primary() {
        let mut cfg =
            config_with_connections(&[("local", ollama_local()), ("aws", bedrock_work())]);
        cfg.purposes.set(
            PurposeKind::Interactive,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("llama3".into()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        cfg.purposes.set(
            PurposeKind::Dreaming,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("aws").unwrap()),
                model: ModelRef::Named("claude".into()),
                effort: None,
                max_context_tokens: None,
            }),
        );

        let handle = make_handle_with(cfg);
        let svc = DaemonConnectionsService::new(Arc::clone(&handle));
        svc.delete_connection("aws".to_string(), true)
            .await
            .unwrap();

        let cfg = handle.snapshot_config();
        assert!(!cfg.connections.contains_key("aws"));
        let dreaming = cfg
            .purposes
            .get(PurposeKind::Dreaming)
            .expect("dreaming still set");
        assert!(matches!(dreaming.connection, ConnectionRef::Primary));
    }

    #[tokio::test]
    async fn set_purpose_rejects_primary_in_interactive() {
        let cfg = config_with_connections(&[("local", ollama_local())]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let err = svc
            .set_purpose(
                PurposeKind::Interactive,
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
        cfg.purposes.set(
            PurposeKind::Interactive,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("llama3".into()),
                effort: Some(Effort::Medium),
                max_context_tokens: None,
            }),
        );
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        let view = svc.get_purposes().await.unwrap();
        let i = view.interactive.expect("interactive set");
        assert_eq!(i.connection, "local");
        assert_eq!(i.model, "llama3");
        assert_eq!(i.effort, Some(Effort::Medium));
    }

    // ----- RegistryHandle lock robustness (DT-9 / #276) ----------------
    //
    // Two invariants:
    //  1. A panic while a holder has the lock must NOT poison it — every
    //     subsequent acquirer must still succeed (no daemon-wide cascade).
    //  2. `mutate_config` must NOT hold the data lock across its blocking
    //     file I/O + registry rebuild — concurrent readers must not stall
    //     for the duration of the disk write.

    /// A panicking lock holder must not poison the lock: the next acquirer
    /// (here a `snapshot_config` read) must still succeed rather than
    /// inheriting a poisoned-lock panic.
    #[test]
    fn panicked_holder_does_not_poison_lock() {
        let cfg = config_with_connections(&[("local", ollama_local())]);
        let handle = make_handle_with(cfg);

        // Spawn a thread that panics from *inside* `mutate_config`'s
        // closure — i.e. while the write lock is held in the old code. With
        // a poisoning std::RwLock this leaves the lock permanently poisoned.
        let h = Arc::clone(&handle);
        let res = std::thread::spawn(move || {
            let _ = h.mutate_config(|_cfg| {
                panic!("holder panicked while holding the write lock");
            });
        })
        .join();
        assert!(res.is_err(), "the holder thread should have panicked");

        // With a poisoning std::RwLock this read would itself panic
        // (poison cascade). It must succeed.
        let snap = handle.snapshot_config();
        assert!(snap.connections.contains_key("local"));

        // A subsequent mutate must also still work.
        let svc = DaemonConnectionsService::new(Arc::clone(&handle));
        // mutate via set_personality (cheap, no connection rebuild needed)
        handle
            .set_personality(snap.personality)
            .expect("mutate after a poisoned-holder panic must still succeed");
        // and a read path through the service:
        let _ = svc; // service constructed fine; lock usable
    }

    /// `mutate_config` must drop the data lock before doing its blocking
    /// file write + registry rebuild. We prove it by pointing the config
    /// path at a FIFO with no reader: `save_daemon_config`'s `open(O_WRONLY)`
    /// blocks forever. A concurrent `snapshot_config` read must still
    /// complete promptly — it would hang if the write lock were held across
    /// the I/O.
    #[cfg(unix)]
    #[test]
    fn mutate_config_does_not_hold_lock_across_blocking_io() {
        use std::sync::mpsc;
        use std::time::Duration;

        // Build a FIFO path. open(O_WRONLY) on a FIFO blocks until a reader
        // appears, which never happens here — a deterministic "slow I/O".
        let dir = std::env::temp_dir();
        let fifo = dir.join(format!(
            "da-test-fifo-{}.toml",
            uuid::Uuid::new_v4().simple()
        ));
        let cstr = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        let rc = unsafe { libc::mkfifo(cstr.as_ptr(), 0o600) };
        assert_eq!(rc, 0, "mkfifo failed");

        let cfg = config_with_connections(&[("local", ollama_local())]);
        let registry = build_registry(&cfg);
        let handle = Arc::new(RegistryHandle::new(cfg, registry).with_config_path(fifo.clone()));

        // Writer thread: this mutate will block inside the file write
        // (open on the readerless FIFO) and never return.
        let writer = Arc::clone(&handle);
        std::thread::spawn(move || {
            let _ = writer.set_personality(desktop_assistant_core::prompts::Personality::default());
        });

        // Give the writer time to reach (and block in) the file write.
        std::thread::sleep(Duration::from_millis(200));

        // Reader: must complete promptly. If the write lock were held across
        // the blocked I/O, this read would hang and the recv would time out.
        let reader = Arc::clone(&handle);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let snap = reader.snapshot_config();
            let _ = tx.send(snap.connections.contains_key("local"));
        });

        match rx.recv_timeout(Duration::from_secs(2)) {
            Ok(found) => assert!(found, "reader saw the expected config"),
            Err(_) => panic!("snapshot_config hung — the write lock is held across blocking I/O"),
        }

        let _ = std::fs::remove_file(&fifo);
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
        let cfg =
            config_with_connections(&[("local1", ollama_local()), ("local2", ollama_local())]);
        let svc = DaemonConnectionsService::new(make_handle_with(cfg));
        // Either the network fails (empty list) or succeeds — both are OK
        // since we're just checking we don't hard-error when aggregating.
        let _ = svc.list_available_models(None, false).await;
    }

    // ----- Hot-reload (apply_reload) tests (#222) ----------------------
    //
    // These cover the state-preserving swap:
    // - an in-flight turn's cloned `Arc<dyn LlmClient>` stays alive across a
    //   reload (registry swap drops only the registry's own handles)
    // - a malformed config is refused without disturbing the running state
    // - a valid edit swaps the config + registry in place

    mod hot_reload {
        use super::*;

        /// Write `toml` to a fresh temp path and return a handle whose
        /// `config_path` points at it, so `apply_reload` reads our file.
        fn handle_for_toml(toml: &str) -> (Arc<RegistryHandle>, std::path::PathBuf) {
            let path = tmp_config_path();
            std::fs::write(&path, toml).expect("write initial config");
            let cfg = crate::config::load_daemon_config(&path)
                .expect("initial config parses")
                .expect("initial config present");
            let registry = build_registry(&cfg);
            let handle =
                Arc::new(RegistryHandle::new(cfg, registry).with_config_path(path.clone()));
            (handle, path)
        }

        const OLLAMA_A: &str = r#"
[connections.local]
type = "ollama"
base_url = "http://localhost:11434"
"#;

        #[test]
        fn in_flight_turn_client_survives_reload() {
            // Simulate an in-flight turn: dispatch clones the `Arc<dyn
            // LlmClient>` before awaiting. Hold that clone across a reload and
            // assert the underlying client is NOT dropped — the registry swap
            // must rely on refcounts, not forcibly tear clients down.
            let (handle, path) = handle_for_toml(OLLAMA_A);
            let id = ConnectionId::new("local").unwrap();

            // The "in-flight turn" grabs its client up front.
            let in_flight = handle.client_for(&id).expect("client present");
            let weak = Arc::downgrade(&in_flight);
            assert!(weak.upgrade().is_some());

            // Edit the connection's base_url and reload. This rebuilds the
            // registry — the swap drops the registry's own Arc but our
            // in-flight clone must keep the old client alive.
            std::fs::write(
                &path,
                r#"
[connections.local]
type = "ollama"
base_url = "http://localhost:9999"
"#,
            )
            .unwrap();
            let plan = handle.apply_reload().expect("valid reload applies");
            assert!(plan.rebuild_registry, "a connection edit rebuilds");
            assert!(!plan.needs_restart());

            // The in-flight turn's client is still alive (refcount held by our
            // clone), even though the registry now serves a different client.
            assert!(
                weak.upgrade().is_some(),
                "the registry swap must not drop a client an in-flight turn still holds"
            );
            // New turns resolve through the freshly built registry.
            assert!(handle.client_for(&id).is_some());

            // Drop the in-flight clone; now the old client can be reclaimed.
            drop(in_flight);
            assert!(
                weak.upgrade().is_none(),
                "once the in-flight turn finishes, the old client is reclaimed"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn malformed_config_is_refused_without_disturbing_running_state() {
            let (handle, path) = handle_for_toml(OLLAMA_A);
            let id = ConnectionId::new("local").unwrap();
            let before = handle.snapshot_config();
            let live_before = handle.client_for(&id).is_some();
            assert!(live_before, "the good config has a live client");

            // Garbage TOML on disk.
            std::fs::write(&path, "this is not = valid toml [[[").unwrap();
            let err = handle
                .apply_reload()
                .expect_err("a malformed config must be refused");
            assert!(!format!("{err:#}").is_empty());

            // Running state is untouched: same config, same live client.
            let after = handle.snapshot_config();
            assert_eq!(
                toml::to_string(&before).unwrap(),
                toml::to_string(&after).unwrap(),
                "a refused reload must leave the last-good config in place"
            );
            assert!(
                handle.client_for(&id).is_some(),
                "a refused reload must not drop the running registry's clients"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn reload_with_no_changes_is_a_noop() {
            let (handle, path) = handle_for_toml(OLLAMA_A);
            // Rewrite identical content (an editor save with no edits).
            std::fs::write(&path, OLLAMA_A).unwrap();
            let plan = handle.apply_reload().expect("identical config applies");
            assert!(plan.is_empty(), "an unchanged config is a no-op reload");
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn valid_edit_swaps_config_and_registry() {
            let (handle, path) = handle_for_toml(OLLAMA_A);
            assert!(
                handle
                    .client_for(&ConnectionId::new("local").unwrap())
                    .is_some()
            );

            // Add a second connection.
            std::fs::write(
                &path,
                r#"
[connections.local]
type = "ollama"
base_url = "http://localhost:11434"

[connections.other]
type = "ollama"
base_url = "http://localhost:11435"
"#,
            )
            .unwrap();
            let plan = handle.apply_reload().expect("valid reload applies");
            assert!(plan.rebuild_registry);
            // The new connection is now routable.
            assert!(
                handle
                    .client_for(&ConnectionId::new("other").unwrap())
                    .is_some(),
                "a reload that adds a connection makes it routable for new turns"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn reload_refused_when_new_config_has_no_usable_connection() {
            // Start good (ollama is healthy), then edit to an openai
            // connection with no api key — every connection fails to build.
            // The reload must be refused so new turns don't all break.
            let unused = format!("DA_TEST_RELOAD_KEY_{}", uuid::Uuid::new_v4().simple());
            // SAFETY: unique name, single-threaded test.
            unsafe {
                std::env::remove_var(&unused);
            }
            let (handle, path) = handle_for_toml(OLLAMA_A);
            let id = ConnectionId::new("local").unwrap();
            assert!(handle.client_for(&id).is_some());

            std::fs::write(
                &path,
                format!(
                    r#"
[connections.cloud]
type = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "{unused}"
"#
                ),
            )
            .unwrap();
            let err = handle
                .apply_reload()
                .expect_err("a config with no usable connection must be refused");
            assert!(
                format!("{err:#}").contains("no usable LLM connection"),
                "refusal should explain the cause: {err:#}"
            );
            // The original healthy connection is still live.
            assert!(
                handle.client_for(&id).is_some(),
                "a refused reload keeps the last-good registry"
            );
            let _ = std::fs::remove_file(&path);
        }
    }

    // ----- RoutingConversationHandler dispatch-routing tests -----------
    //
    // These tests cover the per-turn routing logic:
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
            /// behaviour for the interactive-purpose fallback path.
            captured_active_client_set: StdMutex<Vec<bool>>,
            /// Snapshot of the `MODEL_OVERRIDE` task-local at each
            /// `send_prompt`. `None` means no override was installed —
            /// connectors will fall back to their baked-in `self.model`.
            captured_model_override: StdMutex<Vec<Option<String>>>,
            /// Snapshot of the `PERSONALITY` task-local (#227) at each
            /// `send_prompt`. Asserting on this proves the routing wrapper
            /// resolved the conversation override against the global config and
            /// installed the effective personality on the dispatch scope.
            captured_personality: StdMutex<Vec<Personality>>,
            /// Snapshot of the `CONTEXT_BUDGET` task-local (#343) at each
            /// `send_prompt` — proves the resolved (and possibly learned-capped)
            /// budget reaches the dispatch scope.
            captured_budget:
                StdMutex<Vec<Option<desktop_assistant_core::ports::llm::ContextBudget>>>,
        }

        impl CapturingInner {
            fn new() -> Self {
                Self {
                    captured_reasoning: StdMutex::new(Vec::new()),
                    captured_active_client_set: StdMutex::new(Vec::new()),
                    captured_model_override: StdMutex::new(Vec::new()),
                    captured_personality: StdMutex::new(Vec::new()),
                    captured_budget: StdMutex::new(Vec::new()),
                }
            }
        }

        #[async_trait::async_trait]
        impl ConversationService for CapturingInner {
            async fn create_conversation(&self, title: String, _tags: Vec<String>) -> Result<Conversation, CoreError> {
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
            async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn rename_conversation(
                &self,
                _id: &ConversationId,
                _title: String,
            ) -> Result<(), CoreError> {
                Ok(())
            }
            async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
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
                let model = desktop_assistant_core::ports::llm::current_model_override();
                self.captured_model_override.lock().unwrap().push(model);
                let personality = desktop_assistant_core::ports::llm::current_personality();
                self.captured_personality.lock().unwrap().push(personality);
                let budget = desktop_assistant_core::ports::llm::current_context_budget();
                self.captured_budget.lock().unwrap().push(budget);
                Ok("ok".to_string())
            }
        }

        fn local_ollama_cfg() -> DaemonConfig {
            let mut cfg =
                config_with_connections(&[("local", ollama_local()), ("aws", bedrock_work())]);
            cfg.purposes.set(
                PurposeKind::Interactive,
                Some(PurposeConfig {
                    connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                    model: ModelRef::Named("llama3".into()),
                    effort: None,
                    max_context_tokens: None,
                }),
            );
            cfg
        }

        // One-off test fixture tuple; a type alias would only add indirection
        // for a helper used solely within this test module.
        #[allow(clippy::type_complexity)]
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

        // ─── Issue #227: per-conversation personality resolution at send ───

        #[tokio::test]
        async fn send_installs_global_personality_when_no_conversation_override() {
            // With no stored override, the personality task-local the inner
            // handler observes must equal the registry's global personality —
            // identical to Phase-1 behaviour.
            let (routing, inner, registry, _store) = make_handler();
            let global = registry.personality();

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

            let captured = inner.captured_personality.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(
                captured[0], global,
                "no override → the global personality must be installed verbatim"
            );
        }

        // ─── Issue #343: learned context-window cap at dispatch ───────────

        /// Window-store double returning a fixed learned observation.
        struct FixedWindowStore(Option<desktop_assistant_core::ports::store::LearnedWindow>);
        #[async_trait::async_trait]
        impl LearnedWindowStore for FixedWindowStore {
            async fn lookup(
                &self,
                _connector: &str,
                _model: &str,
            ) -> Result<Option<desktop_assistant_core::ports::store::LearnedWindow>, CoreError>
            {
                Ok(self.0)
            }
            async fn record_overflow(
                &self,
                _connector: &str,
                _model: &str,
                _observed_limit: u64,
                _configured_window: u64,
            ) -> Result<(), CoreError> {
                Ok(())
            }
            async fn record_success(
                &self,
                _connector: &str,
                _model: &str,
                _input_tokens: u64,
            ) -> Result<(), CoreError> {
                Ok(())
            }
        }

        /// End-to-end (issue #343): a turn-1 overflow learned a 4096 ceiling
        /// under the same configured window the resolver produces (8192, the
        /// Ollama effective num_ctx). On the NEXT turn budget resolution caps
        /// DOWN to 4096, so the dispatch scope sees the smaller budget — the
        /// turn no longer assumes the too-large window that overflowed.
        #[tokio::test]
        async fn learned_window_caps_budget_down_on_next_turn() {
            let cfg = local_ollama_cfg();
            let registry = make_handle_with(cfg);
            let inner = Arc::new(CapturingInner::new());
            let store = Arc::new(InMemoryConversationSelectionStore::default());
            // Resolver yields 8192 for this dead-ollama connection (the
            // configured effective num_ctx). The learned row matches that
            // configured window and observed 4096, so it must cap DOWN.
            let window = Arc::new(FixedWindowStore(Some(
                desktop_assistant_core::ports::store::LearnedWindow {
                    observed_limit: Some(4_096),
                    configured_window: Some(8_192),
                    max_success_input: None,
                },
            )));
            let routing = Arc::new(
                RoutingConversationHandler::new(
                    Arc::clone(&inner),
                    Arc::clone(&store),
                    Arc::clone(&registry),
                )
                .with_window_store(window),
            );

            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("send_prompt");

            let captured = inner.captured_budget.lock().unwrap();
            let budget = captured[0].expect("budget installed");
            assert_eq!(
                budget.max_input_tokens, 4_096,
                "next turn must start under the learned ceiling, not the 8192 window that overflowed"
            );
            assert_eq!(
                budget.source,
                desktop_assistant_core::ports::llm::BudgetSource::LearnedCap
            );
        }

        /// Invalidation end-to-end: a learned observation recorded under a
        /// DIFFERENT configured window than the resolver now produces is stale
        /// and must NOT cap — the budget reflects the fresh resolved window.
        #[tokio::test]
        async fn stale_learned_window_does_not_cap_budget() {
            let cfg = local_ollama_cfg();
            let registry = make_handle_with(cfg);
            let inner = Arc::new(CapturingInner::new());
            let store = Arc::new(InMemoryConversationSelectionStore::default());
            // Observed 4096, but under an OLD 2048 configured window — the
            // resolver now produces 8192, so this row is stale and ignored.
            let window = Arc::new(FixedWindowStore(Some(
                desktop_assistant_core::ports::store::LearnedWindow {
                    observed_limit: Some(4_096),
                    configured_window: Some(2_048),
                    max_success_input: None,
                },
            )));
            let routing = Arc::new(
                RoutingConversationHandler::new(
                    Arc::clone(&inner),
                    Arc::clone(&store),
                    Arc::clone(&registry),
                )
                .with_window_store(window),
            );

            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("send_prompt");

            let captured = inner.captured_budget.lock().unwrap();
            let budget = captured[0].expect("budget installed");
            assert_eq!(
                budget.max_input_tokens, 8_192,
                "a learned row under a different configured window is stale and must not cap"
            );
            assert_ne!(
                budget.source,
                desktop_assistant_core::ports::llm::BudgetSource::LearnedCap
            );
        }

        #[tokio::test]
        async fn send_installs_resolved_override_over_global() {
            // A stored partial override must be resolved against the global
            // config (override wins per-trait, unspecified traits fall back)
            // and the *resolved* personality installed on the dispatch scope.
            let (routing, inner, registry, store) = make_handler();
            let global = registry.personality();

            // "No-nonsense" override: force humor off, directness max; leave the
            // rest to fall back to the global.
            let ovr = PersonalityOverride {
                humor: Some(PersonalityLevel::Never),
                directness: Some(PersonalityLevel::Always),
                ..PersonalityOverride::default()
            };
            store
                .set_personality(&ConversationId::from("c1"), Some(&ovr))
                .await
                .unwrap();

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

            let captured = inner.captured_personality.lock().unwrap();
            assert_eq!(captured.len(), 1);
            let installed = captured[0];
            // Pinned traits win.
            assert_eq!(installed.humor, PersonalityLevel::Never);
            assert_eq!(installed.directness, PersonalityLevel::Always);
            // Unspecified traits fall back to the global.
            assert_eq!(installed.professionalism, global.professionalism);
            assert_eq!(installed.warmth, global.warmth);
            assert_eq!(installed.sarcasm, global.sarcasm);
            // Exactly the per-trait merge of the override over the global.
            assert_eq!(installed, ovr.resolve(&global));
        }

        #[tokio::test]
        async fn set_then_get_conversation_personality_round_trips_and_clears() {
            // The routing wrapper's setter/getter persist through the store;
            // an empty override clears it (getter reports None).
            let (routing, _inner, _reg, _store) = make_handler();
            let id = ConversationId::from("c1");

            let ovr = PersonalityOverride {
                sarcasm: Some(PersonalityLevel::Never),
                ..PersonalityOverride::default()
            };
            routing
                .set_conversation_personality(&id, ovr)
                .await
                .unwrap();
            assert_eq!(
                routing.get_conversation_personality(&id).await.unwrap(),
                Some(ovr)
            );

            // Empty override clears the stored value.
            routing
                .set_conversation_personality(&id, PersonalityOverride::default())
                .await
                .unwrap();
            assert_eq!(
                routing.get_conversation_personality(&id).await.unwrap(),
                None,
                "an all-None override must clear the stored override"
            );
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
                    String::new(),
                    on_chunk,
                    on_status,
                    tokio_util::sync::CancellationToken::new(),
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
                c.purposes.set(
                    PurposeKind::Interactive,
                    Some(PurposeConfig {
                        connection: ConnectionRef::Named(ConnectionId::new("aws").unwrap()),
                        model: ModelRef::Named("us.anthropic.claude-sonnet-4-6".into()),
                        effort: None,
                        max_context_tokens: None,
                    }),
                );
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
            >::apply_effort_mapping(
                "anthropic", "claude-sonnet-4-6", Some(Effort::Low)
            );
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
            >::apply_effort_mapping(
                "mystery-vendor", "m1", Some(Effort::High)
            );
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
        async fn interactive_purpose_installs_active_client_for_concrete_connection() {
            // No override, no stored selection → the interactive purpose drives
            // the turn. When that purpose names a concrete, *live* connection
            // (the fixture's `local`/`llama3`), dispatch now routes through the
            // registry client (ACTIVE_CLIENT set) and pins the model per-call —
            // the SAME live resolution the budget uses — instead of falling
            // through to a construction-time static primary that could be stale.
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
                active[0],
                "an interactive purpose naming a concrete live connection must \
                 route through the registry client so dispatch matches the budget"
            );
        }

        #[tokio::test]
        async fn interactive_purpose_effort_still_applies() {
            // The purpose's effort flows through the reasoning task-local. Use
            // ollama so the connector mapping is a no-op (default
            // ReasoningConfig) — the assertion is that we got the expected
            // default, not that we lost the effort entirely. A non-ollama
            // connector can't be exercised end-to-end without a live model list,
            // so the bedrock-effort case is covered by the unit test on
            // `apply_effort_mapping` above.
            let mut cfg = local_ollama_cfg();
            cfg.purposes.set(
                PurposeKind::Interactive,
                Some(PurposeConfig {
                    connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                    model: ModelRef::Named("llama3".into()),
                    effort: Some(Effort::High),
                    max_context_tokens: None,
                }),
            );
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
            // the test is that the effort still flowed through the resolution.
            assert_eq!(reasoning[0], ReasoningConfig::default());

            // The concrete live connection routes through the registry client.
            let active = inner.captured_active_client_set.lock().unwrap();
            assert!(active[0]);
        }

        #[tokio::test]
        async fn interactive_purpose_dispatch_installs_model_override() {
            // With no user-driven selection, the interactive purpose drives the
            // turn. When it names a concrete live connection, `MODEL_OVERRIDE` is
            // pinned to the purpose's model (`llama3`) so the connector sends
            // exactly that — the same model the budget was computed for — rather
            // than relying on the static primary's construction-time model.
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
            let captured = inner.captured_model_override.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(
                captured[0],
                Some("llama3".to_string()),
                "interactive purpose must pin its model so dispatch matches the budget"
            );
        }

        #[tokio::test]
        async fn override_dispatch_installs_model_override_task_local() {
            // Issue #34 happy path: a `send_prompt_with_override` whose
            // resolved selection picks a non-default model results in that
            // `model_id` reaching the per-turn `MODEL_OVERRIDE` task-local
            // observed inside the inner `send_prompt`. We use httpmock to
            // satisfy the `connection_lists_model` validation gate.
            let server = httpmock::MockServer::start();

            // Validation calls `list_models()` which on Ollama hits
            // `/api/tags` and (for models with details) `/api/show`. We
            // need both `llama3.2` (the connection default) and our
            // override target `qwen3` to be present.
            let _tags = server.mock(|when, then| {
                when.method(httpmock::Method::GET).path("/api/tags");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(
                        r#"{"models":[
                            {"name":"llama3.2","model":"llama3.2","digest":"sha256:aaa"},
                            {"name":"qwen3","model":"qwen3","digest":"sha256:bbb"}
                        ]}"#,
                    );
            });
            // `/api/show` is called per-model to enrich context limits;
            // a 404 is harmless — the connector skips context limits.
            let _show = server.mock(|when, then| {
                when.method(httpmock::Method::POST).path("/api/show");
                then.status(404).body("not found");
            });

            let cfg = {
                let mut c = config_with_connections(&[(
                    "local",
                    ConnectionConfig::Ollama(OllamaConnection {
                        base_url: Some(server.url("")),
                        ..Default::default()
                    }),
                )]);
                c.purposes.set(
                    PurposeKind::Interactive,
                    Some(PurposeConfig {
                        connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                        model: ModelRef::Named("llama3.2".into()),
                        effort: None,
                        max_context_tokens: None,
                    }),
                );
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

            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt_with_override(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    Some(PromptSelectionOverride {
                        connection_id: "local".into(),
                        // Pick a model that differs from the connection's
                        // baked-in default (`llama3.2`) so the assertion
                        // is meaningful — without `MODEL_OVERRIDE` the
                        // connector would dispatch `self.model` and
                        // silently drop this.
                        model_id: "qwen3".into(),
                        effort: None,
                    }),
                    String::new(),
                    on_chunk,
                    on_status,
                    tokio_util::sync::CancellationToken::new(),
                )
                .await
                .expect("override dispatch should succeed via mocked /api/tags");

            let captured = inner.captured_model_override.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(
                captured[0],
                Some("qwen3".to_string()),
                "MODEL_OVERRIDE must carry the resolved override model id"
            );
            // And the active-client task-local must also be set, since
            // the override-driven path always routes through the
            // registry rather than the primary llm.
            let active = inner.captured_active_client_set.lock().unwrap();
            assert!(active[0]);
        }

        #[tokio::test]
        async fn override_with_default_model_still_installs_override() {
            // Determinism: even when the user picks the connection's
            // default model, `send_prompt_with_override` installs
            // `MODEL_OVERRIDE` so dispatch does not silently rely on
            // `self.model`. Eliminates a sometimes-set state.
            let server = httpmock::MockServer::start();
            let _tags = server.mock(|when, then| {
                when.method(httpmock::Method::GET).path("/api/tags");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(
                        r#"{"models":[{"name":"llama3.2","model":"llama3.2","digest":"sha256:aaa"}]}"#,
                    );
            });
            let _show = server.mock(|when, then| {
                when.method(httpmock::Method::POST).path("/api/show");
                then.status(404).body("not found");
            });

            let cfg = {
                let mut c = config_with_connections(&[(
                    "local",
                    ConnectionConfig::Ollama(OllamaConnection {
                        base_url: Some(server.url("")),
                        ..Default::default()
                    }),
                )]);
                c.purposes.set(
                    PurposeKind::Interactive,
                    Some(PurposeConfig {
                        connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                        model: ModelRef::Named("llama3.2".into()),
                        effort: None,
                        max_context_tokens: None,
                    }),
                );
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

            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt_with_override(
                    &ConversationId::from("c1"),
                    "hi".into(),
                    Some(PromptSelectionOverride {
                        connection_id: "local".into(),
                        model_id: "llama3.2".into(),
                        effort: None,
                    }),
                    String::new(),
                    on_chunk,
                    on_status,
                    tokio_util::sync::CancellationToken::new(),
                )
                .await
                .expect("default-model override should succeed");

            let captured = inner.captured_model_override.lock().unwrap();
            assert_eq!(
                captured[0],
                Some("llama3.2".to_string()),
                "MODEL_OVERRIDE installs even when override matches the default"
            );
        }

        #[tokio::test]
        async fn dangling_stored_selection_falls_back_to_interactive() {
            // A stored selection pointing at a connection that's no longer
            // declared is cleared and falls back to the interactive purpose.
            // Since that purpose names a concrete live connection, the fallback
            // now routes through the registry client (ACTIVE_CLIENT set) and
            // pins its model — the dangling pick is not user-driven, so this is
            // the same fallback path the plain interactive case takes.
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
                    String::new(),
                    on_chunk,
                    on_status,
                    tokio_util::sync::CancellationToken::new(),
                )
                .await
                .expect("dispatch must succeed via fallback");

            {
                let active = inner.captured_active_client_set.lock().unwrap();
                assert_eq!(active.len(), 1);
                assert!(
                    active[0],
                    "dangling selection falls back to the interactive purpose, \
                     which routes through its concrete live connection's client"
                );
            } // drop std::sync::MutexGuard before the next .await — clippy::await_holding_lock

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

        #[tokio::test]
        async fn interactive_purpose_with_primary_ref_falls_through_to_static_primary() {
            // #33 passthrough preserved: when the interactive purpose defers to
            // the `[llm]` primary (`connection`/`model = primary`), there is no
            // concrete registry connection to pin, so `resolve_turn` leaves
            // ACTIVE_CLIENT / MODEL_OVERRIDE unset and dispatch falls through to
            // the static primary llm — exactly as before.
            let mut cfg = local_ollama_cfg();
            cfg.purposes.set(
                PurposeKind::Interactive,
                Some(PurposeConfig {
                    connection: ConnectionRef::Primary,
                    model: ModelRef::Primary,
                    effort: None,
                    max_context_tokens: None,
                }),
            );
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
                .expect("dispatch must succeed via the static primary");

            let active = inner.captured_active_client_set.lock().unwrap();
            assert_eq!(active.len(), 1);
            assert!(
                !active[0],
                "a Primary-ref interactive purpose must pass through to the \
                 static primary, not pin a registry client"
            );
            let overrides = inner.captured_model_override.lock().unwrap();
            assert_eq!(
                overrides[0], None,
                "no model override for the primary passthrough"
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
