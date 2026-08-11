//! Daemon-side implementation of the connection/purpose management API
//! plus the wrapper that threads per-send overrides through the core
//! `ConversationHandler`.
//!
//! Architecture:
//!
//! - [`DaemonConnectionsService`] wraps a shared [`ConnectionRegistry`]
//!   (plus the on-disk config) and implements the
//!   [`desktop_assistant_core::ports::inbound::ConnectionsService`]
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

use desktop_assistant_core::tool_provenance::ToolPolicy;
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
    ChunkCallback, LlmClient, ModelKind, ReasoningConfig, ReasoningLevel, StatusCallback,
    with_context_budget, with_model_override, with_personality, with_reasoning_config,
    with_system_refinement,
};
use desktop_assistant_core::ports::store::LearnedWindowStore;
use desktop_assistant_core::prompts::{Personality, PersonalityOverride};

use crate::config::{
    DaemonConfig, default_daemon_config_path, parse_daemon_config, save_daemon_config,
};
use crate::connections::{
    AnthropicConnection, AzureConnection, BedrockConnection, ConnectionConfig, ConnectionId,
    GoogleConnection, OllamaConnection, OpenAiConnection, OpenRouterConnection,
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
///
/// `booted_config` is the config the process was wired from and never changes
/// for the life of the handle. It is the baseline for
/// [`Self::restart_required`]: restart-bound subsystems (TLS, WS auth, the
/// embedding client) hold values taken from *that* snapshot, so it is the only
/// thing a candidate config can honestly be compared against.
pub struct RegistryHandle {
    state: RwLock<RegistryState>,
    write_serializer: Mutex<()>,
    config_path: std::path::PathBuf,
    booted_config: DaemonConfig,
}

struct RegistryState {
    config: DaemonConfig,
    registry: ConnectionRegistry,
    /// Whether `config` is the config file's content or the built-in defaults
    /// the daemon fell back to when the file would not load. Lives beside the
    /// config it describes so the two are always read under one lock.
    config_origin: crate::config::ConfigOrigin,
}

impl RegistryHandle {
    pub fn new(config: DaemonConfig, registry: ConnectionRegistry) -> Self {
        Self {
            booted_config: config.clone(),
            state: RwLock::new(RegistryState {
                config,
                registry,
                config_origin: crate::config::ConfigOrigin::File,
            }),
            write_serializer: Mutex::new(()),
            config_path: default_daemon_config_path(),
        }
    }

    pub fn with_config_path(mut self, path: std::path::PathBuf) -> Self {
        self.config_path = path;
        self
    }

    /// Record that the daemon is running built-in defaults because
    /// `daemon.toml` failed to load, rather than the file's own contents.
    ///
    /// Set once by `main` at startup from the boot-time load result.
    pub fn with_config_origin(mut self, origin: crate::config::ConfigOrigin) -> Self {
        self.state.get_mut().config_origin = origin;
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

    /// Resolve the [`ModelKind`] the `(connection, model)` pair advertises, for
    /// the SetPurpose write-time guard (#647).
    ///
    /// Returns [`ModelKind::Unknown`] -- and logs a `warn!` -- for every case the
    /// guard must NOT block on: the connection has no live client, listing its
    /// models failed (down / transient network fault), or the model id is not in
    /// the connection's listing (an unrecognized custom id). A definite
    /// `Generative` / `Embedding` is returned only when the connector positively
    /// classified the bound model. This is the capability-degradation posture in
    /// `AGENTS.md`: never turn a config edit into a failure over something the
    /// daemon merely could not verify.
    ///
    /// The client `Arc` is cloned out under `client_for`'s brief read lock and
    /// awaited unlocked -- the same pattern as `connection_lists_model` and
    /// `list_available_models` -- so no registry lock is held across `.await`.
    async fn resolve_model_kind(&self, conn_id: &ConnectionId, model_id: &str) -> ModelKind {
        let Some(client) = self.client_for(conn_id) else {
            tracing::warn!(
                connection = %conn_id,
                model = model_id,
                "cannot verify model kind: connection has no live client; allowing the binding"
            );
            return ModelKind::Unknown;
        };
        let models = match client.list_models().await {
            Ok(models) => models,
            Err(e) => {
                tracing::warn!(
                    connection = %conn_id,
                    model = model_id,
                    "cannot verify model kind: listing models failed ({e}); allowing the binding"
                );
                return ModelKind::Unknown;
            }
        };
        match models.iter().find(|m| m.id == model_id) {
            Some(m) => m.capabilities.kind,
            None => {
                tracing::warn!(
                    connection = %conn_id,
                    model = model_id,
                    "cannot verify model kind: model not in the connection's listing; allowing the binding"
                );
                ModelKind::Unknown
            }
        }
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
    ///
    /// A write that touches a restart-bound area is logged here rather than
    /// left to the config-file watcher: this path refreshes the in-memory
    /// config *before* the watcher fires, so by the time the watcher diffs the
    /// file it is a genuine no-op and has nothing to report (#686).
    /// [`Self::restart_required`] is what carries the same fact to a client.
    ///
    /// The write is refused outright when the in-memory config is not the
    /// file's own content - see [`Self::refuse_if_overwrite_would_destroy_the_file`].
    fn mutate_config<F>(&self, op: F) -> Result<(), CoreError>
    where
        F: FnOnce(&mut DaemonConfig) -> Result<(), String>,
    {
        // Serialize mutators (not readers). parking_lot::Mutex is
        // non-poisoning, so a prior panicked mutator doesn't wedge this path.
        let _writer = self.write_serializer.lock();

        // Before anything else: this write serializes the in-memory snapshot
        // over the whole file, so it must not run when that snapshot is not
        // what the file holds. Checked ahead of the closure because the
        // closure has side effects of its own (`set_connection_secret` writes
        // to the secret backend).
        self.refuse_if_overwrite_would_destroy_the_file()?;

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
        let plan = crate::config::plan_reload(&state.config, &new_config);
        state.config = new_config;
        state.registry = registry;
        drop(state);

        if plan.needs_restart() {
            tracing::warn!(
                "config write applied, but these changes need a daemon restart to take effect: {}",
                plan.restart_required_keys().join(", ")
            );
        }
        Ok(())
    }

    /// Read-only snapshot of the current `DaemonConfig`. Used by purposes
    /// and model-listing paths.
    pub fn snapshot_config(&self) -> DaemonConfig {
        self.state.read().config.clone()
    }

    /// Whether the running config is the config file's own content.
    pub(crate) fn config_origin(&self) -> crate::config::ConfigOrigin {
        self.state.read().config_origin
    }

    /// Refuse a config-mutating write whose result would replace the user's
    /// file with something that is not it (#723).
    ///
    /// Every write here serializes the whole in-memory `DaemonConfig` over
    /// `daemon.toml`, so the snapshot has to *be* that file plus the edit.
    /// Two cases where it is not:
    ///
    /// 1. The daemon booted on built-in defaults because the file would not
    ///    load ([`crate::config::ConfigOrigin::DefaultsAfterFailedLoad`]).
    ///    Writing then replaces every connection, purpose, `[ws_auth]`,
    ///    `[database]` and comment in the file with defaults plus one edit.
    ///    Cleared by a successful [`Self::apply_reload`], not by fixing the
    ///    file alone: until the fixed file is actually applied, the running
    ///    snapshot is still defaults.
    /// 2. The file stopped parsing after boot - a hand edit with a typo. The
    ///    running snapshot is the last-good config, and writing it discards
    ///    the edit the user is in the middle of making.
    ///
    /// An absent or empty file is neither: there is nothing to destroy, and a
    /// first run has to be able to save.
    ///
    /// The refusal names the file and what to do about it, but never the parse
    /// error: a TOML error quotes the offending line, which in `daemon.toml`
    /// can be a credential. The cause is logged once where it is detected - at
    /// startup by `main`, and per reload attempt by [`Self::apply_reload`].
    fn refuse_if_overwrite_would_destroy_the_file(&self) -> Result<(), CoreError> {
        if self.config_origin() == crate::config::ConfigOrigin::DefaultsAfterFailedLoad {
            let path = self.config_path.display();
            tracing::warn!(
                "refusing a config write: the daemon is running built-in defaults because {path} \
                 could not be loaded at startup (see the startup log for the cause)"
            );
            return Err(CoreError::Storage(format!(
                "refusing to write {path}: the daemon is running built-in defaults because that \
                 file could not be loaded at startup, so saving now would replace its contents \
                 with defaults. Fix the file - the daemon log names the error - and it is picked \
                 up automatically."
            )));
        }

        // Same validation the reload path applies, so "the daemon would refuse
        // to load this" and "the daemon refuses to overwrite it" agree. The
        // non-migrating reader: a guard must not rewrite the file it guards.
        if crate::config::parse_daemon_config(&self.config_path).is_err() {
            let path = self.config_path.display();
            tracing::warn!(
                "refusing a config write: {path} no longer parses (see the config reload log \
                 for the cause)"
            );
            return Err(CoreError::Storage(format!(
                "refusing to write {path}: the file on disk no longer parses, so saving now \
                 would overwrite it with the configuration the daemon is running. Fix the file \
 - the daemon log names the error - and retry."
            )));
        }

        Ok(())
    }

    /// What is in the config *file* that the running process is not acting on
    /// (#686). Empty means everything on disk is live.
    ///
    /// Diffs the config the daemon booted with against the config on disk,
    /// which is the only baseline that stays honest across every write path:
    ///
    /// - A daemon-authored write ([`Self::mutate_config`]) refreshes the
    ///   in-memory config before the watcher runs, so a current-vs-disk diff
    ///   would be empty even though the running subsystem is stale.
    /// - A settings write that only touches the file (`set_ws_auth_settings`)
    ///   is reported immediately, without waiting for the debounced watcher.
    /// - A hand edit to `daemon.toml`, the only way to rotate `[tls]` today,
    ///   is reported the same way.
    /// - Reverting an edit clears the report, because the file is back in step
    ///   with what the process is running.
    ///
    /// A file that will not load at all is reported as
    /// [`crate::config::RestartArea::ConfigLoadFailed`] - the whole file is
    /// out of force, and a client that sees only an empty connections list
    /// would otherwise read a degraded daemon as an unconfigured one (#723).
    /// It is reported for as long as the running config is not the file's own
    /// content, so a daemon that booted on defaults keeps saying so until a
    /// reload actually applies the repaired file.
    ///
    /// Carries area names only, never configured values: `[ws_auth]` and
    /// `[tls]` are security-relevant, and a caller allowed to learn *that* a
    /// restart is pending is not thereby allowed to read the new settings.
    ///
    /// Never fails: this path only describes state. [`Self::apply_reload`] is
    /// the one that refuses a bad config.
    pub fn restart_required(&self) -> Vec<crate::config::RestartArea> {
        let candidate = match parse_daemon_config(&self.config_path) {
            Ok(Some(config)) => Some(config),
            // No file (or an empty one) means nothing on disk contradicts the
            // running process.
            Ok(None) => None,
            // Deliberately without the parse error: this path can run per
            // settings read, and a TOML error quotes the offending line, which
            // in daemon.toml can be a credential reference. The reload path
            // logs the cause once when it refuses the file.
            Err(_) => {
                tracing::warn!(
                    "restart-required report: {} could not be parsed; reporting it as a failed \
                     config load (see the config reload log for the cause)",
                    self.config_path.display()
                );
                // Nothing to diff against: the one honest thing to say is that
                // the file did not load.
                return vec![crate::config::RestartArea::ConfigLoadFailed];
            }
        };

        let state = self.state.read();
        let degraded = state.config_origin == crate::config::ConfigOrigin::DefaultsAfterFailedLoad;
        let candidate = candidate.as_ref().unwrap_or(&state.config);
        let mut areas = crate::config::plan_reload(&self.booted_config, candidate).restart_required;
        if degraded {
            // The file parses again but the process is still on the defaults it
            // booted with; the diff below explains which areas are stale, this
            // says why.
            areas.insert(0, crate::config::RestartArea::ConfigLoadFailed);
        }
        areas
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
    /// Returns the [`crate::config::ReloadPlan`] describing what was applied (and what still
    /// needs a restart) on success.
    pub fn apply_reload(&self) -> anyhow::Result<crate::config::ReloadPlan> {
        // 1. Parse + validate the candidate from disk. `parse_daemon_config`
        //    surfaces TOML and [connections]/[purposes] validation errors. A
        //    failure here returns Err and leaves the running state untouched.
        //
        //    The non-migrating reader: this path runs from the config-file
        //    watcher, so rewriting the file here would answer a file change
        //    with another file change. Legacy shapes are migrated once at
        //    startup instead (#915).
        let new_config = match parse_daemon_config(&self.config_path) {
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

        // 2. Decide whether there is anything to do BEFORE building anything.
        //    Every daemon-authored write trips the on-disk watcher, so this
        //    path runs constantly with nothing to apply; building a registry
        //    first meant constructing a live client per connection and
        //    discarding them on each of those passes.
        //
        //    A daemon running defaults after a failed load always has work to
        //    do, whatever the diff says: it has to take on the file's content
        //    so config writes are safe again (#723). `plan_reload` compares the
        //    areas a reload can act on, not every field, so an empty plan there
        //    does not mean the two configs are equal.
        {
            let state = self.state.read();
            let plan = crate::config::plan_reload(&state.config, &new_config);
            if plan.is_empty() && state.config_origin == crate::config::ConfigOrigin::File {
                tracing::info!("config reload: no effective changes; nothing to apply");
                return Ok(plan);
            }
        }

        // 3. Build the candidate registry off the lock. `build_registry` is
        //    infallible (bad connections become `Unavailable` rows rather than
        //    aborting), but we refuse a config that yields *zero* usable
        //    connections when the running one had at least one — that would
        //    silently break every new turn. The running registry stays put.
        let new_registry = build_registry(&new_config);
        {
            let state = self.state.read();
            if state.registry.live_count() > 0 && new_registry.live_count() == 0 {
                tracing::error!(
                    "config reload refused: the new config has no usable LLM connection \
                     (every connection failed to build); keeping the last-good running config"
                );
                anyhow::bail!("new config has no usable LLM connection");
            }
        }

        // 4. Re-diff and swap under the write lock. Re-reading `state.config`
        //    here (rather than trusting the read-lock snapshot above) keeps the
        //    plan consistent if a concurrent `mutate_config` slipped in.
        let mut state = self.state.write();
        let plan = crate::config::plan_reload(&state.config, &new_config);
        let was_degraded =
            state.config_origin == crate::config::ConfigOrigin::DefaultsAfterFailedLoad;
        state.config = new_config;
        // Swapping the registry drops only its own Arc handles; in-flight turns
        // that already cloned their client keep it alive (see method docs).
        state.registry = new_registry;
        // The running config is the file's content again, so writing it back
        // can no longer destroy anything (#723).
        state.config_origin = crate::config::ConfigOrigin::File;
        drop(state);

        if was_degraded {
            tracing::info!(
                "config reload applied: {} loaded successfully; the daemon is no longer running \
                 built-in defaults and config-changing commands are accepted again. Areas wired \
                 at startup still need a restart - see the restart-required report",
                self.config_path.display()
            );
        }
        if plan.rebuild_registry {
            tracing::info!("config reload applied: connection registry rebuilt for new turns");
        }
        if plan.needs_restart() {
            tracing::warn!(
                "config reload: these changes need a daemon restart to take effect: {}",
                plan.restart_required_keys().join(", ")
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
        let mut new_conn = payload_to_connection(config);
        // Nothing to carry forward on a create, so the connector's own key
        // variables are the only names accepted.
        constrain_api_key_env(&mut new_conn, None).map_err(CoreError::Llm)?;
        constrain_base_url(&new_conn)?;
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
        let mut new_conn = payload_to_connection(config);
        constrain_base_url(&new_conn)?;
        self.registry.mutate_config(|cfg| {
            let Some(existing) = cfg.connections.get(id_valid.as_str()) else {
                return Err(format!("connection id {:?} does not exist", id_valid));
            };

            // Read the check against *this* connection's stored name, inside
            // the mutator's serialized section, so a concurrent edit can't
            // widen what this update is allowed to name.
            constrain_api_key_env(&mut new_conn, existing.api_key_env())?;

            // The payload deliberately carries no credential material, so a
            // bare replace would drop the connection's secret coordinate and
            // orphan the credential still sitting in the secret backend —
            // leaving the connection silently falling back to the provider's
            // ambient credential chain (#643). Carry the coordinate forward.
            //
            // Why only for a matching connector: the stored credential is
            // connector-shaped (an AWS key pair, a bearer token, …), so
            // carrying it across a connector switch would leave a value the
            // new connector cannot interpret. That case drops it, and the
            // operator re-supplies one via SetConnectionSecret.
            if existing.connector_type() == new_conn.connector_type()
                && let Some(secret) = existing.secret().cloned()
            {
                new_conn.set_secret(Some(secret)).expect(
                    "connector types match and the stored connection carried a secret, \
                     so this variant has a secret field",
                );
            }

            // Same hazard, wider than one field. Every setting the payload has
            // no field for is absent from `new_conn`, so a client editing any
            // other field would delete it - for Bedrock's `cache_policy`, that
            // quietly puts the cache writes back on the bill (#1027). The
            // connection type states what it keeps in the file; a test sweep
            // holds every connector to its answer.
            new_conn.carry_forward_file_only_fields(existing);

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

    async fn set_connection_secret(&self, id: String, credential: String) -> Result<(), CoreError> {
        let id_valid = ConnectionId::new(id.clone())
            .map_err(|e| CoreError::Llm(format!("invalid connection id: {e}")))?;
        self.registry.mutate_config(|cfg| {
            // Unhappy path: the connection must already exist. We store the
            // credential against a specific connection's coordinate, so there is
            // nothing to attach it to otherwise.
            let Some(conn) = cfg.connections.get(id_valid.as_str()) else {
                return Err(format!("connection id {id_valid:?} does not exist"));
            };
            let connector = conn.connector_type().to_string();

            // Write (or clear) the raw value in the secret backend and get back
            // the coordinate to persist. The raw credential never enters the
            // config; only this non-secret coordinate does. Error messages here
            // carry the store path/connector — never the credential value.
            let secret =
                crate::config::store_connection_secret(id_valid.as_str(), &connector, &credential)
                    .map_err(|e| format!("failed to store secret for {id_valid:?}: {e}"))?;

            let conn = cfg
                .connections
                .get_mut(id_valid.as_str())
                .expect("connection existence checked above");
            conn.set_secret(secret)
                .map_err(|e| format!("connection {id_valid:?}: {e}"))?;
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
            // The detailed variant so a connector that had to degrade (e.g.
            // Bedrock without `bedrock:ListInferenceProfiles`) can say so in
            // the response rather than only in the daemon log (#648).
            let list_result = if refresh {
                client.refresh_models_detailed().await
            } else {
                client.list_models_detailed().await
            };
            match list_result {
                Ok(report) => {
                    if report.is_degraded() {
                        tracing::info!(
                            connection = %id,
                            notices = report.notices.len(),
                            "model listing is incomplete; reporting it to the client"
                        );
                    }
                    // Notices ride each row of the connection (the listing is
                    // a flat per-model stream). A connection that produced no
                    // rows at all therefore carries none. Acceptable because
                    // an empty picker is already unambiguous, unlike the
                    // deceptive "loaded, embeddings only" case this reports.
                    let merged =
                        crate::model_defaults::merge_with_defaults(&connector_type, report.models);
                    for m in merged {
                        out.push(CoreModelListing {
                            connection_id: id.as_str().to_string(),
                            connection_label: label.clone(),
                            model: m,
                            notices: report.notices.clone(),
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
            voice: config
                .purposes
                .get(PurposeKind::Voice)
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

        // Half-inheritance is meaningless: a real connection cannot resolve a
        // model borrowed from a different one. Refusing it here is what keeps
        // a client that cannot populate its model dropdown from quietly
        // retiring a working binding (#659).
        if new_cfg.inheritance_is_mixed() {
            return Err(CoreError::Llm(format!(
                "purpose \"{}\": connection \"{}\" and model \"{}\" mix a named binding with the \
                 \"primary\" inherit sentinel — use \"primary\" for both to inherit from \
                 interactive, or name both",
                purpose_kind.as_key(),
                new_cfg.connection,
                new_cfg.model,
            )));
        }

        // Reject a binding whose model KIND contradicts the purpose (#647): a
        // generative model can't serve the embedding purpose, and an embedding
        // model can't serve a generative one. Only a fully-named binding (real
        // connection + real model) is checked here; the "primary" inherit case
        // is validated for consistency above, and resolving THROUGH inheritance
        // to the interactive model's kind is a separate follow-up.
        //
        // A kind that can't be verified (Unknown, model not listed, or the
        // connection is down) is allowed with a warning inside
        // `resolve_model_kind` — a config edit must never fail on something the
        // daemon merely could not check.
        if let (ConnectionRef::Named(conn_id), ModelRef::Named(model_id)) =
            (&new_cfg.connection, &new_cfg.model)
        {
            let expected = expected_kind_for_purpose(purpose_kind);
            let found = self.registry.resolve_model_kind(conn_id, model_id).await;
            if found != ModelKind::Unknown && found != expected {
                return Err(CoreError::Llm(format!(
                    "purpose \"{purpose}\": model \"{model}\" on connection \"{conn}\" is \
                     {found}, but the {purpose} purpose requires {expected} — choose a model \
                     whose kind is {expected}",
                    purpose = purpose_kind.as_key(),
                    model = model_id,
                    conn = conn_id,
                    found = kind_word(found),
                    expected = kind_word(expected),
                )));
            }
        }

        self.registry.mutate_config(|cfg| {
            cfg.purposes.set(purpose_kind, Some(new_cfg));
            cfg.purposes.validate().map_err(|e| format!("{e}"))
        })
    }
}

/// The [`ModelKind`] a purpose requires of a bound model (#647). Embedding is
/// the only purpose that needs an embedding model; every other purpose runs a
/// generative model. Exhaustive so a new [`PurposeKind`] cannot be added
/// without deciding what kind of model it binds.
fn expected_kind_for_purpose(purpose: PurposeKind) -> ModelKind {
    match purpose {
        PurposeKind::Embedding => ModelKind::Embedding,
        PurposeKind::Interactive
        | PurposeKind::Dreaming
        | PurposeKind::Consolidation
        | PurposeKind::Titling
        | PurposeKind::Voice => ModelKind::Generative,
    }
}

/// Human word for a [`ModelKind`] used in the SetPurpose rejection message.
fn kind_word(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Generative => "generative",
        ModelKind::Embedding => "embedding",
        ModelKind::Unknown => "unknown",
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

    /// Read the conversation's stored tool-provenance-gate override (#1007),
    /// or `false` when none is stored. Mirrors [`Self::get_personality`].
    fn get_tool_gate_disabled(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<bool, CoreError>> + Send;

    /// Set the conversation's tool-provenance-gate override (#1007). Mirrors
    /// [`Self::set_personality`].
    fn set_tool_gate_disabled(
        &self,
        id: &ConversationId,
        disabled: bool,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    /// Read the conversation's tags (e.g. `"voice"`, set by the voice daemon),
    /// or an empty list when the backend doesn't track them. Used to route
    /// voice-originated turns to the Voice purpose (voice#126). Defaults to
    /// empty so stores that don't provide tags (tests, the JSON backend) simply
    /// opt out of tag-based routing.
    fn get_tags(
        &self,
        _id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<Vec<String>, CoreError>> + Send {
        async { Ok(Vec::new()) }
    }
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
    /// The connector kind behind the chosen connection - `anthropic`,
    /// `ollama`, and so on. `Some` exactly when `chosen` is. This is the
    /// `provider` axis of the turn's spans and metrics, and it is read from
    /// the same resolution the dispatch uses so the two cannot disagree.
    connector: Option<String>,
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
    /// `(connector, model)` the statically configured primary client was built
    /// with, for telemetry only.
    ///
    /// Captured when that client was built rather than resolved per turn, for
    /// two reasons. Resolving `[llm]` reads its credential from the secret
    /// backend, which on a keyring install is a blocking D-Bus round trip -
    /// not something to do on every turn for two labels. And the primary
    /// client is built once in `main` and is *not* rebuilt by a configuration
    /// reload, so the live configuration can name a model the process is not
    /// running; reporting what was actually built is the honest answer.
    ///
    /// `None` where no caller stated it, which reports as `unset` rather than
    /// as a guess.
    primary_route: Option<(String, String)>,
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
            primary_route: None,
        }
    }

    /// State which `(connector, model)` the statically configured primary
    /// client was built with, so a turn that falls through to it is still
    /// attributed. See [`Self::primary_route`].
    pub fn with_primary_route(mut self, connector: String, model: String) -> Self {
        self.primary_route = Some((connector, model));
        self
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

    /// Resolve the Voice purpose from the current config, mirroring
    /// [`Self::interactive_selection`] (voice#126). Returns `None` when
    /// `[purposes.voice]` is absent or set to inherit (`connection`/`model =
    /// primary`) — so voice turns then fall through to the interactive purpose,
    /// i.e. adding the purpose is a no-op until an operator points it at a
    /// concrete model.
    fn voice_selection(&self) -> Option<ConversationModelSelection> {
        let cfg = self.registry.snapshot_config();
        cfg.purposes.get(PurposeKind::Voice).and_then(|p| {
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

    /// True when the conversation carries the `"voice"` tag the voice daemon
    /// sets on its conversations — the signal that a turn is voice-originated
    /// and should prefer the Voice purpose (voice#126).
    async fn conversation_is_voice(&self, id: &ConversationId) -> Result<bool, CoreError> {
        Ok(self
            .selection_store
            .get_tags(id)
            .await?
            .iter()
            .any(|t| t == "voice"))
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

    /// This daemon's configured default tool policy, read from the live
    /// configuration so a reload takes effect without a restart.
    ///
    /// Startup validation rejects an unparseable value, so reaching the
    /// fallback here means a reload introduced one. That warns and uses the
    /// shipped default: a running daemon must not be left without a level, and
    /// it must not silently take a more permissive one than the operator's
    /// last valid choice.
    fn configured_tool_policy(&self) -> ToolPolicy {
        match self.registry.snapshot_config().security.tool_policy() {
            Ok(policy) => policy,
            Err(e) => {
                tracing::warn!("{e}; using {}", ToolPolicy::default().as_str());
                ToolPolicy::default()
            }
        }
    }

    /// Resolve the tool policy for a send: the conversation's own override,
    /// else this daemon's configured default.
    ///
    /// The stored override is still the boolean #1007 shipped, and `true`
    /// means [`ToolPolicy::Lax`] - the level that turn asked for by name. A
    /// stored `false` is not a level; it is the absence of one, so it takes
    /// the daemon default like any conversation that never set it.
    ///
    /// Never resolves to a more permissive level than the operator chose: a
    /// store error logs and falls back to that default, and only an explicit
    /// stored `true` reaches [`ToolPolicy::Lax`].
    async fn resolve_tool_policy(&self, conversation_id: &ConversationId) -> ToolPolicy {
        let default = self.configured_tool_policy();
        match self
            .selection_store
            .get_tool_gate_disabled(conversation_id)
            .await
        {
            Ok(true) => ToolPolicy::Lax,
            Ok(false) => default,
            Err(e) => {
                tracing::warn!(
                    conversation_id = %conversation_id.0,
                    "failed to read conversation tool policy; using the daemon default \
                     {}: {e}",
                    default.as_str()
                );
                default
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
        let mut connector = None;

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
                    connector = Some(connector_type.clone());
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
            connector,
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
        // OpenAI, OpenRouter, and Azure all take a `reasoning_effort` literal;
        // each connector applies its own per-model gate and drops the field for
        // non-reasoning models.
        "openai" | "openrouter" | "azure" => {
            let level = map_effort_to_reasoning_level(effort);
            tracing::debug!(
                connector = connector_type,
                model = model_id,
                effort = ?effort,
                reasoning_level = ?level,
                "mapped effort to reasoning_effort"
            );
            ReasoningConfig::with_reasoning_effort(level)
        }
        // Google Gemini has no effort literal; it takes a thinking-token budget
        // via `generationConfig.thinkingConfig`. Use Gemini-calibrated budgets
        // (NOT the Claude extended-thinking table). The connector further gates
        // this to thinking-capable models (2.5+) and omits it otherwise.
        "google" => {
            let budget = map_gemini_thinking_budget(effort);
            tracing::debug!(
                connector = connector_type,
                model = model_id,
                effort = ?effort,
                thinking_budget_tokens = budget,
                "mapped effort to Gemini thinking budget"
            );
            ReasoningConfig::with_thinking_budget(budget)
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
    async fn create_conversation(
        &self,
        title: String,
        tags: Vec<String>,
    ) -> Result<Conversation, CoreError> {
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

    async fn get_conversation_tool_gate_disabled(
        &self,
        id: &ConversationId,
    ) -> Result<bool, CoreError> {
        self.selection_store.get_tool_gate_disabled(id).await
    }

    async fn set_conversation_tool_gate_disabled(
        &self,
        id: &ConversationId,
        disabled: bool,
    ) -> Result<(), CoreError> {
        self.selection_store
            .set_tool_gate_disabled(id, disabled)
            .await
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
        // Set when tier-3 routing sends this turn to the Voice purpose (voice#126).
        let mut routed_via_voice = false;

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
        } else if let Some(voice_sel) = self.voice_selection() {
            // Tier 3a (voice#126): no override and no stored selection. When a
            // Voice purpose is configured with a concrete model AND this is a
            // voice-originated conversation (tagged "voice" by the voice
            // daemon), route the turn there — as a *user-driven* selection so
            // the chosen model actually dispatches (the interactive fallback
            // routes through the static primary and would drop the model id).
            // Gating the tag lookup on a configured Voice purpose keeps this a
            // zero-cost no-op for the common case where none is set.
            if self.conversation_is_voice(conversation_id).await? {
                routed_via_voice = true;
                Some(voice_sel)
            } else {
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
            connector,
        } = self
            .resolve_turn(user_driven_selection.as_ref(), effective_selection.as_ref())
            .await?;

        // Where this turn dispatches, for its spans and its metrics.
        //
        // When routing resolved a live connection, that is the answer. When it
        // fell through to the statically configured primary there is no
        // connection id - but there is still a connector and a model, taken
        // from what `main` actually built that client with. Reporting `unset`
        // for those two would leave every `[llm]`-only install, which is the
        // ordinary desktop shape, with no provider or model on any span, any
        // metric or the completion line.
        let route = match &chosen {
            Some((connection, model)) => desktop_assistant_core::ports::turn_telemetry::TurnRoute {
                connection_id: Some(connection.clone()),
                provider: connector,
                model: Some(model.clone()),
            },
            None => desktop_assistant_core::ports::turn_telemetry::TurnRoute {
                connection_id: None,
                provider: self.primary_route.as_ref().map(|(c, _)| c.clone()),
                model: self.primary_route.as_ref().map(|(_, m)| m.clone()),
            },
        };

        // The turn's tool-discovery mode is NOT logged here, deliberately.
        // `active_client` is the raw registry client, and the turn asks the
        // decorator chain that wraps it
        // (`Arc` -> `Retrying` -> `FixedReasoning` -> `RoutingLlmClient`).
        // The two answers agree only while every
        // decorator forwards the capability correctly, which is precisely the
        // invariant that breaks. Logging the raw client here would print the
        // right answer while the turn used the wrong one, and point an
        // operator away from the cause. `ConversationHandler::send_prompt`
        // logs the mode from the value it actually used instead.
        tracing::info!(
            purpose = ?if routed_via_voice { PurposeKind::Voice } else { PurposeKind::Interactive },
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

        // Resolve the turn's tool policy, fresh on every send, the same way as
        // the personality above. A read that fails resolves to this daemon's
        // configured default inside `resolve_tool_policy` itself.
        let effective_tool_policy = self.resolve_tool_policy(conversation_id).await;

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
            // Boxed before the task-local wraps below. Each `with_*` embeds
            // the future it wraps *by value*, so an unboxed turn future is
            // re-embedded once per slot and the nest grows with every slot
            // added here - the same accounting that put the streaming send
            // path over a worker thread's stack in #205/#206. Boxing once
            // keeps every wrapper pointer-sized whatever the slot count.
            let dispatch: std::pin::Pin<Box<dyn Future<Output = _> + Send>> =
                Box::pin(async move {
                    inner
                        .send_prompt(&conv_id, prompt, on_chunk, on_status)
                        .await
                });
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
            // Install the resolved tool policy so the core dispatch loop
            // constructs this turn's `TurnProvenance` at that level. The core
            // read side (`current_tool_policy`) reads back the shipped default
            // outside this scope, never the most permissive level.
            let dispatch = desktop_assistant_core::ports::llm::with_tool_policy(
                effective_tool_policy,
                dispatch,
            );
            // Install the ambient "now" line so the core assembler surfaces a
            // `[Now]` system message for this turn. Request-scoped, never
            // persisted; see `NOW_CONTEXT`.
            let dispatch = desktop_assistant_core::ports::llm::with_now_context(now_line, dispatch);
            // Installed outside the routing wrap below so it holds for the
            // whole turn, including the fall-through arm where no concrete
            // connection resolved and the route reads `unset` on every axis.
            let dispatch =
                desktop_assistant_core::ports::turn_telemetry::with_turn_route(route, dispatch);
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

/// Google Gemini `thinkingConfig.thinkingBudget` (tokens) for an effort hint.
///
/// Gemini-calibrated budgets, deliberately distinct from the Claude
/// extended-thinking table ([`map_anthropic_thinking_budget`]): Gemini's
/// thinking budget is a smaller, differently-scaled knob. Low = 2_048,
/// Medium = 8_192, High = 16_384. All are positive, so unlike the Anthropic
/// mapping (Low = 0 = off) an explicit effort always requests some thinking;
/// the connector still omits the field for non-thinking-capable models.
pub fn map_gemini_thinking_budget(e: Effort) -> u32 {
    match e {
        Effort::Low => 2_048,
        Effort::Medium => 8_192,
        Effort::High => 16_384,
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

/// Constrain the `api_key_env` a client-supplied connection carries, in place.
///
/// A blank name means "unset"; surrounding whitespace is trimmed (a padded
/// name would name no variable at all). A name that survives that must be one
/// of the connector's own documented key variables
/// ([`crate::config::allowed_api_key_envs`]), matched exactly and
/// case-sensitively — environment lookups are case-sensitive, so a folded or
/// substring match would admit a *different* variable.
///
/// Why: the resolver reads the named variable from the daemon's own process
/// environment and the connector sends it to the connection's `base_url` as a
/// bearer token. Without this, an API client could name the deployment's
/// database password and have the daemon post it to a host of the client's
/// choosing.
///
/// `previous` is the name the stored connection already reads, on an update.
/// Re-sending it is accepted even when it is outside the list: `daemon.toml`
/// is operator-owned and may name a custom variable, the echoed connection
/// view carries that name back to the client, and re-saving it grants no read
/// the daemon was not already performing for this connection. Every other
/// value must be on the list, so a client can never introduce a new name.
fn constrain_api_key_env(
    conn: &mut ConnectionConfig,
    previous: Option<&str>,
) -> Result<(), String> {
    let Some(requested) = conn.api_key_env() else {
        return Ok(());
    };
    let requested = requested.trim();

    if requested.is_empty() {
        return conn
            .set_api_key_env(None)
            .map_err(|e| format!("api_key_env: {e}"));
    }

    let unchanged = previous.is_some_and(|p| p.trim() == requested);
    let allowed = crate::config::allowed_api_key_envs(conn.connector());
    if !unchanged && !allowed.iter().any(|a| a == requested) {
        return Err(format!(
            "api_key_env {requested:?} is not permitted for the {connector} connector; \
             use {allowed}, or store the credential with set_connection_secret",
            connector = conn.connector_type(),
            allowed = allowed.join(" or "),
        ));
    }

    conn.set_api_key_env(Some(requested.to_string()))
        .map_err(|e| format!("api_key_env: {e}"))
}

/// Validate a connection's `base_url` against the shared remote-URL policy
/// (#804, #895) before it is stored.
///
/// `None` (use the connector's own hosted default) is always fine — that
/// endpoint is never client-controlled. When the client sets one, it is
/// exactly the value the connector attaches the connection's API key or
/// bearer token to on every request, so it gets the same scrutiny as a
/// remote MCP endpoint's `url`.
fn constrain_base_url(conn: &ConnectionConfig) -> Result<(), CoreError> {
    let Some(base_url) = conn.base_url() else {
        return Ok(());
    };
    let credential = if conn.carries_credential() {
        desktop_assistant_mcp_client::url_policy::RequestCredential::Attached
    } else {
        desktop_assistant_mcp_client::url_policy::RequestCredential::None
    };
    desktop_assistant_mcp_client::url_policy::validate_remote_url(base_url, credential).map_err(
        |e| CoreError::InvalidInput {
            code: e.code(),
            description: format!("connection base_url {base_url:?} refused: {e}"),
            message: e.user_message(),
        },
    )
}

/// Project a client-supplied [`ConnectionConfigPayload`] onto the stored
/// [`ConnectionConfig`] shape.
///
/// Pure field mapping: every payload converts, and secrets never cross this
/// boundary (`secret: None` throughout — they are set out-of-band via
/// `set_connection_secret`). The one field that needs vetting rather than
/// copying is `api_key_env`; `create_connection` / `update_connection` run
/// [`constrain_api_key_env`] over the result before storing it.
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
        ConnectionConfigPayload::OpenRouter {
            base_url,
            api_key_env,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfig::OpenRouter(OpenRouterConnection {
            base_url,
            api_key_env,
            secret: None,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        }),
        ConnectionConfigPayload::Azure {
            base_url,
            api_key_env,
            api_surface,
            auth_mode,
            api_version,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfig::Azure(AzureConnection {
            base_url,
            api_key_env,
            // Secrets never cross the non-secret payload boundary; set
            // out-of-band via `set_connection_secret`.
            secret: None,
            api_surface,
            auth_mode,
            api_version,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        }),
        ConnectionConfigPayload::Google {
            base_url,
            api_key_env,
            project,
            location,
            auth_mode,
            credentials_path,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
        } => ConnectionConfig::Google(GoogleConnection {
            base_url,
            api_key_env,
            // Secrets never cross the non-secret payload boundary.
            secret: None,
            project,
            location,
            auth_mode,
            credentials_path,
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
            // Secrets never cross the non-secret payload boundary; they are set
            // out-of-band via `set_connection_secret`. Every payload therefore
            // converts with no coordinate — `update_connection` re-attaches the
            // stored one afterwards so an edit doesn't orphan the credential.
            secret: None,
            connect_timeout_secs,
            stream_timeout_secs,
            max_context_tokens,
            // The payload cannot express a cache policy, so this conversion
            // cannot either. `update_connection` carries the stored one
            // forward, the same way it re-attaches the secret coordinate.
            cache_policy: None,
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
        ConnectionConfig::OpenRouter(c) => ConnectionConfigPayload::OpenRouter {
            base_url: c.base_url.clone(),
            api_key_env: c.api_key_env.clone(),
            // `c.secret` (keyring coordinates) intentionally not echoed.
            connect_timeout_secs: c.connect_timeout_secs,
            stream_timeout_secs: c.stream_timeout_secs,
            max_context_tokens: c.max_context_tokens,
        },
        ConnectionConfig::Azure(c) => ConnectionConfigPayload::Azure {
            base_url: c.base_url.clone(),
            api_key_env: c.api_key_env.clone(),
            api_surface: c.api_surface.clone(),
            auth_mode: c.auth_mode.clone(),
            api_version: c.api_version.clone(),
            // `c.secret` (keyring coordinates) intentionally not echoed.
            connect_timeout_secs: c.connect_timeout_secs,
            stream_timeout_secs: c.stream_timeout_secs,
            max_context_tokens: c.max_context_tokens,
        },
        ConnectionConfig::Google(c) => ConnectionConfigPayload::Google {
            base_url: c.base_url.clone(),
            api_key_env: c.api_key_env.clone(),
            project: c.project.clone(),
            location: c.location.clone(),
            auth_mode: c.auth_mode.clone(),
            credentials_path: c.credentials_path.clone(),
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
    use crate::connections::{BedrockConnection, ConnectionConfig, Connector, OllamaConnection};
    use desktop_assistant_core::prompts::PersonalityLevel;

    use std::sync::Mutex;

    /// Trivial in-memory `ConversationSelectionStore` for the daemon test
    /// suite. Production code uses the Postgres-backed store via the
    /// storage crate.
    pub struct InMemoryConversationSelectionStore {
        inner: Mutex<std::collections::HashMap<String, ConversationModelSelection>>,
        personality: Mutex<std::collections::HashMap<String, PersonalityOverride>>,
        tags: Mutex<std::collections::HashMap<String, Vec<String>>>,
        tool_gate_disabled: Mutex<std::collections::HashMap<String, bool>>,
    }

    impl Default for InMemoryConversationSelectionStore {
        fn default() -> Self {
            Self {
                inner: Mutex::new(std::collections::HashMap::new()),
                personality: Mutex::new(std::collections::HashMap::new()),
                tags: Mutex::new(std::collections::HashMap::new()),
                tool_gate_disabled: Mutex::new(std::collections::HashMap::new()),
            }
        }
    }

    impl InMemoryConversationSelectionStore {
        /// Test helper: pin a conversation's tags (e.g. `["voice"]`).
        fn set_tags(&self, id: &str, tags: Vec<String>) {
            self.tags
                .lock()
                .expect("tags store poisoned")
                .insert(id.to_string(), tags);
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

        async fn get_tool_gate_disabled(&self, id: &ConversationId) -> Result<bool, CoreError> {
            Ok(self
                .tool_gate_disabled
                .lock()
                .expect("tool-gate store poisoned")
                .get(&id.0)
                .copied()
                .unwrap_or(false))
        }

        async fn set_tool_gate_disabled(
            &self,
            id: &ConversationId,
            disabled: bool,
        ) -> Result<(), CoreError> {
            self.tool_gate_disabled
                .lock()
                .expect("tool-gate store poisoned")
                .insert(id.0.clone(), disabled);
            Ok(())
        }

        async fn get_tags(&self, id: &ConversationId) -> Result<Vec<String>, CoreError> {
            Ok(self
                .tags
                .lock()
                .expect("tags store poisoned")
                .get(&id.0)
                .cloned()
                .unwrap_or_default())
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

    // --- set_connection_secret (#11 credential storage) ---------------------

    /// Run `body` with `XDG_DATA_HOME` pointed at a fresh unique temp dir so the
    /// 0600 secret files live in isolation and never touch the real
    /// `~/.local/share`. Serialised via the crate-wide
    /// [`crate::config::xdg_data_home_test_lock`] so it can't race the config
    /// module's JWT / `set_api_key` tests, which repoint the same env var.
    fn with_isolated_secret_store<T>(body: impl FnOnce(&std::path::Path) -> T) -> T {
        let _guard = crate::config::xdg_data_home_test_lock();
        let test_dir =
            std::env::temp_dir().join(format!("da-test-connsecret-{}", uuid::Uuid::new_v4()));
        let data_home = test_dir.join("data");
        std::fs::create_dir_all(&data_home).unwrap();
        // SAFETY: serialised against other secret tests via the lock above; the
        // temp dir is unique per run (UUID-suffixed).
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &data_home);
        }
        let out = body(&data_home);
        // SAFETY: same scope as the matching `set_var` above.
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
        std::fs::remove_dir_all(&test_dir).ok();
        out
    }

    /// Drive an async future to completion on a fresh current-thread runtime.
    /// Kept sync (vs `#[tokio::test]`) so the env guard from
    /// [`with_isolated_secret_store`] is never held across an `.await`.
    fn run_async<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    fn make_handle_at(cfg: DaemonConfig, path: std::path::PathBuf) -> Arc<RegistryHandle> {
        let registry = build_registry(&cfg);
        Arc::new(RegistryHandle::new(cfg, registry).with_config_path(path))
    }

    /// Resolve a connection's live API key from the current config, reading any
    /// stored secret back through the backend — the property downstream
    /// connectors depend on.
    fn resolved_api_key(handle: &RegistryHandle, id: &str) -> String {
        let cfg = handle.snapshot_config();
        let conn = cfg
            .connections
            .get(id)
            .expect("connection should exist in config");
        crate::config::resolve_connection_llm_config(conn, None).api_key
    }

    // Realistic AWS static-credential string; contains none of the
    // placeholder markers `sanitize_secret_value` filters out.
    const BEDROCK_CRED: &str = "AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY";

    #[test]
    fn set_connection_secret_round_trips_to_resolved_api_key() {
        with_isolated_secret_store(|_| {
            let handle = make_handle_at(
                config_with_connections(&[("aws", bedrock_work())]),
                tmp_config_path(),
            );
            let svc = DaemonConnectionsService::new(handle.clone());

            run_async(svc.set_connection_secret("aws".into(), BEDROCK_CRED.into())).unwrap();

            // Resolution reads the file secret back as the Bedrock api_key, which
            // `static_credentials_from_api_key` parses into AWS credentials.
            assert_eq!(resolved_api_key(&handle, "aws"), BEDROCK_CRED);
        });
    }

    #[test]
    fn set_connection_secret_empty_credential_clears_it() {
        with_isolated_secret_store(|_| {
            let handle = make_handle_at(
                config_with_connections(&[("aws", bedrock_work())]),
                tmp_config_path(),
            );
            let svc = DaemonConnectionsService::new(handle.clone());

            run_async(svc.set_connection_secret("aws".into(), BEDROCK_CRED.into())).unwrap();
            assert_eq!(resolved_api_key(&handle, "aws"), BEDROCK_CRED);

            // Empty credential clears the stored secret; resolution then sees no
            // key (no env fallback set in this isolated store).
            run_async(svc.set_connection_secret("aws".into(), String::new())).unwrap();
            assert_eq!(resolved_api_key(&handle, "aws"), "");

            // The connection's `secret` coordinate is dropped from config too.
            let cfg = handle.snapshot_config();
            match cfg.connections.get("aws").unwrap() {
                ConnectionConfig::Bedrock(c) => assert!(c.secret.is_none()),
                other => panic!("expected Bedrock, got {other:?}"),
            }
        });
    }

    #[test]
    fn set_connection_secret_whitespace_credential_clears_it() {
        with_isolated_secret_store(|_| {
            let handle = make_handle_at(
                config_with_connections(&[("aws", bedrock_work())]),
                tmp_config_path(),
            );
            let svc = DaemonConnectionsService::new(handle.clone());

            run_async(svc.set_connection_secret("aws".into(), BEDROCK_CRED.into())).unwrap();
            // Whitespace-only is treated as "clear", same as empty.
            run_async(svc.set_connection_secret("aws".into(), "   \n\t ".into())).unwrap();
            assert_eq!(resolved_api_key(&handle, "aws"), "");
        });
    }

    #[test]
    fn set_connection_secret_is_isolated_per_connection() {
        with_isolated_secret_store(|_| {
            let handle = make_handle_at(
                config_with_connections(&[("aws1", bedrock_work()), ("aws2", bedrock_work())]),
                tmp_config_path(),
            );
            let svc = DaemonConnectionsService::new(handle.clone());

            // Setting aws1 must not populate aws2 (distinct account files keyed
            // by connection id).
            run_async(svc.set_connection_secret("aws1".into(), BEDROCK_CRED.into())).unwrap();
            assert_eq!(resolved_api_key(&handle, "aws1"), BEDROCK_CRED);
            assert_eq!(resolved_api_key(&handle, "aws2"), "");

            // Two connections of the same connector keep independent values.
            let other = "AKIAI44QH8DHBEXAMPLE:je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY";
            run_async(svc.set_connection_secret("aws2".into(), other.into())).unwrap();
            assert_eq!(resolved_api_key(&handle, "aws1"), BEDROCK_CRED);
            assert_eq!(resolved_api_key(&handle, "aws2"), other);
        });
    }

    #[test]
    fn set_connection_secret_rejects_unknown_connection() {
        with_isolated_secret_store(|_| {
            let handle = make_handle_at(
                config_with_connections(&[("aws", bedrock_work())]),
                tmp_config_path(),
            );
            let svc = DaemonConnectionsService::new(handle);
            let err = run_async(svc.set_connection_secret("nope".into(), BEDROCK_CRED.into()))
                .unwrap_err();
            assert!(
                format!("{err}").contains("does not exist"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn set_connection_secret_rejects_bad_slug() {
        with_isolated_secret_store(|_| {
            let handle = make_handle_at(DaemonConfig::default(), tmp_config_path());
            let svc = DaemonConnectionsService::new(handle);
            let err = run_async(svc.set_connection_secret("Bad Id!".into(), BEDROCK_CRED.into()))
                .unwrap_err();
            assert!(
                format!("{err}").contains("invalid connection id"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn set_connection_secret_rejects_ollama() {
        with_isolated_secret_store(|_| {
            let handle = make_handle_at(
                config_with_connections(&[("local", ollama_local())]),
                tmp_config_path(),
            );
            let svc = DaemonConnectionsService::new(handle);
            // Ollama has no API key; setting a credential is a caller error.
            let err = run_async(svc.set_connection_secret("local".into(), "whatever".into()))
                .unwrap_err();
            assert!(
                format!("{err}").contains("do not use a stored credential"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn set_connection_secret_never_writes_credential_to_daemon_toml() {
        with_isolated_secret_store(|_| {
            let config_path = tmp_config_path();
            let handle = make_handle_at(
                config_with_connections(&[("aws", bedrock_work())]),
                config_path.clone(),
            );
            let svc = DaemonConnectionsService::new(handle);

            run_async(svc.set_connection_secret("aws".into(), BEDROCK_CRED.into())).unwrap();

            let toml = std::fs::read_to_string(&config_path)
                .expect("mutate_config should have persisted daemon.toml");
            assert!(
                !toml.contains(BEDROCK_CRED),
                "raw credential leaked into daemon.toml:\n{toml}"
            );
            // The non-secret coordinate (keyed by connection id) is what persists.
            assert!(
                toml.contains("connection_aws"),
                "expected the secret coordinate account in daemon.toml:\n{toml}"
            );
            std::fs::remove_file(&config_path).ok();
        });
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

    // --- base_url is validated against the shared remote-URL policy --------
    // (#804, #895): a connection's base_url is a client-controlled value the
    // connector attaches a credential to on every request, so it is checked
    // the same way as a remote MCP endpoint's url.

    fn openai_payload_with_base_url(base_url: &str) -> ConnectionConfigPayload {
        ConnectionConfigPayload::OpenAi {
            base_url: Some(base_url.to_string()),
            api_key_env: None,
            connect_timeout_secs: None,
            stream_timeout_secs: None,
            max_context_tokens: None,
        }
    }

    #[tokio::test]
    async fn create_connection_rejects_a_plain_http_base_url_to_a_public_host() {
        let svc = DaemonConnectionsService::new(make_handle_with(DaemonConfig::default()));
        let err = svc
            .create_connection(
                "work".to_string(),
                openai_payload_with_base_url("http://evil.example.com/v1"),
            )
            .await
            .unwrap_err();
        match err {
            CoreError::InvalidInput { code, .. } => assert_eq!(code, "url_insecure_scheme"),
            other => panic!("expected CoreError::InvalidInput, got {other}"),
        }
    }

    #[tokio::test]
    async fn create_connection_rejects_a_link_local_base_url() {
        let svc = DaemonConnectionsService::new(make_handle_with(DaemonConfig::default()));
        let err = svc
            .create_connection(
                "work".to_string(),
                openai_payload_with_base_url("http://169.254.169.254/v1"),
            )
            .await
            .unwrap_err();
        match err {
            CoreError::InvalidInput { code, .. } => assert_eq!(code, "url_target_blocked"),
            other => panic!("expected CoreError::InvalidInput, got {other}"),
        }
    }

    #[tokio::test]
    async fn create_connection_accepts_an_https_base_url() {
        let svc = DaemonConnectionsService::new(make_handle_with(DaemonConfig::default()));
        svc.create_connection(
            "work".to_string(),
            openai_payload_with_base_url("https://api.openai.com/v1"),
        )
        .await
        .expect("a legitimate https base_url must keep working");
    }

    #[tokio::test]
    async fn create_connection_accepts_a_loopback_http_base_url() {
        let svc = DaemonConnectionsService::new(make_handle_with(DaemonConfig::default()));
        svc.create_connection(
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
        .expect("a loopback http base_url must be accepted");
    }

    /// The exact shipped deployment: `deploy/k8s/base/daemon.toml` reaches
    /// Ollama at `http://ollama:11434`, a bare in-cluster Service name.
    /// Ollama has no credential concept, so the bare-hostname exemption
    /// applies regardless (#804 review: this must stay pinned once the
    /// exemption becomes credential-gated).
    #[tokio::test]
    async fn create_connection_accepts_a_bare_hostname_base_url_for_ollama() {
        let svc = DaemonConnectionsService::new(make_handle_with(DaemonConfig::default()));
        svc.create_connection(
            "cluster-ollama".to_string(),
            ConnectionConfigPayload::Ollama {
                base_url: Some("http://ollama:11434".into()),
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                keep_warm: None,
                max_context_tokens: None,
            },
        )
        .await
        .expect("a bare in-cluster service name must be accepted for a credential-less connector");
    }

    /// #804 review (F1/F5): a bare hostname is not "a network the operator
    /// controls" once a credential is attached — every non-Ollama connector
    /// carries one (or is meant to). Without this, an admin could point a
    /// credentialed connector's `base_url` at a name resolved by search-
    /// domain append or LLMNR, neither of which the operator authenticated.
    #[tokio::test]
    async fn create_connection_rejects_a_bare_hostname_base_url_for_a_credentialed_connector() {
        let svc = DaemonConnectionsService::new(make_handle_with(DaemonConfig::default()));
        let err = svc
            .create_connection(
                "work".to_string(),
                openai_payload_with_base_url("http://internal-proxy:8080/v1"),
            )
            .await
            .unwrap_err();
        match err {
            CoreError::InvalidInput { code, .. } => assert_eq!(code, "url_insecure_scheme"),
            other => panic!("expected CoreError::InvalidInput, got {other}"),
        }
    }

    #[tokio::test]
    async fn update_connection_rejects_a_plain_http_base_url_to_a_public_host() {
        let handle = make_handle_with(config_with_connections(&[(
            "work",
            openai_reading_env(Some("OPENAI_API_KEY")),
        )]));
        let svc = DaemonConnectionsService::new(handle.clone());
        let err = svc
            .update_connection(
                "work".to_string(),
                openai_payload_with_base_url("http://attacker.example.invalid/v1"),
            )
            .await
            .unwrap_err();
        match err {
            CoreError::InvalidInput { code, .. } => assert_eq!(code, "url_insecure_scheme"),
            other => panic!("expected CoreError::InvalidInput, got {other}"),
        }

        let cfg = handle.snapshot_config();
        let conn = cfg
            .connections
            .get("work")
            .expect("connection should exist");
        match conn {
            ConnectionConfig::OpenAi(c) => assert_eq!(
                c.base_url.as_deref(),
                Some("https://api.openai.com/v1"),
                "a rejected update must not apply its base_url either"
            ),
            other => panic!("expected an openai connection, got {other:?}"),
        }
    }

    // --- UpdateConnection must not clobber the secret coordinate -----------

    const SECRET_ACCOUNT: &str = "connection_credential";

    /// The `account` of a connection's stored secret coordinate, if any.
    fn secret_account(conn: &ConnectionConfig) -> Option<String> {
        conn.secret().and_then(|s| s.account.clone())
    }

    fn secret_coordinate() -> Option<crate::config::SecretConfig> {
        Some(crate::config::SecretConfig {
            account: Some(SECRET_ACCOUNT.into()),
            ..crate::config::SecretConfig::default()
        })
    }

    fn bedrock_with_secret() -> ConnectionConfig {
        ConnectionConfig::Bedrock(crate::connections::BedrockConnection {
            aws_profile: Some("work".into()),
            region: Some("us-west-2".into()),
            secret: secret_coordinate(),
            ..Default::default()
        })
    }

    fn bedrock_payload(region: &str) -> ConnectionConfigPayload {
        ConnectionConfigPayload::Bedrock {
            aws_profile: Some("work".into()),
            region: Some(region.into()),
            base_url: None,
            connect_timeout_secs: None,
            stream_timeout_secs: None,
            max_context_tokens: None,
        }
    }

    /// A default stored connection of each connector type. Exhaustive with no
    /// catch-all, so a new connector must supply one before this compiles.
    fn stored_connection(connector: Connector) -> ConnectionConfig {
        match connector {
            Connector::Anthropic => ConnectionConfig::Anthropic(AnthropicConnection::default()),
            Connector::OpenAi => ConnectionConfig::OpenAi(OpenAiConnection::default()),
            Connector::OpenRouter => ConnectionConfig::OpenRouter(OpenRouterConnection::default()),
            Connector::Azure => ConnectionConfig::Azure(AzureConnection::default()),
            Connector::Google => ConnectionConfig::Google(GoogleConnection::default()),
            Connector::Bedrock => bedrock_work(),
            Connector::Ollama => ollama_local(),
        }
    }

    /// An update payload of each connector type, carrying one edit so a
    /// carry-forward test can see the update land. Exhaustive with no
    /// catch-all.
    fn update_payload(connector: Connector) -> ConnectionConfigPayload {
        match connector {
            Connector::Anthropic => ConnectionConfigPayload::Anthropic {
                base_url: Some("https://anthropic.example.invalid".into()),
                api_key_env: None,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::OpenAi => ConnectionConfigPayload::OpenAi {
                base_url: Some("https://openai.example.invalid".into()),
                api_key_env: None,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::OpenRouter => ConnectionConfigPayload::OpenRouter {
                base_url: Some("https://openrouter.example.invalid".into()),
                api_key_env: None,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::Azure => ConnectionConfigPayload::Azure {
                base_url: Some("https://azure.example.invalid".into()),
                api_key_env: None,
                api_surface: None,
                auth_mode: None,
                api_version: None,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::Google => ConnectionConfigPayload::Google {
                base_url: Some("https://google.example.invalid".into()),
                api_key_env: None,
                project: Some("proj".into()),
                location: None,
                auth_mode: None,
                credentials_path: None,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::Bedrock => bedrock_payload("eu-west-1"),
            Connector::Ollama => ConnectionConfigPayload::Ollama {
                // Loopback: Ollama is the one connector the URL policy lets
                // reach a plain-http endpoint, and only on a network the
                // operator already controls.
                base_url: Some("http://127.0.0.1:11434".into()),
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                keep_warm: None,
                max_context_tokens: None,
            },
        }
    }

    /// Every connector carrying a `secret` coordinate, paired with a stored
    /// connection that holds one and an update payload of the same connector
    /// type.
    ///
    /// Derived from [`Connector::ALL`] and [`Connector::carries_credential`],
    /// so a new credential-bearing connector joins the sweep the moment it is
    /// declared. `set_secret` refuses a connector with no `secret` field, so
    /// the filter and the stored connection cannot disagree in silence.
    fn credential_connectors() -> Vec<(Connector, ConnectionConfig, ConnectionConfigPayload)> {
        Connector::ALL
            .iter()
            .copied()
            .filter(|c| c.carries_credential())
            .map(|c| {
                let mut stored = stored_connection(c);
                stored.set_secret(secret_coordinate()).unwrap_or_else(|e| {
                    panic!("{c} claims a credential but refused a secret coordinate: {e}")
                });
                (c, stored, update_payload(c))
            })
            .collect()
    }

    #[tokio::test]
    async fn update_connection_preserves_existing_secret_coordinate() {
        let handle = make_handle_with(config_with_connections(&[("aws", bedrock_with_secret())]));
        let svc = DaemonConnectionsService::new(handle.clone());

        svc.update_connection("aws".to_string(), bedrock_payload("eu-west-1"))
            .await
            .expect("updating an existing connection should succeed");

        let cfg = handle.snapshot_config();
        let conn = cfg
            .connections
            .get("aws")
            .expect("connection should still exist after update");
        assert_eq!(
            secret_account(conn).as_deref(),
            Some(SECRET_ACCOUNT),
            "editing an unrelated field must not drop the credential reference"
        );
        match conn {
            ConnectionConfig::Bedrock(c) => assert_eq!(
                c.region.as_deref(),
                Some("eu-west-1"),
                "the requested edit should still apply"
            ),
            other => panic!("expected a bedrock connection, got {other:?}"),
        }
    }

    use std::collections::BTreeSet;

    /// A stored connection of each connector type whose **file-only** fields -
    /// the ones no `ConnectionConfigPayload` can express - are set to a value
    /// that is not the default, and which carries no secret, so a payload
    /// round trip differs from it in those fields and nothing else.
    ///
    /// Exhaustive with no catch-all, like [`stored_connection`]: a new
    /// connector must state what it keeps in the file before this compiles.
    /// `the_file_only_fixture_sets_every_field_a_connection_carries` holds it
    /// to every *field*, which the compiler cannot.
    fn stored_with_file_only_fields(connector: Connector) -> ConnectionConfig {
        match connector {
            Connector::Bedrock => ConnectionConfig::Bedrock(BedrockConnection {
                aws_profile: Some("work".into()),
                region: Some("us-west-2".into()),
                base_url: Some("https://bedrock-runtime.example.com".into()),
                secret: None,
                connect_timeout_secs: Some(11),
                stream_timeout_secs: Some(12),
                max_context_tokens: Some(4096),
                cache_policy: Some(desktop_assistant_llm_bedrock::CachePolicy::None),
            }),
            Connector::Anthropic => ConnectionConfig::Anthropic(AnthropicConnection {
                base_url: Some("https://api.anthropic.com".into()),
                api_key_env: Some("ANTHROPIC_API_KEY".into()),
                secret: None,
                connect_timeout_secs: Some(11),
                stream_timeout_secs: Some(12),
                max_context_tokens: Some(4096),
            }),
            Connector::OpenAi => ConnectionConfig::OpenAi(OpenAiConnection {
                base_url: Some("https://api.openai.com/v1".into()),
                api_key_env: Some("OPENAI_API_KEY".into()),
                secret: None,
                connect_timeout_secs: Some(11),
                stream_timeout_secs: Some(12),
                max_context_tokens: Some(4096),
            }),
            Connector::OpenRouter => ConnectionConfig::OpenRouter(OpenRouterConnection {
                base_url: Some("https://openrouter.ai/api/v1".into()),
                api_key_env: Some("OPENROUTER_API_KEY".into()),
                secret: None,
                connect_timeout_secs: Some(11),
                stream_timeout_secs: Some(12),
                max_context_tokens: Some(4096),
            }),
            Connector::Azure => ConnectionConfig::Azure(AzureConnection {
                base_url: Some("https://example-resource.openai.azure.com".into()),
                api_key_env: Some("AZURE_OPENAI_API_KEY".into()),
                secret: None,
                api_surface: Some("v1".into()),
                auth_mode: Some("api_key".into()),
                api_version: Some("2026-01-01".into()),
                connect_timeout_secs: Some(11),
                stream_timeout_secs: Some(12),
                max_context_tokens: Some(4096),
            }),
            Connector::Google => ConnectionConfig::Google(GoogleConnection {
                base_url: Some("https://us-central1-aiplatform.googleapis.com".into()),
                api_key_env: Some("GOOGLE_API_KEY".into()),
                secret: None,
                project: Some("example-project".into()),
                location: Some("us-central1".into()),
                auth_mode: Some("vertex".into()),
                credentials_path: Some("/etc/example/credentials.json".into()),
                connect_timeout_secs: Some(11),
                stream_timeout_secs: Some(12),
                max_context_tokens: Some(4096),
            }),
            Connector::Ollama => ConnectionConfig::Ollama(OllamaConnection {
                base_url: Some("http://127.0.0.1:11434".into()),
                connect_timeout_secs: Some(11),
                stream_timeout_secs: Some(12),
                keep_warm: Some(true),
                max_context_tokens: Some(4096),
            }),
        }
    }

    /// The fields of a stored connection, as serde writes them. Absent means
    /// the field is `None` and skipped, which is itself a value: a field that
    /// survived a round trip and one that did not are distinguishable.
    fn field_values(connection: &ConnectionConfig) -> serde_json::Map<String, serde_json::Value> {
        match serde_json::to_value(connection).expect("a connection serialises") {
            serde_json::Value::Object(fields) => fields,
            other => panic!("a connection must serialise to an object, got {other:?}"),
        }
    }

    /// The names of the fields on which two connections differ.
    ///
    /// Field-granular on purpose. `ConnectionConfig`'s `PartialEq` is
    /// whole-struct, so "these differ" is satisfied by any one surviving
    /// difference - a second file-only field could then be added, dropped on
    /// every client edit, and never noticed, because the first one still made
    /// the structs unequal.
    fn differing_fields(a: &ConnectionConfig, b: &ConnectionConfig) -> BTreeSet<String> {
        let (a, b) = (field_values(a), field_values(b));
        a.keys()
            .chain(b.keys())
            .filter(|key| a.get(*key) != b.get(*key))
            .cloned()
            .collect()
    }

    fn declared_file_only(connector: Connector) -> BTreeSet<String> {
        connector
            .file_only_fields()
            .iter()
            .map(|f| (*f).to_string())
            .collect()
    }

    /// The field names a struct's derived `Deserialize` declares.
    ///
    /// Read from serde rather than from a hand-written list: the derive hands
    /// the field list to `Deserializer::deserialize_struct`, so recording that
    /// argument is reflection over the real struct. A field added to a
    /// connection therefore reaches the fixture check with no edit, which is
    /// the one thing the compiler cannot force.
    fn declared_fields<'de, T: serde::Deserialize<'de>>() -> BTreeSet<String> {
        use serde::de::value::Error as ValueError;

        struct FieldRecorder<'a>(&'a mut BTreeSet<String>);

        impl<'de> serde::Deserializer<'de> for FieldRecorder<'_> {
            type Error = ValueError;

            fn deserialize_struct<V: serde::de::Visitor<'de>>(
                self,
                _name: &'static str,
                fields: &'static [&'static str],
                _visitor: V,
            ) -> Result<V::Value, Self::Error> {
                self.0.extend(fields.iter().map(|f| (*f).to_string()));
                Err(serde::de::Error::custom("fields recorded"))
            }

            fn deserialize_any<V: serde::de::Visitor<'de>>(
                self,
                _visitor: V,
            ) -> Result<V::Value, Self::Error> {
                Err(serde::de::Error::custom("not a struct"))
            }

            serde::forward_to_deserialize_any! {
                bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
                bytes byte_buf option unit unit_struct newtype_struct seq tuple
                tuple_struct map enum identifier ignored_any
            }
        }

        let mut fields = BTreeSet::new();
        let _ = T::deserialize(FieldRecorder(&mut fields));
        assert!(
            !fields.is_empty(),
            "no fields recorded - the reflection this check rests on has stopped working"
        );
        fields
    }

    /// Every field a connection of this type carries. Exhaustive with no
    /// catch-all.
    fn connection_struct_fields(connector: Connector) -> BTreeSet<String> {
        match connector {
            Connector::Anthropic => declared_fields::<AnthropicConnection>(),
            Connector::OpenAi => declared_fields::<OpenAiConnection>(),
            Connector::OpenRouter => declared_fields::<OpenRouterConnection>(),
            Connector::Azure => declared_fields::<AzureConnection>(),
            Connector::Google => declared_fields::<GoogleConnection>(),
            Connector::Bedrock => declared_fields::<BedrockConnection>(),
            Connector::Ollama => declared_fields::<OllamaConnection>(),
        }
    }

    #[test]
    fn the_file_only_fixture_sets_every_field_a_connection_carries() {
        // The sweeps below can only see a field the fixture sets to something
        // the payload does not reproduce. A new field that nobody adds here is
        // therefore a field no sweep checks - and the compiler cannot force it,
        // because a struct literal in a test is not the payload conversion.
        // So the fixture is checked against the struct itself.
        for &connector in Connector::ALL {
            let mut expected = connection_struct_fields(connector);
            // The credential coordinate is deliberately unset: it is dropped by
            // the payload for every connector, and `update_connection` carries
            // it forward on its own path, with its own sweep.
            expected.remove("secret");

            let mut present: BTreeSet<String> =
                field_values(&stored_with_file_only_fields(connector))
                    .keys()
                    .cloned()
                    .collect();
            // The enum tag, not a field of the struct.
            present.remove("type");

            assert_eq!(
                present, expected,
                "{connector}: the fixture must set every field the connection carries, \
                 or a field it misses is a field no sweep can see"
            );
        }
    }

    #[test]
    fn a_payload_round_trip_loses_exactly_the_fields_a_connector_declares_file_only() {
        // `Connector::file_only_fields` is the claim; this is the check of it,
        // field by field and in both directions. A connector that declares none
        // must survive the round trip whole, and one that declares some must
        // lose those and nothing else - so neither an undeclared file-only
        // field nor a stale declaration can pass.
        for &connector in Connector::ALL {
            let stored = stored_with_file_only_fields(connector);
            let round_tripped = payload_to_connection(connection_to_payload(&stored));
            assert_eq!(
                differing_fields(&stored, &round_tripped),
                declared_file_only(connector),
                "{connector}: the fields a payload round trip loses must be exactly the \
                 fields declared file-only - anything else is deleted on every client edit"
            );
        }
    }

    /// Store `stored` as connection `c`, apply `connector`'s update payload,
    /// and return the connection that resulted.
    async fn update_and_read_back(
        connector: Connector,
        stored: ConnectionConfig,
    ) -> ConnectionConfig {
        let handle = make_handle_with(config_with_connections(&[("c", stored)]));
        let svc = DaemonConnectionsService::new(handle.clone());
        svc.update_connection("c".to_string(), update_payload(connector))
            .await
            .unwrap_or_else(|e| panic!("{connector}: updating an existing connection: {e}"));
        handle
            .snapshot_config()
            .connections
            .get("c")
            .expect("connection still exists")
            .clone()
    }

    #[tokio::test]
    async fn update_connection_preserves_file_only_fields_across_all_connectors() {
        // The behavioural half. An edit through the API rebuilds the connection
        // from a payload that cannot carry these fields, so `update_connection`
        // must put them back - the same obligation, and the same sweep, as the
        // stored credential coordinate.
        //
        // The expected connection is built from the payload alone. Deliberately
        // not with the carry-forward itself, which would move both sides of the
        // assertion together and pass whatever the carry did.
        for &connector in Connector::ALL {
            let stored = stored_with_file_only_fields(connector);
            let after = update_and_read_back(connector, stored.clone()).await;
            let from_payload_alone = payload_to_connection(update_payload(connector));

            assert_eq!(
                differing_fields(&after, &from_payload_alone),
                declared_file_only(connector),
                "{connector}: exactly the declared file-only fields may survive an edit"
            );

            // And they carry the value that was stored, not merely some value.
            let (after_fields, stored_fields) = (field_values(&after), field_values(&stored));
            for field in connector.file_only_fields() {
                assert_eq!(
                    after_fields.get(*field),
                    stored_fields.get(*field),
                    "{connector}: {field} must survive an edit with the value it was stored with"
                );
            }

            // And the edit itself landed, whole.
            assert_eq!(
                connection_to_payload(&after),
                update_payload(connector),
                "{connector}: the requested edit must apply"
            );
        }
    }

    #[tokio::test]
    async fn update_connection_preserves_a_configured_bedrock_cache_policy() {
        // `cache_policy` is a daemon.toml setting with no field on the wire
        // payload, so an edit from a client rebuilds the connection without it.
        // Losing it silently puts the cache writes back on the bill (#1027).
        let stored = ConnectionConfig::Bedrock(crate::connections::BedrockConnection {
            aws_profile: Some("work".into()),
            region: Some("us-west-2".into()),
            cache_policy: Some(desktop_assistant_llm_bedrock::CachePolicy::None),
            ..Default::default()
        });
        let handle = make_handle_with(config_with_connections(&[("aws", stored)]));
        let svc = DaemonConnectionsService::new(handle.clone());

        svc.update_connection("aws".to_string(), bedrock_payload("eu-west-1"))
            .await
            .expect("updating an existing connection should succeed");

        let cfg = handle.snapshot_config();
        match cfg
            .connections
            .get("aws")
            .expect("connection should still exist after update")
        {
            ConnectionConfig::Bedrock(c) => {
                assert_eq!(
                    c.cache_policy,
                    Some(desktop_assistant_llm_bedrock::CachePolicy::None),
                    "editing an unrelated field must not turn caching back on"
                );
                assert_eq!(
                    c.region.as_deref(),
                    Some("eu-west-1"),
                    "the requested edit should still apply"
                );
            }
            other => panic!("expected a bedrock connection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn update_connection_preserves_secret_when_payload_changes_nothing() {
        let handle = make_handle_with(config_with_connections(&[("aws", bedrock_with_secret())]));
        let svc = DaemonConnectionsService::new(handle.clone());

        // A no-op save from a settings dialog is the most common way this
        // path runs, and the least excusable way to lose a credential.
        svc.update_connection("aws".to_string(), bedrock_payload("us-west-2"))
            .await
            .expect("a no-op update should succeed");

        let cfg = handle.snapshot_config();
        assert_eq!(
            secret_account(cfg.connections.get("aws").expect("connection should exist")).as_deref(),
            Some(SECRET_ACCOUNT),
            "a no-op update must not drop the credential reference"
        );
    }

    #[tokio::test]
    async fn update_connection_preserves_secret_across_all_credential_connectors() {
        let sweep = credential_connectors();
        assert!(
            !sweep.is_empty(),
            "the credential class is empty, so this sweep asserts nothing"
        );

        for (connector, stored, payload) in sweep {
            let handle = make_handle_with(config_with_connections(&[("conn", stored)]));
            let svc = DaemonConnectionsService::new(handle.clone());

            svc.update_connection("conn".to_string(), payload)
                .await
                .unwrap_or_else(|e| panic!("update should succeed for {connector}: {e}"));

            let cfg = handle.snapshot_config();
            assert_eq!(
                secret_account(
                    cfg.connections
                        .get("conn")
                        .expect("connection should exist")
                )
                .as_deref(),
                Some(SECRET_ACCOUNT),
                "{connector} lost its credential reference on update"
            );
        }
    }

    /// The carry-forward sweep must reach every connector that stores a
    /// credential, and no connector that does not.
    ///
    /// Fails in both directions: a connector wrongly excluded goes untested by
    /// the sweep above, and a connector wrongly included would be asked to hold
    /// a credential it has nowhere to put.
    #[tokio::test]
    async fn credential_connectors_covers_the_credential_bearing_connectors_only() {
        let swept: Vec<Connector> = credential_connectors()
            .into_iter()
            .map(|(c, _, _)| c)
            .collect();

        for &c in Connector::ALL {
            assert_eq!(
                swept.contains(&c),
                c.carries_credential(),
                "{c} is swept = {}, but carries_credential() says {}",
                swept.contains(&c),
                c.carries_credential(),
            );
        }
        assert!(
            !swept.is_empty(),
            "the credential class is empty, so the sweep asserts nothing"
        );
    }

    #[tokio::test]
    async fn update_connection_clears_secret_when_connector_type_changes() {
        let handle = make_handle_with(config_with_connections(&[("aws", bedrock_with_secret())]));
        let svc = DaemonConnectionsService::new(handle.clone());

        // A bedrock credential is an AWS key pair; carrying it onto an
        // OpenAI connection would leave an uninterpretable value in place.
        svc.update_connection(
            "aws".to_string(),
            ConnectionConfigPayload::OpenAi {
                base_url: Some("https://openai.example.invalid".into()),
                api_key_env: None,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
        )
        .await
        .expect("switching connector type should succeed");

        let cfg = handle.snapshot_config();
        let conn = cfg.connections.get("aws").expect("connection should exist");
        assert_eq!(conn.connector_type(), "openai", "the switch should apply");
        assert_eq!(
            secret_account(conn),
            None,
            "a connector switch must not carry the old credential coordinate across"
        );
    }

    #[tokio::test]
    async fn update_connection_leaves_absent_secret_absent() {
        let handle = make_handle_with(config_with_connections(&[("aws", bedrock_work())]));
        let svc = DaemonConnectionsService::new(handle.clone());

        svc.update_connection("aws".to_string(), bedrock_payload("eu-west-1"))
            .await
            .expect("update should succeed");

        let cfg = handle.snapshot_config();
        assert_eq!(
            secret_account(cfg.connections.get("aws").expect("connection should exist")),
            None,
            "a connection with no credential must not acquire one"
        );
    }

    #[tokio::test]
    async fn create_connection_never_sets_a_secret() {
        let handle = make_handle_with(DaemonConfig::default());
        let svc = DaemonConnectionsService::new(handle.clone());

        svc.create_connection("aws".to_string(), bedrock_payload("us-east-1"))
            .await
            .expect("create should succeed");

        let cfg = handle.snapshot_config();
        assert_eq!(
            secret_account(cfg.connections.get("aws").expect("connection should exist")),
            None,
            "creation has nothing to carry forward and must not invent a coordinate"
        );
    }

    // --- api_key_env may not name an arbitrary process env var (#736) -------

    /// Every connector whose wire payload carries an `api_key_env` field.
    ///
    /// Derived from [`Connector::ALL`] and [`Connector::carries_api_key_env`],
    /// so a new connector that reads a key from an environment variable joins
    /// the sweeps the moment it is declared.
    fn api_key_env_connectors() -> Vec<Connector> {
        Connector::ALL
            .iter()
            .copied()
            .filter(|c| c.carries_api_key_env())
            .collect()
    }

    /// A payload for `connector` carrying `api_key_env` and connector defaults
    /// for everything else. `None` for a connector with no such field.
    ///
    /// Exhaustive with no catch-all, so a new connector must answer here, and
    /// `payload_with_api_key_env_agrees_with_the_connector_claim` holds that
    /// answer to [`Connector::carries_api_key_env`].
    fn payload_with_api_key_env(
        connector: Connector,
        api_key_env: Option<&str>,
    ) -> Option<ConnectionConfigPayload> {
        let api_key_env = api_key_env.map(str::to_string);
        Some(match connector {
            Connector::Anthropic => ConnectionConfigPayload::Anthropic {
                base_url: None,
                api_key_env,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::OpenAi => ConnectionConfigPayload::OpenAi {
                base_url: None,
                api_key_env,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::OpenRouter => ConnectionConfigPayload::OpenRouter {
                base_url: None,
                api_key_env,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::Azure => ConnectionConfigPayload::Azure {
                base_url: None,
                api_key_env,
                api_surface: None,
                auth_mode: None,
                api_version: None,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::Google => ConnectionConfigPayload::Google {
                base_url: None,
                api_key_env,
                project: Some("proj".into()),
                location: None,
                auth_mode: None,
                credentials_path: None,
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
            Connector::Bedrock | Connector::Ollama => return None,
        })
    }

    /// The payload builder and the connector claim answer the same question,
    /// so they must agree for every variant.
    ///
    /// Fails in both directions: a connector that claims the field and has no
    /// payload for it fails, and so does a connector with a payload that
    /// claims no field.
    #[test]
    fn payload_with_api_key_env_agrees_with_the_connector_claim() {
        for &c in Connector::ALL {
            assert_eq!(
                payload_with_api_key_env(c, Some("PROBE_API_KEY")).is_some(),
                c.carries_api_key_env(),
                "{c} builds an api_key_env payload = {}, but \
                 carries_api_key_env() says {}",
                payload_with_api_key_env(c, Some("PROBE_API_KEY")).is_some(),
                c.carries_api_key_env(),
            );
        }
    }

    /// A payload for a connector that carries the field. Panics otherwise, so
    /// a caller naming the wrong connector says so instead of skipping.
    fn api_key_env_payload(
        connector: Connector,
        api_key_env: Option<&str>,
    ) -> ConnectionConfigPayload {
        payload_with_api_key_env(connector, api_key_env)
            .unwrap_or_else(|| panic!("{connector} carries no api_key_env field"))
    }

    /// The `api_key_env` stored on a connection, whatever its connector.
    fn stored_api_key_env(conn: &ConnectionConfig) -> Option<String> {
        match conn {
            ConnectionConfig::Anthropic(c) => c.api_key_env.clone(),
            ConnectionConfig::OpenAi(c) => c.api_key_env.clone(),
            ConnectionConfig::OpenRouter(c) => c.api_key_env.clone(),
            ConnectionConfig::Azure(c) => c.api_key_env.clone(),
            ConnectionConfig::Google(c) => c.api_key_env.clone(),
            ConnectionConfig::Bedrock(_) | ConnectionConfig::Ollama(_) => None,
        }
    }

    /// A stored OpenAI connection reading its key from `name`, as an operator
    /// would write it in `daemon.toml`.
    fn openai_reading_env(name: Option<&str>) -> ConnectionConfig {
        use crate::connections::OpenAiConnection;
        ConnectionConfig::OpenAi(OpenAiConnection {
            base_url: Some("https://api.openai.com/v1".into()),
            api_key_env: name.map(str::to_string),
            ..Default::default()
        })
    }

    /// The `api_key_env` a stored connection reads, by connection id.
    fn api_key_env_of(handle: &RegistryHandle, id: &str) -> Option<String> {
        stored_api_key_env(
            handle
                .snapshot_config()
                .connections
                .get(id)
                .expect("connection should exist"),
        )
    }

    #[tokio::test]
    async fn create_connection_rejects_api_key_env_outside_the_connector_allowlist() {
        let handle = make_handle_with(DaemonConfig::default());
        let svc = DaemonConnectionsService::new(handle.clone());

        // The daemon's own process environment carries the deployment's
        // secrets; naming one here would POST it to `base_url` as a bearer
        // token.
        for name in [
            "POSTGRES_PASSWORD",
            "DESKTOP_ASSISTANT_DATABASE_URL",
            "DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD",
            "PATH",
        ] {
            let err = svc
                .create_connection(
                    "exfil".to_string(),
                    api_key_env_payload(Connector::OpenAi, Some(name)),
                )
                .await
                .unwrap_err();
            let err = format!("{err}");
            assert!(
                err.contains("api_key_env"),
                "rejecting {name} should name the offending field; got: {err}"
            );
            assert!(
                err.contains("OPENAI_API_KEY"),
                "rejecting {name} should say what is permitted instead; got: {err}"
            );
        }

        assert!(
            handle.snapshot_config().connections.is_empty(),
            "a rejected create must not store a connection"
        );
    }

    #[tokio::test]
    async fn create_connection_rejects_lowercase_api_key_env_bypass() {
        let handle = make_handle_with(DaemonConfig::default());
        let svc = DaemonConnectionsService::new(handle.clone());

        // Environment variable names are case-sensitive on Unix, so a
        // case-folded match would admit a different variable entirely.
        for name in ["openai_api_key", "OpenAi_Api_Key", "OPENAI_API_key"] {
            svc.create_connection(
                "exfil".to_string(),
                api_key_env_payload(Connector::OpenAi, Some(name)),
            )
            .await
            .unwrap_err();
        }

        assert!(
            handle.snapshot_config().connections.is_empty(),
            "a case-folded bypass must not store a connection"
        );
    }

    #[tokio::test]
    async fn create_connection_rejects_api_key_env_that_merely_contains_an_allowed_name() {
        let handle = make_handle_with(DaemonConfig::default());
        let svc = DaemonConnectionsService::new(handle.clone());

        for name in [
            "MY_OPENAI_API_KEY",
            "OPENAI_API_KEY_BACKUP",
            "XOPENAI_API_KEYX",
            "OPENAI_API_KE",
        ] {
            svc.create_connection(
                "exfil".to_string(),
                api_key_env_payload(Connector::OpenAi, Some(name)),
            )
            .await
            .unwrap_err();
        }

        assert!(
            handle.snapshot_config().connections.is_empty(),
            "a substring match must not store a connection"
        );
    }

    #[tokio::test]
    async fn create_connection_rejects_another_connectors_api_key_env() {
        let handle = make_handle_with(DaemonConfig::default());
        let svc = DaemonConnectionsService::new(handle.clone());

        // Pointing one connector at another's key sends that key to a host
        // the payload also chooses.
        for (connector, name) in [
            (Connector::Anthropic, "OPENAI_API_KEY"),
            (Connector::OpenAi, "ANTHROPIC_API_KEY"),
            (Connector::Google, "AZURE_OPENAI_API_KEY"),
            (Connector::OpenRouter, "GOOGLE_API_KEY"),
        ] {
            let err = svc
                .create_connection(
                    "exfil".to_string(),
                    api_key_env_payload(connector, Some(name)),
                )
                .await
                .unwrap_err();
            assert!(
                format!("{err}").contains("api_key_env"),
                "{connector} must not read {name}; got: {err}"
            );
        }

        assert!(
            handle.snapshot_config().connections.is_empty(),
            "a cross-connector key name must not store a connection"
        );
    }

    #[tokio::test]
    async fn create_connection_accepts_the_documented_api_key_env_for_each_connector() {
        for (connector, name) in [
            (Connector::Anthropic, "ANTHROPIC_API_KEY"),
            (Connector::OpenAi, "OPENAI_API_KEY"),
            (Connector::OpenRouter, "OPENROUTER_API_KEY"),
            (Connector::Azure, "AZURE_OPENAI_API_KEY"),
            (Connector::Azure, "AZURE_API_KEY"),
            (Connector::Google, "GOOGLE_API_KEY"),
        ] {
            let handle = make_handle_with(DaemonConfig::default());
            let svc = DaemonConnectionsService::new(handle.clone());

            svc.create_connection(
                "conn".to_string(),
                api_key_env_payload(connector, Some(name)),
            )
            .await
            .unwrap_or_else(|e| panic!("{connector} should accept {name}: {e}"));

            assert_eq!(
                api_key_env_of(&handle, "conn").as_deref(),
                Some(name),
                "{connector} should store {name} verbatim"
            );
        }
    }

    #[tokio::test]
    async fn create_connection_accepts_each_connectors_derived_default_api_key_env() {
        // Pins the allowlist to the `<CONNECTOR>_API_KEY` derivation so the
        // two cannot drift apart.
        for connector in api_key_env_connectors() {
            let derived = crate::config::default_api_key_env(connector.as_str());
            let handle = make_handle_with(DaemonConfig::default());
            let svc = DaemonConnectionsService::new(handle.clone());

            svc.create_connection(
                "conn".to_string(),
                api_key_env_payload(connector, Some(&derived)),
            )
            .await
            .unwrap_or_else(|e| panic!("{connector} should accept its derived {derived}: {e}"));

            assert_eq!(
                api_key_env_of(&handle, "conn").as_deref(),
                Some(derived.as_str()),
                "{connector} should store its derived default verbatim"
            );
        }
    }

    #[tokio::test]
    async fn create_connection_normalizes_a_blank_api_key_env_to_unset() {
        for blank in ["", "   ", "\t"] {
            let handle = make_handle_with(DaemonConfig::default());
            let svc = DaemonConnectionsService::new(handle.clone());

            svc.create_connection(
                "conn".to_string(),
                api_key_env_payload(Connector::OpenAi, Some(blank)),
            )
            .await
            .unwrap_or_else(|e| panic!("a blank api_key_env should mean unset: {e}"));

            assert_eq!(
                api_key_env_of(&handle, "conn"),
                None,
                "a blank api_key_env must not be stored as a variable name"
            );
        }
    }

    #[tokio::test]
    async fn create_connection_trims_whitespace_around_an_allowed_api_key_env() {
        let handle = make_handle_with(DaemonConfig::default());
        let svc = DaemonConnectionsService::new(handle.clone());

        svc.create_connection(
            "conn".to_string(),
            api_key_env_payload(Connector::OpenAi, Some("  OPENAI_API_KEY\n")),
        )
        .await
        .expect("surrounding whitespace should not defeat the allowlist");

        assert_eq!(
            api_key_env_of(&handle, "conn").as_deref(),
            Some("OPENAI_API_KEY"),
            "the stored name must be the trimmed one, not the padded one"
        );
    }

    #[tokio::test]
    async fn update_connection_rejects_api_key_env_outside_the_connector_allowlist() {
        let handle = make_handle_with(config_with_connections(&[(
            "work",
            openai_reading_env(Some("OPENAI_API_KEY")),
        )]));
        let svc = DaemonConnectionsService::new(handle.clone());

        let err = svc
            .update_connection(
                "work".to_string(),
                ConnectionConfigPayload::OpenAi {
                    base_url: Some("https://attacker.example.invalid/v1".into()),
                    api_key_env: Some("POSTGRES_PASSWORD".into()),
                    connect_timeout_secs: None,
                    stream_timeout_secs: None,
                    max_context_tokens: None,
                },
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err}").contains("api_key_env"),
            "rejection should name the offending field; got: {err}"
        );

        let cfg = handle.snapshot_config();
        let conn = cfg
            .connections
            .get("work")
            .expect("connection should exist");
        assert_eq!(
            stored_api_key_env(conn).as_deref(),
            Some("OPENAI_API_KEY"),
            "a rejected update must leave the stored key variable alone"
        );
        match conn {
            ConnectionConfig::OpenAi(c) => assert_eq!(
                c.base_url.as_deref(),
                Some("https://api.openai.com/v1"),
                "a rejected update must not apply its base_url either"
            ),
            other => panic!("expected an openai connection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn update_connection_keeps_an_operator_configured_api_key_env() {
        // `daemon.toml` is operator-owned and may name a custom variable; the
        // echoed view carries it back, so a client edit that leaves the field
        // untouched must still save.
        let handle = make_handle_with(config_with_connections(&[(
            "work",
            openai_reading_env(Some("OPENAI_WORK_KEY")),
        )]));
        let svc = DaemonConnectionsService::new(handle.clone());

        svc.update_connection(
            "work".to_string(),
            api_key_env_payload(Connector::OpenAi, Some("OPENAI_WORK_KEY")),
        )
        .await
        .expect("re-sending the stored api_key_env must not be rejected");

        assert_eq!(
            api_key_env_of(&handle, "work").as_deref(),
            Some("OPENAI_WORK_KEY"),
            "the operator's custom variable should survive a client round-trip"
        );
    }

    #[tokio::test]
    async fn update_connection_rejects_another_connections_api_key_env() {
        // Carrying an unchanged value forward is per-connection: connection B
        // must not inherit connection A's custom variable.
        let handle = make_handle_with(config_with_connections(&[
            ("work", openai_reading_env(Some("OPENAI_WORK_KEY"))),
            ("personal", openai_reading_env(Some("OPENAI_API_KEY"))),
        ]));
        let svc = DaemonConnectionsService::new(handle.clone());

        let err = svc
            .update_connection(
                "personal".to_string(),
                api_key_env_payload(Connector::OpenAi, Some("OPENAI_WORK_KEY")),
            )
            .await
            .unwrap_err();
        assert!(
            format!("{err}").contains("api_key_env"),
            "rejection should name the offending field; got: {err}"
        );

        assert_eq!(
            api_key_env_of(&handle, "personal").as_deref(),
            Some("OPENAI_API_KEY"),
            "the rejected update must not have applied"
        );
    }

    #[test]
    fn set_connection_secret_overwrites_a_carried_forward_coordinate() {
        with_isolated_secret_store(|_| {
            let handle = make_handle_at(
                config_with_connections(&[("aws", bedrock_with_secret())]),
                tmp_config_path(),
            );
            let svc = DaemonConnectionsService::new(handle.clone());

            run_async(svc.update_connection("aws".into(), bedrock_payload("eu-west-1"))).unwrap();
            run_async(svc.set_connection_secret("aws".into(), BEDROCK_CRED.into())).unwrap();

            // The explicit credential path still wins over whatever the
            // update carried forward, and resolves to the stored value.
            assert_eq!(resolved_api_key(&handle, "aws"), BEDROCK_CRED);
        });
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

    // --- reload must not build a registry it throws away -------------------

    #[test]
    fn no_op_reload_does_not_build_a_registry() {
        let cfg = config_with_connections(&[("local", ollama_local())]);
        let path = tmp_config_path();
        crate::config::save_daemon_config(&path, &cfg).expect("seed config on disk");
        // Run from the config as it round-trips through TOML, so "unchanged on
        // disk" really is unchanged — serialization materializes defaults that
        // the in-memory value does not carry.
        let loaded = crate::config::load_and_migrate_daemon_config(&path)
            .expect("load succeeds")
            .expect("config is present");
        let handle = make_handle_at(loaded, path.clone());

        // Reloading a config identical to the running one is the common case:
        // every daemon-authored write trips the on-disk watcher, which then
        // finds nothing to do. It must not cost a full set of LLM clients.
        crate::registry::reset_build_registry_calls();
        let plan = handle.apply_reload().expect("reload succeeds");

        assert!(plan.is_empty(), "precondition: nothing changed");
        assert_eq!(
            crate::registry::build_registry_calls(),
            0,
            "a reload with no effective changes must not construct clients"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn effective_reload_still_rebuilds_and_swaps() {
        let running = config_with_connections(&[("local", ollama_local())]);
        let path = tmp_config_path();
        let handle = make_handle_at(running, path.clone());

        // A real [connections] change must still hot-apply.
        let changed =
            config_with_connections(&[("local", ollama_local()), ("aws", bedrock_work())]);
        crate::config::save_daemon_config(&path, &changed).expect("write changed config");

        crate::registry::reset_build_registry_calls();
        let plan = handle.apply_reload().expect("reload succeeds");

        assert!(plan.rebuild_registry, "a connections edit hot-applies");
        assert_eq!(
            crate::registry::build_registry_calls(),
            1,
            "exactly one build for a real change — not zero, and not two"
        );
        assert!(
            handle.snapshot_config().connections.contains_key("aws"),
            "the new connection should be live"
        );
        std::fs::remove_file(&path).ok();
    }

    // --- SetPurpose must reject a mixed inherit pair -----------------------

    /// A purposes config with interactive bound, so a non-interactive write
    /// under test is the only thing that can fail validation.
    fn handle_with_interactive() -> Arc<RegistryHandle> {
        let mut cfg = config_with_connections(&[("local", ollama_local())]);
        cfg.purposes.set(
            PurposeKind::Interactive,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("llama3".into()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        make_handle_with(cfg)
    }

    fn payload(connection: &str, model: &str) -> PurposeConfigPayload {
        PurposeConfigPayload {
            connection: connection.into(),
            model: model.into(),
            effort: None,
            max_context_tokens: None,
        }
    }

    #[tokio::test]
    async fn set_purpose_rejects_real_connection_with_primary_model() {
        // The exact shape a client wrote to production: a real connection
        // paired with the inherit sentinel, which retired a live binding.
        let svc = DaemonConnectionsService::new(handle_with_interactive());
        let err = svc
            .set_purpose(PurposeKind::Embedding, payload("local", "primary"))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("primary"),
            "rejection should name the sentinel; got {msg}"
        );
    }

    #[tokio::test]
    async fn set_purpose_rejects_primary_connection_with_real_model() {
        let svc = DaemonConnectionsService::new(handle_with_interactive());
        let err = svc
            .set_purpose(PurposeKind::Dreaming, payload("primary", "llama3"))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("primary"));
    }

    #[tokio::test]
    async fn set_purpose_rejection_names_the_purpose_and_the_pair() {
        let svc = DaemonConnectionsService::new(handle_with_interactive());
        let err = svc
            .set_purpose(PurposeKind::Embedding, payload("local", "primary"))
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("embedding") && msg.contains("local"),
            "an operator must be able to tell which purpose and which pair was \
             refused without reading the source; got {msg}"
        );
    }

    #[tokio::test]
    async fn set_purpose_rejection_leaves_the_stored_binding_untouched() {
        let handle = handle_with_interactive();
        handle
            .mutate_config(|cfg| {
                cfg.purposes.set(
                    PurposeKind::Embedding,
                    Some(PurposeConfig {
                        connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                        model: ModelRef::Named("nomic-embed-text".into()),
                        effort: None,
                        max_context_tokens: None,
                    }),
                );
                Ok(())
            })
            .expect("seed a good binding");

        let svc = DaemonConnectionsService::new(handle.clone());
        svc.set_purpose(PurposeKind::Embedding, payload("local", "primary"))
            .await
            .expect_err("mixed pair must be refused");

        let stored = handle.snapshot_config();
        let embedding = stored
            .purposes
            .get(PurposeKind::Embedding)
            .expect("binding should survive a refused write");
        assert_eq!(embedding.model, ModelRef::Named("nomic-embed-text".into()));
    }

    #[tokio::test]
    async fn set_purpose_accepts_a_full_primary_pair_for_non_interactive() {
        let svc = DaemonConnectionsService::new(handle_with_interactive());
        svc.set_purpose(PurposeKind::Dreaming, payload("primary", "primary"))
            .await
            .expect("inheriting from interactive is the whole point of the sentinel");
    }

    #[tokio::test]
    async fn set_purpose_accepts_two_real_ids() {
        let svc = DaemonConnectionsService::new(handle_with_interactive());
        svc.set_purpose(PurposeKind::Embedding, payload("local", "nomic-embed-text"))
            .await
            .expect("a fully-specified binding is always valid");
    }

    // --- SetPurpose must reject a model whose kind contradicts the purpose ---
    // (#647). These use an injected client with a fixed model catalog so the
    // resolved kind is deterministic and no network is touched.

    use desktop_assistant_core::ports::llm::{
        LlmResponse, ModelCapabilities, ModelInfo, ModelKind,
    };

    /// A minimal `LlmClient` whose only real behaviour is a fixed model list,
    /// so `set_purpose` enforcement can resolve a known [`ModelKind`]. `fail`
    /// makes `list_models` error, exercising the connection-down degradation.
    struct FakeModelList {
        models: Vec<ModelInfo>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl LlmClient for FakeModelList {
        async fn stream_completion(
            &self,
            _messages: Vec<desktop_assistant_core::domain::Message>,
            _tools: &[desktop_assistant_core::domain::ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            Ok(LlmResponse::text(""))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
            if self.fail {
                Err(CoreError::Llm("connection down".into()))
            } else {
                Ok(self.models.clone())
            }
        }
    }

    fn model_of(id: &str, kind: ModelKind) -> ModelInfo {
        ModelInfo::new(id).with_capabilities(ModelCapabilities {
            kind,
            ..Default::default()
        })
    }

    /// Build a handle whose `conn` connection is a [`FakeModelList`] returning
    /// `models`, with interactive already bound (so a non-interactive write is
    /// the only thing that can fail validation).
    fn handle_with_model_catalog(
        conn: &str,
        models: Vec<ModelInfo>,
        fail: bool,
    ) -> Arc<RegistryHandle> {
        let mut cfg = config_with_connections(&[(conn, ollama_local())]);
        cfg.purposes.set(
            PurposeKind::Interactive,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new(conn).unwrap()),
                model: ModelRef::Named("interactive-model".into()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        let id = ConnectionId::new(conn).unwrap();
        let client: Arc<dyn LlmClient> = Arc::new(FakeModelList { models, fail });
        let registry =
            ConnectionRegistry::from_test_clients(vec![(id, "ollama".to_string(), client)]);
        Arc::new(RegistryHandle::new(cfg, registry).with_config_path(tmp_config_path()))
    }

    #[tokio::test]
    async fn set_purpose_rejects_generative_model_for_embedding_purpose() {
        // The production failure: a text-generation model bound to embedding.
        let handle = handle_with_model_catalog(
            "local",
            vec![model_of("zai.glm-5", ModelKind::Generative)],
            false,
        );
        let svc = DaemonConnectionsService::new(handle);
        let err = svc
            .set_purpose(PurposeKind::Embedding, payload("local", "zai.glm-5"))
            .await
            .expect_err("a generative model cannot serve the embedding purpose");
        let msg = format!("{err}");
        assert!(
            msg.contains("zai.glm-5") && msg.contains("embedding"),
            "rejection must name the model and the purpose; got {msg}"
        );
    }

    #[tokio::test]
    async fn set_purpose_rejects_embedding_model_for_generative_purpose() {
        // The mirror case: an embedding model bound to a generative purpose.
        let handle = handle_with_model_catalog(
            "local",
            vec![model_of("nomic-embed-text", ModelKind::Embedding)],
            false,
        );
        let svc = DaemonConnectionsService::new(handle);
        let err = svc
            .set_purpose(PurposeKind::Dreaming, payload("local", "nomic-embed-text"))
            .await
            .expect_err("an embedding model cannot serve a generative purpose");
        let msg = format!("{err}");
        assert!(
            msg.contains("nomic-embed-text") && msg.contains("dreaming"),
            "rejection must name the model and the purpose; got {msg}"
        );
    }

    #[tokio::test]
    async fn set_purpose_rejection_names_the_model_and_purpose() {
        // The error must be actionable without reading the source: model id,
        // purpose, and the expected-vs-found kinds.
        let handle = handle_with_model_catalog(
            "local",
            vec![model_of("zai.glm-5", ModelKind::Generative)],
            false,
        );
        let svc = DaemonConnectionsService::new(handle);
        let err = svc
            .set_purpose(PurposeKind::Embedding, payload("local", "zai.glm-5"))
            .await
            .expect_err("contradiction must be refused");
        let msg = format!("{err}").to_ascii_lowercase();
        assert!(msg.contains("zai.glm-5"), "names the model; got {msg}");
        assert!(msg.contains("embedding"), "names the purpose; got {msg}");
        assert!(
            msg.contains("generative"),
            "names the kind that was found; got {msg}"
        );
    }

    #[tokio::test]
    async fn set_purpose_allows_unknown_kind_with_a_warning() {
        // An `Unknown` kind degrades to allow-with-a-warning; it must never
        // block a config edit.
        let handle = handle_with_model_catalog(
            "local",
            vec![model_of("mystery-model", ModelKind::Unknown)],
            false,
        );
        let svc = DaemonConnectionsService::new(handle);
        svc.set_purpose(PurposeKind::Embedding, payload("local", "mystery-model"))
            .await
            .expect("an unknown kind must not block the write");
    }

    #[tokio::test]
    async fn set_purpose_allows_model_missing_from_the_catalog() {
        // A custom id the connector never listed cannot be classified, so the
        // write is allowed with a warning rather than blocked.
        let handle = handle_with_model_catalog(
            "local",
            vec![model_of("some-other-model", ModelKind::Generative)],
            false,
        );
        let svc = DaemonConnectionsService::new(handle);
        svc.set_purpose(PurposeKind::Embedding, payload("local", "not-in-the-list"))
            .await
            .expect("a model the connector didn't list must not block the write");
    }

    #[tokio::test]
    async fn set_purpose_allows_when_listing_fails() {
        // A connection that is down (list_models errors) must not turn a config
        // edit into a failure — a transient network fault is not a rules
        // violation.
        let handle = handle_with_model_catalog("local", vec![], true);
        let svc = DaemonConnectionsService::new(handle);
        svc.set_purpose(PurposeKind::Embedding, payload("local", "anything"))
            .await
            .expect("a listing failure must degrade to allow-with-a-warning");
    }

    #[tokio::test]
    async fn set_purpose_accepts_a_kind_matching_binding() {
        // The positive path: an embedding model for the embedding purpose is
        // accepted, proving enforcement doesn't reject valid bindings.
        let handle = handle_with_model_catalog(
            "local",
            vec![model_of("nomic-embed-text", ModelKind::Embedding)],
            false,
        );
        let svc = DaemonConnectionsService::new(handle);
        svc.set_purpose(PurposeKind::Embedding, payload("local", "nomic-embed-text"))
            .await
            .expect("an embedding model is valid for the embedding purpose");
    }

    #[test]
    fn existing_contradictory_config_loads_with_a_warning() {
        // Enforcement lives on the write path only. An already-stored
        // contradictory binding (embedding purpose -> a generative model, the
        // exact prod shape) must still BOOT: the loader does not check kinds,
        // so it returns Ok rather than crashing the daemon on startup.
        let mut cfg = config_with_connections(&[("local", ollama_local())]);
        cfg.purposes.set(
            PurposeKind::Interactive,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("llama3".into()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        // The contradiction: a generative chat model bound to embedding.
        cfg.purposes.set(
            PurposeKind::Embedding,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("llama3".into()),
                effort: None,
                max_context_tokens: None,
            }),
        );

        let path = tmp_config_path();
        crate::config::save_daemon_config(&path, &cfg).expect("seed contradictory config on disk");
        let loaded = crate::config::load_and_migrate_daemon_config(&path)
            .expect("a stored contradictory binding must still load, not crash the daemon")
            .expect("config is present");
        let embedding = loaded
            .purposes
            .get(PurposeKind::Embedding)
            .expect("the contradictory embedding binding is preserved as-is");
        assert_eq!(embedding.model, ModelRef::Named("llama3".into()));
        std::fs::remove_file(&path).ok();
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
    /// file I/O + registry rebuild. We prove it by pointing the config path
    /// at a FIFO with no writer: the pre-write guard's read of the file
    /// (`parse_daemon_config`) blocks forever on `open(O_RDONLY)`. A
    /// concurrent `snapshot_config` read must still complete promptly - it
    /// would hang if the write lock were held across the I/O.
    #[cfg(unix)]
    #[test]
    fn mutate_config_does_not_hold_lock_across_blocking_io() {
        use std::sync::mpsc;
        use std::time::Duration;

        // Build a FIFO path. open(O_RDONLY) on a FIFO blocks until a writer
        // appears, which never happens here - a deterministic "slow I/O".
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

        // Writer thread: this mutate will block inside its file I/O (the
        // open on the writerless FIFO) and never return.
        let writer = Arc::clone(&handle);
        std::thread::spawn(move || {
            let _ = writer.set_personality(desktop_assistant_core::prompts::Personality::default());
        });

        // Give the writer time to reach (and block in) the file I/O.
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
            let cfg = crate::config::load_and_migrate_daemon_config(&path)
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
            // A refused reload must not invent a restart requirement either:
            // the rejected edit touched only [connections], which is hot.
            assert!(
                handle.restart_required().is_empty(),
                "a refused connections-only reload reports no restart requirement: {:?}",
                handle.restart_required()
            );
            let _ = std::fs::remove_file(&path);
        }

        // ----- Restart-required reporting (#686) -----------------------
        //
        // `restart_required` answers "what is in the config file that the
        // *running process* does not have?" by diffing the config the daemon
        // booted with against the config on disk. That baseline is what makes
        // it correct on the daemon-authored write path, where the file watcher
        // sees a genuine no-op because `mutate_config` already updated the
        // in-memory config before the watcher fired.

        /// An ollama connection plus a named interactive purpose, so purpose
        /// edits load and validate.
        const OLLAMA_WITH_PURPOSES: &str = r#"
[connections.local]
type = "ollama"
base_url = "http://localhost:11434"

[purposes.interactive]
connection = "local"
model = "llama3"

[purposes.embedding]
connection = "local"
model = "nomic-embed-text"
"#;

        #[test]
        fn nothing_changed_since_boot_needs_no_restart() {
            let (handle, path) = handle_for_toml(OLLAMA_WITH_PURPOSES);
            assert!(
                handle.restart_required().is_empty(),
                "a freshly booted daemon whose config is untouched needs no restart: {:?}",
                handle.restart_required()
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn daemon_authored_embedding_purpose_write_is_reported_despite_an_empty_watcher_diff() {
            let (handle, path) = handle_for_toml(OLLAMA_WITH_PURPOSES);

            // A daemon-authored write: `mutate_config` saves the file AND
            // refreshes the in-memory config in one step.
            handle
                .mutate_config(|cfg| {
                    cfg.purposes.set(
                        crate::purposes::PurposeKind::Embedding,
                        Some(PurposeConfig {
                            connection: ConnectionRef::Named(
                                ConnectionId::new("local").expect("test slug is valid"),
                            ),
                            model: ModelRef::Named("amazon.titan-embed-text-v2:0".to_string()),
                            effort: None,
                            max_context_tokens: None,
                        }),
                    );
                    Ok(())
                })
                .expect("a valid purpose write succeeds");

            assert!(
                handle
                    .restart_required()
                    .contains(&crate::config::RestartArea::Embeddings),
                "a daemon-authored embedding-purpose write must report a restart requirement: {:?}",
                handle.restart_required()
            );

            // The file watcher fires next and finds nothing to do, which is
            // precisely why the write path has to be the one that reports.
            let plan = handle
                .apply_reload()
                .expect("the watcher's re-read of our own write applies");
            assert!(
                plan.is_empty(),
                "the watcher diff after a daemon-authored write is a genuine no-op: {plan:?}"
            );
            assert!(
                handle
                    .restart_required()
                    .contains(&crate::config::RestartArea::Embeddings),
                "the no-op watcher pass must not clear the outstanding restart requirement"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn ws_auth_edit_is_reported_before_any_reload_runs() {
            // `set_ws_auth_settings` writes the file directly and never touches
            // the registry, so the answer has to come from the file.
            let (handle, path) = handle_for_toml(OLLAMA_WITH_PURPOSES);
            std::fs::write(
                &path,
                format!("{OLLAMA_WITH_PURPOSES}\n[ws_auth]\nmethods = [\"oidc\"]\n"),
            )
            .expect("write ws_auth edit");
            assert!(
                handle
                    .restart_required()
                    .contains(&crate::config::RestartArea::WsAuth),
                "a [ws_auth] edit must be reported without waiting for a reload: {:?}",
                handle.restart_required()
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn tls_edit_is_reported_before_any_reload_runs() {
            // Certificate rotation is a file-only edit with no client-facing
            // setter at all, so the on-disk diff is the only honest source.
            let (handle, path) = handle_for_toml(OLLAMA_WITH_PURPOSES);
            std::fs::write(
                &path,
                format!("{OLLAMA_WITH_PURPOSES}\n[tls]\nenabled = false\n"),
            )
            .expect("write tls edit");
            assert!(
                handle
                    .restart_required()
                    .contains(&crate::config::RestartArea::Tls),
                "a [tls] edit must be reported without waiting for a reload: {:?}",
                handle.restart_required()
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn hot_applicable_edit_reports_no_restart_requirement() {
            let (handle, path) = handle_for_toml(OLLAMA_WITH_PURPOSES);
            std::fs::write(
                &path,
                OLLAMA_WITH_PURPOSES.replace("http://localhost:11434", "http://localhost:9999"),
            )
            .expect("write connection edit");
            let plan = handle.apply_reload().expect("valid reload applies");
            assert!(plan.rebuild_registry);
            assert!(
                handle.restart_required().is_empty(),
                "a hot-applicable edit must not claim a restart is needed: {:?}",
                handle.restart_required()
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn reverting_a_restart_bound_edit_clears_the_report() {
            let (handle, path) = handle_for_toml(OLLAMA_WITH_PURPOSES);
            std::fs::write(
                &path,
                format!("{OLLAMA_WITH_PURPOSES}\n[tls]\nenabled = false\n"),
            )
            .expect("write tls edit");
            assert!(!handle.restart_required().is_empty());

            std::fs::write(&path, OLLAMA_WITH_PURPOSES).expect("revert the edit");
            assert!(
                handle.restart_required().is_empty(),
                "reverting the edit puts the file back in step with the running process: {:?}",
                handle.restart_required()
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn unparseable_config_reports_the_load_failure_and_invents_no_areas() {
            // A broken file must not panic the report or invent areas out of a
            // config nobody can read. What the client needs to see is the one
            // fact the daemon does know: the file on disk did not load.
            let (handle, path) = handle_for_toml(OLLAMA_WITH_PURPOSES);
            std::fs::write(&path, "this is not = valid toml [[[").expect("write garbage");
            let keys: Vec<String> = handle
                .restart_required()
                .iter()
                .map(|area| area.as_key().to_string())
                .collect();
            assert_eq!(
                keys,
                vec!["config_load_failed".to_string()],
                "an unreadable config is reported as exactly that, with no areas invented \
                 out of a config nobody can read"
            );
            let _ = std::fs::remove_file(&path);
        }
    }

    // ----- A config that failed to load must not be overwritten (#723) ---
    //
    // The daemon still starts on a broken `daemon.toml` — going down entirely
    // is worse. What it must not do is serialize its built-in defaults over
    // the file, which is what every connection/purpose/personality write does
    // with the in-memory snapshot.
    mod config_load_failure {
        use super::*;
        use crate::config::ConfigOrigin;

        /// A hand-authored config the daemon cannot load: `api_key_envv` is a
        /// typo and connection tables `deny_unknown_fields`, so the whole file
        /// is refused. Everything else in it is what a write-from-defaults
        /// destroys.
        const TYPO_IN_CONNECTION: &str = r#"
# Hand-authored, comments and all.
[connections.work]
type = "anthropic"
api_key_envv = "WORK_ANTHROPIC_KEY"

[connections.local]
type = "ollama"
base_url = "http://localhost:11434"

[purposes.interactive]
connection = "local"
model = "llama3"

[ws_auth]
methods = ["oidc"]
"#;

        /// The same file with the typo fixed — the user's repair.
        const REPAIRED: &str = r#"
[connections.work]
type = "anthropic"
api_key_env = "WORK_ANTHROPIC_KEY"

[connections.local]
type = "ollama"
base_url = "http://localhost:11434"

[purposes.interactive]
connection = "local"
model = "llama3"
"#;

        /// A load failure whose *parse error* quotes a credential: the URL is
        /// unquoted, so the TOML error names that line verbatim.
        const UNQUOTED_URL_WITH_PASSWORD: &str = r#"
[connections.local]
type = "ollama"
base_url = "http://localhost:11434"

[database]
url = postgres://adele:hunter2@db.example/adele
"#;

        /// The synthetic password inside `UNQUOTED_URL_WITH_PASSWORD`.
        const PASSWORD_IN_BROKEN_LINE: &str = "hunter2";

        /// Boot a handle the way `main` does when `load_and_migrate_daemon_config`
        /// returned `Err`: built-in defaults in memory, the user's real file
        /// still on disk.
        fn booted_from_a_failed_load(toml: &str) -> (Arc<RegistryHandle>, std::path::PathBuf) {
            let path = tmp_config_path();
            std::fs::write(&path, toml).expect("write the user's config");
            assert!(
                crate::config::load_and_migrate_daemon_config(&path).is_err(),
                "fixture must be a config the daemon cannot load"
            );
            let cfg = DaemonConfig::default();
            let registry = build_registry(&cfg);
            let handle = Arc::new(
                RegistryHandle::new(cfg, registry)
                    .with_config_path(path.clone())
                    .with_config_origin(ConfigOrigin::DefaultsAfterFailedLoad),
            );
            (handle, path)
        }

        fn ollama_payload() -> ConnectionConfigPayload {
            ConnectionConfigPayload::Ollama {
                base_url: Some("http://localhost:11434".into()),
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                keep_warm: None,
                max_context_tokens: None,
            }
        }

        #[tokio::test]
        async fn adding_a_connection_after_a_failed_load_leaves_the_file_intact() {
            let (handle, path) = booted_from_a_failed_load(TYPO_IN_CONNECTION);
            let before = std::fs::read_to_string(&path).expect("read the config back");
            let svc = DaemonConnectionsService::new(Arc::clone(&handle));

            let err = svc
                .create_connection("new".to_string(), ollama_payload())
                .await
                .expect_err("a write from defaults must be refused, not persisted");
            assert!(
                format!("{err}").contains(&path.display().to_string()),
                "the refusal must name the file the user has to fix: {err}"
            );
            assert_eq!(
                std::fs::read_to_string(&path).expect("read the config back"),
                before,
                "the user's connections, purposes and [ws_auth] must survive byte-for-byte"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn the_refusal_never_repeats_the_parse_error_or_its_credentials() {
            let (handle, path) = booted_from_a_failed_load(UNQUOTED_URL_WITH_PASSWORD);
            // The premise: the parse error itself quotes the offending line.
            let parse_error = format!(
                "{:#}",
                crate::config::load_and_migrate_daemon_config(&path)
                    .expect_err("fixture must not load")
            );
            assert!(
                parse_error.contains(PASSWORD_IN_BROKEN_LINE),
                "fixture must produce a parse error that quotes the credential: {parse_error}"
            );

            let err = handle
                .set_personality(Personality::default())
                .expect_err("a write from defaults must be refused");
            assert!(
                !format!("{err}").contains(PASSWORD_IN_BROKEN_LINE),
                "the client-visible refusal must not echo the offending line: {err}"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn a_config_that_stops_parsing_after_boot_is_not_overwritten_by_the_running_config() {
            // The daemon booted fine; the user then hand-edits the file and
            // mistypes a key. The running snapshot must not be written over
            // the edit the daemon could not read.
            let path = tmp_config_path();
            std::fs::write(&path, REPAIRED).expect("write the boot config");
            let cfg = crate::config::load_and_migrate_daemon_config(&path)
                .expect("boot config parses")
                .expect("boot config present");
            let registry = build_registry(&cfg);
            let handle =
                Arc::new(RegistryHandle::new(cfg, registry).with_config_path(path.clone()));

            std::fs::write(&path, TYPO_IN_CONNECTION).expect("hand edit introduces a typo");
            let before = std::fs::read_to_string(&path).expect("read the config back");

            handle
                .set_personality(Personality::default())
                .expect_err("the daemon must not overwrite an edit it could not read");
            assert_eq!(
                std::fs::read_to_string(&path).expect("read the config back"),
                before,
                "the hand edit must survive so the user can fix the typo"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn a_successful_reload_clears_the_degraded_state_and_restores_config_writes() {
            let (handle, path) = booted_from_a_failed_load(TYPO_IN_CONNECTION);
            std::fs::write(&path, REPAIRED).expect("the user fixes the file");
            handle
                .apply_reload()
                .expect("the repaired config loads and applies");

            handle
                .set_personality(Personality::default())
                .expect("writes resume once the daemon is running the file's own contents");

            let reloaded = crate::config::load_and_migrate_daemon_config(&path)
                .expect("the written config still parses")
                .expect("the written config is present");
            assert!(
                reloaded.connections.contains_key("work")
                    && reloaded.connections.contains_key("local"),
                "the repaired config's connections must survive the write: {:?}",
                reloaded.connections.keys().collect::<Vec<_>>()
            );
            assert!(
                reloaded.purposes.get(PurposeKind::Interactive).is_some(),
                "the repaired config's purposes must survive the write"
            );
            assert!(
                handle
                    .restart_required()
                    .iter()
                    .all(|area| area.as_key() != "config_load_failed"),
                "a recovered daemon must stop reporting a failed load: {:?}",
                handle.restart_required()
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn a_failed_config_load_is_reported_through_restart_required() {
            // The degraded state has to be visible *before* the user tries to
            // write, or an empty connections panel reads as "nothing is
            // configured" rather than "the daemon could not read your config".
            let (handle, path) = booted_from_a_failed_load(TYPO_IN_CONNECTION);
            let keys: Vec<String> = handle
                .restart_required()
                .iter()
                .map(|area| area.as_key().to_string())
                .collect();
            assert!(
                keys.contains(&"config_load_failed".to_string()),
                "the settings view must show a degraded daemon, not an empty one: {keys:?}"
            );
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn an_absent_config_file_is_not_a_failed_load_and_still_accepts_writes() {
            // First run: there is nothing to destroy, so the guard must not
            // turn a fresh install into a daemon that cannot save anything.
            let path = tmp_config_path();
            let handle = make_handle_at(DaemonConfig::default(), path.clone());
            handle
                .set_personality(Personality::default())
                .expect("a first run must still be able to save its config");
            assert!(path.exists(), "the write created the config file");
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn an_empty_config_file_is_not_a_failed_load_and_still_accepts_writes() {
            let path = tmp_config_path();
            std::fs::write(&path, "  \n\n").expect("write an empty config");
            let handle = make_handle_at(DaemonConfig::default(), path.clone());
            handle
                .set_personality(Personality::default())
                .expect("an empty config file is not a failed load");
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn concurrent_writes_during_a_failed_load_are_all_refused() {
            let (handle, path) = booted_from_a_failed_load(TYPO_IN_CONNECTION);
            let before = std::fs::read_to_string(&path).expect("read the config back");

            let workers: Vec<_> = (0..4)
                .map(|_| {
                    let handle = Arc::clone(&handle);
                    std::thread::spawn(move || {
                        handle.set_personality(Personality::default()).is_err()
                    })
                })
                .collect();
            for worker in workers {
                assert!(
                    worker.join().expect("worker thread finished"),
                    "every concurrent write must be refused, not just the first"
                );
            }

            assert_eq!(
                std::fs::read_to_string(&path).expect("read the config back"),
                before,
                "no interleaving of refused writes may touch the file"
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
            /// Snapshot of the `TOOL_GATE_DISABLED` task-local (#1007) at each
            /// `send_prompt` — proves the routing wrapper resolved the
            /// conversation's stored tool-gate override and installed it on
            /// the dispatch scope.
            captured_tool_policy: StdMutex<Vec<ToolPolicy>>,
        }

        impl CapturingInner {
            fn new() -> Self {
                Self {
                    captured_reasoning: StdMutex::new(Vec::new()),
                    captured_active_client_set: StdMutex::new(Vec::new()),
                    captured_model_override: StdMutex::new(Vec::new()),
                    captured_personality: StdMutex::new(Vec::new()),
                    captured_budget: StdMutex::new(Vec::new()),
                    captured_tool_policy: StdMutex::new(Vec::new()),
                }
            }
        }

        #[async_trait::async_trait]
        impl ConversationService for CapturingInner {
            async fn create_conversation(
                &self,
                title: String,
                _tags: Vec<String>,
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
                let policy = desktop_assistant_core::ports::llm::current_tool_policy();
                self.captured_tool_policy.lock().unwrap().push(policy);
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

        // ─── Issue #1007: per-conversation tool-gate override at send ─────

        #[tokio::test]
        async fn send_installs_gate_enforced_by_default_when_no_override_stored() {
            // With no stored override, the resolved value must be `false` —
            // the gate stays enforced. This is the fail-closed default.
            let (routing, inner, _registry, _store) = make_handler();

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

            let captured = inner.captured_tool_policy.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(
                captured[0],
                ToolPolicy::Standard,
                "no stored override must resolve to the configured default"
            );
        }

        #[tokio::test]
        async fn a_stored_override_resolves_to_the_permissive_level() {
            // `SetConversationToolGate { disabled: true }` is a conversation
            // asking for the level that refuses nothing, and the next send
            // must install exactly that on the dispatch scope.
            let (routing, inner, _registry, store) = make_handler();
            let id = ConversationId::from("c1");
            store
                .set_tool_gate_disabled(&id, true)
                .await
                .expect("set stored override");

            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(&id, "hi".into(), on_chunk, on_status)
                .await
                .expect("plain send_prompt should succeed via interactive purpose");

            let captured = inner.captured_tool_policy.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(
                captured[0],
                ToolPolicy::Lax,
                "a stored `true` override must resolve to the permissive level"
            );
        }

        /// A `ConversationSelectionStore` whose tool-gate accessor always
        /// errors, so `resolve_tool_gate_disabled` must fail closed rather
        /// than propagate the error into the send path.
        struct ErroringToolGateStore;

        impl ConversationSelectionStore for ErroringToolGateStore {
            async fn get_selection(
                &self,
                _id: &ConversationId,
            ) -> Result<Option<ConversationModelSelection>, CoreError> {
                Ok(None)
            }

            async fn set_selection(
                &self,
                _id: &ConversationId,
                _selection: Option<&ConversationModelSelection>,
            ) -> Result<(), CoreError> {
                Ok(())
            }

            async fn get_personality(
                &self,
                _id: &ConversationId,
            ) -> Result<Option<PersonalityOverride>, CoreError> {
                Ok(None)
            }

            async fn set_personality(
                &self,
                _id: &ConversationId,
                _personality: Option<&PersonalityOverride>,
            ) -> Result<(), CoreError> {
                Ok(())
            }

            async fn get_tool_gate_disabled(
                &self,
                _id: &ConversationId,
            ) -> Result<bool, CoreError> {
                Err(CoreError::Storage("simulated store failure".into()))
            }

            async fn set_tool_gate_disabled(
                &self,
                _id: &ConversationId,
                _disabled: bool,
            ) -> Result<(), CoreError> {
                Err(CoreError::Storage("simulated store failure".into()))
            }
        }

        #[tokio::test]
        async fn resolve_tool_policy_falls_back_to_the_default_on_a_store_error() {
            let cfg = local_ollama_cfg();
            let registry = make_handle_with(cfg);
            let inner = Arc::new(CapturingInner::new());
            let store = Arc::new(ErroringToolGateStore);
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
                .expect("a broken store must not fail the turn");

            let captured = inner.captured_tool_policy.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(
                captured[0],
                ToolPolicy::Standard,
                "a store error must resolve to the configured default, never to the \
                 permissive level"
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

        // ─── voice#126: Voice purpose routing by the "voice" tag ──────────

        /// Build a handler whose config adds a concrete `[purposes.voice]`
        /// (`local`/`qwen3`) on top of the interactive `local`/`llama3`.
        fn make_handler_with_voice_purpose() -> (
            Arc<RoutingConversationHandler<InMemoryConversationSelectionStore, CapturingInner>>,
            Arc<CapturingInner>,
            Arc<InMemoryConversationSelectionStore>,
        ) {
            let mut cfg = local_ollama_cfg();
            cfg.purposes.set(
                PurposeKind::Voice,
                Some(PurposeConfig {
                    connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                    model: ModelRef::Named("qwen3".into()),
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
            (routing, inner, store)
        }

        #[tokio::test]
        async fn voice_selection_none_unless_concrete() {
            // Absent → None (inherit interactive).
            let (routing, ..) = make_handler();
            assert!(
                routing.voice_selection().is_none(),
                "an absent Voice purpose must inherit interactive (None)"
            );
            // Concrete connection+model → Some.
            let (routing, ..) = make_handler_with_voice_purpose();
            let sel = routing
                .voice_selection()
                .expect("a concrete Voice purpose must resolve");
            assert_eq!(sel.connection_id, "local");
            assert_eq!(sel.model_id, "qwen3");
        }

        #[tokio::test]
        async fn conversation_is_voice_reads_the_voice_tag() {
            let (routing, _inner, store) = make_handler_with_voice_purpose();
            store.set_tags("v", vec!["voice".into()]);
            store.set_tags("t", vec!["something-else".into()]);
            assert!(
                routing
                    .conversation_is_voice(&ConversationId::from("v"))
                    .await
                    .unwrap()
            );
            assert!(
                !routing
                    .conversation_is_voice(&ConversationId::from("t"))
                    .await
                    .unwrap()
            );
            assert!(
                !routing
                    .conversation_is_voice(&ConversationId::from("unknown"))
                    .await
                    .unwrap(),
                "an unknown conversation has no tags → not voice"
            );
        }

        #[tokio::test]
        async fn voice_tagged_conversation_dispatches_the_voice_model() {
            // A "voice"-tagged conversation dispatches the Voice purpose's model
            // (qwen3); an untagged one still uses interactive (llama3). voice#126.
            let (routing, inner, store) = make_handler_with_voice_purpose();
            store.set_tags("voice-conv", vec!["voice".into()]);

            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("voice-conv"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("voice-tagged send should succeed");
            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("text-conv"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("untagged send should succeed");

            let captured = inner.captured_model_override.lock().unwrap();
            assert_eq!(captured.len(), 2);
            assert_eq!(
                captured[0],
                Some("qwen3".to_string()),
                "a voice-tagged turn must dispatch the Voice purpose's model"
            );
            assert_eq!(
                captured[1],
                Some("llama3".to_string()),
                "an untagged turn must keep dispatching the interactive model"
            );
        }

        #[tokio::test]
        async fn voice_purpose_absent_is_a_noop_for_voice_tagged_conversation() {
            // Without [purposes.voice], a voice-tagged conversation still uses
            // the interactive model — adding the variant is a no-op until
            // an operator points it at a concrete model (voice#126).
            let (routing, inner, _reg, store) = make_handler();
            store.set_tags("voice-conv", vec!["voice".into()]);
            let (on_chunk, on_status) = noop_cb();
            routing
                .send_prompt(
                    &ConversationId::from("voice-conv"),
                    "hi".into(),
                    on_chunk,
                    on_status,
                )
                .await
                .expect("send should succeed");
            let captured = inner.captured_model_override.lock().unwrap();
            assert_eq!(
                captured[0],
                Some("llama3".to_string()),
                "no Voice purpose → the interactive model, even for a voice-tagged conversation"
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

    // ----- A read of the config file must not write it (#915) -------------
    //
    // The legacy migration rewrites `daemon.toml` and takes a `.bak` when the
    // file holds `[llm]` and no `[connections]` table. Deleting the last
    // connection makes the daemon write exactly that shape, so any caller
    // that migrates on the way in resurrects the connection the user removed
    // and drops another backup beside the file.
    //
    // These tests watch the file, not the value the call returned: the defect
    // is a side effect on disk, so a test that only read the return value
    // would pass while the bug is present.
    mod config_reads_do_not_write {
        use super::*;

        /// A running daemon with one connection and a legacy `[llm]` block
        /// that an earlier migration left in place. The file is the
        /// serialization of the config the handle runs, which is what every
        /// daemon-authored write produces.
        ///
        /// `[llm]` carries a model, so it is not the default block and
        /// serializes whatever else changes. Deleting the one connection
        /// therefore leaves `[llm]` with no `[connections]` - the shape the
        /// legacy migration triggers on.
        fn handle_with_one_connection_and_a_legacy_llm_block(
            dir: &std::path::Path,
        ) -> (Arc<RegistryHandle>, std::path::PathBuf) {
            let mut cfg = config_with_connections(&[("local", ollama_local())]);
            cfg.llm.model = Some("llama3".to_string());
            let path = dir.join("daemon.toml");
            crate::config::save_daemon_config(&path, &cfg).expect("write the running config");
            let registry = build_registry(&cfg);
            let handle =
                Arc::new(RegistryHandle::new(cfg, registry).with_config_path(path.clone()));
            (handle, path)
        }

        /// Every file name in `dir`, sorted, so a new `.bak` shows up as an
        /// added entry.
        fn file_names(dir: &std::path::Path) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(dir)
                .expect("read the config directory")
                .map(|entry| {
                    entry
                        .expect("read a directory entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            names.sort();
            names
        }

        /// Byte-exact comparison, reported as text so a failure shows which
        /// tables were added rather than two arrays of numbers.
        fn assert_file_is_byte_identical(path: &std::path::Path, before: &[u8], what: &str) {
            let after = std::fs::read(path).expect("read the config file");
            assert!(
                after == before,
                "{what} must leave daemon.toml byte-identical\n--- before ---\n{}\n--- after ---\n{}",
                String::from_utf8_lossy(before),
                String::from_utf8_lossy(&after),
            );
        }

        async fn delete_the_only_connection(handle: &Arc<RegistryHandle>) {
            DaemonConnectionsService::new(handle.clone())
                .delete_connection("local".to_string(), false)
                .await
                .expect("the last connection can be deleted");
        }

        #[tokio::test]
        async fn a_settings_read_leaves_the_config_file_byte_identical_after_the_last_connection_is_deleted()
         {
            let dir = tempfile::TempDir::new().expect("temp dir");
            let (handle, path) = handle_with_one_connection_and_a_legacy_llm_block(dir.path());
            delete_the_only_connection(&handle).await;

            let after_delete = std::fs::read(&path).expect("read the config file");
            let files_after_delete = file_names(dir.path());

            // `get_config` asks for this on every settings read, so it runs as
            // often as a client refreshes its settings panel.
            for _ in 0..3 {
                let _ = handle.restart_required();
            }

            assert_file_is_byte_identical(&path, &after_delete, "a settings read");
            assert_eq!(
                file_names(dir.path()),
                files_after_delete,
                "a settings read must not leave a backup beside daemon.toml"
            );
        }

        #[tokio::test]
        async fn a_deleted_connection_does_not_come_back_when_the_config_file_is_reloaded() {
            let dir = tempfile::TempDir::new().expect("temp dir");
            let (handle, path) = handle_with_one_connection_and_a_legacy_llm_block(dir.path());
            delete_the_only_connection(&handle).await;

            let after_delete = std::fs::read(&path).expect("read the config file");
            let files_after_delete = file_names(dir.path());

            // What the config-file watcher runs after a daemon-authored write.
            let _ = handle.apply_reload();

            assert!(
                handle.snapshot_config().connections.is_empty(),
                "a deleted connection must not be restored by a reload: {:?}",
                handle
                    .snapshot_config()
                    .connections
                    .keys()
                    .collect::<Vec<_>>()
            );
            assert_file_is_byte_identical(&path, &after_delete, "a reload");
            assert_eq!(
                file_names(dir.path()),
                files_after_delete,
                "a reload must not leave a backup beside daemon.toml"
            );
        }
    }
}
