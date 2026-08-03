//! Per-turn dispatch client used by [`crate::api_surface::RoutingConversationHandler`]
//! to swap the underlying [`AnyLlmClient`] based on the resolved
//! `(connection_id, model_id, effort)` triple for each `send_prompt`.
//!
//! Rationale: the core `ConversationHandler` owns a single `llm: L` field
//! baked into its type parameters. Rebuilding the handler per turn is
//! impractical (shared `namespace_cache`, non-`Clone` `id_generator`), and
//! plumbing a per-call client argument through the ~450-line `send_prompt`
//! would be a very invasive change.
//!
//! Instead we install this wrapper as the handler's `L`. It looks up the
//! target `AnyLlmClient` on each call via a [`tokio::task_local!`] slot
//! populated by the daemon-side routing wrapper. When the slot is unset
//! (e.g. background jobs), dispatch falls through to a
//! statically-configured fallback — the interactive-purpose client at
//! daemon startup.
//!
//! The same wrapper serves the daemon's backend-task slot, in a mode that
//! deliberately does the opposite: it ignores the per-turn slot and pins
//! its own model. Titling and context summary run inside the caller's
//! `send_prompt` scope, so "the slot is unset" is not true for them, and
//! reading it would bill every title to the interactive connection and
//! model. [`FallbackMode`] states which mode does which.
//!
//! Concurrency: `tokio::task_local!` is per-task, so two concurrent
//! `send_prompt` calls on different conversations each see their own
//! routing target without coupling.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, HostedToolSearch, LlmClient, LlmResponse, ModelInfo, ModelListingReport,
    ReasoningConfig, dispatch_namespaced, with_model_override,
};

use crate::api_surface::RegistryHandle;
use crate::purposes::PurposeKind;

tokio::task_local! {
    /// Per-turn routing override. When set, dispatch uses the contained
    /// `Arc<dyn LlmClient>` (resolved from the registry) instead of the
    /// [`RoutingLlmClient`]'s static fallback. Populated by
    /// [`with_active_client`] from inside the routing wrapper.
    static ACTIVE_CLIENT: Arc<dyn LlmClient>;
}

/// Run `fut` with `client` installed as the current turn's active LLM
/// client. All `stream_completion(_with_namespaces)` calls on the
/// enclosing [`RoutingLlmClient`] observe `client` and dispatch to it.
pub async fn with_active_client<F, T>(client: Arc<dyn LlmClient>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    ACTIVE_CLIENT.scope(client, fut).await
}

/// Whether an [`ACTIVE_CLIENT`] task-local is set for the current
/// scope. Used by tests in the api_surface dispatch module to assert
/// that interactive-purpose fallbacks correctly *do not* install an
/// override — dispatch should fall through to the primary llm in that
/// case so the interactive purpose's model takes effect.
#[cfg(test)]
pub(crate) fn active_client_is_set() -> bool {
    ACTIVE_CLIENT.try_with(|_| ()).is_ok()
}

/// Fallback resolution mode for [`RoutingLlmClient`]. Controls what the
/// wrapper dispatches to when no per-turn [`ACTIVE_CLIENT`] task-local is
/// installed.
#[derive(Clone)]
pub enum FallbackMode {
    /// Follows the turn. Dispatch reads `ACTIVE_CLIENT` first, then falls
    /// back to this client, which is captured at construction, for callers
    /// that installed no per-turn override. Used by the primary
    /// (interactive) slot, and the only mode that reads the task-local.
    PerTurn { client: Arc<dyn LlmClient> },
    /// Resolve the target client from a [`RegistryHandle`] on every
    /// dispatch by re-reading the named purpose's config. Used by the
    /// backend-tasks slot so titling/dreaming pick up control-panel
    /// edits without a daemon restart. Always ignores `ACTIVE_CLIENT`
    /// — backend tasks must not inherit the user's per-turn model
    /// override even when invoked inside a `send_prompt` scope.
    DynamicPurpose {
        registry: Arc<RegistryHandle>,
        purpose: PurposeKind,
    },
    /// Client and model both captured at construction. Used by the legacy
    /// `[backend_tasks.llm]` slot, which names one connector and one model
    /// in config and has no purpose to re-resolve.
    ///
    /// Holds the same invariant as [`Self::DynamicPurpose`], and holds it
    /// the same way: never read `ACTIVE_CLIENT`, and shadow the turn's
    /// `MODEL_OVERRIDE` with `model` for the whole dispatch. A backend task
    /// runs inside the caller's `send_prompt` scope by design (titling and
    /// context summary both do), so both task-locals are in force when it
    /// starts, and both would otherwise send the request to the user's
    /// interactive client and interactive model.
    ///
    /// Both halves are load-bearing. Reading `ACTIVE_CLIENT` bills the
    /// title to the interactive connection. Leaving `MODEL_OVERRIDE` in
    /// place asks the backend connection for a model it does not serve.
    Pinned {
        client: Arc<dyn LlmClient>,
        model: String,
    },
}

/// The handler's LLM facade. Delegates to the per-turn active client when
/// one is installed (PerTurn mode only), or to the configured fallback
/// otherwise.
///
/// Only [`FallbackMode::PerTurn`] follows the turn. The two backend-task
/// modes, [`FallbackMode::DynamicPurpose`] and
/// [`FallbackMode::Pinned`], never read `ACTIVE_CLIENT` and always
/// install their own model override, so a title or a summary goes to the
/// configured backend model whether or not it started inside a
/// `send_prompt` scope.
///
/// Two groups of accessors resolve differently, and the split is
/// deliberate. Anything that changes what a request contains or where it
/// goes — `stream_completion`, the hosted-search dispatch,
/// `hosted_tool_search`, `max_context_tokens`, the model-listing
/// calls — resolves through the per-turn active client, so the answer
/// always describes the client the turn dispatches to. The borrowing
/// accessors `get_default_model` and `get_default_base_url` return
/// `Option<&str>` tied to `self`, which no task-local lookup can satisfy,
/// so they report the statically configured value. The DynamicPurpose mode
/// has no single captured client and answers `None`/`false`/empty for all
/// of them. The Pinned mode answers from its captured client, and
/// reports its pinned model as the default model.
///
/// One gap in the Pinned mode, and it is deliberate: the sync
/// accessors cannot install a model override, because `with_model_override`
/// is async. So a connector whose `max_context_tokens` consults
/// `current_model_override` answers for the turn's model there. Nothing
/// reads `max_context_tokens` on the backend slot, because the dispatch loop
/// takes its budget from the `CONTEXT_BUDGET` task-local, and DynamicPurpose
/// already reports `None` for the same call, so the gap is unobservable
/// today.
///
/// One method sits in neither group: `estimate_tokens` is not overridden
/// here at all, so the trait's own `chars/4` estimate answers whatever
/// client is resolved. That is harmless while no connector overrides it
/// with a better tokeniser, and wrong the moment one does.
#[derive(Clone)]
pub struct RoutingLlmClient {
    fallback: FallbackMode,
}

impl RoutingLlmClient {
    /// Per-turn constructor. Used by the primary (interactive) slot. The
    /// argument serves any call made outside a turn scope.
    pub fn new(fallback: Arc<dyn LlmClient>) -> Self {
        Self {
            fallback: FallbackMode::PerTurn { client: fallback },
        }
    }

    /// Dynamic-purpose constructor. Each `stream_completion` call resolves
    /// the named purpose against the live `RegistryHandle.snapshot_config`
    /// and dispatches to the registry's client for the resolved
    /// connection, with the resolved model override and effort-mapped
    /// reasoning installed for the duration of the call.
    pub fn new_dynamic_purpose(registry: Arc<RegistryHandle>, purpose: PurposeKind) -> Self {
        Self {
            fallback: FallbackMode::DynamicPurpose { registry, purpose },
        }
    }

    /// Pinned-client constructor. Used by the legacy `[backend_tasks.llm]`
    /// slot. `client` serves every dispatch and `model` is installed as the
    /// model override for its duration, so neither half of the caller's turn
    /// reaches the backend task. See [`FallbackMode::Pinned`].
    pub fn new_pinned(client: Arc<dyn LlmClient>, model: String) -> Self {
        Self {
            fallback: FallbackMode::Pinned { client, model },
        }
    }

    /// The client captured at construction, for accessor delegation
    /// (`list_models`, `max_context_tokens`, etc.). Returns `None` for
    /// dynamic-purpose wrappers, which intentionally have no single
    /// captured client to delegate to.
    fn captured_client(&self) -> Option<&Arc<dyn LlmClient>> {
        match &self.fallback {
            FallbackMode::PerTurn { client, .. } => Some(client),
            FallbackMode::Pinned { client, .. } => Some(client),
            FallbackMode::DynamicPurpose { .. } => None,
        }
    }

    /// Resolve the current turn's active client for PerTurn mode. Returns
    /// the task-local override if set, or the captured client otherwise.
    /// Only meaningful for PerTurn mode. DynamicPurpose dispatches via
    /// [`Self::dispatch_dynamic`] and Pinned via [`Self::dispatch_pinned`].
    fn resolve_per_turn(&self) -> Arc<dyn LlmClient> {
        let FallbackMode::PerTurn { client, .. } = &self.fallback else {
            unreachable!("resolve_per_turn called on a backend-task mode");
        };
        ACTIVE_CLIENT
            .try_with(Arc::clone)
            .unwrap_or_else(|_| Arc::clone(client))
    }
}

impl RoutingLlmClient {
    /// Dispatch path for [`FallbackMode::DynamicPurpose`]. Resolves the
    /// purpose against the live config snapshot, installs the resolved
    /// model override for the connector, and runs `op` against the
    /// registry's client. Returns a `CoreError::Llm` describing the
    /// failure mode if resolution can't proceed (purpose unconfigured,
    /// connections invalid, connection missing from the registry).
    async fn dispatch_dynamic<F, Fut, T>(&self, op: F) -> Result<T, CoreError>
    where
        F: FnOnce(Arc<dyn LlmClient>, ReasoningConfig) -> Fut,
        Fut: std::future::Future<Output = Result<T, CoreError>>,
    {
        let FallbackMode::DynamicPurpose { registry, purpose } = &self.fallback else {
            unreachable!("dispatch_dynamic called on a non-purpose mode");
        };
        let config = registry.snapshot_config();
        // Resolve the purpose to a concrete `ResolvedPurpose` carrying
        // the connection id (not the connector type) so the registry
        // lookup hits the right entry. `resolve_purpose_dispatch` /
        // `resolve_purpose_llm_config` flatten the connection id away
        // into a `ResolvedLlmConfig.connector` field that holds the
        // connector type string — useless for registry indexing. We
        // call the lower-level resolver here and re-derive reasoning
        // ourselves.
        let connections = config.validated_connections().map_err(|e| {
            CoreError::Llm(format!(
                "purpose {:?}: [connections] failed validation: {e}",
                purpose.as_key()
            ))
        })?;
        let resolved = crate::purposes::resolve_purpose(*purpose, &config.purposes, &connections)
            .map_err(|e| {
            CoreError::Llm(format!(
                "purpose {:?} resolution failed: {e}",
                purpose.as_key()
            ))
        })?;
        let connection = connections.get(&resolved.connection_id).ok_or_else(|| {
            CoreError::Llm(format!(
                "purpose {:?} resolved connection {:?} is missing from \
                 [connections] (post-resolution invariant violated)",
                purpose.as_key(),
                resolved.connection_id
            ))
        })?;
        let connector_type = connection.connector_type().to_string();
        let reasoning = crate::api_surface::map_effort_to_reasoning_config(
            &connector_type,
            &resolved.model_id,
            resolved.effort,
        );
        let client = registry
            .client_for(&resolved.connection_id)
            .ok_or_else(|| {
                CoreError::Llm(format!(
                    "purpose {:?} references connection {:?} which is not present in the registry",
                    purpose.as_key(),
                    resolved.connection_id
                ))
            })?;
        with_model_override(resolved.model_id, op(client, reasoning)).await
    }

    /// Dispatch path for [`FallbackMode::Pinned`]. Runs `op` against
    /// the captured client with the captured model installed as the model
    /// override, so the connector reads the configured backend model and
    /// not the turn's. `ACTIVE_CLIENT` is never consulted.
    async fn dispatch_pinned<F, Fut, T>(&self, op: F) -> Result<T, CoreError>
    where
        F: FnOnce(Arc<dyn LlmClient>) -> Fut,
        Fut: std::future::Future<Output = Result<T, CoreError>>,
    {
        let FallbackMode::Pinned { client, model } = &self.fallback else {
            unreachable!("dispatch_pinned called on a non-pinned mode");
        };
        with_model_override(model.clone(), op(Arc::clone(client))).await
    }
}

#[async_trait::async_trait]
impl LlmClient for RoutingLlmClient {
    fn get_default_model(&self) -> Option<&str> {
        // `Option<&str>` borrows from `self`; we can't delegate through the
        // task-local (which returns an Arc) or a dynamic registry lookup.
        // PerTurn mode delegates to the captured client; dynamic-purpose
        // mode has no single captured client so reports `None`. This
        // accessor reports the statically configured default and is not
        // meaningfully per-turn or per-purpose.
        //
        // Pinned mode answers with the model it pins, which is the model
        // every one of its dispatches carries.
        match &self.fallback {
            FallbackMode::Pinned { model, .. } => Some(model.as_str()),
            _ => self.captured_client().and_then(|c| c.get_default_model()),
        }
    }

    fn get_default_base_url(&self) -> Option<&str> {
        self.captured_client()
            .and_then(|c| c.get_default_base_url())
    }

    fn max_context_tokens(&self) -> Option<u64> {
        // The dispatch loop reads token-pressure budgets from the
        // `CONTEXT_BUDGET` task-local installed by the daemon's wrapper,
        // not from this trait method, so the resolution chain no longer
        // lives here. PerTurn mode delegates to the resolved client;
        // dynamic-purpose mode has no single client to ask without a
        // config snapshot, and callers (capability probes, debug paths)
        // tolerate `None`. Pinned mode asks its captured client, never the
        // turn's.
        match &self.fallback {
            FallbackMode::PerTurn { .. } => self.resolve_per_turn().max_context_tokens(),
            FallbackMode::Pinned { client, .. } => client.max_context_tokens(),
            FallbackMode::DynamicPurpose { .. } => None,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // Callers of `list_models` are typically the connections-management
        // API, which resolves clients directly from the registry — not
        // through the routing wrapper. Keep this consistent with
        // connector-level behaviour and delegate to whichever client is
        // currently active (task-local or captured client). The
        // dynamic-purpose wrapper isn't used by listing paths, so report
        // an empty list there. Pinned mode asks its captured client, never
        // the turn's.
        match &self.fallback {
            FallbackMode::PerTurn { .. } => self.resolve_per_turn().list_models().await,
            FallbackMode::Pinned { client, .. } => client.list_models().await,
            FallbackMode::DynamicPurpose { .. } => Ok(Vec::new()),
        }
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        match &self.fallback {
            FallbackMode::PerTurn { .. } => self.resolve_per_turn().refresh_models().await,
            FallbackMode::Pinned { client, .. } => client.refresh_models().await,
            FallbackMode::DynamicPurpose { .. } => Ok(Vec::new()),
        }
    }

    async fn list_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        match &self.fallback {
            FallbackMode::PerTurn { .. } => self.resolve_per_turn().list_models_detailed().await,
            FallbackMode::Pinned { client, .. } => client.list_models_detailed().await,
            FallbackMode::DynamicPurpose { .. } => Ok(ModelListingReport::default()),
        }
    }

    async fn refresh_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        match &self.fallback {
            FallbackMode::PerTurn { .. } => self.resolve_per_turn().refresh_models_detailed().await,
            FallbackMode::Pinned { client, .. } => client.refresh_models_detailed().await,
            FallbackMode::DynamicPurpose { .. } => Ok(ModelListingReport::default()),
        }
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match &self.fallback {
            FallbackMode::PerTurn { .. } => {
                let client = self.resolve_per_turn();
                client
                    .stream_completion(messages, tools, reasoning, on_chunk)
                    .await
            }
            FallbackMode::Pinned { .. } => {
                // The legacy `[backend_tasks.llm]` block carries no effort,
                // so the caller's reasoning stands. Only the client and the
                // model are pinned.
                self.dispatch_pinned(|client| async move {
                    client
                        .stream_completion(messages, tools, reasoning, on_chunk)
                        .await
                })
                .await
            }
            FallbackMode::DynamicPurpose { .. } => {
                // Backend tasks pass `ReasoningConfig::default()`; the
                // resolved purpose's reasoning takes precedence so we
                // discard the caller's config in dynamic mode.
                let _ = reasoning;
                self.dispatch_dynamic(|client, resolved_reasoning| async move {
                    client
                        .stream_completion(messages, tools, resolved_reasoning, on_chunk)
                        .await
                })
                .await
            }
        }
    }

    /// Hands back `self`, never the resolved client's object, so the router
    /// stays in the call path for a namespaced turn. Handing back the
    /// resolved client's object would also freeze the resolution at the
    /// moment of the capability read, while dispatch must resolve again at
    /// the moment it sends. See [`LlmClient::hosted_tool_search`].
    fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
        let resolved_has = match &self.fallback {
            FallbackMode::PerTurn { .. } => self.resolve_per_turn().hosted_tool_search().is_some(),
            FallbackMode::Pinned { client, .. } => client.hosted_tool_search().is_some(),
            FallbackMode::DynamicPurpose { .. } => false,
        };
        resolved_has.then_some(self as &dyn HostedToolSearch)
    }
}

#[async_trait::async_trait]
impl HostedToolSearch for RoutingLlmClient {
    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match &self.fallback {
            FallbackMode::PerTurn { .. } => {
                let client = self.resolve_per_turn();
                dispatch_namespaced(
                    &client, messages, core_tools, namespaces, reasoning, on_chunk,
                )
                .await
            }
            FallbackMode::Pinned { .. } => {
                self.dispatch_pinned(|client| async move {
                    dispatch_namespaced(
                        &client, messages, core_tools, namespaces, reasoning, on_chunk,
                    )
                    .await
                })
                .await
            }
            // Unreachable through `dispatch_namespaced`, because
            // `hosted_tool_search` answers `None` in this mode: a backend
            // task must not inherit the user's per-turn model override. The
            // arm still dispatches correctly, so a direct call cannot send a
            // backend task somewhere else.
            FallbackMode::DynamicPurpose { .. } => {
                let _ = reasoning;
                self.dispatch_dynamic(|client, resolved_reasoning| async move {
                    dispatch_namespaced(
                        &client,
                        messages,
                        core_tools,
                        namespaces,
                        resolved_reasoning,
                        on_chunk,
                    )
                    .await
                })
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connections::{ConnectionConfig, ConnectionId, OllamaConnection};
    use crate::registry::build_registry;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::domain::Message;
    use desktop_assistant_core::ports::llm::ReasoningConfig;
    use indexmap::IndexMap;

    /// Ollama base URL for tests: a closed loopback port. Connecting here
    /// yields an instant connection-refused, so the connector returns `Err`
    /// fast regardless of whether a real Ollama is running on `:11434`. Using
    /// the real port made these tests hang on dev machines (#186).
    const DEAD_OLLAMA: &str = "http://127.0.0.1:1";

    /// Hard cap on a test-side network call: panic (fail the test) if it does
    /// not finish in time, so a hung endpoint can never stall the whole suite
    /// (#186). `DEAD_OLLAMA` makes the call fail fast in practice; this is the
    /// backstop in case some future endpoint blocks instead of refusing.
    async fn no_hang<F: std::future::Future>(label: &str, fut: F) -> F::Output {
        match tokio::time::timeout(std::time::Duration::from_secs(15), fut).await {
            Ok(v) => v,
            Err(_) => panic!("{label} did not complete within 15s — network call hung?"),
        }
    }

    fn build_ollama_registry() -> Arc<dyn LlmClient> {
        let cfg = crate::config::DaemonConfig {
            connections: IndexMap::from([(
                "local".to_string(),
                ConnectionConfig::Ollama(OllamaConnection {
                    base_url: Some(DEAD_OLLAMA.into()),
                    ..Default::default()
                }),
            )]),
            ..crate::config::DaemonConfig::default()
        };
        let registry = build_registry(&cfg);
        let id = ConnectionId::new("local").unwrap();
        registry.get(&id).unwrap()
    }

    #[tokio::test]
    async fn falls_back_to_static_when_no_task_local() {
        let fallback = build_ollama_registry();
        let client = RoutingLlmClient::new(Arc::clone(&fallback));
        // Without a task-local override, `resolve()` must equal the
        // fallback pointer.
        let resolved = client.resolve_per_turn();
        assert!(
            Arc::ptr_eq(&resolved, &fallback),
            "resolve() should return fallback when task-local is unset"
        );
    }

    #[tokio::test]
    async fn uses_task_local_override_when_set() {
        let fallback = build_ollama_registry();
        // Build a second distinct Ollama client so we can Arc-ptr compare.
        let override_client = build_ollama_registry();
        assert!(
            !Arc::ptr_eq(&fallback, &override_client),
            "test setup: fallback and override must be distinct"
        );

        let client = RoutingLlmClient::new(Arc::clone(&fallback));

        let override_clone = Arc::clone(&override_client);
        let resolved =
            with_active_client(override_client, async move { client.resolve_per_turn() }).await;
        assert!(
            Arc::ptr_eq(&resolved, &override_clone),
            "resolve() must return the task-local override when set"
        );
    }

    /// A mock `AnyLlmClient` variant is overkill for this test; we simply
    /// verify the dispatch does not panic and returns the fallback's
    /// error (there's no real server), which proves the delegation
    /// compiles and reaches the inner client.
    #[tokio::test]
    async fn stream_completion_delegates_to_resolved_client() {
        let fallback = build_ollama_registry();
        let client = RoutingLlmClient::new(fallback);
        let _ = no_hang(
            "stream_completion_delegates_to_resolved_client",
            client.stream_completion(
                vec![Message::new(
                    desktop_assistant_core::domain::Role::User,
                    "hi",
                )],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            ),
        )
        .await;
        // Result will be an `Err` (connection refused at DEAD_OLLAMA), but the
        // call path itself must complete without panicking.
    }

    fn _assert_llm_client_impl<L: LlmClient>() {}
    fn _assert_routing_client_implements_llm_client() {
        _assert_llm_client_impl::<RoutingLlmClient>();
    }

    fn build_local_ollama_handle() -> Arc<crate::api_surface::RegistryHandle> {
        let cfg = crate::config::DaemonConfig {
            connections: IndexMap::from([(
                "local".to_string(),
                ConnectionConfig::Ollama(OllamaConnection {
                    base_url: Some(DEAD_OLLAMA.into()),
                    ..Default::default()
                }),
            )]),
            ..crate::config::DaemonConfig::default()
        };
        let reg = build_registry(&cfg);
        Arc::new(crate::api_surface::RegistryHandle::new(cfg, reg))
    }

    #[test]
    fn missing_connection_id_returns_none() {
        let registry_handle = build_local_ollama_handle();
        let missing = ConnectionId::new("nonexistent").unwrap();
        assert!(registry_handle.client_for(&missing).is_none());
    }

    #[test]
    fn existing_connection_id_resolves() {
        let registry_handle = build_local_ollama_handle();
        let id = ConnectionId::new("local").unwrap();
        assert!(registry_handle.client_for(&id).is_some());
    }

    #[test]
    fn unused_core_error_type_still_compiles() {
        // Make sure the CoreError import isn't elided by mistake.
        let _e: Option<CoreError> = None;
    }

    #[tokio::test]
    async fn max_context_delegates_to_resolved_client() {
        // `RoutingLlmClient::max_context_tokens` is plain delegation to
        // the resolved client — no overlay, no tier fallback (the
        // three-tier budget resolution lives in
        // `config::resolve_context_budget`). Post-#342 an un-warmed Ollama
        // reports its configured-default effective `num_ctx` (never `None`),
        // so the wrapper must surface that same value.
        let fallback = build_ollama_registry();
        let client = RoutingLlmClient::new(fallback);
        assert_eq!(
            client.max_context_tokens(),
            Some(desktop_assistant_llm_ollama::DEFAULT_OLLAMA_NUM_CTX)
        );
    }

    // --- DynamicPurpose mode -------------------------------------------------

    /// Build a `RegistryHandle` with `[purposes.titling]` pointed at the
    /// "local" Ollama connection — exercises the full purpose-resolution
    /// path used by the backend slot.
    fn build_handle_with_titling(model: &str) -> Arc<crate::api_surface::RegistryHandle> {
        use crate::purposes::{ConnectionRef, ModelRef, PurposeConfig, Purposes};
        let mut purposes = Purposes::default();
        purposes.set(
            PurposeKind::Interactive,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("interactive-model".to_string()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        purposes.set(
            PurposeKind::Titling,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named(model.to_string()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        let cfg = crate::config::DaemonConfig {
            connections: IndexMap::from([(
                "local".to_string(),
                ConnectionConfig::Ollama(OllamaConnection {
                    base_url: Some(DEAD_OLLAMA.into()),
                    ..Default::default()
                }),
            )]),
            purposes,
            ..crate::config::DaemonConfig::default()
        };
        let reg = build_registry(&cfg);
        Arc::new(crate::api_surface::RegistryHandle::new(cfg, reg))
    }

    #[tokio::test]
    async fn dynamic_purpose_unconfigured_returns_error() {
        // Connections present but `[purposes.titling]` is missing: the
        // resolver itself surfaces a clean error rather than panicking.
        // (An empty config also errors but earlier — at connections
        // validation — which is a separate path covered by config-level
        // tests.)
        use crate::purposes::{ConnectionRef, ModelRef, PurposeConfig, Purposes};
        let mut purposes = Purposes::default();
        purposes.set(
            PurposeKind::Interactive,
            Some(PurposeConfig {
                connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                model: ModelRef::Named("interactive-model".to_string()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        // Note: titling intentionally absent.
        let cfg = crate::config::DaemonConfig {
            connections: IndexMap::from([(
                "local".to_string(),
                ConnectionConfig::Ollama(OllamaConnection {
                    base_url: Some(DEAD_OLLAMA.into()),
                    ..Default::default()
                }),
            )]),
            purposes,
            ..crate::config::DaemonConfig::default()
        };
        let reg = build_registry(&cfg);
        let handle = Arc::new(crate::api_surface::RegistryHandle::new(cfg, reg));
        let client = RoutingLlmClient::new_dynamic_purpose(handle, PurposeKind::Titling);
        let err = no_hang(
            "dynamic_purpose_unconfigured_returns_error",
            client.stream_completion(
                vec![Message::new(
                    desktop_assistant_core::domain::Role::User,
                    "hi",
                )],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            ),
        )
        .await
        .expect_err("dispatch should fail when purpose is unconfigured");
        assert!(
            matches!(err, CoreError::Llm(ref msg) if msg.contains("titling")
                && msg.contains("resolution failed")),
            "expected titling-resolution-failed error, got: {err}"
        );
    }

    #[tokio::test]
    async fn dynamic_purpose_looks_up_registry_by_connection_id() {
        // Regression: an earlier draft of `dispatch_dynamic` looked up
        // the registry's client using `ResolvedLlmConfig.connector`,
        // which carries the connector *type* string (e.g. "ollama"),
        // not the connection id ("local"). The lookup missed and every
        // backend dispatch errored with "connection not present in
        // registry" — title generation fell back to the placeholder.
        //
        // This test fixes the connection slug and connector type to
        // distinct strings ("local" vs "ollama") so a regression that
        // confuses the two would fail here.
        let handle = build_handle_with_titling("titling-model");
        let client =
            RoutingLlmClient::new_dynamic_purpose(Arc::clone(&handle), PurposeKind::Titling);
        let result = no_hang(
            "dynamic_purpose_looks_up_registry_by_connection_id",
            client.stream_completion(
                vec![Message::new(
                    desktop_assistant_core::domain::Role::User,
                    "hi",
                )],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            ),
        )
        .await;
        // The Ollama connection in the test registry isn't backed by a
        // real server, so dispatch reaches the connector and errors out
        // there — that's fine. What we *must not* see is a registry-
        // lookup error mentioning the connector type as a missing
        // connection.
        if let Err(CoreError::Llm(msg)) = &result {
            assert!(
                !msg.contains("\"ollama\""),
                "registry lookup must use connection id 'local', \
                 not connector type 'ollama' — got error: {msg}"
            );
            assert!(
                !msg.contains("not present in the registry"),
                "registry lookup with the correct id should succeed; \
                 got: {msg}"
            );
        }
    }

    #[tokio::test]
    async fn dynamic_purpose_resolves_against_live_config() {
        // The point of #68: a single dynamic-purpose client must read
        // the registry's current config on every call, not a snapshot
        // captured at construction. Build a handle, swap the titling
        // model in-place, and verify resolution observes the new value.
        use crate::api_surface::resolve_purpose_dispatch;

        let handle = build_handle_with_titling("model-v1");
        let _client =
            RoutingLlmClient::new_dynamic_purpose(Arc::clone(&handle), PurposeKind::Titling);

        let cfg = handle.snapshot_config();
        let (resolved, _) = resolve_purpose_dispatch(Some(&cfg), PurposeKind::Titling)
            .expect("titling resolves before mutation");
        assert_eq!(resolved.model, "model-v1");

        // Swap the in-memory config — same path `mutate_config` takes
        // after the control panel writes a new value, minus the disk
        // persistence (covered by the connections-management API tests).
        let mut new_cfg = handle.snapshot_config();
        new_cfg.purposes.set(
            PurposeKind::Titling,
            Some(crate::purposes::PurposeConfig {
                connection: crate::purposes::ConnectionRef::Named(
                    ConnectionId::new("local").unwrap(),
                ),
                model: crate::purposes::ModelRef::Named("model-v2".to_string()),
                effort: None,
                max_context_tokens: None,
            }),
        );
        handle.replace_config_for_test(new_cfg);

        let cfg2 = handle.snapshot_config();
        let (resolved2, _) = resolve_purpose_dispatch(Some(&cfg2), PurposeKind::Titling)
            .expect("titling resolves after mutation");
        assert_eq!(resolved2.model, "model-v2");
    }

    // --- Hosted-tool-search capability --------------------------------------

    /// `LlmClient` double whose hosted-tool-search answer is fixed at
    /// construction. It records the tool names each dispatch offered, so a
    /// test can assert what the request actually carried.
    ///
    /// Its hosted dispatch flattens on purpose. A connector without hosted
    /// search (Bedrock, Ollama) sends every namespace tool in one ordinary
    /// request, and that is what makes a wrong capability answer expensive,
    /// so the double reproduces it on both paths and lets `offered()` compare
    /// them.
    struct CapabilityLlm {
        hosted: bool,
        offered: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl CapabilityLlm {
        fn new(hosted: bool) -> Self {
            Self {
                hosted,
                offered: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// One entry per dispatch: the tool names that request carried.
        fn offered(&self) -> Vec<Vec<String>> {
            self.offered.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for CapabilityLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.offered
                .lock()
                .unwrap()
                .push(tools.iter().map(|t| t.name.clone()).collect());
            Ok(LlmResponse::text("done"))
        }

        fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
            self.hosted.then_some(self as &dyn HostedToolSearch)
        }
    }

    #[async_trait::async_trait]
    impl HostedToolSearch for CapabilityLlm {
        async fn stream_completion_with_namespaces(
            &self,
            messages: Vec<Message>,
            core_tools: &[ToolDefinition],
            namespaces: &[ToolNamespace],
            reasoning: ReasoningConfig,
            on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let all =
                desktop_assistant_core::ports::llm::flatten_namespaces(core_tools, namespaces);
            self.stream_completion(messages, &all, reasoning, on_chunk)
                .await
        }
    }

    #[tokio::test]
    async fn hosted_tool_search_follows_the_active_client_override() {
        // Fallback supports hosted search; the turn's override does not.
        let router = RoutingLlmClient::new(Arc::new(CapabilityLlm::new(true)));
        let without_search: Arc<dyn LlmClient> = Arc::new(CapabilityLlm::new(false));
        let answer = with_active_client(without_search, async {
            router.hosted_tool_search().is_some()
        })
        .await;
        assert!(
            !answer,
            "capability must come from the active client, which has no hosted search"
        );

        // The other direction, so a constant answer cannot pass this test.
        let router = RoutingLlmClient::new(Arc::new(CapabilityLlm::new(false)));
        let with_search: Arc<dyn LlmClient> = Arc::new(CapabilityLlm::new(true));
        let answer =
            with_active_client(with_search, async { router.hosted_tool_search().is_some() }).await;
        assert!(
            answer,
            "capability must come from the active client, which has hosted search"
        );
    }

    #[tokio::test]
    async fn hosted_tool_search_uses_the_fallback_when_no_override_is_installed() {
        let router = RoutingLlmClient::new(Arc::new(CapabilityLlm::new(true)));
        assert!(
            router.hosted_tool_search().is_some(),
            "with no override installed the static fallback answers"
        );

        let router = RoutingLlmClient::new(Arc::new(CapabilityLlm::new(false)));
        assert!(
            !router.hosted_tool_search().is_some(),
            "with no override installed the static fallback answers"
        );
    }

    #[tokio::test]
    async fn dynamic_purpose_mode_still_reports_false() {
        let handle = build_handle_with_titling("titling-model");
        let client = RoutingLlmClient::new_dynamic_purpose(handle, PurposeKind::Titling);
        assert!(
            !client.hosted_tool_search().is_some(),
            "dynamic-purpose wrappers report the connector default"
        );

        // Backend tasks must not inherit the user's per-turn override even
        // when they start inside a `send_prompt` scope.
        let hosted: Arc<dyn LlmClient> = Arc::new(CapabilityLlm::new(true));
        let answer =
            with_active_client(hosted, async { client.hosted_tool_search().is_some() }).await;
        assert!(
            !answer,
            "dynamic-purpose wrappers must ignore the per-turn override"
        );
    }

    /// Decorator-in-the-path criterion for `RoutingLlmClient` (#1033).
    ///
    /// The router's job is to send the turn to the per-turn active client.
    /// If it hands the caller its fallback client's hosted-search dispatch
    /// object, the namespaced turn goes to the wrong connection - billed to
    /// the wrong account, and against a model the user did not choose.
    #[tokio::test]
    async fn routing_decorator_stays_in_the_namespaced_path() {
        use crate::hosted_search_probe::{NamespaceProbe, noop_chunk, probe_namespace};
        use desktop_assistant_core::domain::Role;
        use desktop_assistant_core::ports::llm::dispatch_namespaced;

        let fallback = Arc::new(NamespaceProbe::plain());
        let active = Arc::new(NamespaceProbe::hosted());
        let router = RoutingLlmClient::new(Arc::clone(&fallback) as Arc<dyn LlmClient>);

        with_active_client(Arc::clone(&active) as Arc<dyn LlmClient>, async {
            dispatch_namespaced(
                &router,
                vec![Message::new(Role::User, "hi")],
                &[],
                &[probe_namespace()],
                ReasoningConfig::default(),
                noop_chunk(),
            )
            .await
            .expect("probe turn");
        })
        .await;

        assert_eq!(
            active.namespaced_calls(),
            1,
            "the namespaced turn must go to the per-turn active client"
        );
        assert_eq!(
            fallback.plain_calls() + fallback.namespaced_calls(),
            0,
            "the fallback client must see nothing while an override is installed"
        );
    }

    // --- End-to-end turn through `send_prompt` ------------------------------

    /// In-memory conversation store for the end-to-end turn test.
    #[derive(Default)]
    struct MemStore {
        data: std::sync::Mutex<
            std::collections::HashMap<String, desktop_assistant_core::domain::Conversation>,
        >,
    }

    impl desktop_assistant_core::ports::store::ConversationStore for MemStore {
        async fn create(
            &self,
            conv: desktop_assistant_core::domain::Conversation,
        ) -> Result<(), CoreError> {
            self.data.lock().unwrap().insert(conv.id.0.clone(), conv);
            Ok(())
        }

        async fn get(
            &self,
            id: &desktop_assistant_core::domain::ConversationId,
        ) -> Result<desktop_assistant_core::domain::Conversation, CoreError> {
            self.data
                .lock()
                .unwrap()
                .get(&id.0)
                .cloned()
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
        }

        async fn list(
            &self,
        ) -> Result<Vec<desktop_assistant_core::domain::ConversationSummary>, CoreError> {
            Ok(Vec::new())
        }

        async fn update(
            &self,
            conv: desktop_assistant_core::domain::Conversation,
        ) -> Result<(), CoreError> {
            self.data.lock().unwrap().insert(conv.id.0.clone(), conv);
            Ok(())
        }

        async fn delete(
            &self,
            id: &desktop_assistant_core::domain::ConversationId,
        ) -> Result<(), CoreError> {
            self.data.lock().unwrap().remove(&id.0);
            Ok(())
        }

        async fn archive(
            &self,
            _id: &desktop_assistant_core::domain::ConversationId,
        ) -> Result<(), CoreError> {
            Ok(())
        }

        async fn unarchive(
            &self,
            _id: &desktop_assistant_core::domain::ConversationId,
        ) -> Result<(), CoreError> {
            Ok(())
        }

        async fn create_summary(
            &self,
            _conversation_id: &desktop_assistant_core::domain::ConversationId,
            _summary: String,
            _start_ordinal: usize,
            _end_ordinal: usize,
        ) -> Result<String, CoreError> {
            Ok("summary-1".to_string())
        }

        async fn expand_summary(&self, _summary_id: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    /// Tool executor with one core tool and a small namespaced fleet.
    ///
    /// The fleet has three tools on purpose. `categorize_tool_namespaces`
    /// returns its input unchanged when the namespaced set holds ten tools
    /// or fewer (`crates/core/src/tools.rs`), so the turn under test needs no
    /// categorization LLM round-trip and the namespaces reach dispatch as
    /// written. Raising that threshold is safe, and so is lowering it to
    /// three. Lowering it below three turns this into a categorization test,
    /// and the failure would name the tool list rather than the threshold
    /// that changed.
    struct FleetTools;

    impl FleetTools {
        fn tool(name: &str) -> ToolDefinition {
            ToolDefinition::new(name, "test tool", serde_json::json!({"type": "object"}))
        }
    }

    impl desktop_assistant_core::ports::tools::ToolExecutor for FleetTools {
        async fn core_tools(&self) -> Vec<ToolDefinition> {
            vec![Self::tool("builtin_tool_search")]
        }

        async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(Vec::new())
        }

        async fn tool_definition(&self, _name: &str) -> Result<Option<ToolDefinition>, CoreError> {
            Ok(None)
        }

        async fn tool_namespaces(&self) -> Vec<ToolNamespace> {
            vec![ToolNamespace::new(
                "fleet",
                "the whole tool fleet",
                vec![
                    Self::tool("fleet_alpha"),
                    Self::tool("fleet_beta"),
                    Self::tool("fleet_gamma"),
                ],
            )]
        }

        async fn execute_tool(
            &self,
            _name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            Ok(String::new())
        }
    }

    /// The #1021 failure, end to end. The static fallback supports hosted
    /// tool search, the turn overrides to a connection that does not, and
    /// the request must therefore keep `builtin_tool_search` and must not
    /// flatten the namespaced fleet into one call.
    #[tokio::test]
    async fn per_turn_override_keeps_tool_search_and_does_not_flatten_the_fleet() {
        use desktop_assistant_core::ports::inbound::ConversationService;
        use desktop_assistant_core::service::ConversationHandler;

        let hosted_fallback = Arc::new(CapabilityLlm::new(true));
        let overridden = Arc::new(CapabilityLlm::new(false));

        let router = RoutingLlmClient::new(Arc::clone(&hosted_fallback) as Arc<dyn LlmClient>);
        let handler = ConversationHandler::with_tools(
            MemStore::default(),
            router,
            FleetTools,
            Box::new(|| "conv-1".to_string()),
        );
        let conv = handler
            .create_conversation("routing".to_string(), vec![])
            .await
            .expect("conversation is created");

        with_active_client(
            Arc::clone(&overridden) as Arc<dyn LlmClient>,
            handler.send_prompt(
                &conv.id,
                "hello".to_string(),
                Box::new(|_| true),
                Box::new(|_| {}),
            ),
        )
        .await
        .expect("the turn completes");

        assert!(
            hosted_fallback.offered().is_empty(),
            "the turn must dispatch to the override, not the fallback"
        );

        // Later dispatches in the turn are the title-generation round, which
        // carries no tools. The turn's own request is the first one.
        let offered = overridden.offered();
        let first = offered.first().expect("the turn dispatched to the LLM");
        assert!(
            first.iter().any(|n| n == "builtin_tool_search"),
            "a connection without hosted search keeps builtin_tool_search; got {first:?}"
        );
        assert!(
            !offered.iter().flatten().any(|n| n.starts_with("fleet_")),
            "the namespaced fleet must not be flattened into a request; got {offered:?}"
        );
    }

    // --- Backend-tasks slot: the turn must not reach it ---------------------

    /// `LlmClient` double that records, for every dispatch, the
    /// `MODEL_OVERRIDE` task-local a real connector would have read.
    ///
    /// Every connector resolves its wire model as
    /// `current_model_override().unwrap_or(self.model)`, so this recorded
    /// value is exactly what decides which model the request is billed to.
    /// Recording it (rather than answering a fixed capability) is what makes
    /// these tests fail in both directions: a slot that starts reading the
    /// task-local again, and a slot that stops reading it.
    #[derive(Default)]
    struct ModelRecordingLlm {
        seen: std::sync::Mutex<Vec<Option<String>>>,
    }

    impl ModelRecordingLlm {
        /// One entry per dispatch: the model override in force at the time.
        fn seen(&self) -> Vec<Option<String>> {
            self.seen.lock().unwrap().clone()
        }

        fn dispatches(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for ModelRecordingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.seen
                .lock()
                .unwrap()
                .push(desktop_assistant_core::ports::llm::current_model_override());
            Ok(LlmResponse::text("done"))
        }
    }

    fn one_user_message() -> Vec<Message> {
        vec![Message::new(
            desktop_assistant_core::domain::Role::User,
            "hi",
        )]
    }

    /// Acceptance criterion (#1031): with an `ACTIVE_CLIENT` installed, a
    /// legacy `[backend_tasks.llm]` slot dispatches to its own configured
    /// client, never to the turn's interactive client.
    #[tokio::test]
    async fn legacy_backend_tasks_slot_ignores_the_per_turn_client() {
        let backend = Arc::new(ModelRecordingLlm::default());
        let interactive = Arc::new(ModelRecordingLlm::default());

        // Built exactly the way `main.rs` builds the legacy
        // `[backend_tasks.llm]` slot.
        let slot = RoutingLlmClient::new_pinned(
            Arc::clone(&backend) as Arc<dyn LlmClient>,
            "backend-model".to_string(),
        );

        with_active_client(Arc::clone(&interactive) as Arc<dyn LlmClient>, async {
            slot.stream_completion(
                one_user_message(),
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
        })
        .await
        .expect("the backend dispatch completes");

        assert_eq!(
            backend.dispatches(),
            1,
            "the backend-tasks slot must dispatch to its own configured client"
        );
        assert_eq!(
            interactive.dispatches(),
            0,
            "the turn's interactive client must never serve a backend task"
        );
    }

    /// Acceptance criterion (#1031): the turn's `MODEL_OVERRIDE` must not
    /// reach the legacy backend-tasks slot. The connector must see the
    /// configured backend model instead.
    #[tokio::test]
    async fn legacy_backend_tasks_slot_ignores_the_per_turn_model_override() {
        let backend = Arc::new(ModelRecordingLlm::default());
        let interactive = Arc::new(ModelRecordingLlm::default());

        // Built exactly the way `main.rs` builds the legacy
        // `[backend_tasks.llm]` slot.
        let slot = RoutingLlmClient::new_pinned(
            Arc::clone(&backend) as Arc<dyn LlmClient>,
            "backend-model".to_string(),
        );

        with_model_override(
            "interactive-model".to_string(),
            with_active_client(Arc::clone(&interactive) as Arc<dyn LlmClient>, async {
                slot.stream_completion(
                    one_user_message(),
                    &[],
                    ReasoningConfig::default(),
                    Box::new(|_| true),
                )
                .await
            }),
        )
        .await
        .expect("the backend dispatch completes");

        assert_eq!(
            backend.seen(),
            vec![Some("backend-model".to_string())],
            "the backend-tasks slot must pin its configured model, not the turn's"
        );
    }

    /// The mirror case (#1021, which must keep working): the *primary* slot
    /// is supposed to read both task-locals. A change that stopped it
    /// following the turn fails here.
    #[tokio::test]
    async fn primary_slot_still_follows_the_per_turn_client_and_model_override() {
        let startup_default = Arc::new(ModelRecordingLlm::default());
        let interactive = Arc::new(ModelRecordingLlm::default());

        let primary = RoutingLlmClient::new(Arc::clone(&startup_default) as Arc<dyn LlmClient>);

        with_model_override(
            "interactive-model".to_string(),
            with_active_client(Arc::clone(&interactive) as Arc<dyn LlmClient>, async {
                primary
                    .stream_completion(
                        one_user_message(),
                        &[],
                        ReasoningConfig::default(),
                        Box::new(|_| true),
                    )
                    .await
            }),
        )
        .await
        .expect("the turn dispatch completes");

        assert_eq!(
            interactive.seen(),
            vec![Some("interactive-model".to_string())],
            "the primary slot must dispatch to the turn's client with the turn's model"
        );
        assert_eq!(
            startup_default.dispatches(),
            0,
            "the startup default must not serve a turn that resolved a client"
        );
    }

    /// The reported configuration, end to end: a legacy `[backend_tasks.llm]`
    /// naming a different connector/model, no `[purposes.titling]`, and a turn
    /// that installs the interactive client and model. Title generation runs
    /// inside `send_prompt`, so it must still reach the backend slot.
    #[tokio::test]
    async fn legacy_backend_tasks_titling_runs_on_the_configured_backend_model() {
        use desktop_assistant_core::ports::inbound::ConversationService;
        use desktop_assistant_core::service::ConversationHandler;

        let startup_default = Arc::new(ModelRecordingLlm::default());
        let interactive = Arc::new(ModelRecordingLlm::default());
        let backend = Arc::new(ModelRecordingLlm::default());

        let primary = RoutingLlmClient::new(Arc::clone(&startup_default) as Arc<dyn LlmClient>);
        let backend_slot = RoutingLlmClient::new_pinned(
            Arc::clone(&backend) as Arc<dyn LlmClient>,
            "backend-model".to_string(),
        );

        let handler = ConversationHandler::with_tools(
            MemStore::default(),
            primary,
            FleetTools,
            Box::new(|| "conv-1".to_string()),
        )
        .with_backend_llm(backend_slot);

        let conv = handler
            .create_conversation("routing".to_string(), vec![])
            .await
            .expect("conversation is created");

        with_model_override(
            "interactive-model".to_string(),
            with_active_client(
                Arc::clone(&interactive) as Arc<dyn LlmClient>,
                handler.send_prompt(
                    &conv.id,
                    "hello".to_string(),
                    Box::new(|_| true),
                    Box::new(|_| {}),
                ),
            ),
        )
        .await
        .expect("the turn completes");

        let billed = backend.seen();
        assert!(
            !billed.is_empty(),
            "title generation must reach the configured backend-tasks client"
        );
        assert!(
            billed.iter().all(|m| m.as_deref() == Some("backend-model")),
            "every backend-task request must carry the configured backend model; got {billed:?}"
        );
        assert_eq!(
            interactive.seen(),
            vec![Some("interactive-model".to_string())],
            "the turn's own request must still go to the interactive client and model"
        );
        assert_eq!(
            startup_default.dispatches(),
            0,
            "neither slot may fall through to the startup default here"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn no_hang_fails_fast_when_inner_call_times_out() {
        // The backstop must *fail* (panic) — never hang — when a wrapped call
        // doesn't complete (#186). With the clock paused, the runtime
        // auto-advances to the 15s deadline so this runs instantly; the
        // spawned task's panic surfaces as a JoinError.
        let handle = tokio::spawn(async {
            no_hang("never-completes", std::future::pending::<()>()).await;
        });
        assert!(
            handle.await.is_err(),
            "no_hang must fail (panic) when the inner call times out, not hang"
        );
    }
}
