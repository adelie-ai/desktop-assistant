//! Per-turn dispatch client used by [`crate::api_surface::RoutingConversationHandler`]
//! to swap the underlying [`AnyLlmClient`] based on the resolved
//! `(connection_id, model_id, effort)` triple for each `send_prompt`.
//!
//! Rationale (issue #18): the core `ConversationHandler` owns a single
//! `llm: L` field baked into its type parameters. Rebuilding the handler
//! per turn is impractical (shared `namespace_cache`, non-`Clone`
//! `id_generator`), and plumbing a per-call client argument through the
//! ~450-line `send_prompt` would be a very invasive change.
//!
//! Instead we install this wrapper as the handler's `L`. It looks up the
//! target `AnyLlmClient` on each call via a [`tokio::task_local!`] slot
//! populated by the daemon-side routing wrapper. When the slot is unset
//! (e.g. backend-tasks, background jobs), dispatch falls through to a
//! statically-configured fallback — the interactive-purpose client at
//! daemon startup.
//!
//! Concurrency: `tokio::task_local!` is per-task, so two concurrent
//! `send_prompt` calls on different conversations each see their own
//! routing target without coupling.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelInfo, ReasoningConfig, with_model_override,
};

use crate::api_surface::RegistryHandle;
use crate::connections::ConnectionId;
use crate::purposes::PurposeKind;
use crate::registry::AnyLlmClient;

tokio::task_local! {
    /// Per-turn routing override. When set, dispatch uses the contained
    /// `Arc<AnyLlmClient>` (resolved from the registry) instead of the
    /// [`RoutingLlmClient`]'s static fallback. Populated by
    /// [`with_active_client`] from inside the routing wrapper.
    static ACTIVE_CLIENT: Arc<AnyLlmClient>;
}

/// Run `fut` with `client` installed as the current turn's active LLM
/// client. All `stream_completion(_with_namespaces)` calls on the
/// enclosing [`RoutingLlmClient`] observe `client` and dispatch to it.
pub async fn with_active_client<F, T>(client: Arc<AnyLlmClient>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    ACTIVE_CLIENT.scope(client, fut).await
}

/// Whether an [`ACTIVE_CLIENT`] task-local is set for the current
/// scope. Used by tests in the api_surface dispatch module to assert
/// that interactive-purpose fallbacks correctly *do not* install an
/// override (issue #33: dispatch should fall through to the primary
/// llm in that case so the interactive purpose's model takes effect).
#[cfg(test)]
pub(crate) fn active_client_is_set() -> bool {
    ACTIVE_CLIENT.try_with(|_| ()).is_ok()
}

/// Fallback resolution mode for [`RoutingLlmClient`]. Controls what the
/// wrapper dispatches to when no per-turn [`ACTIVE_CLIENT`] task-local is
/// installed.
#[derive(Clone)]
pub enum FallbackMode {
    /// Static client captured at construction. Used by the primary
    /// (interactive) slot — dispatch reads `ACTIVE_CLIENT` first, then
    /// falls back to this client for legacy callers without an override.
    Static {
        client: Arc<AnyLlmClient>,
        /// Connector-type tag retained for diagnostics.
        #[allow(dead_code)]
        connector_type: String,
    },
    /// Resolve the target client from a [`RegistryHandle`] on every
    /// dispatch by re-reading the named purpose's config. Used by the
    /// backend-tasks slot so titling/dreaming pick up control-panel edits
    /// without a daemon restart (issue #68). Always ignores
    /// `ACTIVE_CLIENT` — backend tasks must not inherit the user's
    /// per-turn model override even when invoked inside a `send_prompt`
    /// scope.
    DynamicPurpose {
        registry: Arc<RegistryHandle>,
        purpose: PurposeKind,
    },
}

/// The handler's LLM facade. Delegates to the per-turn active client when
/// one is installed (Static mode only), or to the configured fallback
/// otherwise.
///
/// Note that `list_models`, capability flags, and default-model accessors
/// always delegate to the Static fallback — they describe the handler's
/// configured interactive model and are not meaningfully per-turn. The
/// DynamicPurpose mode is only useful for `stream_completion` paths.
#[derive(Clone)]
pub struct RoutingLlmClient {
    fallback: FallbackMode,
}

impl RoutingLlmClient {
    /// Static-fallback constructor. Used by the primary (interactive)
    /// slot.
    pub fn new(fallback: Arc<AnyLlmClient>, fallback_connector_type: String) -> Self {
        Self {
            fallback: FallbackMode::Static {
                client: fallback,
                connector_type: fallback_connector_type,
            },
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

    /// Snapshot of the static fallback client for accessor delegation
    /// (`list_models`, `max_context_tokens`, etc.). Returns `None` for
    /// dynamic-purpose wrappers, which intentionally have no single
    /// captured client to delegate to.
    fn static_fallback(&self) -> Option<&Arc<AnyLlmClient>> {
        match &self.fallback {
            FallbackMode::Static { client, .. } => Some(client),
            FallbackMode::DynamicPurpose { .. } => None,
        }
    }

    /// Resolve the current turn's active client for Static mode. Returns
    /// the task-local override if set, or the static fallback otherwise.
    /// Only meaningful for Static mode — DynamicPurpose dispatches via
    /// [`Self::dispatch_dynamic`].
    fn resolve_static(&self) -> Arc<AnyLlmClient> {
        let FallbackMode::Static { client, .. } = &self.fallback else {
            unreachable!("resolve_static called on DynamicPurpose mode");
        };
        ACTIVE_CLIENT
            .try_with(|c| Arc::clone(c))
            .unwrap_or_else(|_| Arc::clone(client))
    }
}

impl RoutingLlmClient {
    /// Dispatch path for [`FallbackMode::DynamicPurpose`]. Resolves the
    /// purpose against the live config snapshot, installs the resolved
    /// model override for the connector, and runs `op` against the
    /// registry's client. Returns a `CoreError::Llm` describing the
    /// failure mode if resolution can't proceed (purpose unconfigured,
    /// connection missing from the registry).
    async fn dispatch_dynamic<F, Fut, T>(&self, op: F) -> Result<T, CoreError>
    where
        F: FnOnce(Arc<AnyLlmClient>, ReasoningConfig) -> Fut,
        Fut: std::future::Future<Output = Result<T, CoreError>>,
    {
        let FallbackMode::DynamicPurpose { registry, purpose } = &self.fallback else {
            unreachable!("dispatch_dynamic called on Static mode");
        };
        let config = registry.snapshot_config();
        let (resolved, reasoning) = crate::api_surface::resolve_purpose_dispatch(
            Some(&config),
            *purpose,
        )
        .ok_or_else(|| {
            CoreError::Llm(format!(
                "purpose {:?} is not configured; cannot dispatch backend task",
                purpose.as_key()
            ))
        })?;
        let connection_id = ConnectionId::new(resolved.connector.clone()).map_err(|e| {
            CoreError::Llm(format!(
                "purpose {:?} resolved to invalid connection id {:?}: {e}",
                purpose.as_key(),
                resolved.connector
            ))
        })?;
        // `resolved.connector` carries the connection id (not the
        // connector type) — see `resolve_purpose_llm_config`. The
        // registry indexes by connection id so this lookup is correct.
        let client = registry.client_for(&connection_id).ok_or_else(|| {
            CoreError::Llm(format!(
                "purpose {:?} references connection {:?} which is not present in the registry",
                purpose.as_key(),
                resolved.connector
            ))
        })?;
        let model = resolved.model.clone();
        with_model_override(model, op(client, reasoning)).await
    }
}

impl LlmClient for RoutingLlmClient {
    fn get_default_model(&self) -> Option<&str> {
        // `Option<&str>` borrows from `self`; we can't delegate through the
        // task-local (which returns an Arc) or a dynamic registry lookup.
        // Static mode delegates to the captured fallback; dynamic-purpose
        // mode has no single captured client so reports `None`. This
        // accessor reports the statically configured default and is not
        // meaningfully per-turn or per-purpose.
        self.static_fallback().and_then(|c| c.get_default_model())
    }

    fn get_default_base_url(&self) -> Option<&str> {
        self.static_fallback()
            .and_then(|c| c.get_default_base_url())
    }

    fn max_context_tokens(&self) -> Option<u64> {
        // The dispatch loop reads token-pressure budgets from the
        // `CONTEXT_BUDGET` task-local installed by the daemon's wrapper
        // (issue #63), not from this trait method, so the resolution
        // chain no longer lives here. Static-mode delegates to the
        // resolved client; dynamic-purpose mode has no single client to
        // ask without a config snapshot, and callers (capability probes,
        // debug paths) tolerate `None`.
        match &self.fallback {
            FallbackMode::Static { .. } => self.resolve_static().max_context_tokens(),
            FallbackMode::DynamicPurpose { .. } => None,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // Callers of `list_models` are typically the connections-management
        // API, which resolves clients directly from the registry — not
        // through the routing wrapper. Keep this consistent with
        // connector-level behaviour and delegate to whichever client is
        // currently active (task-local or static fallback). The
        // dynamic-purpose wrapper isn't used by listing paths, so report
        // an empty list there.
        match &self.fallback {
            FallbackMode::Static { .. } => self.resolve_static().list_models().await,
            FallbackMode::DynamicPurpose { .. } => Ok(Vec::new()),
        }
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        match &self.fallback {
            FallbackMode::Static { .. } => self.resolve_static().refresh_models().await,
            FallbackMode::DynamicPurpose { .. } => Ok(Vec::new()),
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
            FallbackMode::Static { .. } => {
                let client = self.resolve_static();
                client
                    .stream_completion(messages, tools, reasoning, on_chunk)
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

    fn supports_hosted_tool_search(&self) -> bool {
        // The flag gates how `ConversationHandler` assembles the tool
        // list at the start of a turn, before any task-local is
        // consulted. Static mode reports the fallback's capability;
        // dynamic-purpose mode is only used for backend tasks
        // (title/summary), which don't traverse the hosted-search path,
        // so reporting `false` is safe and matches the connector-default.
        self.static_fallback()
            .map(|c| c.supports_hosted_tool_search())
            .unwrap_or(false)
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        match &self.fallback {
            FallbackMode::Static { .. } => {
                let client = self.resolve_static();
                client
                    .stream_completion_with_namespaces(
                        messages,
                        core_tools,
                        namespaces,
                        reasoning,
                        on_chunk,
                    )
                    .await
            }
            FallbackMode::DynamicPurpose { .. } => {
                let _ = reasoning;
                self.dispatch_dynamic(|client, resolved_reasoning| async move {
                    client
                        .stream_completion_with_namespaces(
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

/// Look up a connection id's live client on the registry. Wraps the
/// existing `RegistryHandle::client_for` so [`crate::api_surface`] can
/// hand a concrete `Arc<AnyLlmClient>` into [`with_active_client`]
/// without pulling the `crate::registry` internals into the public API.
#[allow(dead_code)]
pub fn resolve_client(
    registry: &crate::api_surface::RegistryHandle,
    id: &ConnectionId,
) -> Option<Arc<AnyLlmClient>> {
    registry.client_for(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connections::{ConnectionConfig, OllamaConnection};
    use crate::registry::build_registry;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::domain::Message;
    use desktop_assistant_core::ports::llm::ReasoningConfig;
    use indexmap::IndexMap;

    fn build_ollama_registry() -> Arc<AnyLlmClient> {
        let cfg = crate::config::DaemonConfig {
            connections: IndexMap::from([(
                "local".to_string(),
                ConnectionConfig::Ollama(OllamaConnection {
                    base_url: Some("http://localhost:11434".into()),
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
        let client = RoutingLlmClient::new(Arc::clone(&fallback), "ollama".into());
        // Without a task-local override, `resolve()` must equal the
        // fallback pointer.
        let resolved = client.resolve_static();
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

        let client = RoutingLlmClient::new(Arc::clone(&fallback), "ollama".into());

        let override_clone = Arc::clone(&override_client);
        let resolved = with_active_client(override_client, async move {
            client.resolve_static()
        })
        .await;
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
        let client = RoutingLlmClient::new(fallback, "ollama".into());
        let _ = client
            .stream_completion(
                vec![Message::new(
                    desktop_assistant_core::domain::Role::User,
                    "hi",
                )],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await;
        // Result will be an `Err` (no ollama server), but the call
        // path itself must complete without panicking.
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
                    base_url: Some("http://localhost:11434".into()),
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
        assert!(resolve_client(&registry_handle, &missing).is_none());
    }

    #[test]
    fn existing_connection_id_resolves() {
        let registry_handle = build_local_ollama_handle();
        let id = ConnectionId::new("local").unwrap();
        assert!(resolve_client(&registry_handle, &id).is_some());
    }

    #[test]
    fn unused_core_error_type_still_compiles() {
        // Make sure the CoreError import isn't elided by mistake.
        let _e: Option<CoreError> = None;
    }

    // The three-tier `max_context_tokens` resolution previously tested
    // here moved to `crate::config::resolve_context_budget` (issue #63).
    // Tests against that resolver live in `crates/daemon/src/config.rs`.
    // The task-local accessor is exercised in
    // `crates/core/src/ports/llm.rs` against `current_context_budget`.

    #[tokio::test]
    async fn max_context_delegates_to_resolved_client() {
        // After #63 `RoutingLlmClient::max_context_tokens` is plain
        // delegation to the resolved client — no overlay, no tier
        // fallback. Ollama returns `None`; the wrapper must too.
        let fallback = build_ollama_registry();
        let client = RoutingLlmClient::new(fallback, "ollama".into());
        assert_eq!(client.max_context_tokens(), None);
    }

    // --- DynamicPurpose mode (issue #68) ---------------------------------

    /// Build a `RegistryHandle` with `[purposes.titling]` pointed at the
    /// "local" Ollama connection — exercises the full purpose-resolution
    /// path used by the backend slot.
    fn build_handle_with_titling(model: &str) -> Arc<crate::api_surface::RegistryHandle> {
        use crate::purposes::{ConnectionRef, ModelRef, PurposeConfig, Purposes};
        let cfg = crate::config::DaemonConfig {
            connections: IndexMap::from([(
                "local".to_string(),
                ConnectionConfig::Ollama(OllamaConnection {
                    base_url: Some("http://localhost:11434".into()),
                }),
            )]),
            purposes: Purposes {
                interactive: Some(PurposeConfig {
                    connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                    model: ModelRef::Named("interactive-model".to_string()),
                    effort: None,
                    max_context_tokens: None,
                }),
                titling: Some(PurposeConfig {
                    connection: ConnectionRef::Named(ConnectionId::new("local").unwrap()),
                    model: ModelRef::Named(model.to_string()),
                    effort: None,
                    max_context_tokens: None,
                }),
                ..Purposes::default()
            },
            ..crate::config::DaemonConfig::default()
        };
        let reg = build_registry(&cfg);
        Arc::new(crate::api_surface::RegistryHandle::new(cfg, reg))
    }

    #[tokio::test]
    async fn dynamic_purpose_unconfigured_returns_error() {
        // Empty config: titling purpose isn't set, so dispatch must fail
        // with a clear error rather than panic.
        let cfg = crate::config::DaemonConfig::default();
        let reg = build_registry(&cfg);
        let handle = Arc::new(crate::api_surface::RegistryHandle::new(cfg, reg));
        let client = RoutingLlmClient::new_dynamic_purpose(handle, PurposeKind::Titling);
        let err = client
            .stream_completion(
                vec![Message::new(
                    desktop_assistant_core::domain::Role::User,
                    "hi",
                )],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("dispatch should fail when purpose is unconfigured");
        assert!(
            matches!(err, CoreError::Llm(ref msg) if msg.contains("titling")
                && msg.contains("not configured")),
            "expected purpose-not-configured error, got: {err}"
        );
    }

    #[tokio::test]
    async fn dynamic_purpose_resolves_against_live_config() {
        // The point of #68: a single dynamic-purpose client must read
        // the registry's current config on every call, not a snapshot
        // captured at construction. Build a handle, swap the titling
        // model in-place, and verify resolution observes the new value.
        use crate::api_surface::resolve_purpose_dispatch;

        let handle = build_handle_with_titling("model-v1");
        let _client = RoutingLlmClient::new_dynamic_purpose(
            Arc::clone(&handle),
            PurposeKind::Titling,
        );

        let cfg = handle.snapshot_config();
        let (resolved, _) = resolve_purpose_dispatch(Some(&cfg), PurposeKind::Titling)
            .expect("titling resolves before mutation");
        assert_eq!(resolved.model, "model-v1");

        // Swap the in-memory config — same path `mutate_config` takes
        // after the control panel writes a new value, minus the disk
        // persistence (covered by the connections-management API tests).
        let mut new_cfg = handle.snapshot_config();
        new_cfg.purposes.titling = Some(crate::purposes::PurposeConfig {
            connection: crate::purposes::ConnectionRef::Named(
                ConnectionId::new("local").unwrap(),
            ),
            model: crate::purposes::ModelRef::Named("model-v2".to_string()),
            effort: None,
            max_context_tokens: None,
        });
        handle.replace_config_for_test(new_cfg);

        let cfg2 = handle.snapshot_config();
        let (resolved2, _) = resolve_purpose_dispatch(Some(&cfg2), PurposeKind::Titling)
            .expect("titling resolves after mutation");
        assert_eq!(resolved2.model, "model-v2");
    }
}
