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
    ChunkCallback, LlmClient, LlmResponse, ModelInfo, ReasoningConfig,
};

use crate::connections::ConnectionId;
use crate::registry::AnyLlmClient;

tokio::task_local! {
    /// Per-turn routing override. When set, dispatch uses the contained
    /// `Arc<AnyLlmClient>` (resolved from the registry) instead of the
    /// [`RoutingLlmClient`]'s static fallback. Populated by
    /// [`with_active_client`] from inside the routing wrapper.
    static ACTIVE_CLIENT: Arc<AnyLlmClient>;

    /// Per-turn purpose-driven `max_context_tokens` override (issue #51).
    /// When set, [`RoutingLlmClient::max_context_tokens`] uses this value
    /// as tier 1 of the three-tier resolution (user override beats the
    /// connector's curated table beats the universal fallback). Populated
    /// by [`with_max_context_override`].
    static MAX_CONTEXT_OVERRIDE: Option<u64>;
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

/// Run `fut` with `override_value` installed as the per-turn
/// `max_context_tokens` override (issue #51). The wrapper's
/// [`LlmClient::max_context_tokens`] consults this slot first, falling
/// through to the resolved client's curated value and then to
/// [`crate::config::DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS`].
///
/// `override_value` is `Option<u64>` so callers can install the slot
/// unconditionally — `None` simply means "no purpose-level override; use
/// the connector's table or the universal fallback."
pub async fn with_max_context_override<F, T>(override_value: Option<u64>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    MAX_CONTEXT_OVERRIDE.scope(override_value, fut).await
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

/// The handler's LLM facade. Delegates to the per-turn active client when
/// one is installed, or to the static fallback otherwise.
///
/// Note that `list_models`, capability flags, and default-model accessors
/// always delegate to the fallback — they describe the handler's
/// configured interactive model and are not meaningfully per-turn.
#[derive(Clone)]
pub struct RoutingLlmClient {
    /// Client used when no task-local override is installed (e.g. title
    /// generation run outside `send_prompt`, dreaming jobs that own
    /// their own `llm` handle).
    fallback: Arc<AnyLlmClient>,
    /// Static fallback connector-type tag. Only used for diagnostics.
    #[allow(dead_code)]
    fallback_connector_type: String,
}

impl RoutingLlmClient {
    pub fn new(fallback: Arc<AnyLlmClient>, fallback_connector_type: String) -> Self {
        Self {
            fallback,
            fallback_connector_type,
        }
    }

    /// Resolve the current turn's active client. Returns the task-local
    /// override if set, or the static fallback otherwise.
    fn resolve(&self) -> Arc<AnyLlmClient> {
        ACTIVE_CLIENT
            .try_with(|c| Arc::clone(c))
            .unwrap_or_else(|_| Arc::clone(&self.fallback))
    }
}

impl LlmClient for RoutingLlmClient {
    fn get_default_model(&self) -> Option<&str> {
        // `Option<&str>` borrows from `self`; we can't delegate through the
        // task-local (which returns an Arc). Delegation to the fallback is
        // correct since this accessor reports the statically configured
        // default, not the per-turn model.
        self.fallback.get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        self.fallback.get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        // Three-tier resolution (issue #51):
        //   1. Per-turn purpose override from `MAX_CONTEXT_OVERRIDE`.
        //   2. Resolved client's curated value (`max_context_tokens` on the
        //      underlying `AnyLlmClient`, e.g. `BedrockClient`'s table).
        //   3. `DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS` (200k) — conservative
        //      universal fallback so token-based compaction stays on even
        //      when the connector has no curated entry.
        let purpose_override = MAX_CONTEXT_OVERRIDE.try_with(|v| *v).unwrap_or(None);
        let client_max = self.resolve().max_context_tokens();
        Some(crate::config::resolve_max_context_tokens(
            purpose_override,
            client_max,
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // Callers of `list_models` are typically the connections-management
        // API, which resolves clients directly from the registry — not
        // through the routing wrapper. Keep this consistent with
        // connector-level behaviour and delegate to whichever client is
        // currently active (task-local or fallback).
        self.resolve().list_models().await
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.resolve().refresh_models().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let client = self.resolve();
        client
            .stream_completion(messages, tools, reasoning, on_chunk)
            .await
    }

    fn supports_hosted_tool_search(&self) -> bool {
        // The flag gates how `ConversationHandler` assembles the tool
        // list at the start of a turn, before any task-local is
        // consulted. Report the fallback's capability so the handler's
        // choice is consistent with what dispatch will actually support
        // in the absence of per-turn routing.
        self.fallback.supports_hosted_tool_search()
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let client = self.resolve();
        client
            .stream_completion_with_namespaces(
                messages, core_tools, namespaces, reasoning, on_chunk,
            )
            .await
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
        let resolved = client.resolve();
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
            client.resolve()
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

    // --- max_context_tokens three-tier resolution (issue #51) ------------

    #[tokio::test]
    async fn max_context_falls_back_to_universal_when_unset() {
        // Tier 3: ollama exposes no curated `max_context_tokens` and we
        // installed no override → 200k universal fallback. This is the
        // safety net that keeps token-based compaction on for non-curated
        // providers instead of silently disabling.
        let fallback = build_ollama_registry();
        let client = RoutingLlmClient::new(fallback, "ollama".into());
        let resolved = client.max_context_tokens();
        assert_eq!(
            resolved,
            Some(crate::config::DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS)
        );
    }

    #[tokio::test]
    async fn max_context_uses_purpose_override_when_set() {
        // Tier 1: an installed `MAX_CONTEXT_OVERRIDE` task-local takes
        // precedence over the connector's value (or lack thereof).
        let fallback = build_ollama_registry();
        let client = RoutingLlmClient::new(fallback, "ollama".into());
        let resolved = with_max_context_override(Some(1_000_000), async {
            client.max_context_tokens()
        })
        .await;
        assert_eq!(resolved, Some(1_000_000));
    }

    #[tokio::test]
    async fn max_context_override_none_falls_through_to_curated_or_universal() {
        // An installed-but-`None` override means "no purpose-level
        // override; use tier 2/3." For ollama (no curated value) that's
        // the universal fallback.
        let fallback = build_ollama_registry();
        let client = RoutingLlmClient::new(fallback, "ollama".into());
        let resolved = with_max_context_override(None, async {
            client.max_context_tokens()
        })
        .await;
        assert_eq!(
            resolved,
            Some(crate::config::DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS)
        );
    }
}
