use crate::CoreError;
use crate::domain::{Conversation, ConversationId, ConversationSummary, KnowledgeEntry};
use crate::ports::llm::{ChunkCallback, ModelInfo, StatusCallback, with_cancellation_token};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct LlmSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub hosted_tool_search: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingsSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
    pub available: bool,
    pub is_default: bool,
}

#[derive(Debug, Clone)]
pub struct ConnectorDefaultsView {
    pub llm_model: String,
    pub llm_base_url: String,
    pub backend_llm_model: String,
    pub embeddings_model: String,
    pub embeddings_base_url: String,
    pub embeddings_available: bool,
    pub hosted_tool_search_available: bool,
}

#[derive(Debug, Clone)]
pub struct PersistenceSettingsView {
    pub enabled: bool,
    /// Empty string means no remote is configured.
    pub remote_url: String,
    pub remote_name: String,
    pub push_on_update: bool,
}

#[derive(Debug, Clone)]
pub struct DatabaseSettingsView {
    /// Empty string means no URL is configured.
    pub url: String,
    pub max_connections: u32,
}

/// Resolved assistant-personality settings (issue #226). A thin alias for the
/// canonical [`crate::prompts::Personality`] so the settings surface carries
/// the same typed trait set the prompt assembler uses — no parallel schema.
pub type PersonalitySettingsView = crate::prompts::Personality;

#[derive(Debug, Clone)]
pub struct McpServerView {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub namespace: Option<String>,
    pub enabled: bool,
    /// "running" | "stopped" | "disabled"
    pub status: String,
    pub tool_count: u32,
}

#[derive(Debug, Clone)]
pub struct WsAuthSettingsView {
    pub methods: Vec<String>,
    pub oidc_issuer: String,
    pub oidc_auth_endpoint: String,
    pub oidc_token_endpoint: String,
    pub oidc_client_id: String,
    pub oidc_scopes: String,
}

#[derive(Debug, Clone)]
pub struct BackendTasksSettingsView {
    /// Whether `[backend_tasks.llm]` is explicitly configured (vs. falling back to primary LLM).
    pub has_separate_llm: bool,
    /// Resolved connector (from backend_tasks.llm or fallback).
    pub llm_connector: String,
    /// Resolved model (from backend_tasks.llm or fallback).
    pub llm_model: String,
    /// Resolved base URL (from backend_tasks.llm or fallback).
    pub llm_base_url: String,
    /// Whether periodic fact extraction ("dreaming") is enabled.
    pub dreaming_enabled: bool,
    /// Interval in seconds between dreaming cycles.
    pub dreaming_interval_secs: u64,
    /// Archive conversations older than this many days (0 = disabled).
    pub archive_after_days: u32,
}

/// Inbound port for health/status queries.
///
/// Any adapter that wants to expose assistant status (D-Bus, HTTP, etc.)
/// implements a handler that calls through this trait.
pub trait AssistantService: Send + Sync {
    /// Returns a version string for the running assistant.
    fn version(&self) -> &str;

    /// Simple liveness check.
    fn ping(&self) -> &str;
}

/// Inbound port for conversation management.
///
/// Uses [`async_trait::async_trait`] so the per-turn future is boxed
/// (`Pin<Box<dyn Future>>`) rather than a deeply nested generic state
/// machine. The interactive turn spawns `send_prompt[_with_override]` on
/// a tokio worker (the transport dispatch loop / subagents); the
/// unboxed RPITIT form monomorphized into a multi-MB frame that
/// overflowed the 2 MB worker stack (#205/#206). Boxing keeps the
/// spawned future thin — the one heap allocation per call is negligible
/// next to an LLM round-trip, the same trade-off
/// [`crate::ports::llm::LlmClient`] already makes (#207).
#[async_trait::async_trait]
pub trait ConversationService: Send + Sync {
    async fn create_conversation(
        &self,
        title: String,
        tags: Vec<String>,
    ) -> Result<Conversation, CoreError>;

    async fn list_conversations(
        &self,
        max_age_days: Option<u32>,
        include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError>;

    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError>;

    /// Read the conversation's currently stored model selection, if one has
    /// been pinned by a prior override. Returns `Ok(None)` when the
    /// conversation has no stored selection (the daemon will fall back to the
    /// `interactive` purpose on the next send).
    ///
    /// The default implementation returns `Ok(None)`; the daemon's routing
    /// wrapper overrides this to consult the persistent selection store.
    async fn get_conversation_model_selection(
        &self,
        id: &ConversationId,
    ) -> Result<Option<ConversationModelSelection>, CoreError> {
        let _ = id;
        Ok(None)
    }

    /// Read the conversation's stored personality override (#227, Phase 2), if
    /// one has been pinned by `SetConversationPersonality`. Returns `Ok(None)`
    /// when the conversation has no override (the daemon falls back to the
    /// global personality on the next send).
    ///
    /// Default returns `Ok(None)`; the daemon's routing wrapper overrides this
    /// to consult the persistent store, mirroring
    /// [`Self::get_conversation_model_selection`].
    async fn get_conversation_personality(
        &self,
        id: &ConversationId,
    ) -> Result<Option<crate::prompts::PersonalityOverride>, CoreError> {
        let _ = id;
        Ok(None)
    }

    /// Set (or clear) the conversation's personality override (#227). Passing an
    /// empty/all-`None` override clears it (back to global-only). Default is a
    /// no-op; the routing wrapper overrides it to persist through the store.
    async fn set_conversation_personality(
        &self,
        id: &ConversationId,
        personality: crate::prompts::PersonalityOverride,
    ) -> Result<(), CoreError> {
        let _ = (id, personality);
        Ok(())
    }

    async fn delete_conversation(&self, id: &ConversationId) -> Result<(), CoreError>;

    async fn rename_conversation(
        &self,
        id: &ConversationId,
        title: String,
    ) -> Result<(), CoreError>;

    async fn archive_conversation(&self, id: &ConversationId) -> Result<(), CoreError>;

    async fn unarchive_conversation(&self, id: &ConversationId) -> Result<(), CoreError>;

    async fn clear_all_history(&self) -> Result<u32, CoreError>;

    async fn send_prompt(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        on_chunk: ChunkCallback,
        on_status: StatusCallback,
    ) -> Result<String, CoreError>;

    /// Send a prompt with optional per-send model/connection override.
    ///
    /// Resolution priority inside the core service:
    /// 1. `override_selection`, when supplied (validated against the
    ///    registry; invalid selections produce `CoreError::Llm`).
    /// 2. The conversation's stored `last_model_selection`, when it still
    ///    resolves to a live connection + listed model.
    /// 3. The `interactive` purpose from the daemon config.
    ///
    /// `cancellation` is a [`tokio_util::sync::CancellationToken`] that the
    /// core service checks at each cooperative checkpoint inside the
    /// agentic loop (between turns, before each tool-round dispatch) and
    /// that LLM adapters watch via `tokio::select!` in their streaming
    /// loops. Pass [`CancellationToken::new`] for callers that never need
    /// to cancel — that's a fresh, never-tripped token and keeps pre-#109
    /// behaviour intact.
    ///
    /// `system_refinement` is an optional, **request-scoped** addition to
    /// the system prompt for this one turn (empty string = none). When
    /// non-empty it is appended *after* the conversation's normal system
    /// prompt for the LLM call only — it is never stored as a message, never
    /// written to the conversation, and so never appears in chat history or
    /// affects later turns. A voice client uses it to attach instructions
    /// like "respond briefly, by voice" to a turn dictated into an existing
    /// chat without permanently changing that conversation's behaviour. It
    /// is provider-agnostic (just system-prompt text).
    ///
    /// Returns the assistant's full response text plus any advisory
    /// warnings that should be surfaced to the client (for example, a
    /// dangling stored selection that was cleared on this call). Default
    /// implementation ignores overrides and delegates to `send_prompt`,
    /// installing the cancellation token and the system refinement as
    /// task-locals so adapters and the context assembler can read them
    /// without per-method threading; the concrete `ConversationHandler`
    /// overrides this with the full resolution path.
    ///
    // Why allow: this is the per-send dispatch entry point. Its arguments are
    // the conversation target plus three independent per-request inputs
    // (model override, system-prompt refinement) and the streaming/cancel
    // plumbing (chunk + status callbacks, cancellation token). They don't
    // cluster into a meaningful struct, and bundling them solely to satisfy
    // the 7-arg lint would obscure every implementor and call site across the
    // daemon, application, and connector layers.
    #[allow(clippy::too_many_arguments)]
    async fn send_prompt_with_override(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        override_selection: Option<PromptSelectionOverride>,
        system_refinement: String,
        on_chunk: ChunkCallback,
        on_status: StatusCallback,
        cancellation: CancellationToken,
    ) -> Result<PromptDispatchOutcome, CoreError> {
        let _ = override_selection;
        let inner = async move {
            crate::ports::llm::with_system_refinement(
                system_refinement,
                self.send_prompt(conversation_id, prompt, on_chunk, on_status),
            )
            .await
        };
        let text = with_cancellation_token(cancellation, inner).await?;
        Ok(PromptDispatchOutcome {
            response: text,
            warnings: Vec::new(),
        })
    }
}

/// Effort hint passed to connectors and mapped to per-connector request
/// parameters at dispatch time. Defined in `desktop-assistant-protocol`
/// (serde wire format: lowercase `"low"`/`"medium"`/`"high"`); re-exported
/// here so existing `core::ports::inbound::Effort` paths are unchanged (#377).
pub use desktop_assistant_protocol::Effort;

/// Per-send selection override (model/connection/effort). Supplied by API
/// clients via `SendPrompt { override: ... }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSelectionOverride {
    pub connection_id: String,
    pub model_id: String,
    pub effort: Option<Effort>,
}

/// Stored per-conversation model selection. Serialized to JSON in the
/// `conversations.last_model_selection` column.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ConversationModelSelection {
    pub connection_id: String,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
}

/// Advisory attached to a `send_prompt_with_override` result. Returned
/// alongside the response text so adapters can surface a one-time UI hint
/// to the client (e.g. "your previous model selection no longer resolves,
/// falling back to purpose default").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchWarning {
    DanglingModelSelection {
        previous: ConversationModelSelection,
        fallback_to: ConversationModelSelection,
    },
}

/// Successful outcome from `send_prompt_with_override`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptDispatchOutcome {
    pub response: String,
    pub warnings: Vec<DispatchWarning>,
}

/// Availability of a connection in the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionAvailability {
    Ok,
    Unavailable { reason: String },
}

/// Protocol-neutral connection config view for the inbound port.
///
/// Mirrors `crates/daemon/src/connections.rs::ConnectionConfig` but is
/// decoupled so the core crate has no dependency on the daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionConfigPayload {
    Anthropic {
        base_url: Option<String>,
        api_key_env: Option<String>,
        connect_timeout_secs: Option<u64>,
        stream_timeout_secs: Option<u64>,
        max_context_tokens: Option<u64>,
    },
    OpenAi {
        base_url: Option<String>,
        api_key_env: Option<String>,
        connect_timeout_secs: Option<u64>,
        stream_timeout_secs: Option<u64>,
        max_context_tokens: Option<u64>,
    },
    Bedrock {
        aws_profile: Option<String>,
        region: Option<String>,
        base_url: Option<String>,
        connect_timeout_secs: Option<u64>,
        stream_timeout_secs: Option<u64>,
        max_context_tokens: Option<u64>,
    },
    Ollama {
        base_url: Option<String>,
        connect_timeout_secs: Option<u64>,
        stream_timeout_secs: Option<u64>,
        keep_warm: Option<bool>,
        max_context_tokens: Option<u64>,
    },
}

impl ConnectionConfigPayload {
    pub fn connector_type(&self) -> &'static str {
        match self {
            Self::Anthropic { .. } => "anthropic",
            Self::OpenAi { .. } => "openai",
            Self::Bedrock { .. } => "bedrock",
            Self::Ollama { .. } => "ollama",
        }
    }
}

/// A single configured connection, including status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionView {
    pub id: String,
    pub connector_type: String,
    pub display_label: String,
    pub availability: ConnectionAvailability,
    pub has_credentials: bool,
    /// Echoed non-secret config (endpoint/profile/region and the env-var
    /// *name* that holds the credential), so clients can pre-fill an edit
    /// dialog without re-deriving it. Carries the same shape as the
    /// create/update input ([`ConnectionConfigPayload`]); raw secret values
    /// and keyring coordinates are never represented in this type.
    /// `None` only if the connection has no stored config entry.
    pub config: Option<ConnectionConfigPayload>,
}

/// A model enumerated under a connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelListing {
    pub connection_id: String,
    pub connection_label: String,
    pub model: ModelInfo,
}

/// Purpose kind identifiers (`Interactive`/`Dreaming`/`Embedding`/`Titling`).
/// Defined in `desktop-assistant-protocol` (serde wire format: `snake_case`);
/// re-exported here — the canonical `core::ports::inbound::PurposeKind` path —
/// so the daemon and api-model keep consuming it from one place (#43, #377).
pub use desktop_assistant_protocol::PurposeKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurposeConfigPayload {
    /// Either a connection id, or the literal `"primary"` (inherit).
    pub connection: String,
    /// Either a model id, or the literal `"primary"`.
    pub model: String,
    pub effort: Option<Effort>,
    /// Optional per-purpose override for the model's context window in
    /// tokens (issue #51). `None` means "use the connector's curated
    /// table, then a conservative universal fallback."
    pub max_context_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PurposesView {
    pub interactive: Option<PurposeConfigPayload>,
    pub dreaming: Option<PurposeConfigPayload>,
    pub consolidation: Option<PurposeConfigPayload>,
    pub embedding: Option<PurposeConfigPayload>,
    pub titling: Option<PurposeConfigPayload>,
    pub voice: Option<PurposeConfigPayload>,
}

/// Inbound port for connection + purpose management (issue #11).
///
/// Implemented by the daemon against its `ConnectionRegistry` and on-disk
/// config. Adapters (D-Bus, WebSocket) dispatch through this trait so the
/// daemon remains the single source of truth for connection state.
pub trait ConnectionsService: Send + Sync {
    fn list_connections(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<ConnectionView>, CoreError>> + Send;

    fn create_connection(
        &self,
        id: String,
        config: ConnectionConfigPayload,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn update_connection(
        &self,
        id: String,
        config: ConnectionConfigPayload,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn delete_connection(
        &self,
        id: String,
        force: bool,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn list_available_models(
        &self,
        connection_id: Option<String>,
        refresh: bool,
    ) -> impl std::future::Future<Output = Result<Vec<ModelListing>, CoreError>> + Send;

    fn get_purposes(
        &self,
    ) -> impl std::future::Future<Output = Result<PurposesView, CoreError>> + Send;

    fn set_purpose(
        &self,
        purpose: PurposeKind,
        config: PurposeConfigPayload,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;
}

/// Inbound port for assistant settings.
///
/// Secret values are write-only through this interface and never returned.
pub trait SettingsService: Send + Sync {
    fn get_llm_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<LlmSettingsView, CoreError>> + Send;

    // Each argument is a distinct, independently-optional LLM settings field;
    // a bundling struct would just mirror this signature and touch every
    // implementor and call site across layers — an out-of-scope refactor.
    #[allow(clippy::too_many_arguments)]
    fn set_llm_settings(
        &self,
        connector: String,
        model: Option<String>,
        base_url: Option<String>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        max_tokens: Option<u32>,
        hosted_tool_search: Option<bool>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn set_api_key(
        &self,
        api_key: String,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn generate_ws_jwt(
        &self,
        subject: Option<String>,
    ) -> impl std::future::Future<Output = Result<String, CoreError>> + Send;

    fn validate_ws_jwt(
        &self,
        token: String,
    ) -> impl std::future::Future<Output = Result<bool, CoreError>> + Send;

    fn get_embeddings_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<EmbeddingsSettingsView, CoreError>> + Send;

    fn set_embeddings_settings(
        &self,
        connector: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn get_connector_defaults(
        &self,
        connector: String,
    ) -> impl std::future::Future<Output = Result<ConnectorDefaultsView, CoreError>> + Send;

    fn get_persistence_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<PersistenceSettingsView, CoreError>> + Send;

    fn set_persistence_settings(
        &self,
        enabled: bool,
        remote_url: Option<String>,
        remote_name: Option<String>,
        push_on_update: bool,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn get_database_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<DatabaseSettingsView, CoreError>> + Send;

    fn set_database_settings(
        &self,
        url: Option<String>,
        max_connections: u32,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    /// Return the active assistant personality (issue #226).
    ///
    /// Default returns the Expressive-7 [`crate::prompts::Personality::default`]
    /// so test mocks and adapters that don't manage personality opt out without
    /// boilerplate; the daemon's real service overrides it to read the resolved
    /// config.
    fn get_personality_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<PersonalitySettingsView, CoreError>> + Send {
        async { Ok(PersonalitySettingsView::default()) }
    }

    /// Update the active assistant personality (issue #226).
    ///
    /// Default is a no-op (`Ok`) so mocks/adapters that don't manage
    /// personality opt out; the daemon's real service overrides it to persist
    /// the change and refresh the in-memory config used by the next send.
    fn set_personality_settings(
        &self,
        personality: PersonalitySettingsView,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send {
        let _ = personality;
        async { Ok(()) }
    }

    fn get_backend_tasks_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<BackendTasksSettingsView, CoreError>> + Send;

    /// Update backend-tasks settings (LLM override + dreaming config).
    ///
    /// Pass `llm_connector = None` to clear the separate LLM override (revert to primary).
    fn set_backend_tasks_settings(
        &self,
        llm_connector: Option<String>,
        llm_model: Option<String>,
        llm_base_url: Option<String>,
        dreaming_enabled: bool,
        dreaming_interval_secs: u64,
        archive_after_days: u32,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    // MCP server management

    fn list_mcp_servers(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<McpServerView>, CoreError>> + Send;

    fn add_mcp_server(
        &self,
        name: String,
        command: String,
        args: Vec<String>,
        namespace: Option<String>,
        enabled: bool,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn remove_mcp_server(
        &self,
        name: String,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn set_mcp_server_enabled(
        &self,
        name: String,
        enabled: bool,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn mcp_server_action(
        &self,
        action: String,
        server: Option<String>,
    ) -> impl std::future::Future<Output = Result<Vec<McpServerView>, CoreError>> + Send;

    fn get_ws_auth_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<WsAuthSettingsView, CoreError>> + Send;

    fn set_ws_auth_settings(
        &self,
        methods: Vec<String>,
        oidc_issuer: String,
        oidc_auth_endpoint: String,
        oidc_token_endpoint: String,
        oidc_client_id: String,
        oidc_scopes: String,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;
}

/// Inbound port for client-facing knowledge base management (#73).
///
/// Distinct from the LLM-facing [`builtin_knowledge_base_*`][1] tools:
/// adapters dispatch through this trait so client UIs (GTK browser, etc.)
/// can browse, search, edit, and delete entries directly. Writes go
/// through the same chunk-and-embed pipeline as the tool path so
/// client-authored entries remain discoverable by the LLM.
///
/// `search_entries` is full-text only (no embedding round-trip on the
/// client) — the LLM tool keeps the hybrid path.
///
/// [1]: crate::ports::knowledge::KnowledgeBaseStore
pub trait KnowledgeService: Send + Sync {
    fn list_entries(
        &self,
        limit: usize,
        offset: usize,
        tag_filter: Option<Vec<String>>,
    ) -> impl std::future::Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send;

    fn get_entry(
        &self,
        id: String,
    ) -> impl std::future::Future<Output = Result<Option<KnowledgeEntry>, CoreError>> + Send;

    fn search_entries(
        &self,
        query: String,
        tag_filter: Option<Vec<String>>,
        limit: usize,
    ) -> impl std::future::Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send;

    /// Create a new entry. The daemon assigns the id, embeds the
    /// content, and records the embedding model used.
    fn create_entry(
        &self,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> impl std::future::Future<Output = Result<KnowledgeEntry, CoreError>> + Send;

    /// Replace an existing entry's content/tags/metadata. Re-embeds.
    fn update_entry(
        &self,
        id: String,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> impl std::future::Future<Output = Result<KnowledgeEntry, CoreError>> + Send;

    fn delete_entry(
        &self,
        id: String,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;
}

/// On-demand knowledge-maintenance passes, triggered from the knowledge panels
/// (the "dream cycle" controls). These mirror the daemon's periodic background
/// passes so a manual trigger shares the same implementation, configured LLM,
/// and per-op mutual exclusion as the timers.
///
/// Each method is long-running and runs as a tracked background task; the
/// supplied `cancellation` token (the task's `ctx.token`) is observed at batch
/// boundaries and before each external (LLM/embedding) call, so the existing
/// task-cancel command stops a run promptly. Returns a count of work done
/// (facts written / entries changed / rows re-embedded). Implementations reject
/// a concurrent run of the same op with `CoreError`.
///
/// Object-safe (`async_trait`) so the API handler can hold it as an optional
/// `Arc<dyn KnowledgeMaintenanceService>` rather than threading another generic.
#[async_trait::async_trait]
pub trait KnowledgeMaintenanceService: Send + Sync {
    /// Run one extraction pass (scan conversations for new facts + archival).
    async fn run_extraction(&self, cancellation: CancellationToken) -> Result<usize, CoreError>;

    /// Run one holistic consolidation pass over the active knowledge base.
    async fn run_consolidation(&self, cancellation: CancellationToken) -> Result<usize, CoreError>;

    /// Force-recompute embeddings for EVERY active knowledge entry, regardless
    /// of model stamp or freshness (for out-of-band cases). Returns the number
    /// of entries re-embedded.
    async fn recalculate_embeddings(
        &self,
        cancellation: CancellationToken,
    ) -> Result<usize, CoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockAssistant;

    impl AssistantService for MockAssistant {
        fn version(&self) -> &str {
            env!("CARGO_PKG_VERSION")
        }

        fn ping(&self) -> &str {
            "pong"
        }
    }

    #[test]
    fn mock_assistant_returns_version() {
        let assistant = MockAssistant;
        assert!(!assistant.version().is_empty(), "version must not be empty");
    }

    #[test]
    fn mock_assistant_responds_to_ping() {
        let assistant = MockAssistant;
        assert_eq!(assistant.ping(), "pong");
    }

    // ConversationService uses impl Future so not dyn-compatible,
    // but we verify it's implementable via the service tests in service.rs.
    fn _assert_conversation_service<T: ConversationService>() {}
    fn _assert_settings_service<T: SettingsService>() {}
    fn _assert_knowledge_service<T: KnowledgeService>() {}
}
