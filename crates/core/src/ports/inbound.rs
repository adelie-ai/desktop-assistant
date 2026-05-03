use crate::CoreError;
use crate::domain::{Conversation, ConversationId, ConversationSummary, KnowledgeEntry};
use crate::ports::llm::{ChunkCallback, ModelInfo, StatusCallback};

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
pub trait ConversationService: Send + Sync {
    fn create_conversation(
        &self,
        title: String,
    ) -> impl std::future::Future<Output = Result<Conversation, CoreError>> + Send;

    fn list_conversations(
        &self,
        max_age_days: Option<u32>,
        include_archived: bool,
    ) -> impl std::future::Future<Output = Result<Vec<ConversationSummary>, CoreError>> + Send;

    fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<Conversation, CoreError>> + Send;

    /// Read the conversation's currently stored model selection, if one has
    /// been pinned by a prior override. Returns `Ok(None)` when the
    /// conversation has no stored selection (the daemon will fall back to the
    /// `interactive` purpose on the next send).
    ///
    /// The default implementation returns `Ok(None)`; the daemon's routing
    /// wrapper overrides this to consult the persistent selection store.
    fn get_conversation_model_selection(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<Option<ConversationModelSelection>, CoreError>> + Send
    where
        Self: Sync,
    {
        async move {
            let _ = id;
            Ok(None)
        }
    }

    fn delete_conversation(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn rename_conversation(
        &self,
        id: &ConversationId,
        title: String,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn archive_conversation(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn unarchive_conversation(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn clear_all_history(&self)
    -> impl std::future::Future<Output = Result<u32, CoreError>> + Send;

    fn send_prompt(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        on_chunk: ChunkCallback,
        on_status: StatusCallback,
    ) -> impl std::future::Future<Output = Result<String, CoreError>> + Send;

    /// Send a prompt with optional per-send model/connection override.
    ///
    /// Resolution priority inside the core service:
    /// 1. `override_selection`, when supplied (validated against the
    ///    registry; invalid selections produce `CoreError::Llm`).
    /// 2. The conversation's stored `last_model_selection`, when it still
    ///    resolves to a live connection + listed model.
    /// 3. The `interactive` purpose from the daemon config.
    ///
    /// Returns the assistant's full response text plus any advisory
    /// warnings that should be surfaced to the client (for example, a
    /// dangling stored selection that was cleared on this call). Default
    /// implementation ignores overrides and delegates to `send_prompt`; the
    /// concrete `ConversationHandler` overrides this with the full
    /// resolution path.
    fn send_prompt_with_override(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        override_selection: Option<PromptSelectionOverride>,
        on_chunk: ChunkCallback,
        on_status: StatusCallback,
    ) -> impl std::future::Future<Output = Result<PromptDispatchOutcome, CoreError>> + Send
    where
        Self: Sync,
    {
        async move {
            let _ = override_selection;
            let text = self
                .send_prompt(conversation_id, prompt, on_chunk, on_status)
                .await?;
            Ok(PromptDispatchOutcome {
                response: text,
                warnings: Vec::new(),
            })
        }
    }
}

/// Effort hint passed to connectors and mapped to per-connector request
/// parameters at dispatch time.
///
/// Serializes as the lowercase variant name (`"low"`, `"medium"`,
/// `"high"`) for JSON columns and wire payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
}

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
    },
    OpenAi {
        base_url: Option<String>,
        api_key_env: Option<String>,
    },
    Bedrock {
        aws_profile: Option<String>,
        region: Option<String>,
        base_url: Option<String>,
    },
    Ollama {
        base_url: Option<String>,
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
}

/// A model enumerated under a connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelListing {
    pub connection_id: String,
    pub connection_label: String,
    pub model: ModelInfo,
}

/// Purpose kind identifiers — mirrors
/// `crates/daemon/src/purposes.rs::PurposeKind` via string keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PurposeKind {
    Interactive,
    Dreaming,
    Embedding,
    Titling,
}

impl PurposeKind {
    pub fn as_key(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Dreaming => "dreaming",
            Self::Embedding => "embedding",
            Self::Titling => "titling",
        }
    }
}

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
    pub embedding: Option<PurposeConfigPayload>,
    pub titling: Option<PurposeConfigPayload>,
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
