//! Protocol-neutral API model shared by adapters (D-Bus, WebSocket, etc.).
//!
//! This crate intentionally contains only:
//! - serializable command/result/event types
//! - stable IDs and small helper types
//!
//! Business logic belongs in core/application crates.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Ping,

    GetStatus,

    // Canonical transport-level config API
    GetConfig,
    SetConfig {
        changes: ConfigChanges,
    },

    // Conversations
    CreateConversation {
        title: String,
    },
    ListConversations {
        max_age_days: Option<u32>,
        #[serde(default)]
        include_archived: bool,
    },
    GetConversation {
        id: String,
    },
    DeleteConversation {
        id: String,
    },
    RenameConversation {
        id: String,
        title: String,
    },
    ArchiveConversation {
        id: String,
    },
    UnarchiveConversation {
        id: String,
    },
    ClearAllHistory,

    /// Send a content to an existing conversation.
    ///
    /// The response is streamed via [`Event::AssistantDelta`] events.
    /// An optional `override` selects a specific connection/model/effort for
    /// this send only; when omitted, the server falls back to (in order) the
    /// conversation's last selection and the `interactive` purpose.
    SendMessage {
        conversation_id: String,
        content: String,
        #[serde(default, rename = "override", skip_serializing_if = "Option::is_none")]
        override_selection: Option<SendPromptOverride>,
    },

    // Settings (legacy `[llm]`-block single-connection surface).
    //
    // The legacy `SetLlmSettings` / `GetLlmSettings` commands have been
    // removed; use the named-connection commands below (`ListConnections`,
    // `CreateConnection`, `UpdateConnection`, `DeleteConnection`,
    // `GetPurposes`, `SetPurpose`) instead.
    SetApiKey {
        api_key: String,
    },

    GetEmbeddingsSettings,
    SetEmbeddingsSettings {
        connector: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    },

    GetConnectorDefaults {
        connector: String,
    },

    GetPersistenceSettings,
    SetPersistenceSettings {
        enabled: bool,
        remote_url: Option<String>,
        remote_name: Option<String>,
        push_on_update: bool,
    },

    // Named connections (issue #11).
    /// Enumerate every configured connection with its availability and
    /// whether credentials are present.
    ListConnections,
    /// Create a new named connection; fails on invalid slug or duplicate id.
    CreateConnection {
        id: String,
        config: ConnectionConfigView,
    },
    /// Replace an existing connection in-place.
    UpdateConnection {
        id: String,
        config: ConnectionConfigView,
    },
    /// Delete a named connection. Refuses with an error when the connection
    /// is referenced by any purpose unless `force` is true, in which case
    /// referencing purposes fall back to the `interactive` purpose.
    DeleteConnection {
        id: String,
        #[serde(default)]
        force: bool,
    },
    /// Enumerate models across one or all configured connections. When
    /// `connection_id` is `None`, aggregates models from every healthy
    /// connection. `refresh=true` bypasses connector caches (e.g. Bedrock).
    ListAvailableModels {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        connection_id: Option<String>,
        #[serde(default)]
        refresh: bool,
    },

    // Purposes (issue #10 + #11).
    GetPurposes,
    SetPurpose {
        purpose: PurposeKindApi,
        config: PurposeConfigView,
    },

    // Knowledge base management (issue #73).
    ListKnowledgeEntries {
        #[serde(default = "default_kb_limit")]
        limit: u32,
        #[serde(default)]
        offset: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tag_filter: Option<Vec<String>>,
    },
    GetKnowledgeEntry {
        id: String,
    },
    SearchKnowledgeEntries {
        query: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tag_filter: Option<Vec<String>>,
        #[serde(default = "default_kb_limit")]
        limit: u32,
    },
    CreateKnowledgeEntry {
        content: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        metadata: serde_json::Value,
    },
    UpdateKnowledgeEntry {
        id: String,
        content: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        metadata: serde_json::Value,
    },
    DeleteKnowledgeEntry {
        id: String,
    },

    // MCP server management
    ListMcpServers,
    AddMcpServer {
        name: String,
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        #[serde(default = "default_true")]
        enabled: bool,
    },
    RemoveMcpServer {
        name: String,
    },
    SetMcpServerEnabled {
        name: String,
        enabled: bool,
    },
    McpServerAction {
        action: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        server: Option<String>,
    },
}

fn default_true() -> bool {
    true
}

fn default_kb_limit() -> u32 {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CommandResult {
    Pong { value: String },

    Status(Status),
    Config(Config),

    ConversationId { id: String },
    Conversations(Vec<ConversationSummary>),
    Conversation(ConversationView),
    Messages(MessagesView),
    Cleared { deleted_count: u32 },

    EmbeddingsSettings(EmbeddingsSettingsView),
    ConnectorDefaults(ConnectorDefaultsView),
    PersistenceSettings(PersistenceSettingsView),

    McpServers(Vec<McpServerView>),

    Connections(Vec<ConnectionView>),
    Models(Vec<ModelListing>),
    Purposes(PurposesView),

    KnowledgeEntries(Vec<KnowledgeEntryView>),
    KnowledgeEntry(Option<KnowledgeEntryView>),
    KnowledgeEntryWritten(KnowledgeEntryView),

    Ack,
}

/// Wire-format view of a knowledge base entry. Mirrors
/// `desktop_assistant_core::domain::KnowledgeEntry` but lives here so
/// transports and clients depend only on `api-model`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeEntryView {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    ConfigChanged {
        config: Config,
    },

    /// Progress status while the assistant is working (tool calls, searches, etc.).
    /// Displayed as transient "working..." indicators, not as chat messages.
    AssistantStatus {
        conversation_id: String,
        request_id: String,
        message: String,
    },

    /// Streaming chunk for a content response.
    AssistantDelta {
        conversation_id: String,
        request_id: String,
        chunk: String,
    },

    /// Full response (terminal event).
    AssistantCompleted {
        conversation_id: String,
        request_id: String,
        full_response: String,
    },

    /// Streaming failure (terminal event).
    AssistantError {
        conversation_id: String,
        request_id: String,
        error: String,
    },

    /// The title of a conversation was changed (e.g. LLM-generated after first message).
    ConversationTitleChanged {
        conversation_id: String,
        title: String,
    },

    /// A one-time advisory for a conversation (e.g. the stored model
    /// selection no longer resolves and was cleared). Emitted at most once
    /// per underlying condition — the server clears the stored state so
    /// the warning does not recur.
    ConversationWarningEmitted {
        conversation_id: String,
        warning: ConversationWarning,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Status {
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub embeddings: EmbeddingsSettingsView,
    pub persistence: PersistenceSettingsView,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ConfigChanges {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings_connector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_remote_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_remote_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_push_on_update: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub message_count: u32,
    pub updated_at: String,
    #[serde(default)]
    pub archived: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationView {
    pub id: String,
    pub title: String,
    pub messages: Vec<MessageView>,
    /// One-time advisories surfaced after `GetConversation` — e.g. the
    /// conversation's last model selection no longer resolves and was cleared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ConversationWarning>,
    /// The conversation's currently stored (connection, model, effort)
    /// selection, when one has been pinned by a prior `SendMessage` override.
    /// `None` means the daemon will fall back to the `interactive` purpose on
    /// the next send. Cleared automatically when the previous selection no
    /// longer resolves (see `ConversationWarning::DanglingModelSelection`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_selection: Option<ConversationModelSelectionView>,
}

/// Advisory conditions attached to a conversation view. Modeled as an enum
/// so additional variants can be added without breaking existing clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationWarning {
    /// The conversation's previous model selection no longer resolves
    /// (connection removed or model not listed by the connector). The
    /// selection has been cleared and the server fell back to the
    /// `fallback_to` target.
    DanglingModelSelection {
        previous_selection: ConversationModelSelectionView,
        fallback_to: ConversationModelSelectionView,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageView {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessagesView {
    pub total_raw_count: u32,
    pub truncated: bool,
    pub messages: Vec<MessageView>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingsSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
    pub available: bool,
    pub is_default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectorDefaultsView {
    pub llm_model: String,
    pub llm_base_url: String,
    pub backend_llm_model: String,
    pub embeddings_model: String,
    pub embeddings_base_url: String,
    pub embeddings_available: bool,
    pub hosted_tool_search_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistenceSettingsView {
    pub enabled: bool,
    /// Empty string means no remote is configured.
    pub remote_url: String,
    pub remote_name: String,
    pub push_on_update: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerView {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub enabled: bool,
    /// "running" | "stopped" | "disabled"
    pub status: String,
    pub tool_count: u32,
}

// --- Named-connection views (#11) ------------------------------------------

/// Opaque, protocol-neutral representation of a connection config.
///
/// This mirrors the daemon's internal `ConnectionConfig` (one variant per
/// connector type) but lives here so clients don't need to depend on the
/// daemon crate. Credentials are represented as `has_credentials` booleans
/// on the view; raw secret values are never serialized back through the API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum ConnectionConfigView {
    Anthropic {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key_env: Option<String>,
    },
    #[serde(rename = "openai")]
    OpenAi {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key_env: Option<String>,
    },
    Bedrock {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aws_profile: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
    },
    Ollama {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
    },
}

impl ConnectionConfigView {
    /// Short connector-type identifier (matches the `type =` tag).
    pub fn connector_type(&self) -> &'static str {
        match self {
            Self::Anthropic { .. } => "anthropic",
            Self::OpenAi { .. } => "openai",
            Self::Bedrock { .. } => "bedrock",
            Self::Ollama { .. } => "ollama",
        }
    }
}

/// Availability of a connection in the registry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ConnectionAvailability {
    Ok,
    Unavailable { reason: String },
}

/// Aggregate view of a single connection for the connections list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectionView {
    pub id: String,
    /// Short connector-type identifier (`"openai"`, `"anthropic"`, etc.).
    pub connector_type: String,
    /// Human-friendly label; defaults to `"<id> (<connector_type>)"` but
    /// daemons can synthesize a more descriptive value.
    pub display_label: String,
    pub availability: ConnectionAvailability,
    /// True when credentials could be resolved during the most recent sanity
    /// check (env var present, keyring lookup succeeded, or Bedrock/Ollama
    /// which auth via ambient credentials / none).
    pub has_credentials: bool,
}

/// A single model enumerated across one or all connections. Mirrors the
/// core `ModelInfo` fields while tagging the connection it came from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelListing {
    pub connection_id: String,
    pub connection_label: String,
    pub model: ModelInfoView,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfoView {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_limit: Option<u64>,
    #[serde(default)]
    pub capabilities: ModelCapabilitiesView,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCapabilitiesView {
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub embedding: bool,
}

// --- Purpose views (#10 + #11) --------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PurposeKindApi {
    Interactive,
    Dreaming,
    Embedding,
    Titling,
}

impl PurposeKindApi {
    pub fn as_key(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Dreaming => "dreaming",
            Self::Embedding => "embedding",
            Self::Titling => "titling",
        }
    }
}

/// Effort hint passed to connectors (mapped at dispatch time).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EffortLevel {
    Low,
    Medium,
    High,
}

/// Protocol-neutral purpose config. String `"primary"` in the connection or
/// model field means "inherit from interactive" — the daemon resolves this
/// before dispatch (see `crates/daemon/src/purposes.rs`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PurposeConfigView {
    /// Either a connection id (slug) or the literal string `"primary"`.
    pub connection: String,
    /// Either a model id or the literal string `"primary"`.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortLevel>,
    /// Optional per-purpose override for the model's context window in
    /// tokens (issue #51). When omitted, the daemon consults the
    /// connector's curated table and a conservative universal fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// Aggregate purpose view. Missing entries mean the purpose is not
/// configured (the daemon falls back to the primary LLM for those).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PurposesView {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interactive: Option<PurposeConfigView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dreaming: Option<PurposeConfigView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<PurposeConfigView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub titling: Option<PurposeConfigView>,
}

impl PurposesView {
    /// Convenience: convert into a BTreeMap keyed by purpose key for clients
    /// that prefer iteration.
    pub fn to_map(&self) -> BTreeMap<String, PurposeConfigView> {
        let mut map = BTreeMap::new();
        if let Some(v) = &self.interactive {
            map.insert("interactive".to_string(), v.clone());
        }
        if let Some(v) = &self.dreaming {
            map.insert("dreaming".to_string(), v.clone());
        }
        if let Some(v) = &self.embedding {
            map.insert("embedding".to_string(), v.clone());
        }
        if let Some(v) = &self.titling {
            map.insert("titling".to_string(), v.clone());
        }
        map
    }
}

// --- Per-send model override (#11) ----------------------------------------

/// Caller-supplied override for a single `SendMessage` (or `SendPrompt`)
/// call. The daemon validates that `connection_id` is live and that the
/// connector lists `model_id`; otherwise the request is rejected with a
/// 400-style error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SendPromptOverride {
    pub connection_id: String,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortLevel>,
}

/// View of a conversation's stored model selection. Same shape as
/// [`SendPromptOverride`] minus the "this is a request" framing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationModelSelectionView {
    pub connection_id: String,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortLevel>,
}

/// WebSocket request envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WsRequest {
    pub id: String,
    pub command: Command,
}

/// WebSocket frames sent from server to client.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WsFrame {
    Result {
        id: String,
        result: CommandResult,
    },
    Error {
        id: String,
        error: String,
    },
    Event {
        event: Event,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_json_roundtrip_ping() {
        let cmd = Command::Ping;
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn event_json_roundtrip_message_delta() {
        let ev = Event::AssistantDelta {
            conversation_id: "c1".into(),
            request_id: "r1".into(),
            chunk: "hello".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn command_json_roundtrip_set_config() {
        let cmd = Command::SetConfig {
            changes: ConfigChanges {
                persistence_enabled: Some(true),
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn list_connections_roundtrip() {
        let cmd = Command::ListConnections;
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("list_connections"));
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn create_connection_roundtrip_openai() {
        let cmd = Command::CreateConnection {
            id: "work".into(),
            config: ConnectionConfigView::OpenAi {
                base_url: Some("https://api.openai.com/v1".into()),
                api_key_env: Some("OPENAI_WORK_KEY".into()),
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn connection_config_view_tagged_type() {
        let c = ConnectionConfigView::Bedrock {
            aws_profile: Some("work".into()),
            region: Some("us-west-2".into()),
            base_url: None,
        };
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("\"type\":\"bedrock\""));
        assert_eq!(c.connector_type(), "bedrock");
        let back: ConnectionConfigView = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn delete_connection_force_flag() {
        let cmd = Command::DeleteConnection {
            id: "old".into(),
            force: true,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);

        // Missing force flag defaults to false.
        let cmd2: Command =
            serde_json::from_str(r#"{"delete_connection":{"id":"old"}}"#).unwrap();
        assert_eq!(
            cmd2,
            Command::DeleteConnection {
                id: "old".into(),
                force: false,
            }
        );
    }

    #[test]
    fn list_available_models_optional_connection_and_refresh() {
        // Both fields omitted.
        let cmd: Command = serde_json::from_str(r#"{"list_available_models":{}}"#).unwrap();
        assert_eq!(
            cmd,
            Command::ListAvailableModels {
                connection_id: None,
                refresh: false,
            }
        );

        // All fields present.
        let cmd2 = Command::ListAvailableModels {
            connection_id: Some("aws".into()),
            refresh: true,
        };
        let json = serde_json::to_string(&cmd2).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd2, back);
    }

    #[test]
    fn set_purpose_roundtrip() {
        let cmd = Command::SetPurpose {
            purpose: PurposeKindApi::Dreaming,
            config: PurposeConfigView {
                connection: "primary".into(),
                model: "primary".into(),
                effort: Some(EffortLevel::Low),
                max_context_tokens: None,
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn set_purpose_view_carries_max_context_tokens() {
        // Issue #51: the wire type carries the user's per-purpose
        // `max_context_tokens` override end-to-end so the KCM can read
        // and write it.
        let cfg = PurposeConfigView {
            connection: "work_bedrock".into(),
            model: "us.amazon.nova-premier-v1:0".into(),
            effort: Some(EffortLevel::Medium),
            max_context_tokens: Some(1_000_000),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("max_context_tokens"));
        let back: PurposeConfigView = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);

        // Round-trip with `None` must omit the field on the wire.
        let cfg_none = PurposeConfigView {
            connection: "work".into(),
            model: "gpt-5".into(),
            effort: None,
            max_context_tokens: None,
        };
        let json_none = serde_json::to_string(&cfg_none).unwrap();
        assert!(!json_none.contains("max_context_tokens"));
    }

    #[test]
    fn send_message_override_is_optional() {
        // Without override.
        let cmd: Command = serde_json::from_str(
            r#"{"send_message":{"conversation_id":"c1","content":"hi"}}"#,
        )
        .unwrap();
        match &cmd {
            Command::SendMessage {
                override_selection, ..
            } => assert!(override_selection.is_none()),
            other => panic!("unexpected {other:?}"),
        }

        // With override.
        let cmd2 = Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hi".into(),
            override_selection: Some(SendPromptOverride {
                connection_id: "aws".into(),
                model_id: "claude-sonnet-4".into(),
                effort: Some(EffortLevel::High),
            }),
        };
        let json = serde_json::to_string(&cmd2).unwrap();
        assert!(json.contains("\"override\":"));
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd2, back);
    }

    #[test]
    fn effort_serialize_lowercase() {
        assert_eq!(serde_json::to_string(&EffortLevel::Low).unwrap(), "\"low\"");
        assert_eq!(
            serde_json::to_string(&EffortLevel::Medium).unwrap(),
            "\"medium\""
        );
        assert_eq!(
            serde_json::to_string(&EffortLevel::High).unwrap(),
            "\"high\""
        );
    }

    #[test]
    fn conversation_view_warnings_default_empty() {
        let json = r#"{"id":"c1","title":"t","messages":[]}"#;
        let v: ConversationView = serde_json::from_str(json).unwrap();
        assert!(v.warnings.is_empty());
    }

    #[test]
    fn conversation_warning_dangling_selection_roundtrip() {
        let w = ConversationWarning::DanglingModelSelection {
            previous_selection: ConversationModelSelectionView {
                connection_id: "old".into(),
                model_id: "gone".into(),
                effort: None,
            },
            fallback_to: ConversationModelSelectionView {
                connection_id: "work".into(),
                model_id: "gpt-5".into(),
                effort: Some(EffortLevel::Medium),
            },
        };
        let json = serde_json::to_string(&w).unwrap();
        assert!(json.contains("\"type\":\"dangling_model_selection\""));
        let back: ConversationWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(w, back);
    }

    #[test]
    fn connection_availability_tagged() {
        let ok = ConnectionAvailability::Ok;
        let json = serde_json::to_string(&ok).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        let un = ConnectionAvailability::Unavailable {
            reason: "x".into(),
        };
        let json2 = serde_json::to_string(&un).unwrap();
        assert!(json2.contains("\"status\":\"unavailable\""));
        let back: ConnectionAvailability = serde_json::from_str(&json2).unwrap();
        assert_eq!(un, back);
    }
}
