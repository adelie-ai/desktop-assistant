//! Protocol-neutral API model shared by adapters (D-Bus, WebSocket, etc.).
//!
//! This crate intentionally contains only:
//! - serializable command/result/event types
//! - stable IDs and small helper types
//!
//! Business logic belongs in core/application crates.

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
    ClearAllHistory,

    /// Send a content to an existing conversation.
    ///
    /// The response is streamed via [`Event::AssistantDelta`] events.
    SendMessage {
        conversation_id: String,
        content: String,
    },

    // Settings
    GetLlmSettings,
    SetLlmSettings {
        connector: String,
        model: Option<String>,
        base_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        temperature: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        top_p: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_tokens: Option<u32>,
    },
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

    LlmSettings(LlmSettingsView),
    EmbeddingsSettings(EmbeddingsSettingsView),
    ConnectorDefaults(ConnectorDefaultsView),
    PersistenceSettings(PersistenceSettingsView),

    McpServers(Vec<McpServerView>),

    Ack,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    ConfigChanged {
        config: Config,
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Status {
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub llm: LlmSettingsView,
    pub embeddings: EmbeddingsSettingsView,
    pub persistence: PersistenceSettingsView,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ConfigChanges {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_connector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_api_key: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_hosted_tool_search: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub message_count: u32,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationView {
    pub id: String,
    pub title: String,
    pub messages: Vec<MessageView>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hosted_tool_search: Option<bool>,
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
                llm_connector: Some("openai".into()),
                persistence_enabled: Some(true),
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }
}
