//! Protocol-neutral API model shared by adapters (D-Bus, WebSocket, etc.).
//!
//! This crate intentionally contains only:
//! - serializable command/result/event types
//! - stable IDs and small helper types
//!
//! Business logic belongs in core/application crates.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Ping,

    GetStatus,

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
    ClearAllHistory,

    /// Send a prompt to an existing conversation.
    ///
    /// The response is streamed via [`Event::MessageDelta`] events.
    SendPrompt {
        conversation_id: String,
        prompt: String,
    },

    // Settings
    GetLlmSettings,
    SetLlmSettings {
        connector: String,
        model: Option<String>,
        base_url: Option<String>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommandResult {
    Pong { value: String },

    Status(Status),

    ConversationId { id: String },
    Conversations(Vec<ConversationSummary>),
    Conversation(ConversationView),
    Messages(MessagesView),
    Cleared { deleted_count: u32 },

    LlmSettings(LlmSettingsView),
    EmbeddingsSettings(EmbeddingsSettingsView),
    ConnectorDefaults(ConnectorDefaultsView),
    PersistenceSettings(PersistenceSettingsView),

    Ack,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// Streaming chunk for a prompt response.
    MessageDelta {
        conversation_id: String,
        request_id: String,
        chunk: String,
    },

    /// Full response (terminal event).
    MessageCompleted {
        conversation_id: String,
        request_id: String,
        full_response: String,
    },

    /// Streaming failure (terminal event).
    MessageError {
        conversation_id: String,
        request_id: String,
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Status {
    pub version: String,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistenceSettingsView {
    pub enabled: bool,
    /// Empty string means no remote is configured.
    pub remote_url: String,
    pub remote_name: String,
    pub push_on_update: bool,
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
        let ev = Event::MessageDelta {
            conversation_id: "c1".into(),
            request_id: "r1".into(),
            chunk: "hello".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }
}
