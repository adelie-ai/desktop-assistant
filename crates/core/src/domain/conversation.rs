use serde::{Deserialize, Serialize};

use super::Message;

/// Opaque identifier for a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationId(pub String);

impl ConversationId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ConversationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ConversationId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// A conversation aggregate containing its messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: ConversationId,
    pub title: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    pub messages: Vec<Message>,
}

impl Conversation {
    pub fn new(id: impl Into<ConversationId>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            created_at: String::new(),
            updated_at: String::new(),
            messages: Vec::new(),
        }
    }
}

/// Lightweight summary for listing conversations.
#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: ConversationId,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: usize,
}

impl From<&Conversation> for ConversationSummary {
    fn from(conv: &Conversation) -> Self {
        Self {
            id: conv.id.clone(),
            title: conv.title.clone(),
            created_at: conv.created_at.clone(),
            updated_at: conv.updated_at.clone(),
            message_count: conv.messages.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Message, Role};

    #[test]
    fn conversation_id_from_string() {
        let id = ConversationId::from("abc-123".to_string());
        assert_eq!(id.as_str(), "abc-123");
    }

    #[test]
    fn conversation_id_from_str() {
        let id = ConversationId::from("abc-123");
        assert_eq!(id.as_str(), "abc-123");
    }

    #[test]
    fn conversation_id_equality() {
        let a = ConversationId::from("same");
        let b = ConversationId::from("same");
        assert_eq!(a, b);
    }

    #[test]
    fn conversation_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ConversationId::from("a"));
        set.insert(ConversationId::from("a"));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn new_conversation_has_empty_messages() {
        let conv = Conversation::new("id-1", "Test Chat");
        assert_eq!(conv.id.as_str(), "id-1");
        assert_eq!(conv.title, "Test Chat");
        assert!(conv.created_at.is_empty());
        assert!(conv.updated_at.is_empty());
        assert!(conv.messages.is_empty());
    }

    #[test]
    fn conversation_summary_from_conversation() {
        let mut conv = Conversation::new("id-1", "Chat");
        conv.messages.push(Message::new(Role::User, "hi"));
        conv.messages.push(Message::new(Role::Assistant, "hello"));

        let summary = ConversationSummary::from(&conv);
        assert_eq!(summary.id.as_str(), "id-1");
        assert_eq!(summary.title, "Chat");
        assert_eq!(summary.created_at, "");
        assert_eq!(summary.updated_at, "");
        assert_eq!(summary.message_count, 2);
    }

    #[test]
    fn conversation_serialization_roundtrip() {
        let mut conv = Conversation::new("id-1", "Chat");
        conv.messages.push(Message::new(Role::User, "test"));
        let json = serde_json::to_string(&conv).unwrap();
        let deserialized: Conversation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, conv.id);
        assert_eq!(deserialized.title, conv.title);
        assert_eq!(deserialized.created_at, conv.created_at);
        assert_eq!(deserialized.updated_at, conv.updated_at);
        assert_eq!(deserialized.messages.len(), 1);
    }

    #[test]
    fn conversation_deserializes_without_timestamps() {
        let json = r#"{"id":"id-1","title":"Chat","messages":[]}"#;
        let conv: Conversation = serde_json::from_str(json).unwrap();
        assert_eq!(conv.created_at, "");
        assert_eq!(conv.updated_at, "");
    }
}
