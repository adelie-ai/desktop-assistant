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

/// A collapsed range of messages replaced by a summary text.
///
/// Why: the range covered by a summary is recovered at render time from
/// the positions of `Message`s whose `summary_id` matches `id`. Storing
/// vec-index ordinals on the summary itself duplicates information already
/// carried by `Message::summary_id` and breaks if any message in the
/// conversation is deleted (the recorded indices would silently drift).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageSummary {
    pub id: String,
    pub summary: String,
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
    /// Rolling summary of messages dropped by context windowing.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub context_summary: String,
    /// Message index up to which compaction has been performed.
    #[serde(default)]
    pub compacted_through: usize,
    /// Collapsed message ranges with their summary text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summaries: Vec<MessageSummary>,
    /// When the conversation was archived (None = active).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived_at: Option<String>,
}

impl Conversation {
    pub fn new(id: impl Into<ConversationId>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            created_at: String::new(),
            updated_at: String::new(),
            messages: Vec::new(),
            context_summary: String::new(),
            compacted_through: 0,
            summaries: Vec::new(),
            archived_at: None,
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
    pub archived: bool,
}

impl From<&Conversation> for ConversationSummary {
    fn from(conv: &Conversation) -> Self {
        Self {
            id: conv.id.clone(),
            title: conv.title.clone(),
            created_at: conv.created_at.clone(),
            updated_at: conv.updated_at.clone(),
            message_count: conv.messages.len(),
            archived: conv.archived_at.is_some(),
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
        assert!(!summary.archived);
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

    #[test]
    fn conversation_deserializes_without_compaction_fields() {
        let json = r#"{"id":"id-1","title":"Chat","messages":[]}"#;
        let conv: Conversation = serde_json::from_str(json).unwrap();
        assert_eq!(conv.context_summary, "");
        assert_eq!(conv.compacted_through, 0);
    }

    #[test]
    fn conversation_serialization_roundtrip_with_compaction() {
        let mut conv = Conversation::new("id-1", "Chat");
        conv.context_summary = "User asked about Rust lifetimes.".to_string();
        conv.compacted_through = 25;
        conv.messages.push(Message::new(Role::User, "test"));

        let json = serde_json::to_string(&conv).unwrap();
        let deserialized: Conversation = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.context_summary,
            "User asked about Rust lifetimes."
        );
        assert_eq!(deserialized.compacted_through, 25);
    }

    #[test]
    fn conversation_skips_empty_context_summary_in_serialization() {
        let conv = Conversation::new("id-1", "Chat");
        let json = serde_json::to_string(&conv).unwrap();
        assert!(!json.contains("context_summary"));
    }

    #[test]
    fn message_summary_tolerates_legacy_ordinal_fields() {
        // Persisted JSON from before the ordinal fields were dropped must
        // still deserialize. Serde tolerates unknown keys by default
        // (no #[serde(deny_unknown_fields)] on the struct).
        let json = r#"{
            "id": "s1",
            "summary": "First batch.",
            "start_ordinal": 1,
            "end_ordinal": 3
        }"#;
        let summary: MessageSummary = serde_json::from_str(json).unwrap();
        assert_eq!(summary.id, "s1");
        assert_eq!(summary.summary, "First batch.");
    }
}
