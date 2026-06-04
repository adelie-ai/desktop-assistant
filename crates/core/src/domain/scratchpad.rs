use serde::{Deserialize, Serialize};

/// A single ephemeral note in a conversation's scratchpad.
///
/// The scratchpad is a small, per-conversation working store the assistant
/// manages itself — distinct from the durable [`crate::domain::KnowledgeEntry`]
/// knowledge base. Each note is addressed by a short `key` within its
/// conversation; writing the same key again replaces the note's content.
/// Notes are linked to the conversation and discarded when it is deleted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScratchpadNote {
    pub id: String,
    pub conversation_id: String,
    pub key: String,
    pub content: String,
    pub created_at: String,
    pub updated_at: String,
}

impl ScratchpadNote {
    /// Construct a note. `created_at` / `updated_at` are left empty for the
    /// storage layer to populate from the database clock on write.
    pub fn new(
        id: impl Into<String>,
        conversation_id: impl Into<String>,
        key: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            conversation_id: conversation_id.into(),
            key: key.into(),
            content: content.into(),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratchpad_note_creation() {
        let note = ScratchpadNote::new("sp-1", "conv-1", "goal", "Ship the scratchpad feature");
        assert_eq!(note.id, "sp-1");
        assert_eq!(note.conversation_id, "conv-1");
        assert_eq!(note.key, "goal");
        assert_eq!(note.content, "Ship the scratchpad feature");
        assert!(note.created_at.is_empty());
        assert!(note.updated_at.is_empty());
    }

    #[test]
    fn scratchpad_note_serialization_roundtrip() {
        let mut note = ScratchpadNote::new("sp-1", "conv-1", "open-questions", "Q: which db?");
        note.created_at = "2026-06-03 00:00:00".to_string();
        note.updated_at = "2026-06-03 00:00:00".to_string();

        let json = serde_json::to_string(&note).unwrap();
        let deserialized: ScratchpadNote = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, note);
    }
}
