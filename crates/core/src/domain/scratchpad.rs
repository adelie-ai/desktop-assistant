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
    /// Materialized subagent-tree path that owns this note (e.g. `"1.1"`);
    /// the root sentinel `""` is the top-level session's own notes (#287).
    /// Storage stamps it from the writer's task-local scope; it is the axis
    /// scratchpad namespacing, snapshot reads, and subtree cleanup key on.
    pub owner_todo: String,
    pub key: String,
    pub content: String,
    /// Free-text category the assistant assigns to organise its notes —
    /// suggested values are `todo` / `note` / `other`, but it is not
    /// constrained so the assistant may invent its own. Defaults to `note`.
    /// Mainly exists so notes can be filtered/grouped (e.g. an ordered plan
    /// of `todo`s) without affecting the keyed-upsert semantics.
    pub note_type: String,
    /// Optional ordering hint, sorted ascending **within a `note_type`**
    /// (nulls last). Lets the assistant keep a sequenced plan of `todo`s.
    pub sequence: Option<i32>,
    /// Whether the assistant (or the user, via a client) has checked this
    /// note off. Orthogonal to `note_type`; a checked-off `todo` stays
    /// visible so completed work isn't redone.
    pub done: bool,
    pub created_at: String,
    pub updated_at: String,
}

/// Default `note_type` when a writer doesn't specify one.
pub const DEFAULT_NOTE_TYPE: &str = "note";

impl ScratchpadNote {
    /// Construct a `note`-typed, unsequenced, not-done note. `created_at` /
    /// `updated_at` are left empty for the storage layer to populate from the
    /// database clock on write. Use the field setters (or struct update
    /// syntax) for `note_type` / `sequence` / `done`.
    pub fn new(
        id: impl Into<String>,
        conversation_id: impl Into<String>,
        key: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            conversation_id: conversation_id.into(),
            owner_todo: String::new(),
            key: key.into(),
            content: content.into(),
            note_type: DEFAULT_NOTE_TYPE.to_string(),
            sequence: None,
            done: false,
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
        // New fields default to a plain, unsequenced, not-done, root note.
        assert_eq!(note.note_type, DEFAULT_NOTE_TYPE);
        assert_eq!(note.sequence, None);
        assert!(!note.done);
        assert_eq!(
            note.owner_todo, "",
            "new notes default to the root namespace"
        );
        assert!(note.created_at.is_empty());
        assert!(note.updated_at.is_empty());
    }

    #[test]
    fn scratchpad_note_serialization_roundtrip() {
        let mut note = ScratchpadNote::new("sp-1", "conv-1", "step-1", "wire the migration");
        note.owner_todo = "1.1".to_string();
        note.note_type = "todo".to_string();
        note.sequence = Some(2);
        note.done = true;
        note.created_at = "2026-06-03 00:00:00".to_string();
        note.updated_at = "2026-06-03 00:00:00".to_string();

        let json = serde_json::to_string(&note).unwrap();
        let deserialized: ScratchpadNote = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, note);
        assert_eq!(deserialized.owner_todo, "1.1");
        assert_eq!(deserialized.note_type, "todo");
        assert_eq!(deserialized.sequence, Some(2));
        assert!(deserialized.done);
    }
}
