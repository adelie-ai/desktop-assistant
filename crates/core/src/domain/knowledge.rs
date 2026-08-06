use serde::{Deserialize, Serialize};

/// A unified knowledge base entry, replacing separate preferences and memory stores.
/// Each entry is prose content with tags and optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub created_at: String,
    pub updated_at: String,
    /// First-class provenance: `extraction` | `consolidation` | `explicit`,
    /// or `None` when unknown (legacy rows, or read paths that don't select
    /// it). On write, `None` preserves any existing value rather than clearing
    /// it.
    #[serde(default)]
    pub source: Option<String>,
    /// A one-line condensation of what this entry says, for a reader that
    /// shows many entries at once and cannot spend the whole body on each.
    ///
    /// `None` means no summary has been written yet, which is true of every
    /// entry stored before the field existed. It is not a definition of the
    /// entry's subject the way a tag's description is: it condenses what this
    /// particular entry says, and is rewritten when the content changes.
    ///
    /// On write, `None` preserves any existing value rather than clearing it -
    /// the same rule [`KnowledgeEntry::source`] follows, so a caller that knows
    /// nothing about summaries cannot wipe one.
    #[serde(default)]
    pub summary: Option<String>,
}

impl KnowledgeEntry {
    pub fn new(id: impl Into<String>, content: impl Into<String>, tags: Vec<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            tags,
            metadata: serde_json::json!({}),
            created_at: String::new(),
            updated_at: String::new(),
            source: None,
            summary: None,
        }
    }

    /// Builder-style setter for provenance.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knowledge_entry_creation() {
        let entry = KnowledgeEntry::new(
            "kb-1",
            "User prefers dark mode",
            vec!["preference".to_string()],
        );
        assert_eq!(entry.id, "kb-1");
        assert_eq!(entry.content, "User prefers dark mode");
        assert_eq!(entry.tags, vec!["preference"]);
        assert_eq!(entry.metadata, serde_json::json!({}));
    }

    #[test]
    fn display_line_returns_the_stored_summary_when_there_is_one() {
        let mut entry = KnowledgeEntry::new(
            "kb-1",
            "A long body that a reader should never see in a list row.",
            vec![],
        );
        entry.summary = Some("Prefers dark themes".to_string());

        assert_eq!(entry.display_line(), "Prefers dark themes");
    }

    #[test]
    fn display_line_falls_back_to_the_content_when_there_is_no_summary() {
        // Until a maintenance pass has written summaries, most entries have
        // none. A render site that skipped them would show almost nothing.
        let entry = KnowledgeEntry::new("kb-1", "User prefers dark mode", vec![]);

        assert_eq!(entry.display_line(), "User prefers dark mode");
    }

    #[test]
    fn display_line_marks_a_cut_body_so_it_reads_as_incomplete() {
        let entry = KnowledgeEntry::new("kb-1", "x".repeat(SUMMARY_MAX_CHARS + 1), vec![]);

        let line = entry.display_line();

        assert!(
            line.ends_with("..."),
            "a body cut short must say so: {line}"
        );
    }

    #[test]
    fn display_line_leaves_a_short_body_unmarked() {
        // The marker means "there is more". A body that fits carries none, so
        // it cannot be mistaken for a cut one.
        let entry = KnowledgeEntry::new("kb-1", "User prefers dark mode", vec![]);

        assert!(!entry.display_line().ends_with("..."));
    }

    #[test]
    fn display_line_never_exceeds_the_cap() {
        // The bound is what keeps one long entry from spending a whole recall
        // budget on its own. It covers the marker too, because the budget
        // counts what is rendered.
        let from_content = KnowledgeEntry::new("kb-1", "x".repeat(10_000), vec![]);
        assert!(from_content.display_line().chars().count() <= SUMMARY_MAX_CHARS);

        let mut from_summary = KnowledgeEntry::new("kb-2", "short", vec![]);
        from_summary.summary = Some("y".repeat(10_000));
        assert!(from_summary.display_line().chars().count() <= SUMMARY_MAX_CHARS);
    }

    #[test]
    fn display_line_truncates_multibyte_content_on_a_character_boundary() {
        // Cutting a UTF-8 character in half panics on a byte-indexed slice, so
        // the cap counts characters. Each of these is three bytes, which puts
        // the naive byte cut inside a character.
        let entry = KnowledgeEntry::new("kb-1", "\u{4e16}".repeat(SUMMARY_MAX_CHARS * 2), vec![]);

        let line = entry.display_line();

        assert!(line.chars().count() <= SUMMARY_MAX_CHARS);
        assert!(line.starts_with('\u{4e16}'));
        assert!(line.ends_with("..."));
    }

    #[test]
    fn display_line_is_empty_for_an_empty_entry() {
        let entry = KnowledgeEntry::new("kb-1", "", vec![]);

        assert_eq!(entry.display_line(), "");
    }

    #[test]
    fn knowledge_entry_deserializes_without_a_summary() {
        // Entries serialized before `summary` existed carry no such key at
        // all. They must still read back, with the field reported as absent
        // rather than failing the whole payload.
        let older = r#"{
            "id": "kb-1",
            "content": "User prefers dark mode",
            "tags": ["preference"],
            "metadata": {},
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00"
        }"#;

        let entry: KnowledgeEntry =
            serde_json::from_str(older).expect("a payload without a summary still deserializes");

        assert_eq!(entry.id, "kb-1");
        assert_eq!(entry.summary, None);
    }

    #[test]
    fn knowledge_entry_serialization_roundtrip() {
        let mut entry = KnowledgeEntry::new("kb-1", "test content", vec!["tag1".to_string()]);
        entry.metadata = serde_json::json!({"key": "editor", "scope": "global"});
        entry.created_at = "2024-01-01 00:00:00".to_string();
        entry.updated_at = "2024-01-01 00:00:00".to_string();

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: KnowledgeEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, entry.id);
        assert_eq!(deserialized.content, entry.content);
        assert_eq!(deserialized.tags, entry.tags);
        assert_eq!(deserialized.metadata, entry.metadata);
    }
}
