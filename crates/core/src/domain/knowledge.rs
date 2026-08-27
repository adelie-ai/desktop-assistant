use serde::{Deserialize, Serialize};

pub use desktop_assistant_protocol::SUMMARY_MAX_CHARS;

/// What consolidation, or a person, has judged a stored claim to be.
///
/// A closed enum rather than a bare string, so an invalid disposition is hard
/// to represent. The database enforces the same six spellings with a CHECK
/// constraint (`knowledge_base_disposition_chk`, migration 056); the storage
/// layer maps [`Self::as_str`] to and from the column, and
/// `disposition_enum_spellings_match_the_schema_check`
/// (`crates/storage/tests`) pins the two vocabularies together so they cannot
/// drift apart silently.
///
/// Disposition is decoupled from deletion: a dispositioned entry is normally
/// still live. Only `deleted_at` says whether a row is in the trash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Disposition {
    /// A live claim, judged nothing else. The default for a new entry, and
    /// what restore sets.
    #[default]
    Active,
    /// Established untrue. `KnowledgeEntry`'s storage-layer counterpart
    /// carries the stated reason in `disposition_reason`. Must never be
    /// rendered as a current fact, but must stay findable when the query is
    /// about its subject - that asymmetry is the point of the value.
    Refuted,
    /// Replaced by a newer statement. The successor's id is recorded
    /// alongside the entry (`superseded_by` in storage); a query that matches
    /// this entry should resolve through the link.
    Superseded,
    /// A duplicate of another entry, which is recorded the same way
    /// `Superseded` records its successor.
    Redundant,
    /// Was true, no longer applies. Excluded from results unless the caller
    /// asks for it.
    Obsolete,
    /// Harmless, not worth surfacing. Ranks below a comparable active entry
    /// rather than being excluded outright.
    Trivial,
}

impl Disposition {
    /// Every value, in the order the database's CHECK constraint lists them.
    pub const ALL: [Disposition; 6] = [
        Self::Active,
        Self::Refuted,
        Self::Superseded,
        Self::Redundant,
        Self::Obsolete,
        Self::Trivial,
    ];

    /// The spelling stored in `knowledge_base.disposition`.
    ///
    /// Stable: it is a value in a database column, so a variant that changed
    /// its spelling here would orphan every row already written under the old
    /// one.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Refuted => "refuted",
            Self::Superseded => "superseded",
            Self::Redundant => "redundant",
            Self::Obsolete => "obsolete",
            Self::Trivial => "trivial",
        }
    }

    /// The disposition a stored spelling names, or `None` for a spelling no
    /// variant claims.
    ///
    /// The database CHECK constraint means a row read back from
    /// `knowledge_base` can never actually carry an unrecognized spelling;
    /// `None` exists for the same reason [`SituationField::parse`]'s does -
    /// so a caller reading a value from somewhere the CHECK does not reach
    /// (a hand-edited row, an older binary's write) degrades to a decision
    /// rather than a panic.
    ///
    /// [`SituationField::parse`]: crate::domain::situation::SituationField::parse
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.as_str() == value)
    }
}

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
    /// What consolidation, or a person, has judged this entry to be.
    /// Defaults to [`Disposition::Active`] - an entry stored before this
    /// field existed, or a read path that does not select it, reads as an
    /// ordinary live claim, which is what it always was.
    #[serde(default)]
    pub disposition: Disposition,
    /// A one-line condensation of what this entry says, for a reader that
    /// shows many entries at once and cannot spend the whole body on each.
    ///
    /// `None` means no summary has been written yet: an entry stored before
    /// the field existed, one whose write named no summary, or one whose
    /// summary was cleared. It is not a definition of the entry's subject the
    /// way a tag's description is - it condenses what this particular entry
    /// says, so it goes stale when the content changes under it.
    ///
    /// Nothing rewrites it on its own. A write that changes the content and
    /// names no summary keeps the old line, deliberately, because the
    /// alternative is wiping one on every partial update. The remedies are a
    /// write that sends a new summary, which the tool schema asks for, and
    /// #1099's pass over the entries that have none.
    ///
    /// On write, `None` preserves any existing value rather than clearing it -
    /// the same rule [`KnowledgeEntry::source`] follows, so a caller that knows
    /// nothing about summaries cannot wipe one. An empty summary is the way to
    /// clear a stored one, and the entry then reads back as `None` again; see
    /// [`crate::ports::knowledge::KnowledgeBaseStore::write`].
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
            disposition: Disposition::Active,
            summary: None,
        }
    }

    /// Builder-style setter for provenance.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// The one line that stands for this entry in a list: the stored
    /// [`summary`](Self::summary) where there is one, otherwise the content.
    /// One physical line, never longer than [`SUMMARY_MAX_CHARS`] characters,
    /// and ending in `...` when it was cut short - see
    /// [`desktop_assistant_protocol::one_line`].
    ///
    /// This is the shared rule for every render site - a client's knowledge
    /// browser, and #1100's recall block once it lands - so the fallback is
    /// decided here instead of once per caller.
    /// `desktop_assistant_api_model::KnowledgeEntryView` carries the same
    /// method for callers that hold the wire type instead.
    ///
    /// Why the fallback lives here and not in the read queries: a
    /// `COALESCE(summary, content)` in SQL would make every read report a
    /// summary for an entry that has none. A pass over the entries that have
    /// none (#1099) finds its work with `WHERE summary IS NULL`, and a list row
    /// that wants to render a stand-in differently from a written summary needs
    /// the same distinction. Both survive only while the read paths stay honest.
    ///
    /// The cap covers a stored summary as well as a fallback body. Nothing in
    /// the schema bounds either, and the budget this protects counts what is
    /// rendered, not where it came from.
    pub fn display_line(&self) -> String {
        let source = self.summary.as_deref().unwrap_or(&self.content);
        desktop_assistant_protocol::one_line(source, SUMMARY_MAX_CHARS)
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
        // Most entries have no summary: every one stored before the field
        // existed, and every write that named none. A render site that skipped
        // them would show almost nothing.
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
    fn display_line_collapses_a_multi_line_body_to_one_physical_line() {
        // The line goes into a line-oriented block, so an embedded newline
        // would break the block's structure rather than the entry's own row.
        let entry = KnowledgeEntry::new(
            "kb-1",
            "First line of the note.\nSecond line.\t\tThird.",
            vec![],
        );

        assert_eq!(
            entry.display_line(),
            "First line of the note. Second line. Third."
        );
    }

    #[test]
    fn display_line_collapses_a_multi_line_stored_summary_too() {
        // Nothing stops a written summary carrying a newline, and it breaks
        // the block just as badly as a body does.
        let mut entry = KnowledgeEntry::new("kb-1", "body", vec![]);
        entry.summary = Some("Prefers dark themes\nin every editor".to_string());

        assert_eq!(entry.display_line(), "Prefers dark themes in every editor");
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
