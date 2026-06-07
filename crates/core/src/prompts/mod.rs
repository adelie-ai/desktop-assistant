/// Semantic kinds for system prompt sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSectionKind {
    // Static (loaded from embedded text files):
    Identity,
    SafetyAndPlanning,
    KnowledgeBase,
    Scratchpad,
    Database,
    Learning,
    ToolUse,
    // Dynamic (built per-turn):
    ToolAvailability,
    ContextSummary,
    MessageSummary,
    /// Per-request, client-supplied addition to the system prompt for a
    /// single turn (e.g. a voice client's "respond briefly, by voice").
    /// Appended last so it can refine/override the static guidance above.
    /// Never persisted; see `crate::ports::llm::SYSTEM_REFINEMENT`.
    SystemRefinement,
}

/// A single section of the system prompt.
#[derive(Debug, Clone)]
pub struct PromptSection {
    pub kind: PromptSectionKind,
    pub content: String,
}

impl PromptSection {
    pub fn new(kind: PromptSectionKind, content: impl Into<String>) -> Self {
        Self {
            kind,
            content: content.into(),
        }
    }
}

const SECTION_IDENTITY: &str = include_str!("sections/identity.txt");
const SECTION_SAFETY_AND_PLANNING: &str = include_str!("sections/safety_and_planning.txt");
const SECTION_KNOWLEDGE_BASE: &str = include_str!("sections/knowledge_base.txt");
const SECTION_SCRATCHPAD: &str = include_str!("sections/scratchpad.txt");
const SECTION_DATABASE: &str = include_str!("sections/database.txt");
const SECTION_LEARNING: &str = include_str!("sections/learning.txt");
const SECTION_TOOL_USE: &str = include_str!("sections/tool_use.txt");

/// Return the static (file-based) prompt sections in order.
pub fn static_sections() -> Vec<PromptSection> {
    vec![
        PromptSection::new(PromptSectionKind::Identity, SECTION_IDENTITY),
        PromptSection::new(
            PromptSectionKind::SafetyAndPlanning,
            SECTION_SAFETY_AND_PLANNING,
        ),
        PromptSection::new(PromptSectionKind::KnowledgeBase, SECTION_KNOWLEDGE_BASE),
        PromptSection::new(PromptSectionKind::Scratchpad, SECTION_SCRATCHPAD),
        PromptSection::new(PromptSectionKind::Database, SECTION_DATABASE),
        PromptSection::new(PromptSectionKind::Learning, SECTION_LEARNING),
        PromptSection::new(PromptSectionKind::ToolUse, SECTION_TOOL_USE),
    ]
}

/// Assemble sections into a single string, joining with double newlines.
pub fn assemble(sections: &[PromptSection]) -> String {
    sections
        .iter()
        .map(|s| s.content.trim_end_matches('\n'))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIGINAL_MONOLITHIC: &str = include_str!("runtime_system_instruction.txt");

    #[test]
    fn assembled_static_sections_match_original() {
        let sections = static_sections();
        let assembled = assemble(&sections);
        assert_eq!(
            assembled, ORIGINAL_MONOLITHIC,
            "assembled sections must exactly match the original monolithic prompt"
        );
    }

    #[test]
    fn static_sections_count() {
        assert_eq!(static_sections().len(), 7);
    }

    #[test]
    fn static_sections_kinds() {
        let sections = static_sections();
        assert_eq!(sections[0].kind, PromptSectionKind::Identity);
        assert_eq!(sections[1].kind, PromptSectionKind::SafetyAndPlanning);
        assert_eq!(sections[2].kind, PromptSectionKind::KnowledgeBase);
        assert_eq!(sections[3].kind, PromptSectionKind::Scratchpad);
        assert_eq!(sections[4].kind, PromptSectionKind::Database);
        assert_eq!(sections[5].kind, PromptSectionKind::Learning);
        assert_eq!(sections[6].kind, PromptSectionKind::ToolUse);
    }

    #[test]
    fn assembled_prompt_advertises_scratchpad_tools() {
        // The scratchpad must be advertised in the always-present system prompt
        // so the model knows the tools exist (#184).
        let assembled = assemble(&static_sections());
        assert!(assembled.contains("== Scratchpad =="));
        assert!(assembled.contains("builtin_scratchpad_write"));
        assert!(assembled.contains("builtin_scratchpad_search"));
        assert!(assembled.contains("builtin_scratchpad_delete"));
        // The reserved goal note must be called out.
        assert!(assembled.contains("\"goal\""));
    }

    // --- Personality (#226) ------------------------------------------------

    /// The fixed adaptation clause is appended to every personality blurb,
    /// regardless of trait levels. Pinned here so the copy can't drift away
    /// from the rest of the suite without a deliberate edit.
    const ADAPTATION_CLAUSE: &str = "Treat this as a starting point, not a script. \
         Take your cues from the conversation and adapt both ways \u{2014} if the user is \
         playful or jokes around, it's fine to loosen up and joke back a bit; if things \
         turn serious or they seem stressed, ease off the humor and sarcasm unless a light \
         touch genuinely helps. Match the user's energy rather than forcing a trait that \
         doesn't fit the moment.";

    #[test]
    fn personality_level_ordinal_round_trip() {
        // The D-Bus contract exposes levels as integers 0..=4. Every level
        // must map to a stable ordinal and back.
        for (level, ordinal) in [
            (PersonalityLevel::Never, 0u8),
            (PersonalityLevel::Rarely, 1),
            (PersonalityLevel::Sometimes, 2),
            (PersonalityLevel::Often, 3),
            (PersonalityLevel::Always, 4),
        ] {
            assert_eq!(level.as_ordinal(), ordinal);
            assert_eq!(PersonalityLevel::from_ordinal(ordinal), Some(level));
        }
        // Out-of-range ordinals are rejected (no silent clamp).
        assert_eq!(PersonalityLevel::from_ordinal(5), None);
    }

    #[test]
    fn personality_defaults_match_expressive_7_table() {
        let p = Personality::default();
        assert_eq!(p.professionalism, PersonalityLevel::Always);
        assert_eq!(p.warmth, PersonalityLevel::Often);
        assert_eq!(p.directness, PersonalityLevel::Often);
        assert_eq!(p.enthusiasm, PersonalityLevel::Sometimes);
        assert_eq!(p.humor, PersonalityLevel::Sometimes);
        assert_eq!(p.sarcasm, PersonalityLevel::Rarely);
        assert_eq!(p.pretentiousness, PersonalityLevel::Rarely);
    }

    #[test]
    fn render_blurb_defaults_emits_disposition_then_adaptation() {
        let blurb = Personality::default().render_blurb();
        // Disposition paragraph mentions each non-Never trait.
        assert!(blurb.contains("professional"), "blurb: {blurb}");
        assert!(blurb.contains("warm"), "blurb: {blurb}");
        assert!(blurb.contains("direct"), "blurb: {blurb}");
        assert!(blurb.contains("enthusias"), "blurb: {blurb}");
        assert!(blurb.contains("humor") || blurb.contains("humour"), "blurb: {blurb}");
        assert!(blurb.contains("sarcas"), "blurb: {blurb}");
        assert!(blurb.contains("pretenti"), "blurb: {blurb}");
        // The adaptation clause is always present and comes last.
        assert!(blurb.contains(ADAPTATION_CLAUSE), "blurb: {blurb}");
        assert!(blurb.trim_end().ends_with(ADAPTATION_CLAUSE), "blurb: {blurb}");
    }

    #[test]
    fn render_blurb_omits_never_traits() {
        // Set Humor and Sarcasm to Never; their clauses must disappear while
        // the other traits remain.
        let p = Personality {
            humor: PersonalityLevel::Never,
            sarcasm: PersonalityLevel::Never,
            ..Personality::default()
        };
        let blurb = p.render_blurb();
        assert!(!blurb.contains("humor") && !blurb.contains("humour"), "blurb: {blurb}");
        assert!(!blurb.contains("sarcas"), "blurb: {blurb}");
        // Remaining traits still rendered.
        assert!(blurb.contains("professional"), "blurb: {blurb}");
        assert!(blurb.contains("warm"), "blurb: {blurb}");
        // Adaptation clause still appended.
        assert!(blurb.contains(ADAPTATION_CLAUSE), "blurb: {blurb}");
    }

    #[test]
    fn render_blurb_all_never_is_adaptation_clause_only() {
        let p = Personality {
            professionalism: PersonalityLevel::Never,
            warmth: PersonalityLevel::Never,
            directness: PersonalityLevel::Never,
            enthusiasm: PersonalityLevel::Never,
            humor: PersonalityLevel::Never,
            sarcasm: PersonalityLevel::Never,
            pretentiousness: PersonalityLevel::Never,
        };
        let blurb = p.render_blurb();
        // No disposition sentence at all — only the adaptation clause.
        assert_eq!(blurb.trim(), ADAPTATION_CLAUSE);
    }

    #[test]
    fn render_blurb_adaptation_clause_always_present() {
        // Property: every possible single-trait setting still appends the
        // adaptation clause. Exhaustive over levels for one representative
        // trait is enough to pin the invariant.
        for level in [
            PersonalityLevel::Never,
            PersonalityLevel::Rarely,
            PersonalityLevel::Sometimes,
            PersonalityLevel::Often,
            PersonalityLevel::Always,
        ] {
            let p = Personality {
                humor: level,
                ..Personality::default()
            };
            assert!(
                p.render_blurb().contains(ADAPTATION_CLAUSE),
                "level {level:?} dropped the adaptation clause"
            );
        }
    }

    #[test]
    fn personality_serde_round_trip_lowercase() {
        // TOML/JSON config persists levels as lowercase strings; round-trip
        // must be lossless so a stored `[personality]` reloads identically.
        let p = Personality {
            humor: PersonalityLevel::Never,
            ..Personality::default()
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"never\""), "json: {json}");
        let parsed: Personality = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, p);
    }
}
