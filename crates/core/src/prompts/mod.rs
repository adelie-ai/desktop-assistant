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
}
