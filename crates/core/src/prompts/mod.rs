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
    /// Configurable disposition blurb (issue #226). Rendered from the active
    /// [`Personality`] and injected before [`Self::ToolAvailability`] and the
    /// per-turn [`Self::SystemRefinement`] so the standing personality is set
    /// up front but a per-turn refinement can still adjust it last.
    Personality,
    ToolAvailability,
    ContextSummary,
    MessageSummary,
    /// Per-request, client-supplied addition to the system prompt for a
    /// single turn (e.g. a voice client's "respond briefly, by voice").
    /// Appended last so it can refine/override the static guidance above.
    /// Never persisted; see `crate::ports::llm::SYSTEM_REFINEMENT`.
    SystemRefinement,
}

// --- Personality (#226) ----------------------------------------------------

/// Qualitative level for a single personality trait.
///
/// Ordered `Never < Rarely < Sometimes < Often < Always`. The numeric
/// ordinal (0..=4, via [`Self::as_ordinal`] / [`Self::from_ordinal`]) is a
/// **stable wire contract**: the D-Bus settings surface exposes each trait as
/// an integer 0..=4 so the KCM can bind a slider directly (Never=0 … Always=4).
/// The serde representation is the lowercase variant name (e.g. `"always"`)
/// for human-friendly TOML/JSON config, mirroring [`crate::ports::llm::ReasoningLevel`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum PersonalityLevel {
    Never,
    Rarely,
    Sometimes,
    Often,
    Always,
}

impl PersonalityLevel {
    /// Stable ordinal 0..=4 used by the D-Bus int contract (Never=0 … Always=4).
    pub fn as_ordinal(self) -> u8 {
        match self {
            Self::Never => 0,
            Self::Rarely => 1,
            Self::Sometimes => 2,
            Self::Often => 3,
            Self::Always => 4,
        }
    }

    /// Inverse of [`Self::as_ordinal`]. Returns `None` for out-of-range input
    /// rather than clamping, so a malformed wire value surfaces as an error at
    /// the boundary instead of silently snapping to a level.
    pub fn from_ordinal(n: u8) -> Option<Self> {
        match n {
            0 => Some(Self::Never),
            1 => Some(Self::Rarely),
            2 => Some(Self::Sometimes),
            3 => Some(Self::Often),
            4 => Some(Self::Always),
            _ => None,
        }
    }
}

/// The assistant's configurable disposition — the "Expressive 7" traits, each
/// at a [`PersonalityLevel`] (issue #226, Phase 1: global).
///
/// Why a typed struct rather than a free `HashMap<String, Level>`: the trait
/// set is fixed and small, so naming each field gives compile-time safety (no
/// typo'd keys, exhaustive `render_blurb` matching) and a **stable wire schema**
/// — config files, the api-model `Config` view, and the D-Bus `ConfigData`
/// tuple all derive from these named fields, so adding/removing a trait is a
/// deliberate, type-checked change rather than a silent string drift.
///
/// The levels are an *initial disposition*, not a rulebook: [`Self::render_blurb`]
/// always appends an adaptation clause telling the model to take cues from the
/// conversation and adapt both ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Personality {
    #[serde(default = "default_professionalism")]
    pub professionalism: PersonalityLevel,
    #[serde(default = "default_warmth")]
    pub warmth: PersonalityLevel,
    #[serde(default = "default_directness")]
    pub directness: PersonalityLevel,
    #[serde(default = "default_enthusiasm")]
    pub enthusiasm: PersonalityLevel,
    #[serde(default = "default_humor")]
    pub humor: PersonalityLevel,
    #[serde(default = "default_sarcasm")]
    pub sarcasm: PersonalityLevel,
    #[serde(default = "default_pretentiousness")]
    pub pretentiousness: PersonalityLevel,
}

// Per-field default fns so a partial `[personality]` TOML block (only some
// traits specified) fills the rest from the Expressive-7 table rather than
// from `PersonalityLevel`'s arbitrary first variant.
fn default_professionalism() -> PersonalityLevel {
    PersonalityLevel::Always
}
fn default_warmth() -> PersonalityLevel {
    PersonalityLevel::Often
}
fn default_directness() -> PersonalityLevel {
    PersonalityLevel::Often
}
fn default_enthusiasm() -> PersonalityLevel {
    PersonalityLevel::Sometimes
}
fn default_humor() -> PersonalityLevel {
    PersonalityLevel::Sometimes
}
fn default_sarcasm() -> PersonalityLevel {
    PersonalityLevel::Rarely
}
fn default_pretentiousness() -> PersonalityLevel {
    PersonalityLevel::Rarely
}

impl Default for Personality {
    /// The "Expressive 7" defaults from the issue table.
    fn default() -> Self {
        Self {
            professionalism: default_professionalism(),
            warmth: default_warmth(),
            directness: default_directness(),
            enthusiasm: default_enthusiasm(),
            humor: default_humor(),
            sarcasm: default_sarcasm(),
            pretentiousness: default_pretentiousness(),
        }
    }
}

/// The fixed adaptation clause appended to every personality blurb. It tells
/// the model the levels are a starting point and to match the user's energy
/// rather than rigidly enforcing a trait.
const ADAPTATION_CLAUSE: &str = "Treat this as a starting point, not a script. \
     Take your cues from the conversation and adapt both ways \u{2014} if the user is \
     playful or jokes around, it's fine to loosen up and joke back a bit; if things \
     turn serious or they seem stressed, ease off the humor and sarcasm unless a light \
     touch genuinely helps. Match the user's energy rather than forcing a trait that \
     doesn't fit the moment.";

impl Personality {
    /// Render the disposition into a natural-language blurb for the system
    /// prompt.
    ///
    /// The blurb is a single disposition sentence — one clause per trait whose
    /// level is not [`PersonalityLevel::Never`], phrased by level — followed by
    /// the fixed [`ADAPTATION_CLAUSE`]. A `Never` trait contributes no clause.
    /// When every trait is `Never`, only the adaptation clause is emitted.
    pub fn render_blurb(&self) -> String {
        // (trait clause builder, level) pairs in a fixed, readable order. Each
        // builder turns a non-Never level into a natural clause; `None` means
        // the trait is omitted (Never).
        let clauses: Vec<String> = [
            trait_clause(self.professionalism, &PROFESSIONALISM_PHRASING),
            trait_clause(self.warmth, &WARMTH_PHRASING),
            trait_clause(self.directness, &DIRECTNESS_PHRASING),
            trait_clause(self.enthusiasm, &ENTHUSIASM_PHRASING),
            trait_clause(self.humor, &HUMOR_PHRASING),
            trait_clause(self.sarcasm, &SARCASM_PHRASING),
            trait_clause(self.pretentiousness, &PRETENTIOUSNESS_PHRASING),
        ]
        .into_iter()
        .flatten()
        .collect();

        if clauses.is_empty() {
            return ADAPTATION_CLAUSE.to_string();
        }

        let disposition = format!("In your default manner, you {}.", join_clauses(&clauses));
        format!("{disposition} {ADAPTATION_CLAUSE}")
    }
}

/// Per-level phrasing for a single trait. Each field is the clause body used at
/// that level; `Never` has no field because a Never trait is omitted entirely.
struct TraitPhrasing {
    rarely: &'static str,
    sometimes: &'static str,
    often: &'static str,
    always: &'static str,
}

/// Pick the clause for a level, or `None` when the trait is `Never`.
fn trait_clause(level: PersonalityLevel, p: &TraitPhrasing) -> Option<String> {
    let body = match level {
        PersonalityLevel::Never => return None,
        PersonalityLevel::Rarely => p.rarely,
        PersonalityLevel::Sometimes => p.sometimes,
        PersonalityLevel::Often => p.often,
        PersonalityLevel::Always => p.always,
    };
    Some(body.to_string())
}

/// Join clauses into a single grammatical list ("a, b, and c").
fn join_clauses(clauses: &[String]) -> String {
    match clauses {
        [] => String::new(),
        [one] => one.clone(),
        [first, second] => format!("{first}, and {second}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}

// Phrasing tables. The clause body completes "you ...": e.g. Always warmth →
// "you are always warm and personable". Rarely uses a hedged "occasionally /
// a touch of"; Sometimes a "now and then / at times"; Often a "usually".
const PROFESSIONALISM_PHRASING: TraitPhrasing = TraitPhrasing {
    rarely: "keep things professional only on rare occasion",
    sometimes: "stay professional at times",
    often: "usually keep a professional tone",
    always: "always keep a professional, polished tone",
};
const WARMTH_PHRASING: TraitPhrasing = TraitPhrasing {
    rarely: "show a touch of warmth on rare occasion",
    sometimes: "come across as warm now and then",
    often: "are usually warm and personable",
    always: "are always warm and personable",
};
const DIRECTNESS_PHRASING: TraitPhrasing = TraitPhrasing {
    rarely: "get straight to the point only on rare occasion",
    sometimes: "are direct at times",
    often: "are usually direct and to the point",
    always: "are always direct and to the point",
};
const ENTHUSIASM_PHRASING: TraitPhrasing = TraitPhrasing {
    rarely: "show enthusiasm only on rare occasion",
    sometimes: "show some enthusiasm now and then",
    often: "are usually enthusiastic",
    always: "are always enthusiastic and energetic",
};
const HUMOR_PHRASING: TraitPhrasing = TraitPhrasing {
    rarely: "crack a bit of humor only on rare occasion",
    sometimes: "bring a little humor now and then",
    often: "usually keep things light with some humor",
    always: "always keep things light with humor",
};
const SARCASM_PHRASING: TraitPhrasing = TraitPhrasing {
    rarely: "let a touch of sarcasm slip only on rare occasion",
    sometimes: "use a bit of dry sarcasm at times",
    often: "are usually a little sarcastic",
    always: "are reliably sarcastic and dry",
};
const PRETENTIOUSNESS_PHRASING: TraitPhrasing = TraitPhrasing {
    rarely: "get a touch pretentious only on rare occasion",
    sometimes: "can be a little pretentious at times",
    often: "are usually somewhat pretentious",
    always: "are consistently pretentious and highbrow",
};

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
        assert!(
            blurb.contains("humor") || blurb.contains("humour"),
            "blurb: {blurb}"
        );
        assert!(blurb.contains("sarcas"), "blurb: {blurb}");
        assert!(blurb.contains("pretenti"), "blurb: {blurb}");
        // The adaptation clause is always present and comes last.
        assert!(blurb.contains(ADAPTATION_CLAUSE), "blurb: {blurb}");
        assert!(
            blurb.trim_end().ends_with(ADAPTATION_CLAUSE),
            "blurb: {blurb}"
        );
    }

    #[test]
    fn render_blurb_omits_never_traits() {
        // Set Humor and Sarcasm to Never; their clauses must disappear from the
        // disposition sentence while the other traits remain. NB: the fixed
        // adaptation clause mentions "humor" and "sarcasm" by design, so we
        // assert against the disposition portion (everything before the
        // adaptation clause), not the whole blurb.
        let p = Personality {
            humor: PersonalityLevel::Never,
            sarcasm: PersonalityLevel::Never,
            ..Personality::default()
        };
        let blurb = p.render_blurb();
        let disposition = blurb
            .split(ADAPTATION_CLAUSE)
            .next()
            .expect("adaptation clause present");
        assert!(
            !disposition.contains("humor") && !disposition.contains("humour"),
            "disposition: {disposition}"
        );
        assert!(
            !disposition.contains("sarcas"),
            "disposition: {disposition}"
        );
        // Remaining traits still rendered.
        assert!(
            disposition.contains("professional"),
            "disposition: {disposition}"
        );
        assert!(disposition.contains("warm"), "disposition: {disposition}");
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
