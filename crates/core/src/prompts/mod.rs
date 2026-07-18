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

/// The personality types — [`Personality`], [`PersonalityLevel`], and
/// [`PersonalityOverride`] — are defined in `desktop-assistant-protocol` (the
/// dependency-light crate that compiles to wasm) and re-exported here at their
/// canonical `core::prompts::*` paths so existing call sites are unchanged
/// (#377). The prompt-rendering logic ([`render_blurb`] + the phrasing tables)
/// stays in this module.
pub use desktop_assistant_protocol::{Personality, PersonalityLevel, PersonalityOverride};

/// The fixed adaptation clause appended to every personality blurb. It tells
/// the model the levels are a starting point and to match the user's energy
/// rather than rigidly enforcing a trait.
const ADAPTATION_CLAUSE: &str = "Treat this as a starting point, not a script. \
     Take your cues from the conversation and adapt both ways \u{2014} if the user is \
     playful or jokes around, it's fine to loosen up and joke back a bit; if things \
     turn serious or they seem stressed, ease off the humor and sarcasm unless a light \
     touch genuinely helps. Match the user's energy rather than forcing a trait that \
     doesn't fit the moment.";

/// Render a [`Personality`] into a natural-language disposition blurb for the
/// system prompt.
///
/// The blurb is a single disposition sentence — one clause per trait whose
/// level is not [`PersonalityLevel::Never`], phrased by level — followed by the
/// fixed [`ADAPTATION_CLAUSE`]. A `Never` trait contributes no clause. When
/// every trait is `Never`, only the adaptation clause is emitted.
///
/// A free function (not an inherent method) because [`Personality`] now lives
/// in `desktop-assistant-protocol`; the prompt-rendering logic stays in `core`.
pub fn render_blurb(p: &Personality) -> String {
    // (trait clause builder, level) pairs in a fixed, readable order. Each
    // builder turns a non-Never level into a natural clause; `None` means
    // the trait is omitted (Never).
    let clauses: Vec<String> = [
        trait_clause(p.professionalism, &PROFESSIONALISM_PHRASING),
        trait_clause(p.warmth, &WARMTH_PHRASING),
        trait_clause(p.directness, &DIRECTNESS_PHRASING),
        trait_clause(p.enthusiasm, &ENTHUSIASM_PHRASING),
        trait_clause(p.humor, &HUMOR_PHRASING),
        trait_clause(p.sarcasm, &SARCASM_PHRASING),
        trait_clause(p.pretentiousness, &PRETENTIOUSNESS_PHRASING),
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
    fn assembled_prompt_urges_web_browsing_and_resourcefulness() {
        // Adele kept forgetting she can browse for live info and waited to be
        // told; the tool-use guidance must push proactive web use for current
        // information and creative combination of general tools when no
        // purpose-built tool exists.
        let assembled = assemble(&static_sections());
        assert!(
            assembled.contains("== Live & external info (web) =="),
            "the always-present prompt must advertise web browsing"
        );
        assert!(
            assembled.contains("browse the web"),
            "and tell her she can browse the web"
        );
        assert!(
            assembled.to_lowercase().contains("news"),
            "naming current-info uses like news"
        );
        assert!(
            assembled.contains("resourceful"),
            "and urge resourcefulness when no purpose-built tool exists"
        );
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
        let blurb = render_blurb(&Personality::default());
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
        let blurb = render_blurb(&p);
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
        let blurb = render_blurb(&p);
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
                render_blurb(&p).contains(ADAPTATION_CLAUSE),
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

    // --- PersonalityOverride (#227, Phase 2) -------------------------------

    #[test]
    fn override_empty_resolves_to_global_unchanged() {
        // No override → resolution is the global disposition verbatim. This is
        // the "no per-conversation override" baseline.
        let global = Personality {
            humor: PersonalityLevel::Often,
            sarcasm: PersonalityLevel::Always,
            ..Personality::default()
        };
        let ovr = PersonalityOverride::default();
        assert!(ovr.is_empty());
        assert_eq!(ovr.resolve(&global), global);
    }

    #[test]
    fn override_some_trait_wins_per_trait() {
        // A "no-nonsense" override forces humor/sarcasm off and directness up;
        // each pinned trait wins over the global value.
        let global = Personality::default();
        let ovr = PersonalityOverride {
            humor: Some(PersonalityLevel::Never),
            sarcasm: Some(PersonalityLevel::Never),
            directness: Some(PersonalityLevel::Always),
            ..PersonalityOverride::default()
        };
        let resolved = ovr.resolve(&global);
        assert_eq!(resolved.humor, PersonalityLevel::Never);
        assert_eq!(resolved.sarcasm, PersonalityLevel::Never);
        assert_eq!(resolved.directness, PersonalityLevel::Always);
    }

    #[test]
    fn override_unspecified_traits_fall_back_to_global() {
        // Traits the override leaves `None` inherit the global value, even when
        // the global differs from the built-in default.
        let global = Personality {
            professionalism: PersonalityLevel::Rarely,
            warmth: PersonalityLevel::Always,
            enthusiasm: PersonalityLevel::Always,
            pretentiousness: PersonalityLevel::Often,
            ..Personality::default()
        };
        let ovr = PersonalityOverride {
            humor: Some(PersonalityLevel::Never),
            ..PersonalityOverride::default()
        };
        let resolved = ovr.resolve(&global);
        // Pinned trait wins.
        assert_eq!(resolved.humor, PersonalityLevel::Never);
        // Every unspecified trait falls back to the (non-default) global.
        assert_eq!(resolved.professionalism, PersonalityLevel::Rarely);
        assert_eq!(resolved.warmth, PersonalityLevel::Always);
        assert_eq!(resolved.directness, global.directness);
        assert_eq!(resolved.enthusiasm, PersonalityLevel::Always);
        assert_eq!(resolved.sarcasm, global.sarcasm);
        assert_eq!(resolved.pretentiousness, PersonalityLevel::Often);
    }

    #[test]
    fn override_is_empty_only_when_all_none() {
        assert!(PersonalityOverride::default().is_empty());
        assert!(
            !PersonalityOverride {
                humor: Some(PersonalityLevel::Never),
                ..PersonalityOverride::default()
            }
            .is_empty()
        );
    }

    #[test]
    fn override_serde_omits_none_traits_and_round_trips() {
        // Only the pinned trait should appear on the wire; the rest are omitted
        // (skip_serializing_if) so a partial override stays compact. Round-trip
        // must be lossless.
        let ovr = PersonalityOverride {
            humor: Some(PersonalityLevel::Never),
            ..PersonalityOverride::default()
        };
        let json = serde_json::to_string(&ovr).unwrap();
        assert!(json.contains("\"humor\""), "json: {json}");
        assert!(json.contains("\"never\""), "json: {json}");
        assert!(!json.contains("\"warmth\""), "json: {json}");
        let parsed: PersonalityOverride = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ovr);
        // An empty override serializes to `{}` and round-trips to empty.
        let empty_json = serde_json::to_string(&PersonalityOverride::default()).unwrap();
        assert_eq!(empty_json, "{}");
        let back: PersonalityOverride = serde_json::from_str("{}").unwrap();
        assert!(back.is_empty());
    }
}
