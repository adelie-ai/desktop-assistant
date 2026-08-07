//! Promoting a completed plan into a skill candidate (#1155).
//!
//! When a plan finishes, the procedure it followed is already written down. The
//! scratchpad holds one `todo` note per step and one `outcome:<step>` note per
//! finding (#240), which is a `## Steps` workflow in everything but name. This
//! module is the pure half of turning that into a skill: the bar a plan must
//! clear, the body rendered from the plan, and the offer the model reads.
//!
//! Three rules shape it.
//!
//! **The plan is the source, never the transcript.** The transcript carries the
//! dead ends; the plan carries what worked. [`render_skill_body`] reads steps
//! and outcomes and nothing else.
//!
//! **The offer is an offer.** Nothing here writes. The model has the context to
//! say whether what it just did generalises, and declining is doing nothing.
//!
//! **Not every success is a procedure.** Answering a question is not a skill,
//! and writing one file is not a skill. [`assess`] holds the bar; the async
//! half (reading the notes, searching the catalog for a skill that already
//! covers this, writing the row) lives in the service dispatch loop.

use crate::domain::skill::{IndexedSkill, SkillError, validate_skill_name};
use crate::domain::tool::ToolDefinition;
use crate::domain::{Message, Role};
use crate::planning::{OUTCOME_KEY_PREFIX, STEP_NOTE_TYPE};
use crate::tool_provenance::WITHHELD_STEP_TEXT;

/// Tool the model calls to accept a promotion offer. A core-loop tool, like the
/// step-control pair it follows: only the loop holds the turn's plan.
pub const PROMOTE_PLAN_TOOL: &str = "promote_plan_to_skill";

/// The builtin tool that reads a skill's body before it is followed.
///
/// A turn that called it followed a skill, and re-saving a skill you just
/// followed is how a library fills with near-duplicates. Defined here rather
/// than in the tool crate because the rule is promotion policy; `mcp-client`
/// advertises the tool under this same name.
pub const SKILL_GET_TOOL: &str = "builtin_skill_get";

/// Fewest steps, each with a recorded outcome, a plan needs before it is worth
/// keeping.
///
/// Three is where "what I just did" becomes "a method". One step is a single
/// act - write the file, answer the question - and two is a pair of acts with
/// no shape between them. The value of a skill is the ordering and the reasons,
/// and neither exists below three.
pub const MIN_PROMOTABLE_STEPS: usize = 3;

/// Largest share of a plan's steps that may be abandoned before the plan stops
/// being evidence of a method: expressed as `abandoned * 3 <= total`, i.e. one
/// third.
///
/// A plan with half its steps abandoned records a search, not a procedure.
pub const MAX_ABANDONED_DENOMINATOR: usize = 3;

/// Prefix `complete_step` writes on an outcome when the step was abandoned.
/// Read back here so an abandoned step is not offered as a working step.
const ABANDONED_PREFIX: &str = "Abandoned: ";

/// How many existing skills the offer names as possible duplicates.
pub const MAX_OFFERED_MATCHES: usize = 3;

/// One scratchpad note, as the promotion pass reads it.
///
/// A borrowed view rather than the stored type, so the pure logic here never
/// depends on the storage shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanNote<'a> {
    /// The note's key: a dotted step path for a step, `outcome:<path>` for a
    /// finding.
    pub key: &'a str,
    /// The note's text.
    pub content: &'a str,
    /// The note's free-text category.
    pub note_type: &'a str,
    /// Whether the note is checked off.
    pub done: bool,
}

/// One completed step of a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanStep {
    /// Dotted step path, e.g. `"1"`, `"1.2"`.
    pub key: String,
    /// What the step set out to do.
    pub goal: String,
    /// What the step found, `None` when the step recorded nothing.
    pub outcome: Option<String>,
    /// Whether the step was abandoned rather than finished.
    pub abandoned: bool,
}

impl PlanStep {
    /// Whether this step is evidence of a working method: finished, not
    /// abandoned, and carrying its own account of what it produced.
    pub fn succeeded(&self) -> bool {
        !self.abandoned && self.outcome.as_ref().is_some_and(|o| !o.trim().is_empty())
    }

    /// Nesting depth from the dotted key: `"1"` is 0, `"1.2"` is 1.
    pub fn depth(&self) -> usize {
        self.key.matches('.').count()
    }
}

/// Why a completed plan is not worth offering as a skill.
///
/// A decline is a normal outcome, not a failure: most plans are not procedures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotPromotable {
    /// The turn read an existing skill before it planned, so this plan is that
    /// skill being followed.
    FollowedAnExistingSkill,
    /// Too few steps recorded a working outcome.
    TooFewSteps {
        /// Steps that succeeded and recorded an outcome.
        succeeded: usize,
        /// How many are needed ([`MIN_PROMOTABLE_STEPS`]).
        needed: usize,
    },
    /// Enough steps, but too many of them were abandoned.
    TooManyAbandoned {
        /// Steps abandoned.
        abandoned: usize,
        /// Steps in the plan.
        total: usize,
    },
    /// The turn's step text was withheld, so there is no procedure to read.
    NothingRecorded,
}

impl NotPromotable {
    /// A short, user-facing reason, for the tool result that declines.
    pub fn reason(&self) -> String {
        match self {
            NotPromotable::FollowedAnExistingSkill => {
                "this plan followed an existing skill, so saving it again would duplicate it"
                    .to_string()
            }
            NotPromotable::TooFewSteps { succeeded, needed } => format!(
                "a skill needs at least {needed} steps with a recorded outcome; this plan has \
                 {succeeded}"
            ),
            NotPromotable::TooManyAbandoned { abandoned, total } => format!(
                "{abandoned} of {total} steps were abandoned, so this plan records a search \
                 rather than a method"
            ),
            NotPromotable::NothingRecorded => {
                "this turn did not record its step text, so there is no procedure to keep"
                    .to_string()
            }
        }
    }
}

/// A completed plan that cleared the bar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotablePlan {
    /// Every step of the plan, in order, abandoned ones included: the body
    /// renders only the working steps, but the count is what the bar was
    /// judged on.
    pub steps: Vec<PlanStep>,
}

impl PromotablePlan {
    /// The steps a skill body is built from: those that finished and recorded
    /// what they produced, in dotted-key order.
    pub fn working_steps(&self) -> Vec<&PlanStep> {
        self.steps.iter().filter(|s| s.succeeded()).collect()
    }
}

/// Read a plan back out of a conversation's scratchpad notes.
///
/// Only checked-off `todo` notes whose key is a dotted step path count as
/// steps, so a hand-written todo (`"buy-milk"`) is never mistaken for one. Each
/// step picks up its `outcome:<path>` note, and an outcome the step abandoned
/// is recognised by the prefix `complete_step` writes.
///
/// Returned in dotted-key order (`1`, `1.1`, `1.2`, `2`), which is the order
/// the work happened in.
pub fn plan_from_notes(notes: &[PlanNote<'_>]) -> Vec<PlanStep> {
    let _ = (notes, OUTCOME_KEY_PREFIX, ABANDONED_PREFIX, STEP_NOTE_TYPE);
    todo!("plan_from_notes")
}

/// Whether the turn read an existing skill before it planned.
///
/// True when any assistant message in the turn called [`SKILL_GET_TOOL`].
pub fn followed_a_skill(messages: &[Message]) -> bool {
    let _ = messages;
    todo!("followed_a_skill")
}

/// Decide whether a completed plan is worth offering as a skill.
///
/// The bar, in one place: the plan was not itself a skill being followed, at
/// least [`MIN_PROMOTABLE_STEPS`] steps finished and recorded what they
/// produced, and no more than a third of the plan was abandoned.
pub fn assess(steps: Vec<PlanStep>, followed_a_skill: bool) -> Result<PromotablePlan, NotPromotable> {
    let _ = (steps, followed_a_skill);
    todo!("assess")
}

/// Render the markdown body of a skill from a plan.
///
/// Built from the plan's steps and their outcomes only. The `## Steps` heading
/// is what makes the result a workflow rather than a prose playbook (see
/// [`crate::domain::skill::detect_kind`]), so it is written exactly.
pub fn render_skill_body(title: &str, summary: Option<&str>, plan: &PromotablePlan) -> String {
    let _ = (title, summary, plan);
    todo!("render_skill_body")
}

/// Render a whole `SKILL.md`: YAML frontmatter plus the body.
///
/// The frontmatter carries the fields the shared cross-product format requires
/// (`name`, `description`, `tags`), so the result parses back through
/// [`crate::domain::skill::parse_skill_md`].
pub fn render_skill_md(name: &str, description: &str, tags: &[String], body: &str) -> String {
    let _ = (name, description, tags, body);
    todo!("render_skill_md")
}

/// The offer appended to a `complete_step` acknowledgement when a finished plan
/// clears the bar.
///
/// `existing` are catalog entries that may already cover this procedure;
/// amending one of those is the useful act, and adding a second is not.
/// Declining is doing nothing.
pub fn render_offer(plan: &PromotablePlan, existing: &[IndexedSkill]) -> serde_json::Value {
    let _ = (plan, existing);
    todo!("render_offer")
}

/// What the model asked [`PROMOTE_PLAN_TOOL`] to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionMode {
    /// Add a new skill.
    New,
    /// Revise an existing skill of the same name.
    Amend,
}

/// The validated arguments of a [`PROMOTE_PLAN_TOOL`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionRequest {
    /// Skill name; also the catalog key.
    pub name: String,
    /// The "when to use" trigger.
    pub description: String,
    /// One short paragraph of what the procedure is for.
    pub summary: Option<String>,
    /// Frontmatter tags.
    pub tags: Vec<String>,
    /// Add or revise.
    pub mode: PromotionMode,
}

/// Parse and validate a [`PROMOTE_PLAN_TOOL`] call's arguments.
///
/// `name` is checked against path traversal with the same guard the scanner
/// uses, because a promoted skill's name is a directory name wherever the
/// catalog is later exported.
pub fn parse_promotion_request(args: &serde_json::Value) -> Result<PromotionRequest, SkillError> {
    let _ = (args, validate_skill_name);
    todo!("parse_promotion_request")
}

/// The tool definition the dispatch loop advertises for accepting an offer.
pub fn promote_plan_tool() -> ToolDefinition {
    todo!("promote_plan_tool")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::skill::{Locality, SkillKind, TrustTier, detect_kind, parse_skill_md};

    fn step(key: &str, goal: &str, outcome: Option<&str>) -> PlanStep {
        PlanStep {
            key: key.to_string(),
            goal: goal.to_string(),
            outcome: outcome.map(str::to_string),
            abandoned: false,
        }
    }

    fn abandoned(key: &str, goal: &str) -> PlanStep {
        PlanStep {
            key: key.to_string(),
            goal: goal.to_string(),
            outcome: Some("ran out of road".to_string()),
            abandoned: true,
        }
    }

    /// A four-step plan whose steps all recorded an outcome.
    fn good_plan() -> Vec<PlanStep> {
        vec![
            step("1", "Find the failing migration", Some("It is 041.")),
            step("2", "Reproduce it on a scratch database", Some("Fails on replay.")),
            step("3", "Guard the backfill", Some("Wrapped it in an existence check.")),
        ]
    }

    fn indexed(name: &str, description: &str) -> IndexedSkill {
        IndexedSkill {
            name: name.to_string(),
            description: description.to_string(),
            kind: SkillKind::Workflow,
            disk_path: String::new(),
            owner_user_id: Some("someone".to_string()),
            locality: Locality::Daemon,
            content_hash: "hash".to_string(),
            trust_tier: TrustTier::Local,
            source: Some("self-authored".to_string()),
            tags: Vec::new(),
            attachments: Vec::new(),
            body: String::new(),
            metadata: serde_json::json!({}),
            present_on_disk: false,
            last_seen_at: None,
            approved_at: None,
            approved_by: None,
        }
    }

    // --- plan_from_notes -----------------------------------------------------

    #[test]
    fn plan_reads_completed_steps_and_their_outcomes() {
        let notes = [
            PlanNote { key: "1", content: "Find the failing migration", note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "outcome:1", content: "It is 041.", note_type: "note", done: false },
            PlanNote { key: "2", content: "Guard the backfill", note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "outcome:2", content: "Wrapped it.", note_type: "note", done: false },
        ];
        let steps = plan_from_notes(&notes);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].key, "1");
        assert_eq!(steps[0].goal, "Find the failing migration");
        assert_eq!(steps[0].outcome.as_deref(), Some("It is 041."));
        assert!(!steps[0].abandoned);
    }

    #[test]
    fn plan_ignores_open_steps_and_hand_written_todos() {
        let notes = [
            PlanNote { key: "1", content: "done step", note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "2", content: "still open", note_type: STEP_NOTE_TYPE, done: false },
            PlanNote { key: "buy-milk", content: "not a step", note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "goal", content: "the overall goal", note_type: "note", done: false },
        ];
        let steps = plan_from_notes(&notes);
        assert_eq!(steps.len(), 1, "only the completed dotted step counts");
        assert_eq!(steps[0].key, "1");
    }

    #[test]
    fn plan_recognises_an_abandoned_step() {
        let notes = [
            PlanNote { key: "1", content: "try the fast path", note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "outcome:1", content: "Abandoned: no index to use", note_type: "note", done: false },
        ];
        let steps = plan_from_notes(&notes);
        assert!(steps[0].abandoned);
        assert_eq!(
            steps[0].outcome.as_deref(),
            Some("no index to use"),
            "the marker is stripped from the recorded finding"
        );
        assert!(!steps[0].succeeded());
    }

    #[test]
    fn plan_orders_steps_by_dotted_key() {
        let notes = [
            PlanNote { key: "2", content: "second", note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "1.10", content: "tenth child", note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "1.2", content: "second child", note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "1", content: "first", note_type: STEP_NOTE_TYPE, done: true },
        ];
        let keys: Vec<String> = plan_from_notes(&notes).into_iter().map(|s| s.key).collect();
        assert_eq!(keys, vec!["1", "1.2", "1.10", "2"], "numeric, not lexical");
    }

    #[test]
    fn plan_treats_withheld_step_text_as_nothing_recorded() {
        let notes = [
            PlanNote { key: "1", content: WITHHELD_STEP_TEXT, note_type: STEP_NOTE_TYPE, done: true },
            PlanNote { key: "outcome:1", content: WITHHELD_STEP_TEXT, note_type: "note", done: false },
        ];
        let steps = plan_from_notes(&notes);
        assert!(steps[0].outcome.is_none(), "a placeholder is not a finding");
        assert!(!steps[0].succeeded());
    }

    // --- followed_a_skill ----------------------------------------------------

    #[test]
    fn a_turn_that_read_a_skill_followed_one() {
        let mut msg = Message::new(Role::Assistant, "");
        msg.tool_calls = vec![crate::domain::tool::ToolCall {
            id: "c1".to_string(),
            name: SKILL_GET_TOOL.to_string(),
            arguments: "{}".to_string(),
        }];
        assert!(followed_a_skill(&[msg]));
    }

    #[test]
    fn a_turn_that_only_searched_did_not_follow_one() {
        let mut msg = Message::new(Role::Assistant, "");
        msg.tool_calls = vec![crate::domain::tool::ToolCall {
            id: "c1".to_string(),
            name: "builtin_skill_search".to_string(),
            arguments: "{}".to_string(),
        }];
        assert!(
            !followed_a_skill(&[msg]),
            "searching the library is not following a skill"
        );
    }

    // --- assess: the acceptance criteria -------------------------------------

    /// Acceptance: a completed multi-step plan whose steps succeeded produces
    /// an offer to write a skill.
    #[test]
    fn completed_multi_step_plan_produces_an_offer() {
        let plan = assess(good_plan(), false).expect("a three-step plan with outcomes clears the bar");
        assert_eq!(plan.working_steps().len(), 3);
        let offer = render_offer(&plan, &[]);
        assert_eq!(offer["tool"], PROMOTE_PLAN_TOOL);
        assert_eq!(offer["steps"], 3);
    }

    /// Acceptance: a plan that was started from an existing skill produces no
    /// offer.
    #[test]
    fn plan_started_from_an_existing_skill_produces_no_offer() {
        let err = assess(good_plan(), true).expect_err("following a skill blocks the offer");
        assert_eq!(err, NotPromotable::FollowedAnExistingSkill);
    }

    /// Acceptance: a single-step or trivially short plan produces no offer.
    #[test]
    fn single_step_or_trivial_plan_produces_no_offer() {
        let one = vec![step("1", "write the file", Some("done"))];
        assert_eq!(
            assess(one, false).expect_err("one step is an act, not a method"),
            NotPromotable::TooFewSteps { succeeded: 1, needed: MIN_PROMOTABLE_STEPS }
        );

        let two = vec![
            step("1", "read the file", Some("read it")),
            step("2", "write the file", Some("wrote it")),
        ];
        assert_eq!(
            assess(two, false).expect_err("two steps is still not a method"),
            NotPromotable::TooFewSteps { succeeded: 2, needed: MIN_PROMOTABLE_STEPS }
        );
    }

    #[test]
    fn steps_without_a_recorded_outcome_do_not_count() {
        let steps = vec![
            step("1", "one", Some("found it")),
            step("2", "two", None),
            step("3", "three", Some("")),
        ];
        assert_eq!(
            assess(steps, false).expect_err("an unrecorded step teaches nothing"),
            NotPromotable::TooFewSteps { succeeded: 1, needed: MIN_PROMOTABLE_STEPS }
        );
    }

    #[test]
    fn a_mostly_abandoned_plan_produces_no_offer() {
        let steps = vec![
            step("1", "one", Some("found it")),
            step("2", "two", Some("found it")),
            step("3", "three", Some("found it")),
            abandoned("4", "four"),
            abandoned("5", "five"),
        ];
        assert_eq!(
            assess(steps, false).expect_err("half a plan abandoned is a search, not a method"),
            NotPromotable::TooManyAbandoned { abandoned: 2, total: 5 }
        );
    }

    #[test]
    fn one_abandoned_step_in_a_long_plan_is_tolerated() {
        let steps = vec![
            step("1", "one", Some("found it")),
            step("2", "two", Some("found it")),
            step("3", "three", Some("found it")),
            abandoned("4", "four"),
        ];
        let plan = assess(steps, false).expect("a dead end does not spoil a working method");
        assert_eq!(plan.working_steps().len(), 3);
    }

    #[test]
    fn a_plan_with_no_recorded_text_produces_no_offer() {
        let steps = vec![
            PlanStep { key: "1".into(), goal: WITHHELD_STEP_TEXT.into(), outcome: None, abandoned: false },
            PlanStep { key: "2".into(), goal: WITHHELD_STEP_TEXT.into(), outcome: None, abandoned: false },
            PlanStep { key: "3".into(), goal: WITHHELD_STEP_TEXT.into(), outcome: None, abandoned: false },
        ];
        assert_eq!(
            assess(steps, false).expect_err("a tainted turn records no procedure"),
            NotPromotable::NothingRecorded
        );
    }

    // --- render_skill_body: the acceptance criterion on provenance -----------

    /// Acceptance: the skill body is built from the plan's steps and outcomes,
    /// not from the raw transcript.
    #[test]
    fn skill_body_is_built_from_plan_steps_and_outcomes() {
        let plan = assess(good_plan(), false).expect("clears the bar");
        let body = render_skill_body("Fix a replayed migration", Some("Use when a migration replays."), &plan);

        for expected in [
            "Find the failing migration",
            "It is 041.",
            "Reproduce it on a scratch database",
            "Fails on replay.",
            "Guard the backfill",
            "Wrapped it in an existence check.",
        ] {
            assert!(body.contains(expected), "the body must carry {expected:?}");
        }
        assert!(body.contains("# Fix a replayed migration"));
        assert!(body.contains("Use when a migration replays."));
        assert_eq!(
            detect_kind(&body),
            SkillKind::Workflow,
            "a plan renders as a runnable workflow, not a prose playbook"
        );
    }

    #[test]
    fn skill_body_omits_abandoned_steps() {
        let steps = vec![
            step("1", "one", Some("found it")),
            step("2", "two", Some("found it")),
            step("3", "three", Some("found it")),
            abandoned("4", "the dead end"),
        ];
        let plan = assess(steps, false).expect("clears the bar");
        let body = render_skill_body("t", None, &plan);
        assert!(
            !body.contains("the dead end"),
            "a skill records what worked, not what was tried and dropped"
        );
    }

    #[test]
    fn skill_body_indents_nested_steps() {
        let steps = vec![
            step("1", "parent", Some("did it")),
            step("1.1", "child", Some("did it")),
            step("2", "sibling", Some("did it")),
        ];
        let plan = assess(steps, false).expect("clears the bar");
        let body = render_skill_body("t", None, &plan);
        let child_line = body
            .lines()
            .find(|l| l.contains("child"))
            .expect("the nested step is rendered");
        assert!(
            child_line.starts_with("    "),
            "a nested step is indented under its parent, got {child_line:?}"
        );
    }

    // --- render_skill_md -----------------------------------------------------

    #[test]
    fn rendered_skill_md_parses_back() {
        let plan = assess(good_plan(), false).expect("clears the bar");
        let body = render_skill_body("Fix a replayed migration", None, &plan);
        let md = render_skill_md(
            "fix-replayed-migration",
            "Use when a migration replays on an old database.",
            &["migrations".to_string(), "postgres".to_string()],
            &body,
        );
        let parsed = parse_skill_md(&md).expect("a rendered skill parses");
        assert_eq!(parsed.frontmatter.name, "fix-replayed-migration");
        assert_eq!(
            parsed.frontmatter.description,
            "Use when a migration replays on an old database."
        );
        assert_eq!(parsed.frontmatter.tags, vec!["migrations", "postgres"]);
        assert!(parsed.body.contains("## Steps"));
    }

    #[test]
    fn rendered_skill_md_quotes_a_description_that_would_break_yaml() {
        let md = render_skill_md("x", "Use when: a thing #happens", &[], "body");
        let parsed = parse_skill_md(&md).expect("a colon in the description must not break parsing");
        assert_eq!(parsed.frontmatter.description, "Use when: a thing #happens");
    }

    // --- render_offer: dedup -------------------------------------------------

    /// Acceptance (offer half): an existing skill covering the same procedure
    /// is named in the offer, so amending it is the obvious act.
    #[test]
    fn offer_names_existing_skills_that_may_cover_it() {
        let plan = assess(good_plan(), false).expect("clears the bar");
        let existing = [indexed("fix-replayed-migration", "Repair a migration that replays.")];
        let offer = render_offer(&plan, &existing);
        let matches = offer["existing"].as_array().expect("existing is an array");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["name"], "fix-replayed-migration");
        assert_eq!(offer["mode_hint"], "amend");
    }

    #[test]
    fn offer_with_no_match_hints_at_a_new_skill() {
        let plan = assess(good_plan(), false).expect("clears the bar");
        assert_eq!(render_offer(&plan, &[])["mode_hint"], "new");
    }

    #[test]
    fn offer_caps_the_matches_it_names() {
        let plan = assess(good_plan(), false).expect("clears the bar");
        let existing: Vec<IndexedSkill> = (0..10)
            .map(|i| indexed(&format!("skill-{i}"), "d"))
            .collect();
        let offer = render_offer(&plan, &existing);
        assert_eq!(
            offer["existing"].as_array().expect("array").len(),
            MAX_OFFERED_MATCHES
        );
    }

    // --- parse_promotion_request ---------------------------------------------

    #[test]
    fn promotion_request_defaults_to_a_new_skill() {
        let req = parse_promotion_request(&serde_json::json!({
            "name": "fix-replayed-migration",
            "description": "Use when a migration replays."
        }))
        .expect("valid");
        assert_eq!(req.mode, PromotionMode::New);
        assert!(req.tags.is_empty());
        assert_eq!(req.summary, None);
    }

    #[test]
    fn promotion_request_reads_amend_mode_and_tags() {
        let req = parse_promotion_request(&serde_json::json!({
            "name": "x",
            "description": "d",
            "mode": "amend",
            "tags": ["a", "b"],
            "summary": "what it is for"
        }))
        .expect("valid");
        assert_eq!(req.mode, PromotionMode::Amend);
        assert_eq!(req.tags, vec!["a", "b"]);
        assert_eq!(req.summary.as_deref(), Some("what it is for"));
    }

    #[test]
    fn promotion_request_rejects_a_traversing_name() {
        for bad in ["../etc", "a/b", "", "."] {
            assert!(
                matches!(
                    parse_promotion_request(&serde_json::json!({"name": bad, "description": "d"})),
                    Err(SkillError::InvalidName(_))
                ),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn promotion_request_rejects_an_empty_description() {
        assert!(
            parse_promotion_request(&serde_json::json!({"name": "x", "description": "  "}))
                .is_err(),
            "a skill with no trigger cannot be found again"
        );
    }

    // --- the tool definition -------------------------------------------------

    #[test]
    fn promote_tool_is_named_and_takes_the_fields_the_model_supplies() {
        let def = promote_plan_tool();
        assert_eq!(def.name, PROMOTE_PLAN_TOOL);
        let props = &def.parameters["properties"];
        for field in ["name", "description", "mode", "tags", "summary"] {
            assert!(props.get(field).is_some(), "{field} must be advertised");
        }
        assert!(
            props.get("body").is_none() && props.get("steps").is_none(),
            "the body comes from the plan, never from the model"
        );
    }
}
