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

use std::collections::HashMap;

use crate::domain::skill::{IndexedSkill, SkillError, validate_skill_name};
use crate::domain::tool::ToolDefinition;
use crate::domain::{Message, Role};
use crate::planning::{OUTCOME_KEY_PREFIX, STEP_NOTE_TYPE};
#[cfg(test)]
use crate::tool_provenance::WITHHELD_STEP_TEXT;
use crate::tool_provenance::is_withheld_step_text;

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

/// `source` recorded on a skill the assistant wrote from its own completed
/// plan, so the catalog can tell self-authored rows from scanned ones.
pub const SELF_AUTHORED_SOURCE: &str = "self-authored";

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
    let outcomes: HashMap<&str, &str> = notes
        .iter()
        .filter_map(|n| {
            n.key
                .strip_prefix(OUTCOME_KEY_PREFIX)
                .map(|k| (k, n.content))
        })
        .collect();

    let mut steps: Vec<PlanStep> = notes
        .iter()
        .filter(|n| n.note_type == STEP_NOTE_TYPE && n.done && is_step_key(n.key))
        .map(|n| {
            let (abandoned, outcome) = read_outcome(outcomes.get(n.key).copied());
            PlanStep {
                key: n.key.to_string(),
                goal: n.content.trim().to_string(),
                outcome,
                abandoned,
            }
        })
        .collect();
    steps.sort_by_key(|s| dotted(&s.key));
    steps
}

/// Whether a note key is a dotted step path (`1`, `1.2`, `1.2.3`).
///
/// The plan's own keys are minted from stack depth, so anything else in the
/// `todo` list is a note the model or the user wrote by hand and is not part of
/// the procedure.
fn is_step_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

/// Split a stored outcome note into `(abandoned, finding)`.
///
/// A withheld placeholder is not a finding: it says a step happened and nothing
/// about what it produced (#741).
fn read_outcome(raw: Option<&str>) -> (bool, Option<String>) {
    let Some(text) = raw.map(str::trim).filter(|t| !t.is_empty()) else {
        return (false, None);
    };
    if is_withheld_step_text(text) {
        return (false, None);
    }
    match text.strip_prefix(ABANDONED_PREFIX) {
        Some(rest) => {
            let rest = rest.trim();
            (true, (!rest.is_empty()).then(|| rest.to_string()))
        }
        None => (false, Some(text.to_string())),
    }
}

/// A dotted key as numbers, so `1.10` sorts after `1.2` rather than before it.
///
/// `is_step_key` has already proved every part is ASCII digits, so the only way
/// a part fails to parse is overflow. Such a key sorts LAST rather than first:
/// it did not come from the step stack, and putting an unexplained key at the
/// head of a rendered procedure would read as its first instruction.
fn dotted(key: &str) -> Vec<u64> {
    key.split('.')
        .map(|p| p.parse().unwrap_or(u64::MAX))
        .collect()
}

/// Keep only the steps a turn opened for itself.
///
/// Step notes live in the conversation's scratchpad, not the turn's, and the
/// step stack continues numbering where the last turn stopped. So a plain read
/// returns every step the conversation ever completed, and a later plan would
/// be judged - and written - as though the earlier ones were part of it: two
/// unrelated two-step plans would clear a three-step bar between them, and the
/// skill body would splice two procedures together.
///
/// `previous_top_level` is the highest top-level step number that existed when
/// this turn began, so a step is this turn's when its top-level number is
/// greater. Nested steps ride on their parent's number, so they are kept or
/// dropped with it.
pub fn steps_this_turn(steps: Vec<PlanStep>, previous_top_level: u32) -> Vec<PlanStep> {
    steps
        .into_iter()
        .filter(|s| {
            s.key
                .split('.')
                .next()
                .and_then(|head| head.parse::<u32>().ok())
                .is_some_and(|top| top > previous_top_level)
        })
        .collect()
}

/// Whether the turn read an existing skill before it planned.
///
/// True when the turn called [`SKILL_GET_TOOL`] and got a skill back. A call
/// that was refused - the name does not exist, or the skill is still awaiting
/// approval - read nothing, so it is not a skill being followed, and treating
/// it as one would decline a perfectly good plan.
///
/// Conservative where it cannot tell: a result the turn has already compacted
/// to a scratchpad pointer no longer says whether it succeeded, and that counts
/// as followed. A missed offer costs nothing; a near-duplicate skill costs
/// attention on every later search.
pub fn followed_a_skill(messages: &[Message]) -> bool {
    let refused: std::collections::HashSet<&str> = messages
        .iter()
        .filter(|m| m.role == Role::Tool && is_refusal(&m.content))
        .filter_map(|m| m.tool_call_id.as_deref())
        .collect();

    messages.iter().any(|m| {
        m.role == Role::Assistant
            && m.tool_calls
                .iter()
                .any(|c| c.name == SKILL_GET_TOOL && !refused.contains(c.id.as_str()))
    })
}

/// Whether a tool result is the skill library saying it returned nothing.
///
/// Reads the structured `ok` field rather than matching English, so the
/// judgement survives a reworded message.
fn is_refusal(content: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(content).is_ok_and(|v| v["ok"] == false)
}

/// Decide whether a completed plan is worth offering as a skill.
///
/// The bar, in one place: the plan was not itself a skill being followed, at
/// least [`MIN_PROMOTABLE_STEPS`] steps finished and recorded what they
/// produced, and no more than a third of the plan was abandoned.
pub fn assess(
    steps: Vec<PlanStep>,
    followed_a_skill: bool,
) -> Result<PromotablePlan, NotPromotable> {
    if followed_a_skill {
        return Err(NotPromotable::FollowedAnExistingSkill);
    }
    // A tainted turn records a placeholder in place of the model's own wording
    // (#741), so there is a plan-shaped set of notes with no procedure in it.
    // Reported on its own rather than as "too few steps", because the fix is
    // different: nothing about the plan was wrong.
    if !steps.is_empty() && steps.iter().all(|s| is_withheld_step_text(&s.goal)) {
        return Err(NotPromotable::NothingRecorded);
    }

    let succeeded = steps.iter().filter(|s| s.succeeded()).count();
    if succeeded < MIN_PROMOTABLE_STEPS {
        return Err(NotPromotable::TooFewSteps {
            succeeded,
            needed: MIN_PROMOTABLE_STEPS,
        });
    }

    let abandoned = steps.iter().filter(|s| s.abandoned).count();
    let total = steps.len();
    if abandoned * MAX_ABANDONED_DENOMINATOR > total {
        return Err(NotPromotable::TooManyAbandoned { abandoned, total });
    }

    Ok(PromotablePlan { steps })
}

/// Render the markdown body of a skill from a plan.
///
/// Built from the plan's steps and their outcomes only. The `## Steps` heading
/// is what makes the result a workflow rather than a prose playbook (see
/// [`crate::domain::skill::detect_kind`]), so it is written exactly.
pub fn render_skill_body(title: &str, summary: Option<&str>, plan: &PromotablePlan) -> String {
    let mut out = format!("# {}\n\n", title.trim());
    if let Some(summary) = summary.map(str::trim).filter(|s| !s.is_empty()) {
        out.push_str(summary);
        out.push_str("\n\n");
    }
    out.push_str("## Steps\n\n");

    // A step's own leaf number, so a nested step reads as `1.` under its
    // parent rather than repeating the whole dotted path.
    for step in plan.working_steps() {
        let indent = "    ".repeat(step.depth());
        let leaf = step.key.rsplit('.').next().unwrap_or(&step.key);
        out.push_str(&format!("{indent}{leaf}. {}\n", step.goal));
        if let Some(outcome) = &step.outcome {
            for line in outcome.lines() {
                out.push_str(&format!("{indent}   {line}\n"));
            }
        }
        out.push('\n');
    }
    out
}

/// Render a whole `SKILL.md`: YAML frontmatter plus the body.
///
/// The frontmatter carries the fields the shared cross-product format requires
/// (`name`, `description`, `tags`), so the result parses back through
/// [`crate::domain::skill::parse_skill_md`].
pub fn render_skill_md(name: &str, description: &str, tags: &[String], body: &str) -> String {
    let tags = tags
        .iter()
        .map(|t| yaml_string(t))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "---\nname: {}\ndescription: {}\ntags: [{tags}]\n---\n\n{body}",
        yaml_string(name),
        yaml_string(description),
    )
}

/// Quote a value as a single-line YAML double-quoted scalar.
///
/// Always quoted, never conditionally: a description is free text the model
/// wrote, and a leading `#`, a `:` or a `-` in it would otherwise change what
/// the document means.
///
/// Control characters are escaped rather than emitted, which is the part that
/// matters for safety as well as fidelity. A raw newline inside a quoted scalar
/// is folded to a space by YAML, so the text would come back changed; and a
/// description carrying a line that is exactly `---` would end the frontmatter
/// early, so the rest of it would be read as the skill's body. Escaping keeps
/// the scalar on one line, which makes both impossible.
fn yaml_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // U+2028/U+2029 are separators, not `char::is_control()`
            // controls, and YAML treats them as line breaks: left raw they
            // would fold the same way a newline does.
            c if c.is_control() || c == '\u{2028}' || c == '\u{2029}' => {
                out.push_str(&format!("\\u{:04x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The offer appended to a `complete_step` acknowledgement when a finished plan
/// clears the bar.
///
/// `existing` are catalog entries that may already cover this procedure;
/// amending one of those is the useful act, and adding a second is not.
/// Declining is doing nothing.
pub fn render_offer(plan: &PromotablePlan, existing: &[IndexedSkill]) -> serde_json::Value {
    let matches: Vec<serde_json::Value> = existing
        .iter()
        .take(MAX_OFFERED_MATCHES)
        .map(|s| {
            serde_json::json!({
                "name": s.name,
                "description": s.description,
                "approved": s.is_approved(),
            })
        })
        .collect();
    // The hint has to agree with what `decide` will actually allow. Amend only
    // works on the assistant's own unadopted drafts, so hinting at it when
    // every match belongs to a person would steer the model into a refusal.
    let amendable = existing.iter().take(MAX_OFFERED_MATCHES).any(is_own_draft);
    let covered = !matches.is_empty();
    serde_json::json!({
        "tool": PROMOTE_PLAN_TOOL,
        "steps": plan.working_steps().len(),
        "mode_hint": if amendable { "amend" } else { "new" },
        "existing": matches,
        "bar": "Offer it only if the method would work again on different inputs. \
                A question answered, a file written, or a one-off fix is not a skill.",
        "note": match (covered, amendable) {
            (_, true) => "The library already has a draft close to this. Amend that draft \
                          rather than adding a second skill about the same thing, or say \
                          nothing and this is dropped.",
            (true, false) => "The library already covers this, and those skills are not \
                              yours to revise. Add one only if your method is genuinely \
                              different; otherwise say nothing and this is dropped.",
            (false, false) => "Call the tool if this generalises. Saying nothing drops the \
                               plan, which costs nothing.",
        },
    })
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
    let name = args
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim();
    validate_skill_name(name)?;

    let description = args
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim();
    if description.is_empty() {
        return Err(SkillError::InvalidFrontmatter(
            "description is required: a skill with no trigger cannot be found again".to_string(),
        ));
    }

    let mode = match args.get("mode").and_then(|v| v.as_str()).unwrap_or("new") {
        "new" => PromotionMode::New,
        "amend" => PromotionMode::Amend,
        other => {
            return Err(SkillError::InvalidFrontmatter(format!(
                "unknown mode {other:?}: expected \"new\" or \"amend\""
            )));
        }
    };

    let tags = args
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|t| t.as_str())
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let summary = args
        .get("summary")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    Ok(PromotionRequest {
        name: name.to_string(),
        description: description.to_string(),
        summary,
        tags,
        mode,
    })
}

/// The tool definition the dispatch loop advertises for accepting an offer.
pub fn promote_plan_tool() -> ToolDefinition {
    ToolDefinition::new(
        PROMOTE_PLAN_TOOL,
        "Keep the plan you just finished as a reusable skill. Call it only when the \
         method would work again on different inputs - a question answered, a single \
         file written, or a one-off fix is not a skill. The steps come from the plan \
         itself, so you supply only how the skill is found and what it is for. The \
         saved skill is UNAPPROVED and cannot be followed until a person approves it. \
         Use mode=\"amend\" with an existing skill's name to revise that skill instead \
         of adding a near-duplicate.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short hyphenated skill name, e.g. \"fix-replayed-migration\". \
                                    With mode=\"amend\", the existing skill's name."
                },
                "description": {
                    "type": "string",
                    "description": "One or two sentences saying WHEN to use the skill. This is \
                                    what a later search matches on."
                },
                "mode": {
                    "type": "string",
                    "enum": ["new", "amend"],
                    "description": "\"new\" adds a skill; \"amend\" revises the existing skill \
                                    of this name. Defaults to \"new\"."
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for grouping."
                },
                "summary": {
                    "type": "string",
                    "description": "Optional short paragraph on what the procedure is for, \
                                    placed above the steps."
                }
            },
            "required": ["name", "description"],
            "additionalProperties": false
        }),
    )
}

/// What a promotion request may do, given the catalog row that already holds
/// the requested name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionAct {
    /// Add a skill of this name.
    Create,
    /// Replace the body of the skill of this name.
    Revise,
    /// Do nothing, and say why.
    Refuse(String),
}

/// Whether a catalog row is one the assistant may revise on its own.
///
/// Only its own drafts. A skill a person put in a skill root, or approved, or
/// installed from anywhere else is theirs: revising it would swap the body
/// under an approval that was given to different words, relabel its provenance
/// as self-authored, and mark it absent from a disk where its file is still
/// sitting. Amending is the useful act only when the thing being amended is a
/// draft nobody has adopted yet.
pub fn is_own_draft(skill: &IndexedSkill) -> bool {
    !skill.is_approved()
        && !skill.present_on_disk
        && matches!(
            skill.source.as_deref(),
            Some(SELF_AUTHORED_SOURCE) | Some(EXTRACTED_SOURCE) | Some(MISFILED_SOURCE)
        )
}

/// `source` recorded on a skill the dream cycle extracted from a transcript.
///
/// Declared here beside [`SELF_AUTHORED_SOURCE`] because [`is_own_draft`] has
/// to recognise both, and a rule that names one marker and forgets the other is
/// the kind of gap that lets an unattended write reach a person's skill.
pub const EXTRACTED_SOURCE: &str = "extraction";

/// `source` recorded on a skill proposed from a knowledge entry that was really
/// a procedure (#1175).
///
/// Its own marker rather than [`EXTRACTED_SOURCE`], because the two answer a
/// different question for the person deciding whether to approve it: one was
/// found in a conversation, and this one says "you already have this written
/// down as a fact, and it does not work as one". The entry it came from is
/// named in the skill's `metadata.from_entry`, so a person can act on both
/// halves of the split.
///
/// [`is_own_draft`] recognises it for the reason it recognises the other two:
/// a later sweep must be able to revise its own unadopted proposal, and must
/// never touch anything else.
pub const MISFILED_SOURCE: &str = "misfiled-knowledge";

/// The one rule that decides which store a piece of learning belongs in
/// (#1155, #1175).
///
/// Stated once, in the domain, and spent by every path that can write a
/// knowledge entry: the dream cycle's extraction prompt, its consolidation
/// prompt, and the write tool the model calls inside a turn. A rule stated in
/// one prompt is a rule the other paths do not have, which is how the knowledge
/// store filled with routines written as facts in the first place.
///
/// It is a sentence rather than a paragraph because two of its three readers
/// pay for it on every call. The paragraph that says what a skill's fields are
/// belongs with the path that asks for those fields.
///
/// **A prompt is not an enforcer.** This makes the rule reach every writer; it
/// does not make any of them obey. What catches what still slips through is the
/// sweep that proposes a mis-filed entry as a skill after the fact.
pub const METHOD_IS_NOT_A_FACT: &str = "Knowledge records what is TRUE. A skill records HOW TO DO something. A method - ordered \
     steps, a repeatable how-to - belongs in the skill library and not here: filed as a fact it \
     reads as neither, competes with real facts for attention, and cannot be followed by \
     anything. A method's PREFERENCES do stay knowledge: which sources to use, in what order, \
     what to skip, and what \"done\" looks like for this person.";

/// Decide what a promotion request may do.
///
/// Two rules, and both are about never doing damage a person did not ask for. A
/// request to add a skill whose name is already taken is refused, never
/// satisfied by a second row, because every skill competes for the same
/// attention budget when the library is searched. And a request to amend is
/// refused unless the row is the assistant's own unadopted draft ([`is_own_draft`]).
pub fn decide(req: &PromotionRequest, existing: Option<&IndexedSkill>) -> PromotionAct {
    match (req.mode, existing) {
        (PromotionMode::New, None) => PromotionAct::Create,
        (PromotionMode::New, Some(found)) => PromotionAct::Refuse(format!(
            "a skill named {:?} already exists ({}). Revise it with mode=\"amend\", or pick a \
             name for a genuinely different procedure.",
            req.name, found.description
        )),
        (PromotionMode::Amend, Some(found)) if is_own_draft(found) => PromotionAct::Revise,
        (PromotionMode::Amend, Some(_)) => PromotionAct::Refuse(format!(
            "{:?} is not yours to revise: it was approved, or it came from a skill root or \
             another source. Pick a different name, or ask the user to change that skill.",
            req.name
        )),
        (PromotionMode::Amend, None) => PromotionAct::Refuse(format!(
            "there is no skill named {:?} to amend. Use mode=\"new\" to add one.",
            req.name
        )),
    }
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
            step(
                "2",
                "Reproduce it on a scratch database",
                Some("Fails on replay."),
            ),
            step(
                "3",
                "Guard the backfill",
                Some("Wrapped it in an existence check."),
            ),
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
            PlanNote {
                key: "1",
                content: "Find the failing migration",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "outcome:1",
                content: "It is 041.",
                note_type: "note",
                done: false,
            },
            PlanNote {
                key: "2",
                content: "Guard the backfill",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "outcome:2",
                content: "Wrapped it.",
                note_type: "note",
                done: false,
            },
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
            PlanNote {
                key: "1",
                content: "done step",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "2",
                content: "still open",
                note_type: STEP_NOTE_TYPE,
                done: false,
            },
            PlanNote {
                key: "buy-milk",
                content: "not a step",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "goal",
                content: "the overall goal",
                note_type: "note",
                done: false,
            },
        ];
        let steps = plan_from_notes(&notes);
        assert_eq!(steps.len(), 1, "only the completed dotted step counts");
        assert_eq!(steps[0].key, "1");
    }

    #[test]
    fn plan_recognises_an_abandoned_step() {
        let notes = [
            PlanNote {
                key: "1",
                content: "try the fast path",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "outcome:1",
                content: "Abandoned: no index to use",
                note_type: "note",
                done: false,
            },
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
            PlanNote {
                key: "2",
                content: "second",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "1.10",
                content: "tenth child",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "1.2",
                content: "second child",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "1",
                content: "first",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
        ];
        let keys: Vec<String> = plan_from_notes(&notes).into_iter().map(|s| s.key).collect();
        assert_eq!(keys, vec!["1", "1.2", "1.10", "2"], "numeric, not lexical");
    }

    #[test]
    fn plan_treats_withheld_step_text_as_nothing_recorded() {
        let notes = [
            PlanNote {
                key: "1",
                content: WITHHELD_STEP_TEXT,
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "outcome:1",
                content: WITHHELD_STEP_TEXT,
                note_type: "note",
                done: false,
            },
        ];
        let steps = plan_from_notes(&notes);
        assert!(steps[0].outcome.is_none(), "a placeholder is not a finding");
        assert!(!steps[0].succeeded());
    }

    // --- steps_this_turn -----------------------------------------------------

    /// Step notes outlive the turn that wrote them, so a plan must be judged on
    /// the steps THIS turn opened. Without this, two unrelated two-step plans
    /// in one conversation would clear a three-step bar between them and be
    /// written as one spliced procedure.
    #[test]
    fn a_later_plan_does_not_inherit_an_earlier_turns_steps() {
        let all = vec![
            step("1", "turn one, step one", Some("a")),
            step("1.1", "turn one, nested", Some("b")),
            step("2", "turn one, step two", Some("c")),
            step("3", "turn two, step one", Some("d")),
            step("3.1", "turn two, nested", Some("e")),
            step("4", "turn two, step two", Some("f")),
        ];
        let mine = steps_this_turn(all.clone(), 2);
        let keys: Vec<&str> = mine.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["3", "3.1", "4"],
            "a nested step rides on its parent"
        );

        assert_eq!(
            steps_this_turn(all.clone(), 0).len(),
            6,
            "a first turn keeps everything it opened"
        );

        // The harm the filter prevents: two unrelated two-step plans in one
        // conversation, neither of them a method, clearing the bar between them
        // and being written as one spliced procedure.
        let two_and_two = vec![
            step("1", "turn one, step one", Some("a")),
            step("2", "turn one, step two", Some("b")),
            step("3", "turn two, step one", Some("c")),
            step("4", "turn two, step two", Some("d")),
        ];
        assert_eq!(
            assess(two_and_two.clone(), false)
                .expect("unfiltered, four steps wrongly clear the bar")
                .working_steps()
                .len(),
            4
        );
        assert_eq!(
            assess(steps_this_turn(two_and_two, 2), false)
                .expect_err("this turn opened two steps, which is not a method"),
            NotPromotable::TooFewSteps {
                succeeded: 2,
                needed: MIN_PROMOTABLE_STEPS
            }
        );
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
    fn a_refused_skill_read_did_not_follow_one() {
        // The library said no - the skill does not exist, or it is awaiting
        // approval - so the turn read nothing and its plan is its own.
        let mut call = Message::new(Role::Assistant, "");
        call.tool_calls = vec![crate::domain::tool::ToolCall {
            id: "c1".to_string(),
            name: SKILL_GET_TOOL.to_string(),
            arguments: "{}".to_string(),
        }];
        let mut result = Message::new(Role::Tool, r#"{"ok":false,"awaiting_approval":true}"#);
        result.tool_call_id = Some("c1".to_string());
        assert!(!followed_a_skill(&[call, result]));
    }

    #[test]
    fn a_compacted_skill_read_counts_as_followed() {
        // The result no longer says whether it succeeded, so the conservative
        // reading stands: a missed offer costs less than a near-duplicate.
        let mut call = Message::new(Role::Assistant, "");
        call.tool_calls = vec![crate::domain::tool::ToolCall {
            id: "c1".to_string(),
            name: SKILL_GET_TOOL.to_string(),
            arguments: "{}".to_string(),
        }];
        let mut result = Message::new(Role::Tool, "<compacted to scratchpad note outcome:1>");
        result.tool_call_id = Some("c1".to_string());
        assert!(followed_a_skill(&[call, result]));
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
        let plan =
            assess(good_plan(), false).expect("a three-step plan with outcomes clears the bar");
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
            NotPromotable::TooFewSteps {
                succeeded: 1,
                needed: MIN_PROMOTABLE_STEPS
            }
        );

        let two = vec![
            step("1", "read the file", Some("read it")),
            step("2", "write the file", Some("wrote it")),
        ];
        assert_eq!(
            assess(two, false).expect_err("two steps is still not a method"),
            NotPromotable::TooFewSteps {
                succeeded: 2,
                needed: MIN_PROMOTABLE_STEPS
            }
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
            NotPromotable::TooFewSteps {
                succeeded: 1,
                needed: MIN_PROMOTABLE_STEPS
            }
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
            NotPromotable::TooManyAbandoned {
                abandoned: 2,
                total: 5
            }
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
            PlanStep {
                key: "1".into(),
                goal: WITHHELD_STEP_TEXT.into(),
                outcome: None,
                abandoned: false,
            },
            PlanStep {
                key: "2".into(),
                goal: WITHHELD_STEP_TEXT.into(),
                outcome: None,
                abandoned: false,
            },
            PlanStep {
                key: "3".into(),
                goal: WITHHELD_STEP_TEXT.into(),
                outcome: None,
                abandoned: false,
            },
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
        let body = render_skill_body(
            "Fix a replayed migration",
            Some("Use when a migration replays."),
            &plan,
        );

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
        let parsed =
            parse_skill_md(&md).expect("a colon in the description must not break parsing");
        assert_eq!(parsed.frontmatter.description, "Use when: a thing #happens");
    }

    // --- render_offer: dedup -------------------------------------------------

    /// Acceptance (offer half): an existing skill covering the same procedure
    /// is named in the offer, so amending it is the obvious act.
    #[test]
    fn offer_names_existing_skills_that_may_cover_it() {
        let plan = assess(good_plan(), false).expect("clears the bar");
        let existing = [indexed(
            "fix-replayed-migration",
            "Repair a migration that replays.",
        )];
        let offer = render_offer(&plan, &existing);
        let matches = offer["existing"].as_array().expect("existing is an array");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["name"], "fix-replayed-migration");
        assert_eq!(offer["mode_hint"], "amend");
    }

    /// The hint must agree with what the write will allow: a match a person
    /// owns cannot be amended, so steering the model at it would only produce
    /// a refusal.
    #[test]
    fn offer_does_not_hint_at_amending_a_skill_a_person_owns() {
        let plan = assess(good_plan(), false).expect("clears the bar");
        let mut theirs = indexed("fix-replayed-migration", "Repair a migration that replays.");
        theirs.approved_at = Some(chrono::Utc::now());
        let offer = render_offer(&plan, &[theirs]);
        assert_eq!(offer["mode_hint"], "new");
        assert_eq!(
            offer["existing"].as_array().expect("array").len(),
            1,
            "the match is still named, so the model can decline knowingly"
        );
        assert!(
            offer["note"]
                .as_str()
                .expect("a note")
                .contains("not yours to revise")
        );
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

/// Fixtures shared by the two test modules below.
#[cfg(test)]
mod tests_support {
    use super::*;
    use crate::domain::skill::{Locality, SkillKind, TrustTier};

    pub fn request(name: &str, mode: PromotionMode) -> PromotionRequest {
        PromotionRequest {
            name: name.to_string(),
            description: "when to use it".to_string(),
            summary: None,
            tags: Vec::new(),
            mode,
        }
    }

    pub fn existing_skill(name: &str) -> IndexedSkill {
        IndexedSkill {
            name: name.to_string(),
            description: "the one already in the catalog".to_string(),
            kind: SkillKind::Workflow,
            disk_path: String::new(),
            owner_user_id: Some("someone".to_string()),
            locality: Locality::Daemon,
            content_hash: "hash".to_string(),
            trust_tier: TrustTier::Local,
            source: Some(SELF_AUTHORED_SOURCE.to_string()),
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
}

#[cfg(test)]
mod dedup_tests {
    use super::tests_support::*;
    use super::*;

    /// Acceptance: an existing skill covering the same procedure is amended or
    /// declined, never duplicated.
    #[test]
    fn adding_over_an_existing_name_is_refused_not_duplicated() {
        let req = request("deploy", PromotionMode::New);
        let act = decide(&req, Some(&existing_skill("deploy")));
        let PromotionAct::Refuse(why) = act else {
            panic!("a second skill of the same name must never be created");
        };
        assert!(
            why.contains("amend"),
            "the refusal names the useful act: {why}"
        );
    }

    #[test]
    fn amending_its_own_unadopted_draft_revises_it() {
        let req = request("deploy", PromotionMode::Amend);
        let draft = existing_skill("deploy");
        assert!(is_own_draft(&draft));
        assert_eq!(decide(&req, Some(&draft)), PromotionAct::Revise);
    }

    /// The damage this guard prevents: amending swaps the body, relabels the
    /// provenance as self-authored, marks the row absent from disk, and drops
    /// the approval. None of that may happen to a skill a person owns.
    #[test]
    fn amending_a_skill_a_person_owns_is_refused() {
        let req = request("deploy", PromotionMode::Amend);

        let mut approved = existing_skill("deploy");
        approved.approved_at = Some(chrono::Utc::now());
        assert!(!is_own_draft(&approved));
        assert!(
            matches!(decide(&req, Some(&approved)), PromotionAct::Refuse(_)),
            "an approved skill is not the assistant's to rewrite"
        );

        let mut on_disk = existing_skill("deploy");
        on_disk.present_on_disk = true;
        assert!(
            matches!(decide(&req, Some(&on_disk)), PromotionAct::Refuse(_)),
            "a skill whose file is still in a skill root is not the assistant's to rewrite"
        );

        let mut installed = existing_skill("deploy");
        installed.source = Some("https://github.com/example/skills".to_string());
        assert!(
            matches!(decide(&req, Some(&installed)), PromotionAct::Refuse(_)),
            "a skill from another source is not the assistant's to rewrite"
        );

        let mut extracted = existing_skill("deploy");
        extracted.source = Some(EXTRACTED_SOURCE.to_string());
        assert_eq!(
            decide(&req, Some(&extracted)),
            PromotionAct::Revise,
            "the dream cycle's own unadopted draft is fair game"
        );
    }

    #[test]
    fn adding_a_new_name_creates_it() {
        assert_eq!(
            decide(&request("fresh", PromotionMode::New), None),
            PromotionAct::Create
        );
    }

    #[test]
    fn amending_a_skill_that_does_not_exist_is_refused() {
        let act = decide(&request("ghost", PromotionMode::Amend), None);
        let PromotionAct::Refuse(why) = act else {
            panic!("there is nothing to amend");
        };
        assert!(
            why.contains("new"),
            "the refusal names the useful act: {why}"
        );
    }
}

#[cfg(test)]
mod yaml_tests {
    use super::*;
    use crate::domain::skill::parse_skill_md;

    /// The model writes the description, so it is arbitrary text. Every shape
    /// it can take must survive the round trip, or a skill lands in the catalog
    /// with a description nothing can read back.
    #[test]
    fn every_awkward_description_round_trips() {
        for description in [
            "Use when: a thing #happens",
            "Two\nlines",
            "A \"quoted\" phrase",
            "A back\\slash",
            "A line that ends the frontmatter:\n---\nand more",
            "- looks like a list item",
            "{braces} and [brackets]",
            "Tab\there",
            "trailing spaces   ",
            "*emphasis* and &anchor and *alias",
            "100%",
            "line\u{2028}separator",
            "paragraph\u{2029}separator",
        ] {
            let md = render_skill_md("x", description, &[], "body");
            let parsed = parse_skill_md(&md)
                .unwrap_or_else(|e| panic!("{description:?} broke the frontmatter: {e}"));
            assert_eq!(
                parsed.frontmatter.description, description,
                "{description:?} did not survive the round trip"
            );
        }
    }

    #[test]
    fn an_awkward_tag_round_trips() {
        let tags = vec![
            "a: b".to_string(),
            "\"quoted\"".to_string(),
            "c\\d".to_string(),
        ];
        let md = render_skill_md("x", "d", &tags, "body");
        let parsed = parse_skill_md(&md).expect("parses");
        assert_eq!(parsed.frontmatter.tags, tags);
    }

    /// A plan whose every step was abandoned cannot reach the renderer through
    /// `assess`, but the renderer must still produce a well-formed document
    /// rather than something `detect_kind` reads as a plain playbook.
    #[test]
    fn a_body_with_no_working_steps_is_still_a_workflow() {
        let plan = PromotablePlan { steps: Vec::new() };
        let body = render_skill_body("t", None, &plan);
        assert_eq!(
            crate::domain::skill::detect_kind(&body),
            crate::domain::SkillKind::Workflow
        );
        assert!(parse_skill_md(&render_skill_md("x", "d", &[], &body)).is_ok());
    }

    /// A step key the store could hold that is not the shape the stack mints.
    #[test]
    fn an_absurd_step_key_sorts_last_rather_than_first() {
        let notes = [
            PlanNote {
                key: "99999999999999999999",
                content: "huge",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
            PlanNote {
                key: "1",
                content: "one",
                note_type: STEP_NOTE_TYPE,
                done: true,
            },
        ];
        let steps = plan_from_notes(&notes);
        assert_eq!(steps.len(), 2, "an unparseable number is kept, not dropped");
        assert_eq!(
            steps[0].key, "1",
            "and the overflowing key sorts as zero-or-last, not randomly"
        );
    }
}
