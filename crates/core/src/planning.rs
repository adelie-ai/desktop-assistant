//! Step-scoped planning and context compaction for long agentic turns (#240).
//!
//! The model works a non-trivial request the way a person with a scratchpad
//! and pen would: break it into ordered steps, work each one, and — when a
//! step turns out to need its own sub-plan — open nested sub-steps. As each
//! step finishes, the *gist* of what was learned is jotted to the scratchpad
//! and the verbose raw work (tool results) is **dropped from working
//! context**, replaced by a short searchable pointer to the note. The plan
//! itself stays cheaply in view; the firehose does not.
//!
//! This module is the pure mechanism behind that behaviour:
//!
//! - [`StepStack`] — a per-turn stack of [`StepFrame`]s. `begin` pushes a
//!   frame and auto-assigns a dotted path from stack depth + a per-frame child
//!   counter (step 1 → 1.1, 1.2, …; 1.2 → 1.2.1 … 1.2.6). `complete` pops it.
//! - [`evict_tool_results`] — replaces the content of sizeable `Role::Tool`
//!   messages in a scope with a pointer to the scratchpad note that distilled
//!   them, **preserving role + `tool_call_id`** so provider ToolUse↔ToolResult
//!   pairing stays valid (Bedrock/Ollama). Idempotent and structure-preserving.
//! - [`render_plan`] — renders the open todos as a compact indented tree for
//!   per-round surfacing.
//! - [`begin_step_tool`] / [`complete_step_tool`] — the tool definitions the
//!   dispatch loop advertises and intercepts (they are core-loop tools, not
//!   MCP/builtin-executor tools, because only the loop owns `conv.messages`).
//!
//! The async orchestration (writing the todo/outcome notes through the wired
//! scratchpad closures, then mutating `conv.messages`) lives in the service
//! dispatch loop; everything here is synchronous and unit-tested in isolation.

use crate::domain::{Message, Role, ToolDefinition};
use crate::ports::scratchpad::SCRATCHPAD_GOAL_KEY;

/// Tool the model calls to begin a (possibly nested) step. Advertised in the
/// per-turn tool set and intercepted by name in the dispatch loop.
pub const BEGIN_STEP_TOOL: &str = "begin_step";

/// Tool the model calls to complete the current step — distil + evict.
pub const COMPLETE_STEP_TOOL: &str = "complete_step";

/// `note_type` used for plan steps so they sort/filter as ordered todos
/// (matching the existing scratchpad `todo`/`sequence`/`done` convention).
pub const STEP_NOTE_TYPE: &str = "todo";

/// `note_type` used for the distilled carry-forward outcome of a step.
pub const OUTCOME_NOTE_TYPE: &str = "note";

/// Key prefix under which a step's distilled outcome note is stored
/// (`outcome:<step-key>`). The plan renderer uses it to attach a step's finding
/// to its todo and to decide when a finding has been rolled up.
pub(crate) const OUTCOME_KEY_PREFIX: &str = "outcome:";

/// Only `Role::Tool` results at least this many bytes are worth evicting —
/// below it the pointer can be larger than the payload, so the savings are
/// negligible. This threshold also conveniently skips the tiny JSON acks of
/// the step-control tools themselves.
pub(crate) const COMPACTION_MIN_EVICT_BYTES: usize = 512;

/// Recognisable opening of an eviction pointer. Used to skip results that are
/// already compacted, so a parent `complete_step` whose scope contains
/// already-compacted child results does not re-stamp them.
pub(crate) const COMPACTION_POINTER_PREFIX: &str = "<compacted to scratchpad";

/// Maximum plan todos rendered into the per-round `[Plan]` surface. Keeps the
/// re-sent-every-round plan cheap; deeper plans show a "… and N more" tail.
pub(crate) const MAX_PLAN_ITEMS: usize = 40;

/// One frame of an in-progress plan: a step and the working scope opened when
/// it began.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepFrame {
    /// Dotted step path, e.g. `"1"`, `"1.2"`, `"1.2.3"`.
    pub key: String,
    /// The step's objective — becomes the `todo` note's content.
    pub goal: String,
    /// `conv.messages.len()` captured when this step began. `complete_step`
    /// evicts `Role::Tool` results from here to the current end of the log.
    pub watermark: usize,
    /// Child steps minted under this frame so far (drives `.1`, `.2`, …).
    pub child_counter: u32,
    /// Ordering hint for the todo note (the leaf number of `key`).
    pub sequence: i32,
}

/// A per-turn stack of plan steps. Auto-numbers dotted paths from structure,
/// so the model never has to track step numbers — it just begins and completes.
#[derive(Debug, Default)]
pub(crate) struct StepStack {
    frames: Vec<StepFrame>,
    /// Top-level steps minted so far (children of the implicit root).
    root_counter: u32,
}

impl StepStack {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a stack whose top-level numbering continues *after*
    /// `root_counter` — i.e. the next top-level step begun will be
    /// `root_counter + 1`. Seeded from the max existing top-level todo key so a
    /// later turn never reuses a key an earlier turn's still-persisted todo
    /// already owns (scratchpad `write` is upsert-by-key, DA-7 / #292).
    pub fn with_root_counter(root_counter: u32) -> Self {
        Self {
            frames: Vec::new(),
            root_counter,
        }
    }

    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// The dotted key of the current (innermost) step, if any.
    pub fn current_key(&self) -> Option<&str> {
        self.frames.last().map(|f| f.key.as_str())
    }

    /// Push a new step capturing `watermark` as its scope start, and return
    /// its assigned `(dotted_key, sequence)`. A new top-level step gets the
    /// next root number; a step begun while another is active becomes its
    /// next numbered child.
    pub fn begin(&mut self, goal: impl Into<String>, watermark: usize) -> (String, i32) {
        let (key, sequence) = match self.frames.last_mut() {
            Some(parent) => {
                parent.child_counter += 1;
                let seq = i32::try_from(parent.child_counter).unwrap_or(i32::MAX);
                (format!("{}.{}", parent.key, parent.child_counter), seq)
            }
            None => {
                self.root_counter += 1;
                let seq = i32::try_from(self.root_counter).unwrap_or(i32::MAX);
                (self.root_counter.to_string(), seq)
            }
        };
        self.frames.push(StepFrame {
            key: key.clone(),
            goal: goal.into(),
            watermark,
            child_counter: 0,
            sequence,
        });
        (key, sequence)
    }

    /// Pop and return the innermost step, or `None` if no step is active.
    pub fn complete(&mut self) -> Option<StepFrame> {
        self.frames.pop()
    }

    /// Drop every frame. Called by the dispatch loop after overflow recovery,
    /// which can drain messages and invalidate the absolute watermarks. The
    /// root counter is intentionally preserved: the todos written before the
    /// clear still live on the scratchpad, so a fresh step must keep advancing
    /// the numbering rather than reuse a key (e.g. `"1"`) that would clobber an
    /// existing todo via upsert.
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

/// The highest *top-level* (un-dotted) numeric step key among `keys`, or `0`
/// when there are none. Used to seed [`StepStack::with_root_counter`] from a
/// conversation's existing `todo` notes so a new turn keeps advancing the
/// numbering instead of restarting at `"1"` (DA-7 / #292). Nested keys
/// (`"1.2"`) and non-numeric keys are ignored.
pub(crate) fn max_top_level_key<'a>(keys: impl IntoIterator<Item = &'a str>) -> u32 {
    keys.into_iter()
        .filter(|k| !k.contains('.')) // top-level only
        .filter_map(|k| k.parse::<u32>().ok())
        .max()
        .unwrap_or(0)
}

/// Truncate `s` to at most `max_bytes`, landing on a UTF-8 char boundary.
pub(crate) fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut cut = max_bytes;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    s[..cut].to_string()
}

/// Build the pointer that replaces an evicted tool result. Addressed to the
/// model so it knows the detail still exists (in the named note, or via a
/// re-run) and was removed only to keep the turn lean.
pub(crate) fn compaction_pointer(tool_name: Option<&str>, note_keys: &[String]) -> String {
    let ran = match tool_name {
        Some(n) if !n.is_empty() => format!(" (ran {n})"),
        _ => String::new(),
    };
    if note_keys.is_empty() {
        return format!(
            "{COMPACTION_POINTER_PREFIX}{ran}: this result was dropped from working \
             context when its step completed (no carry-forward note was recorded). \
             Re-run the tool if you need it again.>"
        );
    }
    let keys = note_keys
        .iter()
        .map(|k| format!("\"{k}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{COMPACTION_POINTER_PREFIX}{ran}: this result was distilled into scratchpad \
         note(s) {keys} and dropped from working context to keep the turn lean. Re-read \
         the note(s) with builtin_scratchpad_search, or re-run the tool for the full output.>"
    )
}

/// Replace the content of every sizeable `Role::Tool` message in
/// `messages[from..]` with a [`compaction_pointer`], freeing context while
/// leaving the message structure (role + `tool_call_id`) intact so provider
/// tool-call/result pairing is never broken.
///
/// Returns `(results_evicted, bytes_freed)`.
///
/// Idempotent: results already bearing a pointer are skipped. `from` is
/// clamped to the slice length. Only the rare overflow-recovery path drains
/// messages mid-turn, and it drains from the left — shifting absolute
/// watermarks so this *under*-evicts (safe) rather than over-evicts; the
/// dispatch loop additionally clears the step stack on overflow recovery, so
/// a stale watermark never reaches here.
pub(crate) fn evict_tool_results(
    messages: &mut [Message],
    from: usize,
    note_keys: &[String],
) -> (usize, usize) {
    let from = from.min(messages.len());

    // Map each tool_call_id to the tool that produced it, from the assistant
    // tool-call requests, so the pointer can name what ran. Owned to avoid
    // holding an immutable borrow across the mutation below.
    let mut names: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for m in messages.iter() {
        if m.role == Role::Assistant {
            for tc in &m.tool_calls {
                names.insert(tc.id.clone(), tc.name.clone());
            }
        }
    }

    let mut evicted = 0usize;
    let mut freed = 0usize;
    for m in messages[from..].iter_mut() {
        if m.role != Role::Tool || m.content.len() < COMPACTION_MIN_EVICT_BYTES {
            continue;
        }
        if m.content.starts_with(COMPACTION_POINTER_PREFIX) {
            continue; // already compacted by an inner step
        }
        let tool_name = m
            .tool_call_id
            .as_deref()
            .and_then(|id| names.get(id))
            .map(String::as_str);
        let pointer = compaction_pointer(tool_name, note_keys);
        freed += m.content.len().saturating_sub(pointer.len());
        evicted += 1;
        m.content = pointer;
    }
    (evicted, freed)
}

/// A single plan entry for [`render_plan`] (a `todo`-typed scratchpad note).
pub(crate) struct PlanItem<'a> {
    pub key: &'a str,
    pub goal: &'a str,
    pub done: bool,
    /// The step's distilled finding, when it is still in view — a completed
    /// step whose parent hasn't yet rolled it up. Rendered nested under the step.
    pub outcome: Option<&'a str>,
}

/// Parse a dotted step key into numeric segments for tree ordering. A
/// non-numeric segment sorts last within its level (defensive — auto-numbered
/// keys are always numeric).
fn dotted_key(key: &str) -> Vec<u64> {
    key.split('.')
        .map(|seg| seg.parse::<u64>().unwrap_or(u64::MAX))
        .collect()
}

/// Render the open plan as a compact indented tree for per-round surfacing.
/// Returns `None` when there are no steps to show. `current` marks the live
/// step (you-are-here); `max_items` caps the rendered size so it stays cheap
/// to re-send every round.
pub(crate) fn render_plan(
    items: &[PlanItem<'_>],
    current: Option<&str>,
    max_items: usize,
) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let mut sorted: Vec<&PlanItem> = items.iter().collect();
    sorted.sort_by_key(|a| dotted_key(a.key));

    // Choose which items to render when there are more than the cap. The naive
    // head-take dropped the live step into the tail once enough old DONE steps
    // accumulated (DA-8 / #293), because done steps sort first. Select instead
    // so the model always sees where it is and what's left:
    //   1. the current step and every ancestor of it (you-are-here + context),
    //   2. then the remaining OPEN steps, most-recent first,
    //   3. then the remaining DONE steps, most-recent first,
    // filling up to `max_items`. The chosen set is then rendered in tree order
    // so the indentation still reads as a plan.
    let elided = sorted.len().saturating_sub(max_items);
    let chosen: Vec<&PlanItem> = if elided == 0 {
        sorted.clone()
    } else {
        select_plan_items(&sorted, current, max_items)
    };

    let mut out = String::from(
        "Your plan (steps on the scratchpad, with findings so far — keep working it; \
         mark steps done as you go, and roll a step's sub-step findings up into its outcome):",
    );
    for item in &chosen {
        let depth = item.key.matches('.').count();
        let indent = "  ".repeat(depth);
        let check = if item.done { "[x]" } else { "[ ]" };
        let here = if current == Some(item.key) {
            "  ← you are here"
        } else {
            ""
        };
        let goal = truncate_on_char_boundary(item.goal, 160);
        out.push_str(&format!("\n{indent}{} {check} {goal}{here}", item.key));
        if let Some(outcome) = item.outcome.filter(|o| !o.is_empty()) {
            let outcome = truncate_on_char_boundary(outcome, 200);
            out.push_str(&format!("\n{indent}  → {outcome}"));
        }
    }
    let shown = chosen.len();
    if sorted.len() > shown {
        out.push_str(&format!("\n… and {} more.", sorted.len() - shown));
    }

    // Wrap-up nudge: when no step is live (the stack has fully unwound) and
    // every step is done, the plan is complete — prompt the model to write its
    // closing summary and clear the stale `goal` note rather than leave it to
    // linger into the next task. Gated on `current.is_none()` so a still-open
    // step (more work pending) never trips it; computed over `sorted` (all
    // items) so cap-elision can't hide an unfinished step and falsely fire it.
    if current.is_none() && sorted.iter().all(|i| i.done) {
        out.push_str(
            "\nAll steps are done. If the task is complete: give the user your closing summary, \
             promote anything worth keeping beyond this conversation to the knowledge base \
             (builtin_knowledge_base_write), then clear your goal note \
             (builtin_scratchpad_delete keys: [\"goal\"]) so it doesn't linger into the next task.",
        );
    }
    Some(out)
}

/// True when `ancestor` is a proper dotted-key prefix of `key`
/// (e.g. `"3"` and `"3.2"` are ancestors of `"3.2.1"`). A key is not its own
/// ancestor.
fn is_ancestor_of(ancestor: &str, key: &str) -> bool {
    key.len() > ancestor.len()
        && key.starts_with(ancestor)
        && key.as_bytes().get(ancestor.len()) == Some(&b'.')
}

/// Pick at most `max_items` of `sorted` (which is already in tree order),
/// keeping the chosen set in tree order. Priority: the current step and its
/// ancestors, then open steps (recent first), then done steps (recent first).
/// See [`render_plan`] for why. Selection is by position in `sorted`, not by
/// key, so duplicate keys are never collapsed.
fn select_plan_items<'a>(
    sorted: &[&'a PlanItem<'a>],
    current: Option<&str>,
    max_items: usize,
) -> Vec<&'a PlanItem<'a>> {
    let mut keep = vec![false; sorted.len()];
    let mut kept = 0usize;

    // 1. Current step + every ancestor of it — always shown, regardless of cap.
    if let Some(cur) = current {
        for (i, item) in sorted.iter().enumerate() {
            if !keep[i] && (item.key == cur || is_ancestor_of(item.key, cur)) {
                keep[i] = true;
                kept += 1;
            }
        }
    }

    // 2 & 3. Fill the rest from open-then-done, most-recent first. "Recent" =
    // later in tree order, so iterate the reverse of `sorted`.
    let fill = |want_done: bool, keep: &mut [bool], kept: &mut usize| {
        for i in (0..sorted.len()).rev() {
            if *kept >= max_items {
                break;
            }
            if !keep[i] && sorted[i].done == want_done {
                keep[i] = true;
                *kept += 1;
            }
        }
    };
    fill(false, &mut keep, &mut kept); // open first
    fill(true, &mut keep, &mut kept); // then done

    // Render in tree order, including only the chosen positions.
    sorted
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, item)| keep[i].then_some(item))
        .collect()
}

/// Maximum free-form note keys named in the per-round `[Scratchpad]` index
/// before the "… and N more" tail. Mirrors [`MAX_PLAN_ITEMS`] — the index is
/// re-sent every round, so it stays cheap; recognition over recall means a
/// generous-but-bounded list of keys is enough to remind the model what it has
/// stashed.
pub(crate) const MAX_SCRATCHPAD_INDEX_KEYS: usize = 40;

/// Select the free-form notepad keys from a conversation's notes (#340).
///
/// "Free-form" = a `note`-typed note that is NOT already surfaced elsewhere:
/// the `goal` note is the `[Current task]` anchor, and `outcome:<step>` notes
/// plus `todo`-typed steps are rendered into `[Plan]`. Filtering by type alone
/// is insufficient (both `goal` and `outcome:*` are `note`-typed), so this also
/// excludes by key. The remaining set is the durable-but-otherwise-invisible
/// notepad that the `[Scratchpad]` index advertises.
pub(crate) fn freeform_note_keys<'a>(notes: &[RawNote<'a>]) -> Vec<&'a str> {
    notes
        .iter()
        .filter(|n| {
            n.note_type == OUTCOME_NOTE_TYPE
                && n.key != SCRATCHPAD_GOAL_KEY
                && !n.key.starts_with(OUTCOME_KEY_PREFIX)
        })
        .map(|n| n.key)
        .collect()
}

/// Render the per-round `[Scratchpad]` index: a sorted, capped list of the
/// free-form note keys, so a note the model stashed earlier survives windowing
/// and compaction as *recognition* (it can `builtin_scratchpad_search` for the
/// key) even after the message that wrote it is gone (#340). Keys only — no
/// content previews. Returns `None` when there are no keys to advertise.
pub(crate) fn render_scratchpad_index(keys: &[&str], max_items: usize) -> Option<String> {
    let mut sorted: Vec<&str> = keys.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.is_empty() {
        return None;
    }

    let total = sorted.len();
    let shown = max_items.min(total);
    let listed = sorted[..shown].join(", ");

    let mut out = format!("Notes you've stashed (read with builtin_scratchpad_search): {listed}");
    if total > shown {
        out.push_str(&format!(" … and {} more.", total - shown));
    } else {
        out.push('.');
    }
    Some(out)
}

/// A scratchpad note as the plan renderer needs it — just the fields it reads,
/// so the renderer stays decoupled from the storage row type.
pub(crate) struct RawNote<'a> {
    pub key: &'a str,
    pub content: &'a str,
    pub note_type: &'a str,
    pub done: bool,
}

/// Build the plan surface from a conversation's scratchpad notes (#240).
///
/// Steps are the `todo`-typed notes; each step's distilled finding lives in a
/// companion `outcome:<step-key>` note. A finding is surfaced (nested under its
/// step) only while it is still *waiting to be rolled up* — i.e. its parent step
/// is not yet done. Once a parent completes (summarising its children up into
/// its own outcome), the children's findings drop from view, so the model always
/// sees exactly the findings pending summary into the currently-open ancestor.
/// Top-level findings (no parent) stay in view as the material for the final
/// summary to the user. Returns `None` when there are no steps.
pub(crate) fn render_plan_from_notes(
    notes: &[RawNote<'_>],
    current: Option<&str>,
    max_items: usize,
) -> Option<String> {
    use std::collections::HashMap;

    let done_by_key: HashMap<&str, bool> = notes
        .iter()
        .filter(|n| n.note_type == STEP_NOTE_TYPE)
        .map(|n| (n.key, n.done))
        .collect();
    if done_by_key.is_empty() {
        return None;
    }

    // Findings still pending roll-up, keyed by their step. Absorbed (dropped)
    // once the parent step is done.
    let outcomes: HashMap<&str, &str> = notes
        .iter()
        .filter_map(|n| {
            n.key
                .strip_prefix(OUTCOME_KEY_PREFIX)
                .map(|step| (step, n.content))
        })
        .filter(|(step, _)| {
            step.rsplit_once('.')
                .map(|(parent, _)| !done_by_key.get(parent).copied().unwrap_or(false))
                .unwrap_or(true)
        })
        .collect();

    let items: Vec<PlanItem> = notes
        .iter()
        .filter(|n| n.note_type == STEP_NOTE_TYPE)
        .map(|n| PlanItem {
            key: n.key,
            goal: n.content,
            done: n.done,
            outcome: outcomes.get(n.key).copied(),
        })
        .collect();
    render_plan(&items, current, max_items)
}

/// The `begin_step` tool definition advertised to the model.
pub(crate) fn begin_step_tool() -> ToolDefinition {
    ToolDefinition::new(
        BEGIN_STEP_TOOL,
        "Begin a step of a multi-step task. Pushes a step onto your plan and opens a \
         fresh working scope. Use it to break a non-trivial request into ordered steps, \
         and again — nested — when a step turns out to need its own sub-plan (a step begun \
         inside step 1.2 becomes 1.2.1, 1.2.2, …). The step is recorded as an ordered todo \
         on the scratchpad and numbered for you. Pair every begin_step with a later \
         complete_step. For small one-shot tasks, don't use steps at all — just answer or act.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What this step aims to accomplish or find out — a short, concrete objective (e.g. 'Get the 7-day forecast for Cary, NC')."
                }
            },
            "required": ["goal"]
        }),
    )
}

/// The `complete_step` tool definition advertised to the model.
pub(crate) fn complete_step_tool() -> ToolDefinition {
    ToolDefinition::new(
        COMPLETE_STEP_TOOL,
        "Complete the current step (the most recently begun one). Marks its todo done, \
         records what you learned as a carry-forward note on the scratchpad, and removes \
         the step's raw tool results from working context — they're distilled into the note, \
         which stays searchable, so nothing important is lost and the turn stays lean. Write \
         the `outcome` whenever the result matters to later steps, or when in doubt; omit it \
         only for trivial steps. If this step had sub-steps, roll their findings up into your \
         outcome — summarise them into one, don't repeat each. Use status \"abandoned\" for a \
         dead end you're backing out of: the wasted exploration is still cleared and the note \
         records why, so you don't repeat it.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "description": "The distilled finding(s) to carry forward — the gist, not the raw output (e.g. 'Cary, NC 7-day: highs low-80s°F, rain likely Tue'). Omit only for trivial steps."
                },
                "status": {
                    "type": "string",
                    "enum": ["done", "abandoned"],
                    "description": "done (default) = the step succeeded. abandoned = a dead end you're backing out of."
                }
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Message, Role, ToolCall};

    #[test]
    fn stack_auto_numbers_roots_and_nested_children() {
        let mut stack = StepStack::new();
        let (k1, s1) = stack.begin("research", 0);
        assert_eq!(k1, "1");
        assert_eq!(s1, 1);
        assert_eq!(stack.current_key(), Some("1"));

        // Nested children of step 1.
        let (k11, _) = stack.begin("sub a", 3);
        assert_eq!(k11, "1.1");
        assert_eq!(stack.depth(), 2);
        // Completing 1.1 pops back to 1.
        let popped = stack.complete().unwrap();
        assert_eq!(popped.key, "1.1");
        assert_eq!(popped.watermark, 3);
        assert_eq!(stack.current_key(), Some("1"));

        // Next child of 1 continues the counter: 1.2, then 1.2.1.
        let (k12, _) = stack.begin("sub b", 5);
        assert_eq!(k12, "1.2");
        let (k121, _) = stack.begin("sub b i", 7);
        assert_eq!(k121, "1.2.1");

        // Unwind fully, then a new root step is 2 (not 1).
        stack.complete();
        stack.complete();
        stack.complete();
        assert_eq!(stack.depth(), 0);
        let (k2, s2) = stack.begin("write up", 9);
        assert_eq!(k2, "2");
        assert_eq!(s2, 2);
    }

    #[test]
    fn complete_on_empty_stack_is_none() {
        let mut stack = StepStack::new();
        assert!(stack.complete().is_none());
    }

    #[test]
    fn clear_drops_frames_but_preserves_numbering() {
        let mut stack = StepStack::new();
        stack.begin("a", 0);
        stack.begin("b", 1);
        stack.clear();
        assert_eq!(stack.depth(), 0);
        // Numbering does NOT reset: a fresh step after a clear must not reuse a
        // key (e.g. "1") that an earlier, still-persisted todo already owns.
        let (k, _) = stack.begin("c", 2);
        assert_eq!(k, "2");
    }

    // --- DA-7: seed root_counter from existing top-level keys ---

    #[test]
    fn max_top_level_key_finds_highest_root_ignoring_children() {
        // Top-level keys are "1", "2", "3"; nested keys ("2.1", "3.4.2") must
        // not bump the root counter.
        let keys = ["1", "2", "2.1", "3", "3.4.2"];
        assert_eq!(max_top_level_key(keys.iter().copied()), 3);
    }

    #[test]
    fn max_top_level_key_is_zero_when_no_top_level_keys() {
        // No top-level keys (empty, or only nested/non-numeric) → 0, so a fresh
        // stack starts numbering at 1.
        assert_eq!(max_top_level_key(std::iter::empty()), 0);
        assert_eq!(max_top_level_key(["1.1", "2.3"].iter().copied()), 0);
        assert_eq!(max_top_level_key(["abc", "outcome:1"].iter().copied()), 0);
    }

    #[test]
    fn seeded_stack_continues_numbering_past_prior_turn_keys() {
        // A new turn whose conversation already has top-level todos "1" and "2"
        // must mint "3" next, not clobber "1" via upsert (DA-7).
        let mut stack = StepStack::with_root_counter(2);
        let (k, s) = stack.begin("third step", 0);
        assert_eq!(k, "3");
        assert_eq!(s, 3);
        // Children of the seeded step still number from .1.
        let (k31, _) = stack.begin("sub", 1);
        assert_eq!(k31, "3.1");
    }

    #[test]
    fn new_stack_starts_at_one_unchanged() {
        // The default (unseeded) stack behaviour is preserved.
        let mut stack = StepStack::new();
        let (k, _) = stack.begin("first", 0);
        assert_eq!(k, "1");
    }

    fn tool_msg(id: &str, content: &str) -> Message {
        Message::tool_result(id, content)
    }

    #[test]
    fn evict_shrinks_large_results_preserving_pairing() {
        let big = "x".repeat(5000);
        let mut messages = vec![
            Message::new(Role::User, "do it"),
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "weather_forecast", "{}")]),
            tool_msg("c1", &big),
        ];
        let keys = vec!["outcome:1".to_string()];
        let (evicted, freed) = evict_tool_results(&mut messages, 1, &keys);
        assert_eq!(evicted, 1);
        assert!(freed > 4000);
        // Structure preserved: still a Tool message with its tool_call_id.
        assert_eq!(messages[2].role, Role::Tool);
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("c1"));
        // Content is now the pointer, naming the tool and the note.
        assert!(messages[2].content.starts_with(COMPACTION_POINTER_PREFIX));
        assert!(messages[2].content.contains("weather_forecast"));
        assert!(messages[2].content.contains("outcome:1"));
        // The assistant tool-call request is untouched.
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[1].tool_calls.len(), 1);
    }

    #[test]
    fn evict_skips_small_and_already_compacted_results() {
        let big = "y".repeat(5000);
        let mut messages = vec![
            Message::assistant_with_tool_calls(vec![
                ToolCall::new("c1", "t", "{}"),
                ToolCall::new("c2", "t", "{}"),
            ]),
            tool_msg("c1", "tiny"), // below threshold
            tool_msg("c2", &big),
        ];
        let keys = vec!["k".to_string()];
        let (evicted, _) = evict_tool_results(&mut messages, 0, &keys);
        assert_eq!(evicted, 1); // only the big one
        assert_eq!(messages[1].content, "tiny");

        // Second pass over the same range is a no-op (idempotent).
        let (evicted2, freed2) = evict_tool_results(&mut messages, 0, &keys);
        assert_eq!(evicted2, 0);
        assert_eq!(freed2, 0);
    }

    #[test]
    fn evict_clamps_out_of_range_watermark() {
        let mut messages = vec![Message::new(Role::User, "hi")];
        let (evicted, freed) = evict_tool_results(&mut messages, 99, &[]);
        assert_eq!((evicted, freed), (0, 0));
    }

    #[test]
    fn pointer_without_notes_says_dropped() {
        let p = compaction_pointer(Some("geocode"), &[]);
        assert!(p.contains("geocode"));
        assert!(p.contains("no carry-forward"));
    }

    #[test]
    fn render_plan_sorts_indents_and_marks_current() {
        let items = vec![
            PlanItem {
                key: "1",
                goal: "research",
                done: true,
                outcome: None,
            },
            PlanItem {
                key: "1.2",
                goal: "draft",
                done: false,
                outcome: None,
            },
            PlanItem {
                key: "1.10",
                goal: "late",
                done: false,
                outcome: None,
            },
            PlanItem {
                key: "1.2.1",
                goal: "pick crate",
                done: true,
                outcome: None,
            },
        ];
        let rendered = render_plan(&items, Some("1.2"), 50).unwrap();
        let lines: Vec<&str> = rendered.lines().collect();
        // Header + 4 items.
        assert_eq!(lines.len(), 5);
        // Numeric (not lexical) ordering: 1, 1.2, 1.2.1, 1.10.
        assert!(lines[1].contains("1 [x] research"));
        assert!(lines[2].contains("1.2 [ ] draft"));
        assert!(lines[2].contains("← you are here"));
        assert!(lines[3].contains("1.2.1 [x] pick crate"));
        assert!(lines[4].trim_start().starts_with("1.10"));
        // Depth-based indentation: 1.2.1 is deeper than 1.2.
        let indent_12 = lines[2].len() - lines[2].trim_start().len();
        let indent_121 = lines[3].len() - lines[3].trim_start().len();
        assert!(indent_121 > indent_12);
    }

    #[test]
    fn render_plan_empty_is_none() {
        assert!(render_plan(&[], None, 10).is_none());
    }

    #[test]
    fn render_plan_nudges_wrap_up_when_all_done_and_no_live_step() {
        // Plan fully unwound (no current step) and every step done → wrap-up
        // nudge to summarise and clear the stale goal note.
        let items = vec![
            PlanItem {
                key: "1",
                goal: "research",
                done: true,
                outcome: None,
            },
            PlanItem {
                key: "2",
                goal: "write up",
                done: true,
                outcome: None,
            },
        ];
        let rendered = render_plan(&items, None, 50).unwrap();
        assert!(rendered.contains("All steps are done"), "{rendered}");
        assert!(rendered.contains(r#"["goal"]"#), "{rendered}");
        // Cleanup reminder covers durable promotion as well as goal clearing.
        assert!(
            rendered.contains("builtin_knowledge_base_write"),
            "{rendered}"
        );
    }

    #[test]
    fn render_plan_no_wrap_up_nudge_while_a_step_is_open() {
        // A live/open step means work is pending — the wrap-up nudge must not
        // fire, whether the open step is the current one or just unfinished.
        let items = vec![
            PlanItem {
                key: "1",
                goal: "research",
                done: true,
                outcome: None,
            },
            PlanItem {
                key: "2",
                goal: "still going",
                done: false,
                outcome: None,
            },
        ];
        // Open step is the current one.
        let with_current = render_plan(&items, Some("2"), 50).unwrap();
        assert!(
            !with_current.contains("All steps are done"),
            "{with_current}"
        );
        // Even with no current step, an unfinished step suppresses the nudge.
        let no_current = render_plan(&items, None, 50).unwrap();
        assert!(!no_current.contains("All steps are done"), "{no_current}");
    }

    #[test]
    fn render_plan_caps_items() {
        let items: Vec<PlanItem> = (1..=10)
            .map(|_| PlanItem {
                key: "1",
                goal: "g",
                done: false,
                outcome: None,
            })
            .collect();
        let rendered = render_plan(&items, None, 3).unwrap();
        assert!(rendered.contains("… and 7 more."));
    }

    #[test]
    fn render_plan_shows_outcome_nested_under_step() {
        let items = vec![PlanItem {
            key: "1",
            goal: "research",
            done: true,
            outcome: Some("API is OAuth2, 100 req/min"),
        }];
        let rendered = render_plan(&items, None, 10).unwrap();
        assert!(rendered.contains("1 [x] research"));
        assert!(rendered.contains("→ API is OAuth2, 100 req/min"));
    }

    // --- DA-8: the live step is always rendered even past the item cap ---

    #[test]
    fn render_plan_always_includes_current_step_when_over_cap() {
        // Many old DONE steps that sort first, plus the live (open) current
        // step that sorts last. With a tiny cap the naive head-take would drop
        // the current step into the "… and N more" tail; the fix must keep it.
        let mut items: Vec<PlanItem> = (1..=50)
            .map(|i| PlanItem {
                key: leak_key(i),
                goal: "old done step",
                done: true,
                outcome: None,
            })
            .collect();
        items.push(PlanItem {
            key: "51",
            goal: "the live step",
            done: false,
            outcome: None,
        });
        let rendered = render_plan(&items, Some("51"), 5).unwrap();
        assert!(
            rendered.contains("51 [ ] the live step"),
            "the current/live step must always be rendered:\n{rendered}"
        );
        assert!(rendered.contains("← you are here"));
        assert!(rendered.contains("… and"), "the cap still elides the rest");
    }

    #[test]
    fn render_plan_includes_current_steps_ancestors() {
        // Current step is "3.2.1"; its ancestors "3" and "3.2" must be shown
        // (and indented under each other) even when older done steps would
        // otherwise consume the whole budget.
        let mut items: Vec<PlanItem> = (1..=2)
            .flat_map(|i| {
                (1..=10).map(move |j| PlanItem {
                    key: leak_key2(i, j),
                    goal: "old done step",
                    done: true,
                    outcome: None,
                })
            })
            .collect();
        items.push(PlanItem {
            key: "3",
            goal: "ancestor root",
            done: false,
            outcome: None,
        });
        items.push(PlanItem {
            key: "3.2",
            goal: "ancestor mid",
            done: false,
            outcome: None,
        });
        items.push(PlanItem {
            key: "3.2.1",
            goal: "live leaf",
            done: false,
            outcome: None,
        });
        let rendered = render_plan(&items, Some("3.2.1"), 4).unwrap();
        assert!(rendered.contains("3 [ ] ancestor root"), "{rendered}");
        assert!(rendered.contains("3.2 [ ] ancestor mid"), "{rendered}");
        assert!(rendered.contains("3.2.1 [ ] live leaf"), "{rendered}");
    }

    #[test]
    fn render_plan_prefers_open_over_done_when_over_cap() {
        // A mix of done and open steps with a tight cap: open steps are
        // preferred over old done ones so the model sees what's left to do.
        let mut items: Vec<PlanItem> = (1..=8)
            .map(|i| PlanItem {
                key: leak_key(i),
                goal: "done",
                done: true,
                outcome: None,
            })
            .collect();
        items.push(PlanItem {
            key: "9",
            goal: "still open A",
            done: false,
            outcome: None,
        });
        items.push(PlanItem {
            key: "10",
            goal: "still open B",
            done: false,
            outcome: None,
        });
        let rendered = render_plan(&items, None, 3).unwrap();
        assert!(rendered.contains("still open A"), "{rendered}");
        assert!(rendered.contains("still open B"), "{rendered}");
    }

    // Tiny helpers to mint 'static keys for the over-cap selection tests.
    fn leak_key(i: u32) -> &'static str {
        Box::leak(i.to_string().into_boxed_str())
    }
    fn leak_key2(i: u32, j: u32) -> &'static str {
        Box::leak(format!("{i}.{j}").into_boxed_str())
    }

    fn raw(
        key: &'static str,
        content: &'static str,
        ty: &'static str,
        done: bool,
    ) -> RawNote<'static> {
        RawNote {
            key,
            content,
            note_type: ty,
            done,
        }
    }

    #[test]
    fn plan_surfaces_findings_until_parent_rolls_them_up() {
        // Step 1 open; 1.1 done with a finding (parent 1 still open → shown).
        // 1.2 done; 1.2.1 done with a finding whose parent 1.2 IS done → that
        // finding was rolled up into 1.2, so it drops from view.
        let notes = vec![
            raw("1", "build it", "todo", false),
            raw("1.1", "research", "todo", true),
            raw("outcome:1.1", "API is OAuth2", "note", false),
            raw("1.2", "wire the client", "todo", true),
            raw("outcome:1.2", "client built on reqwest", "note", false),
            raw("1.2.1", "pick crate", "todo", true),
            raw("outcome:1.2.1", "chose reqwest 0.12", "note", false),
            raw("goal", "the overall goal", "note", false),
        ];
        let rendered = render_plan_from_notes(&notes, Some("1"), 50).unwrap();
        // Pending roll-up into the still-open step 1 → shown.
        assert!(rendered.contains("→ API is OAuth2"));
        // 1.2's own finding is top-of-its-subtree and 1.2's parent (1) is open → shown.
        assert!(rendered.contains("→ client built on reqwest"));
        // 1.2.1's finding was absorbed when 1.2 completed → hidden.
        assert!(!rendered.contains("chose reqwest"));
        // The `goal` note is not a step and must not render as a todo line.
        assert!(!rendered.contains("the overall goal"));
    }

    #[test]
    fn render_plan_from_notes_none_without_todos() {
        let notes = vec![raw("goal", "g", "note", false)];
        assert!(render_plan_from_notes(&notes, None, 10).is_none());
    }

    #[test]
    fn step_tools_have_stable_names() {
        assert_eq!(begin_step_tool().name, "begin_step");
        assert_eq!(complete_step_tool().name, "complete_step");
    }

    // --- Scratchpad index (#340) ---

    #[test]
    fn render_scratchpad_index_empty_is_none() {
        assert!(render_scratchpad_index(&[], 5).is_none());
    }

    #[test]
    fn render_scratchpad_index_sorts_keys() {
        let keys = ["user-prefs", "api-quirks", "deploy-target"];
        let rendered = render_scratchpad_index(&keys, 10).unwrap();
        // Sorted, no "and N more" tail (under cap).
        assert!(
            rendered.contains("api-quirks, deploy-target, user-prefs"),
            "keys must be rendered sorted: {rendered:?}"
        );
        assert!(
            !rendered.contains("more"),
            "no tail under cap: {rendered:?}"
        );
        // Advertises the read tool so the model knows how to recover content.
        assert!(rendered.contains("builtin_scratchpad_search"));
    }

    #[test]
    fn render_scratchpad_index_exactly_at_cap_has_no_tail() {
        let keys = ["a", "b", "c"];
        let rendered = render_scratchpad_index(&keys, 3).unwrap();
        assert!(rendered.contains("a, b, c"));
        assert!(
            !rendered.contains("more"),
            "exactly at cap must not show a tail: {rendered:?}"
        );
    }

    #[test]
    fn render_scratchpad_index_over_cap_shows_remainder_count() {
        let keys = ["e", "d", "c", "b", "a"];
        let rendered = render_scratchpad_index(&keys, 2).unwrap();
        // First two in sort order are shown; the remaining 3 are summarised.
        assert!(
            rendered.contains("a, b"),
            "shows capped sorted head: {rendered:?}"
        );
        assert!(
            rendered.contains("… and 3 more."),
            "over-cap must show remainder count: {rendered:?}"
        );
        assert!(
            !rendered.contains(", c"),
            "elided keys must not render: {rendered:?}"
        );
    }

    #[test]
    fn render_scratchpad_index_dedupes_and_sorts() {
        // Duplicate keys collapse (a key is upsert-by-key in storage, but the
        // renderer should be robust to a caller passing dups).
        let keys = ["b", "a", "b"];
        let rendered = render_scratchpad_index(&keys, 10).unwrap();
        assert!(rendered.contains("a, b"));
        assert!(
            !rendered.contains("a, b, b"),
            "dups must collapse: {rendered:?}"
        );
    }

    #[test]
    fn freeform_note_keys_filters_out_anchors_and_plan_notes() {
        let notes = vec![
            raw("goal", "the goal", "note", false), // excluded: [Current task]
            raw("outcome:1", "finding", "note", false), // excluded: [Plan]
            raw("outcome:1.2", "more", "note", false), // excluded: [Plan]
            raw("1", "a step", "todo", false),      // excluded: [Plan] (todo)
            raw("deploy-target", "prod", "note", false), // KEEP
            raw("api-quirks", "rate limits", "note", false), // KEEP
        ];
        let mut keys = freeform_note_keys(&notes);
        keys.sort();
        assert_eq!(keys, vec!["api-quirks", "deploy-target"]);
    }

    #[test]
    fn freeform_note_keys_empty_when_only_excluded() {
        let notes = vec![
            raw("goal", "g", "note", false),
            raw("outcome:1", "f", "note", false),
            raw("1", "s", "todo", true),
        ];
        assert!(freeform_note_keys(&notes).is_empty());
    }
}
