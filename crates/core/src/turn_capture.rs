//! What the harness keeps of a turn, without asking the model (#1207, #1349).
//!
//! ## The harness test, applied to capture
//!
//! Volunteering asks the model to notice, mid-task and under time pressure,
//! that something is worth keeping - and gives it no feedback when it fails.
//! A turn that forgot to record the decision it just took looks exactly like a
//! turn in which no decision was taken. So capture belongs to the harness: a
//! separate cheap pass with one job, which runs whether or not the model
//! thought of it.
//!
//! ## Three things need no judgment at all
//!
//! The user's own words, the tool calls with their arguments and outcomes, and
//! any decision the user stated. All three are already in the transcript,
//! exactly as they happened, so nothing has to decide what they meant - and
//! nothing may, because a model deciding which half of a turn mattered is the
//! failure this replaces. A decision the user stated is captured because the
//! user's words are captured whole, not because anything recognised it as a
//! decision.
//!
//! ## Why the transcript is not enough on its own
//!
//! The transcript keeps every byte, and the `[Earlier turns]` index makes a
//! dropped turn findable by position. Neither makes it findable by RELEVANCE:
//! `messages` carries no embedding, only a lexical `tsvector` with tool rows
//! excluded, so a past turn is reachable by position, by an always-injected
//! rolling summary, or by a lexical search the model must think to call. All
//! three need the model to already suspect that a past turn matters, which is
//! free recall. A digest is the recognition surface instead, and it costs one
//! write and no model call.
//!
//! ## Where it lands, and why the person rather than the conversation
//!
//! One digest per turn in the episodic turn index
//! ([`crate::ports::turn_digest`]), keyed by the id of the message that opened
//! the turn and scoped to the person.
//!
//! The conversation's own scratchpad is the wrong home for it, for two
//! reasons. A digest sitting beside the transcript it restates duplicates it,
//! which is what a person notices. And a conversation-scoped digest can never
//! answer "when did I last deal with this", because a turn in one conversation
//! is invisible from every other. The store holds the home conversation too,
//! so within-conversation recall survives the move, and nothing is copied onto
//! the pad: one row, one home.
//!
//! The knowledge base is a different scope again - a durable claim that
//! outlives the work - and deciding which facts those are is a judgment, so it
//! stays with the dream cycle, which already makes it.
//!
//! ## When it runs, and what it costs
//!
//! After the turn's work, and after the answer has streamed to the user chunk
//! by chunk - so nothing here sits between a person and the reply they are
//! reading.
//!
//! **Two exits do not stream.** A turn that ends in a provider error, and one
//! the user cancels, hand their text back as a return value rather than
//! through the chunk callback, and the capture is awaited before that return.
//! Those paths pay the write, and in a wired daemon the embedding with it,
//! before the user sees the message. That is a stated cost rather than a
//! property: the alternative is detaching the write, which would take the
//! capture out of the turn's own consistency and make its failure invisible on
//! exactly the exits most likely to need the record.
//!
//! ## Those two exits keep the question and drop the answer
//!
//! The assistant's half of a failed turn is not an answer. It is the provider's
//! error text, or the notice that says the user stopped the turn - an
//! operational failure, which is not a business outcome and is not recorded as
//! one (AGENTS.md 8.2). Stored whole, it becomes a recallable record whose
//! content is an outage: a later question close enough to the one that failed
//! matches it by distance, and `[Recall]` spends a line on a backend message.
//!
//! So those exits capture with [`TurnEnding::Unanswered`], which keeps the
//! `Asked:` half and omits the `Answered:` half. The turn is NOT skipped. A
//! provider can die on the turn where the user said the thing worth keeping,
//! and the failure is the assistant's, not theirs - the same reason the user's
//! words survive the strictest withholding setting below.
//!
//! The omission is disclosed the way a cut is, with a line the digest carries
//! in its own text. Without it a turn that was never answered reads exactly like a
//! turn whose answer was empty, and a reader cannot tell them apart.
//!
//! ## Bounded, and it says what it cut
//!
//! A turn can carry a megabyte of tool output. The digest is bounded, and the
//! budget is spent in priority order: what the user said, then what the
//! assistant answered, then the tool calls. The user's words are never dropped
//! to make room for tool traffic.
//!
//! ## What the digest may hold, and what it may not
//!
//! This is a DURABLE record that a later, clean turn reads back, now across
//! conversations rather than within one. So the same question every stored
//! surface has to answer applies here, and applies harder: can a page the turn
//! read reach a later turn through it?
//!
//! It holds the conversation's own two voices and nothing else. The user's
//! prompt is the user's, and it is never outside content.
//!
//! **The assistant's closing text is a different matter, and the stamp is what
//! accounts for it.** [`crate::tool_provenance`]'s own header records that "the
//! assistant's own reply routinely quotes the page it just read", so a turn
//! that read one and then answered has produced text an outside party
//! influenced. The rolling summary carries the same text unstamped, but it is
//! not the precedent to follow here: it is not embedded, not retrieved by
//! relevance to a later prompt, and not returned by a tool whose grading
//! contract is the marker. The closer precedent is `step_text_to_record`,
//! which stamps model-supplied durable free text with the turn's own
//! provenance, and this does the same.
//!
//! So the digest carries the writing turn's `after_outside_read`, and every
//! path that reads it back accounts for it: a fetch MARKS it, which folds into
//! the reading turn's provenance, and an unprompted render DROPS it at the
//! strict level.
//!
//! **A tool's bytes and a tool call's arguments are deliberately left out.**
//! A result payload is the clearest case of outside content there is, and a
//! call's arguments are text the model wrote after it may have read one - so
//! putting either in a record that renders into a later turn would carry
//! outside influence past the gate that exists to bound it. What the digest
//! keeps of a call is what needs no judgment and carries nobody's bytes: the
//! tool's name, how big its result was, whether that result declared itself a
//! refusal, and the message id the whole of it is one `builtin_transcript_get`
//! away at (#1226). Nothing is lost - the transcript holds every byte - and
//! the digest says exactly where.
//!
//! ## A subagent's conversation is not the person's episodic record
//!
//! A subagent runs its turns in a conversation of its own, and that
//! conversation's opening message is not the person's words: it is a prompt
//! the parent model composed, possibly after the parent read an outside page.
//! The child's own turn can then be clean, so a digest built from it would be
//! unstamped text a parent wrote after reading somebody else's page - and the
//! store is offered across every conversation the person owns, so one such
//! digest widens one conversation's exposure to the whole account.
//!
//! [`capture_turn`] refuses those turns outright, on the conversation's
//! reserved tag rather than on anything the model can set. They are mechanism,
//! not the person's episodic record, and the transcript still holds every byte
//! of them.

use crate::domain::{Message, RESERVED_SUBAGENT_TAG, Role};
use crate::planning::truncate_on_char_boundary;
use crate::ports::transcript::TRANSCRIPT_GET_TOOL;
use crate::ports::turn_digest::{MAX_DIGEST_BYTES, NewTurnDigest};
use crate::tool_provenance::{TurnProvenance, WITHHELD_STEP_TEXT};
use crate::tools::summarize_tool_name;

/// Whether `tags` mark a conversation whose turns stay out of the shared
/// store.
///
/// Read from the conversation's own reserved tag, which the spawn path sets
/// and no tool call can reach. See the module header for why a subagent's
/// conversation is not the person's episodic record.
#[must_use]
pub fn excluded_from_the_shared_store(tags: &[String]) -> bool {
    tags.iter().any(|tag| tag == RESERVED_SUBAGENT_TAG)
}

/// Most bytes of the user's words that reach the digest.
const ASKED_BYTES: usize = 2000;

/// Most bytes of the assistant's closing text that reach the digest.
const ANSWERED_BYTES: usize = 2000;

/// What a digest says when the budget cut it.
///
/// The module claims the digest is bounded AND says what it cut; without this the
/// second half was prose. A reader that cannot tell a complete record from a
/// truncated one reads a claim the record does not hold.
const TRUNCATION_NOTICE: &str = "\n[the rest of this turn's tool calls did not fit]";

/// What a digest says when the turn produced no answer to keep.
///
/// The same disclosure the budget makes when it cuts, for the same reason: a
/// reader that cannot tell a turn that was never answered from a turn whose
/// answer was empty is reading a record that claims more than it holds.
const NO_ANSWER_NOTICE: &str = "\n\n[this turn ended before it was answered; the assistant's half is the failure, not a reply, and is not kept]";

/// How the turn ended, which decides whether it has an answer worth keeping.
///
/// Read at the exit rather than derived here: the loop is the one place that
/// knows why it is leaving, and the last assistant message looks the same
/// either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnEnding {
    /// The turn produced a reply for the user - an answer, or the closing that
    /// an exhausted round budget winds down with.
    Answered,
    /// The turn ended in a provider error or a cancellation. Its last
    /// assistant text is the failure notice, so the digest keeps the question
    /// and says the answer half is absent.
    Unanswered,
}

/// The openings of every message the harness itself writes when a turn failed
/// rather than answered.
///
/// A transcript records what was said and not how the turn ended, so a reader
/// RECONSTRUCTING a finished turn cannot be told which exit it took. These are
/// what it recognises instead: the fixed opening of each message
/// `crate::service`'s `user_visible_llm_error_message` can write, plus the
/// opening of its cancellation notice. Both producers build their text from
/// this list rather than from literals of their own, and
/// `every_failure_notice_the_harness_writes_is_recognised` (in
/// `crate::service`) holds the two together so a reworded notice cannot drift
/// out of recognition silently.
///
/// This recognises the harness's own text and nothing else. A model that
/// opened a genuine reply with one of these sentences would have its answer
/// half omitted and the omission disclosed, which is a small loss and not a
/// false claim; a notice written by an older binary under wording no longer
/// listed here is read as an answer, which is what every such row already is
/// today.
pub const FAILED_TURN_NOTICE_PREFIXES: &[&str] = &[
    "The conversation exceeded the model's context window. We'll truncate older content and      retry. Details: ",
    "The conversation is too large for this model's context window, and shortening it further      would leave nothing to work from. Start a new conversation, or switch to a model with a      larger window. Details: ",
    "The API rate limit was exceeded. Please wait a moment and try again. Details: ",
    "Your API quota is exhausted. Top up the account or switch to a different API key.      Details: ",
    "The model is still downloading or loading. Please wait a moment and try again. Details: ",
    "This model does not support tool use. Please switch to a tool-capable model or disable      tools for this chat. Details: ",
    "I hit an LLM backend error and could not complete this request. Details: ",
    "[Turn cancelled",
];

/// How a stored turn ended, as far as its own transcript can say.
///
/// The live path reads the ending at the exit, where the loop knows why it is
/// leaving. A backfill has only the rows, so it recognises the harness's own
/// failure text by its opening - see [`FAILED_TURN_NOTICE_PREFIXES`].
///
/// A turn with no closing assistant prose at all is [`TurnEnding::Answered`]:
/// there is nothing to drop, and the capture already omits an absent answer
/// half without claiming one was withheld.
#[must_use]
pub fn derive_turn_ending(turn: &[Message]) -> TurnEnding {
    let closing = turn
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant && !m.content.trim().is_empty());
    match closing {
        Some(message)
            if FAILED_TURN_NOTICE_PREFIXES
                .iter()
                .any(|prefix| message.content.starts_with(prefix)) =>
        {
            TurnEnding::Unanswered
        }
        _ => TurnEnding::Answered,
    }
}

/// Whether a tool result declares itself a refusal.
///
/// Read from the payload's own `ok` field, which is the shape every daemon
/// tool answers a decline in (AGENTS.md 8.3). An MCP server that answers some
/// other way reads as answered, which is the safe direction: the digest says how
/// big the result was and where it is, so a reader that needs the truth has an
/// exact route to it.
fn declares_a_refusal(result: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(result)
        .ok()
        .and_then(|v| v.get("ok").and_then(serde_json::Value::as_bool))
        .is_some_and(|ok| !ok)
}

/// Build the digest for the turn that begins at `from`, or `None` when there
/// is nothing to keep.
///
/// `None` means one of two things. The range holds no user message - a range
/// that is not a turn - so there is no key to write under and nothing the
/// digest is about. Or `conversation_tags` mark a conversation whose turns
/// stay out of the shared store, which the module header explains and
/// [`excluded_from_the_shared_store`] decides.
///
/// `provenance` is the writing turn's own, and it decides the stamp. See the
/// module header for why the assistant's closing text needs one.
///
/// `ending` says whether the turn has an answer half at all. A provider error
/// and a cancellation both leave a last assistant message, and neither is a
/// reply - see the module header for why that text is dropped and the question
/// kept.
///
/// `hard_withhold` is the operator setting that destroys rather than stamps.
/// Under it, what the TURN derived is replaced by the placeholder and what the
/// opening message said is not: destroying that would defeat the one thing
/// this exists to keep.
#[must_use]
pub fn capture_turn(
    messages: &[Message],
    from: usize,
    provenance: TurnProvenance,
    hard_withhold: bool,
    ending: TurnEnding,
    conversation_tags: &[String],
) -> Option<NewTurnDigest> {
    if excluded_from_the_shared_store(conversation_tags) {
        return None;
    }
    let turn = messages.get(from.min(messages.len())..)?;
    let opening = turn.iter().find(|m| m.role == Role::User)?;

    let mut content = String::new();
    content.push_str("Asked: ");
    content.push_str(&truncate_on_char_boundary(&opening.content, ASKED_BYTES));

    // The tool calls, with their arguments and their outcomes, whether or not
    // any step claimed them. A step's own distillate is the model's account of
    // what a scope meant; this is the harness's account of what ran.
    let names: std::collections::HashMap<&str, &str> = turn
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .flat_map(|m| m.tool_calls.iter())
        .map(|call| (call.id.as_str(), call.name.as_str()))
        .collect();
    let mut ran = String::new();
    let mut calls = 0usize;
    for message in turn.iter().filter(|m| m.role == Role::Assistant) {
        for call in &message.tool_calls {
            calls += 1;
            let result = turn
                .iter()
                .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some(&call.id));
            let outcome = match result {
                Some(m) => format!(
                    "{} ({} bytes), read it with {TRANSCRIPT_GET_TOOL} message_id=\"{}\"",
                    if declares_a_refusal(&m.content) {
                        "declined"
                    } else {
                        "answered"
                    },
                    m.content.len(),
                    m.id
                ),
                None => "no result: the turn ended first".to_string(),
            };
            // The name comes from the model, so it is bounded before it is
            // stored - the same rule `refusal_text` applies to the same value.
            // A turn that read a page can be told to call a tool whose NAME is
            // the payload; the call fails and the assistant message carrying
            // that name is persisted anyway.
            ran.push_str(&format!(
                "\n- {} -> {outcome}",
                summarize_tool_name(&call.name)
            ));
        }
    }
    // A tool result whose request is no longer in the range still ran.
    let orphans = turn
        .iter()
        .filter(|m| {
            m.role == Role::Tool
                && m.tool_call_id
                    .as_deref()
                    .is_none_or(|id| !names.contains_key(id))
        })
        .count();
    if orphans > 0 {
        ran.push_str(&format!(
            "\n- {orphans} further tool result(s), whose requests are not in this turn"
        ));
    }

    // The assistant's closing text: the commitment half of the turn. The last
    // assistant message that carried prose rather than a tool request. A turn
    // that ended in a provider error or a cancellation has no such half, so
    // nothing is read for it.
    let answered = match ending {
        TurnEnding::Answered => turn
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant && !m.content.trim().is_empty())
            .map(|m| truncate_on_char_boundary(&m.content, ANSWERED_BYTES)),
        TurnEnding::Unanswered => None,
    };

    // Said before the withholding branch, so it survives the operator setting
    // that replaces what the turn derived: the absence of an answer is a fact
    // about the turn, not text the turn produced.
    if ending == TurnEnding::Unanswered {
        content.push_str(NO_ANSWER_NOTICE);
    }

    let after_outside_read = provenance.ingested_external();
    if after_outside_read && hard_withhold {
        // The operator asked for destruction rather than a stamp. Everything
        // below this line is what the TURN produced after it read outside
        // content; what the user said stays above it.
        content.push_str("\n\n");
        content.push_str(WITHHELD_STEP_TEXT);
    } else {
        if let Some(answered) = answered {
            content.push_str("\n\nAnswered: ");
            content.push_str(&answered);
        }
        if calls > 0 || orphans > 0 {
            content.push_str("\n\nRan:");
            content.push_str(&ran);
        }
    }

    // The budget is spent in priority order above, so this only ever cuts the
    // tail - the tool traffic - and never the user's words. A cut says so:
    // a reader that cannot tell a complete record from a truncated one is
    // reading a record that claims more than it holds.
    let content = if content.len() > MAX_DIGEST_BYTES {
        let room = MAX_DIGEST_BYTES.saturating_sub(TRUNCATION_NOTICE.len());
        format!(
            "{}{TRUNCATION_NOTICE}",
            truncate_on_char_boundary(&content, room)
        )
    } else {
        content
    };

    Some(NewTurnDigest {
        opening_message_id: opening.id.clone(),
        content,
        // Filled in by the write closure, the one place every digest write
        // passes through.
        embedding: None,
        // The writing turn's own provenance. A tool's bytes and a call's
        // arguments are kept out of the digest entirely, but the assistant's
        // closing text is in it, and a turn that read a page routinely quotes
        // it - see the module header.
        after_outside_read,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ToolCall;

    fn user(text: &str) -> Message {
        Message::new(Role::User, text)
    }

    fn assistant(text: &str) -> Message {
        Message::new(Role::Assistant, text)
    }

    fn requested(id: &str, name: &str, args: &str) -> Message {
        Message::assistant_with_tool_calls(vec![ToolCall::new(id, name, args)])
    }

    /// A turn that read nothing from outside.
    fn clean() -> TurnProvenance {
        TurnProvenance::new()
    }

    /// A turn that read outside content, as the gate records it.
    fn tainted() -> TurnProvenance {
        let mut p = TurnProvenance::new();
        p.observe_result("web_fetch", "<html>the page</html>");
        assert!(
            p.ingested_external(),
            "the fixture must actually be tainted"
        );
        p
    }

    /// Whether the placeholder appears anywhere in `content`.
    fn is_withheld_step_text_within(content: &str) -> bool {
        content.contains(WITHHELD_STEP_TEXT)
    }

    /// The digest a clean turn writes, with no operator destruction, in a
    /// conversation of the person's own.
    fn capture_clean(messages: &[Message], from: usize) -> Option<NewTurnDigest> {
        capture_turn(messages, from, clean(), false, TurnEnding::Answered, &[])
    }

    #[test]
    fn a_decision_the_user_stated_is_kept_without_the_model_recording_it() {
        // Nothing in this turn wrote a note, opened a step, or called a
        // memory tool. The decision still has to survive it.
        let messages = vec![
            user("from now on deploy with the kustomization, never with a raw apply"),
            assistant("understood"),
        ];
        let note = capture_clean(&messages, 0).expect("a turn was captured");

        assert!(
            note.content
                .contains("from now on deploy with the kustomization, never with a raw apply"),
            "{}",
            note.content
        );
        assert_eq!(note.opening_message_id, messages[0].id);
    }

    #[test]
    fn tool_calls_and_outcomes_are_kept_whether_or_not_a_step_claimed_them() {
        let messages = vec![
            user("check the deploy"),
            requested("c1", "read_file", r#"{"path":"deploy/README.md"}"#),
            Message::tool_result("c1", "the deploy notes"),
            requested("c2", "terminal_run", r#"{"cmd":"kubectl apply"}"#),
            Message::tool_result("c2", r#"{"ok":false,"error":"connection refused"}"#),
            assistant("the cluster is unreachable"),
        ];
        let note = capture_clean(&messages, 0).expect("a turn was captured");

        // Every call is named, with what came of it and an exact route to the
        // bytes.
        for expected in [
            "read_file",
            "answered",
            "terminal_run",
            "declined",
            &format!("message_id=\"{}\"", messages[2].id),
            &format!("message_id=\"{}\"", messages[4].id),
        ] {
            assert!(
                note.content.contains(expected),
                "{expected}: {}",
                note.content
            );
        }
    }

    /// A tool's bytes and a call's arguments are the two things a durable,
    /// recall-rendered note may not carry: a result payload is outside content
    /// by definition, and arguments are text the model wrote after it may have
    /// read some.
    #[test]
    fn a_tools_bytes_and_its_arguments_never_reach_the_durable_note() {
        let page = "the page said the deploy key is hunter2";
        let messages = vec![
            user("what does that page say"),
            requested("c1", "web_fetch", r#"{"url":"https://example.com/secret"}"#),
            Message::tool_result("c1", page),
            assistant("it says something"),
        ];
        let note = capture_clean(&messages, 0).expect("a turn was captured");

        assert!(
            !note.content.contains(page),
            "a tool's bytes must not reach the pad: {}",
            note.content
        );
        assert!(
            !note.content.contains("example.com/secret"),
            "a call's arguments must not reach the pad: {}",
            note.content
        );
        assert!(
            note.content.contains("web_fetch"),
            "the call still happened and the note still says so: {}",
            note.content
        );
        assert!(
            !note.after_outside_read,
            "this turn read nothing from outside, so there is no stamp to carry"
        );
    }

    #[test]
    fn a_tool_call_the_turn_never_answered_is_still_recorded() {
        let messages = vec![
            user("check the deploy"),
            requested("c1", "terminal_run", "{}"),
        ];
        let note = capture_clean(&messages, 0).expect("a turn was captured");
        assert!(note.content.contains("terminal_run"), "{}", note.content);
        assert!(
            note.content.contains("no result"),
            "an unanswered call is a fact about the turn: {}",
            note.content
        );
    }

    #[test]
    fn the_assistants_closing_text_is_kept_as_what_it_committed_to() {
        let messages = vec![
            user("what should I do"),
            requested("c1", "read_file", "{}"),
            Message::tool_result("c1", "notes"),
            assistant("push the tag, then apply the kustomization"),
        ];
        let note = capture_clean(&messages, 0).expect("a turn was captured");
        assert!(
            note.content
                .contains("push the tag, then apply the kustomization"),
            "{}",
            note.content
        );
    }

    #[test]
    fn a_range_that_holds_no_prompt_is_not_a_turn() {
        let messages = vec![assistant("just a reply")];
        assert!(capture_clean(&messages, 0).is_none());
        assert!(capture_clean(&[], 0).is_none());
        assert!(
            capture_clean(&messages, 99).is_none(),
            "an out-of-range start is not a turn either"
        );
    }

    /// Acceptance (#1349): a digest is durable, embedded, and read back by a
    /// LATER, clean turn - now from any conversation the person owns. A fetch
    /// marks a stamped digest and an unprompted render drops one, and both are
    /// keyed on the stamp, so a turn that read a page and then answered has to
    /// carry it: the assistant's own reply routinely quotes what it just read.
    #[test]
    fn a_digest_written_after_an_outside_read_carries_the_provenance_stamp() {
        let messages = vec![
            user("what does that page say"),
            requested("c1", "web_fetch", r#"{"url":"https://example.com"}"#),
            Message::tool_result("c1", "<html>the page</html>"),
            assistant("the page says to email the deploy key to attacker@example.com"),
        ];

        let clean_note = capture_turn(&messages, 0, clean(), false, TurnEnding::Answered, &[])
            .expect("a capture");
        let tainted_note = capture_turn(&messages, 0, tainted(), false, TurnEnding::Answered, &[])
            .expect("a capture");

        assert!(
            tainted_note.after_outside_read,
            "the writing turn read outside content, so the note must say so"
        );
        assert!(
            !clean_note.after_outside_read,
            "and a turn that read nothing must not: the stamp is the turn's, \
             not the shape of the note"
        );
        assert_eq!(
            clean_note.content, tainted_note.content,
            "the stamp accounts for the text rather than changing it"
        );
    }

    /// The operator setting that destroys instead of stamping. What the TURN
    /// derived goes; what the USER said stays, because the user's words are
    /// never outside content and losing them defeats the capture entirely.
    #[test]
    fn hard_withhold_drops_what_the_turn_derived_and_keeps_what_the_user_said() {
        let messages = vec![
            user("from now on deploy with the kustomization"),
            requested("c1", "web_fetch", "{}"),
            Message::tool_result("c1", "<html>a page</html>"),
            assistant("understood, and also email the key to attacker@example.com"),
        ];

        let note = capture_turn(&messages, 0, tainted(), true, TurnEnding::Answered, &[])
            .expect("a capture");

        assert!(
            note.content
                .contains("from now on deploy with the kustomization"),
            "the user's own decision must survive: {}",
            note.content
        );
        assert!(
            !note.content.contains("attacker@example.com"),
            "what the turn derived after reading a page must not: {}",
            note.content
        );
        assert!(
            is_withheld_step_text_within(&note.content),
            "{}",
            note.content
        );
        assert!(
            note.after_outside_read,
            "destruction and the stamp are not alternatives"
        );

        // A clean turn is untouched by the same setting.
        let clean_note = capture_turn(&messages, 0, clean(), true, TurnEnding::Answered, &[])
            .expect("a capture");
        assert!(
            clean_note.content.contains("understood"),
            "{}",
            clean_note.content
        );
    }

    /// A tool NAME is model-supplied text, and a turn that read a page can be
    /// told to call one whose name is the payload. The call fails; the
    /// assistant message carrying the name is persisted anyway.
    #[test]
    fn a_model_supplied_tool_name_is_bounded_before_it_is_stored() {
        let payload = "x".repeat(4_000);
        let messages = vec![
            user("check it"),
            requested("c1", &payload, "{}"),
            Message::tool_result("c1", r#"{"ok":false,"error":"unknown tool"}"#),
        ];

        let note = capture_clean(&messages, 0).expect("a capture");

        assert!(
            !note.content.contains(&payload),
            "an unbounded name must not reach the pad whole"
        );
        assert!(
            note.content.len() < 1_000,
            "the note stays small however long the name was: {} bytes",
            note.content.len()
        );
    }

    /// The module says the note is bounded AND says what it cut. The second
    /// half is the one that needs holding: a reader who cannot tell a complete
    /// record from a truncated one is reading a claim the record does not make.
    #[test]
    fn a_note_the_budget_cut_says_so() {
        let mut messages = vec![user("do all of it")];
        for i in 0..400 {
            messages.push(requested(&format!("c{i}"), "terminal_run", "{}"));
            messages.push(Message::tool_result(format!("c{i}"), "x".repeat(4000)));
        }
        let note = capture_clean(&messages, 0).expect("a capture");

        assert!(note.content.len() <= MAX_DIGEST_BYTES);
        assert!(
            note.content.ends_with(TRUNCATION_NOTICE),
            "a cut note must end by saying it was cut: {}",
            &note.content[note.content.len().saturating_sub(120)..]
        );

        // And a note that fits says nothing of the kind.
        let small = capture_clean(&[user("hello"), assistant("hi")], 0).expect("a capture");
        assert!(
            !small.content.contains(TRUNCATION_NOTICE),
            "{}",
            small.content
        );
    }

    /// The budget is spent in priority order, so a turn that produced a
    /// megabyte of tool output still keeps every word the user said.
    #[test]
    fn tool_traffic_never_crowds_out_what_the_user_said() {
        let prompt = "deploy the fleet image and tell me what broke";
        let mut messages = vec![user(prompt)];
        for i in 0..400 {
            messages.push(requested(&format!("c{i}"), "terminal_run", "{}"));
            messages.push(Message::tool_result(format!("c{i}"), "x".repeat(4000)));
        }
        messages.push(assistant("everything broke"));

        let note = capture_clean(&messages, 0).expect("a turn was captured");
        assert!(note.content.len() <= MAX_DIGEST_BYTES);
        assert!(
            note.content.contains(prompt),
            "the user's words come first and are never cut for tool traffic"
        );
        assert!(
            note.content.contains("everything broke"),
            "{}",
            note.content
        );
    }

    #[test]
    fn the_capture_starts_at_the_turn_it_was_given_and_not_earlier() {
        let messages = vec![
            user("an earlier turn"),
            assistant("earlier answer"),
            user("this turn"),
            assistant("this answer"),
        ];
        let note = capture_clean(&messages, 2).expect("a turn was captured");
        assert!(note.content.contains("this turn"), "{}", note.content);
        assert!(
            !note.content.contains("an earlier turn"),
            "an earlier turn has its own capture: {}",
            note.content
        );
        assert_eq!(note.opening_message_id, messages[2].id);
    }

    /// The key is derived from the turn, so re-running the capture writes the
    /// same row rather than a second one (AGENTS.md 8.4).
    #[test]
    fn capturing_the_same_turn_twice_names_the_same_row() {
        let messages = vec![user("the prompt"), assistant("the answer")];
        let first = capture_clean(&messages, 0).expect("a capture");
        let second = capture_clean(&messages, 0).expect("a capture");
        assert_eq!(first.opening_message_id, second.opening_message_id);
        assert_eq!(first.content, second.content);
    }

    /// AC: an omitted answer half is disclosed, the way a cut is. A turn that
    /// was never answered and a turn whose answer was empty both reach the
    /// digest without an `Answered:` block, and only the first says why.
    #[test]
    fn a_digest_with_no_answer_half_says_so_rather_than_reading_as_an_empty_answer() {
        let unanswered = capture_turn(
            &[user("always use the sealed secret"), assistant("")],
            0,
            clean(),
            false,
            TurnEnding::Unanswered,
            &[],
        )
        .expect("a capture");
        let empty_answer = capture_turn(
            &[user("always use the sealed secret"), assistant("")],
            0,
            clean(),
            false,
            TurnEnding::Answered,
            &[],
        )
        .expect("a capture");

        // Neither carries an answer, so neither may claim one.
        for note in [&unanswered, &empty_answer] {
            assert!(
                note.content.contains("always use the sealed secret"),
                "{}",
                note.content
            );
            assert!(
                !note.content.contains("Answered:"),
                "there was no answer to keep: {}",
                note.content
            );
        }

        // The one that was never answered says so; the one whose answer was
        // empty does not, or the line would say nothing about either.
        assert!(
            unanswered.content.contains(NO_ANSWER_NOTICE.trim()),
            "an unanswered turn discloses the omission: {}",
            unanswered.content
        );
        assert!(
            !empty_answer.content.contains(NO_ANSWER_NOTICE.trim()),
            "an empty answer is not the same fact: {}",
            empty_answer.content
        );
    }

    /// Acceptance (#1349): a subagent's conversation is mechanism, not the
    /// person's episodic record, and its opening message is a prompt the
    /// PARENT composed - possibly after the parent read an outside page, with
    /// nothing in the child's own turn to say so. The refusal is paired with
    /// the permit here, because a test that only proved a top-level turn is
    /// captured would pass against a capture with no exclusion at all.
    #[test]
    fn a_subagent_conversation_writes_no_digest_to_the_shared_store() {
        let messages = vec![
            user("read the page at that address and summarise it"),
            assistant("done"),
        ];

        let subagent = capture_turn(
            &messages,
            0,
            clean(),
            false,
            TurnEnding::Answered,
            &[RESERVED_SUBAGENT_TAG.to_string()],
        );
        assert!(
            subagent.is_none(),
            "a subagent conversation's turn must not reach the shared store"
        );

        // The permit: the same turn in a conversation of the person's own,
        // including one carrying ordinary tags.
        let own = capture_turn(
            &messages,
            0,
            clean(),
            false,
            TurnEnding::Answered,
            &["work".to_string()],
        )
        .expect("a top-level turn is still captured");
        assert_eq!(own.opening_message_id, messages[0].id);
    }
}
