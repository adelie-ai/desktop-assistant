//! How much of the transcript this turn carries verbatim, in tokens (#1208).
//!
//! ## Why a count of messages bounds nothing
//!
//! The window was "the most recent `MAX_CONTEXT_MESSAGES`
//! messages". A turn is not a unit of size: one is "thanks" and the next
//! carries 40 KB of tool output, so forty messages is anywhere between a
//! paragraph and most of a context window. A bound that cannot say how big it
//! is is not a bound.
//!
//! ## The target
//!
//! The lower of a fraction of the effective per-turn input budget and an
//! absolute working ceiling.
//!
//! **Capacity is not a budget.** A third of a million-token window is 330,000
//! tokens per turn because the room existed, which is exactly the growth this
//! epic exists to refuse. The fraction protects a small window and the ceiling
//! protects a large one, and neither does the other's job.
//!
//! **"Effective" is a claim about which number.** Three figures all read as
//! "the context window" and they differ: the model's nominal window, the
//! ceiling an operator configured, and what the assembler actually plans
//! against once the learned-overflow cap has applied. The fraction is taken
//! from the third - [`crate::ports::llm::ContextBudget::max_input_tokens`], the
//! resolved figure, whatever tier produced it. A target measured against
//! either of the others is a claim in the wrong unit.
//!
//! ## Pressure, not a limit
//!
//! Below the target nothing happens. Above it the window carries fewer turns,
//! which raises the pressure on the mechanisms that already exist:
//! [`crate::planning`] evicts superseded results, `[Earlier turns]` keeps a
//! dropped turn distinguishable from one that never happened, and the rolling
//! summary keeps its gist.
//!
//! **Nothing is refused and nothing is truncated.** A turn whose most recent
//! turn alone exceeds the target gets that turn whole: the floor is one
//! complete turn, always. `COMPACTION_TOKEN_RATIO` stays where it is as the
//! emergency; success is that it stops being reached.
//!
//! **This module does not decide what is dropped or in what order.** That is
//! the eviction policy, and #1205 states it once, in [`crate::planning`].

use crate::context::ContextProjection;
use crate::domain::{Message, Role};

/// Fraction of the effective per-turn input budget the verbatim window may
/// hold, before the ceiling applies.
///
/// A third: enough that ordinary work never feels it, small enough that the
/// prompt keeps room for the standing frame, the tool schemas and the turn's
/// own output. Not measured - #1209 is the experiment that would set it from
/// real conversations rather than from judgment.
pub const DEFAULT_WINDOW_RATIO: f64 = 0.33;

/// Absolute ceiling on the verbatim window, whatever the fraction works out to.
///
/// **Chosen, not measured.** #1209 replaces it with the point at which adding
/// more transcript stops changing the answer, per model. Until then this is a
/// working number: comfortably above what an ordinary turn carries, and far
/// below what a large window would otherwise permit.
pub const DEFAULT_WINDOW_CEILING_TOKENS: u64 = 60_000;

/// What bounds this turn's verbatim window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowTarget {
    /// Fraction of the effective per-turn input budget.
    pub ratio: f64,
    /// Absolute ceiling, in estimated tokens.
    pub ceiling_tokens: u64,
}

impl Default for WindowTarget {
    fn default() -> Self {
        Self {
            ratio: DEFAULT_WINDOW_RATIO,
            ceiling_tokens: DEFAULT_WINDOW_CEILING_TOKENS,
        }
    }
}

impl WindowTarget {
    /// The target in estimated tokens, given the effective per-turn input
    /// budget.
    ///
    /// `effective_budget` is
    /// [`crate::ports::llm::ContextBudget::max_input_tokens`] - the figure the
    /// assembler plans against after the learned cap, not the model's nominal
    /// window and not the configured ceiling. The module header says why that
    /// distinction is the whole claim.
    #[must_use]
    pub fn tokens(self, effective_budget: u64) -> u64 {
        let share = (effective_budget as f64 * self.ratio) as u64;
        share.min(self.ceiling_tokens).max(1)
    }
}

/// Whether this daemon bounds the verbatim window by tokens, and to what.
///
/// **Off by default, and off leaves the window byte-for-byte as it was.** This
/// ticket's failure mode presents as "she forgot", which is the one failure
/// this project says must never happen, so the change ships behind a switch
/// that an operator turns on rather than one they must remember to turn off.
///
/// Off is not a claim about the whole upgrade. The `[Earlier turns]` index is
/// gated on message-count windowing rather than on this, and renders whether
/// or not the bound is on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WindowPolicy {
    /// Whether the token bound applies at all.
    pub enabled: bool,
    /// The target for a model nothing else names.
    pub default_target: WindowTarget,
    /// Per-model targets, keyed by the model id the turn route reports.
    ///
    /// Per model because the two numbers this target is built from are per
    /// model: what the model's window costs, and how far it carries a
    /// conversation without the transcript. #1209 measures the second one.
    pub by_model: std::collections::HashMap<String, WindowTarget>,
}

impl WindowPolicy {
    /// The target in force for `model`, or `None` when the bound is off.
    #[must_use]
    pub fn target_for(&self, model: &str) -> Option<WindowTarget> {
        self.enabled.then(|| {
            self.by_model
                .get(model)
                .copied()
                .unwrap_or(self.default_target)
        })
    }
}

/// How many of the most recent messages fit `target_tokens`, snapped back to a
/// turn boundary.
///
/// Answers a message COUNT, because that is what the window start and the
/// compaction range both already take, so the window the model reads and the
/// range the summariser folds cannot disagree about where the boundary is.
///
/// **The floor is one complete turn.** A single turn that exceeds the target on
/// its own is carried whole. Refusing it would drop the thing the user just
/// asked about, which is the failure this epic exists to prevent, and the
/// target is pressure rather than a limit.
///
/// Costed through `projection`, so a result the round already reads as a
/// pointer is counted at the size of the pointer. What the model reads is what
/// the budget counts.
pub(crate) fn messages_within_tokens(
    messages: &[Message],
    projection: &ContextProjection,
    estimate: &dyn Fn(&str) -> u64,
    target_tokens: u64,
) -> usize {
    let starts = turn_starts(messages);
    let Some(&last) = starts.last() else {
        // No turn at all: nothing here can bound what is not a conversation.
        return messages.len();
    };

    // The floor, taken before anything is measured: the most recent turn
    // travels whole whatever it costs.
    let mut count = messages.len() - last;
    let mut spent: u64 = cost_of(&messages[last..], projection, estimate);

    for &start in starts.iter().rev().skip(1) {
        // Everything from this turn's opening up to the window as it stands.
        // The window only ever grows backwards here, so the slice is never
        // inverted.
        let window_start = messages.len() - count;
        let cost = cost_of(&messages[start..window_start], projection, estimate);
        if spent.saturating_add(cost) > target_tokens {
            break;
        }
        spent += cost;
        count = messages.len() - start;
    }
    count
}

/// Estimated tokens the model reads for `messages`.
fn cost_of(
    messages: &[Message],
    projection: &ContextProjection,
    estimate: &dyn Fn(&str) -> u64,
) -> u64 {
    messages
        .iter()
        .map(|m| estimate(projection.content(m)))
        .sum()
}

/// Index of every message that opens a turn.
///
/// A turn runs from a `Role::User` message to the message before the next one.
/// Anything ahead of the first user message belongs to no turn, which is why
/// this can be empty for a conversation that has one.
fn turn_starts(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per four characters, the shape every estimator here uses.
    fn estimate(text: &str) -> u64 {
        (text.len() as u64).div_ceil(4)
    }

    fn user(text: &str) -> Message {
        Message::new(Role::User, text)
    }

    fn assistant(text: &str) -> Message {
        Message::new(Role::Assistant, text)
    }

    /// `turns` turns, each costing about `tokens_each` estimated tokens.
    fn conversation(turns: usize, tokens_each: usize) -> Vec<Message> {
        let body = "x".repeat(tokens_each * 4);
        (0..turns)
            .flat_map(|i| [user(&format!("q{i}")), assistant(&body)])
            .collect()
    }

    #[test]
    fn the_bound_is_off_until_an_operator_turns_it_on() {
        let policy = WindowPolicy::default();
        assert!(!policy.enabled);
        assert_eq!(policy.target_for("claude-opus-5"), None);
    }

    #[test]
    fn a_target_is_settable_per_model() {
        let mut policy = WindowPolicy {
            enabled: true,
            ..Default::default()
        };
        policy.by_model.insert(
            "a-small-local-model".to_string(),
            WindowTarget {
                ratio: 0.2,
                ceiling_tokens: 4_000,
            },
        );
        assert_eq!(
            policy.target_for("a-small-local-model"),
            Some(WindowTarget {
                ratio: 0.2,
                ceiling_tokens: 4_000
            })
        );
        assert_eq!(
            policy.target_for("claude-opus-5"),
            Some(WindowTarget::default()),
            "a model nothing names falls back to the default target"
        );
    }

    #[test]
    fn the_target_is_the_lower_of_the_share_and_the_ceiling() {
        let target = WindowTarget {
            ratio: 0.33,
            ceiling_tokens: 60_000,
        };
        // A small window: the share binds.
        assert_eq!(target.tokens(100_000), 33_000);
        // A large one: the ceiling binds, so capacity alone never grants room.
        assert_eq!(target.tokens(1_000_000), 60_000);
    }

    #[test]
    fn the_window_keeps_the_most_recent_turns_that_fit() {
        let messages = conversation(10, 100);
        // Room for about three turns of ~100 tokens each.
        let count =
            messages_within_tokens(&messages, &ContextProjection::default(), &estimate, 330);
        assert_eq!(count, 6, "three whole turns, two messages each");
        // And the window starts on a prompt, not mid-turn.
        assert_eq!(messages[messages.len() - count].role, Role::User);
    }

    #[test]
    fn one_complete_turn_is_the_floor_however_much_it_costs() {
        let messages = conversation(5, 10_000);
        let count = messages_within_tokens(&messages, &ContextProjection::default(), &estimate, 10);
        assert_eq!(
            count, 2,
            "the most recent turn travels whole; the target never truncates it"
        );
    }

    #[test]
    fn a_target_larger_than_the_conversation_keeps_all_of_it() {
        let messages = conversation(3, 10);
        let count = messages_within_tokens(
            &messages,
            &ContextProjection::default(),
            &estimate,
            1_000_000,
        );
        assert_eq!(count, messages.len());
    }

    /// What the model reads is what the budget counts: a result the round
    /// already reads as a pointer costs the pointer, not the payload.
    #[test]
    fn an_evicted_result_is_counted_at_what_the_round_reads() {
        let mut messages = vec![user("q0"), assistant("a0"), user("q1")];
        let big = Message::new(Role::Tool, "x".repeat(40_000));
        messages.insert(2, big.clone());

        let whole =
            messages_within_tokens(&messages, &ContextProjection::default(), &estimate, 1_000);

        let mut projection = ContextProjection::default();
        projection.replace(&messages[2], "<compacted>".to_string());
        let projected = messages_within_tokens(&messages, &projection, &estimate, 1_000);

        assert!(
            projected > whole,
            "the pointer costs less, so more turns fit: {projected} vs {whole}"
        );
    }

    #[test]
    fn a_conversation_with_no_prompt_is_not_bounded_by_this() {
        let messages = vec![assistant("a preamble")];
        let count = messages_within_tokens(&messages, &ContextProjection::default(), &estimate, 1);
        assert_eq!(count, messages.len());
    }
}
