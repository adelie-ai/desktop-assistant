//! How much transcript this model actually needs, measured (#1209).
//!
//! ## Two numbers this epic left unset on purpose
//!
//! [`crate::verbatim_window`] carries two figures nobody measured: how far a
//! model carries a conversation forward without the transcript, and the point
//! at which adding more transcript stops changing the answer. Picking either by
//! judgment is exactly the hand-fitting this project refuses, so both are left
//! at conservative defaults until an experiment sets them.
//!
//! One experiment answers both. Take real conversations from the store. Replay
//! the next turn against a ladder of window sizes. Compare the answers.
//!
//! ## What it measures, and what it does not
//!
//! It measures the MODEL, not the assembler. The prompt it builds is the
//! conversation's own messages cut to a window and the next thing the user
//! said - no tool schemas, no `[Recall]`, no plan. Adding those would measure
//! how much a particular assembly helps, which changes with every block this
//! project adds and is not what either number is for.
//!
//! ## Offline
//!
//! It reads conversations from the store rather than from any service, and it
//! runs as a batch job outside the request path. It does call a model - it has
//! to, because the question is what the model answers with less context - and
//! the connector it calls is the operator's, which on this project's own
//! hardware is a local one. Its tests call nothing: the replay arrives as a
//! closure, so a scripted answer is the whole fixture.
//!
//! ## Reading the result
//!
//! Both numbers are per model and land in `[context.models."<id>"]`, which is
//! where [`crate::verbatim_window::WindowPolicy`] already reads a per-model
//! target from. A model with no measurement takes the section's default, which
//! is the conservative number rather than an assumed one.

use std::collections::BTreeSet;

use crate::CoreError;
use crate::context::ContextProjection;
use crate::domain::{Conversation, Message, Role};

/// How close two answers must be for the larger window to have changed nothing.
///
/// Not 1.0: a model asked the same question twice varies its wording without
/// varying its answer, and a threshold of exact equality would report every
/// model as needing every token.
pub const CEILING_SIMILARITY: f64 = 0.9;

/// How close an answer must stay for a window to be worth trusting.
///
/// Lower than [`CEILING_SIMILARITY`] on purpose: the trust setting is the point
/// where the answer is still the same answer, not the point where it stops
/// moving at all.
pub const TRUST_SIMILARITY: f64 = 0.7;

/// What fraction of the sampled conversations must agree before a window
/// counts. A single conversation that happens not to need its history says
/// nothing about the model.
pub const AGREEMENT_QUORUM: f64 = 0.8;

/// Fewest conversations a measurement may rest on.
///
/// The quorum alone does not rule out a sample of one - `ceil(1 * 0.8)` is 1,
/// so one agreeing conversation would carry a rung by itself, which is exactly
/// what the quorum exists to prevent. This is the floor that actually does it.
pub const MIN_SAMPLE: usize = 3;

/// Why a conversation was not replayed.
///
/// Stated per conversation rather than counted, because "we sampled 40" and
/// "we sampled 40 of 900, and here is what we could not use" are different
/// claims and only the second one is honest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// Fewer than two turns, so there is no history to withhold.
    NoHistory,
    /// The conversation ends without a prompt to replay.
    NoPromptToReplay,
    /// The replay itself failed, with what the connector said.
    ReplayFailed(String),
}

impl SkipReason {
    /// A stable, machine-readable key.
    #[must_use]
    pub fn as_key(&self) -> &'static str {
        match self {
            Self::NoHistory => "no_history",
            Self::NoPromptToReplay => "no_prompt_to_replay",
            Self::ReplayFailed(_) => "replay_failed",
        }
    }
}

/// One conversation replayed across the ladder.
#[derive(Debug, Clone)]
pub struct ReplayedConversation {
    /// Which conversation, so a reader can go and look at it.
    pub conversation_id: String,
    /// One answer per rung, in the ladder's own order.
    pub answers: Vec<String>,
}

/// What the eval measured for one model.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowMeasurement {
    /// The model the answers came from.
    pub model: String,
    /// The ladder, in tokens, smallest first. A ladder that is not strictly
    /// ascending measures nothing rather than measuring the wrong thing.
    pub ladder: Vec<u64>,
    /// The smallest window at which the answer is still the same answer, or
    /// `None` when no rung reached the quorum.
    pub trust_tokens: Option<u64>,
    /// The smallest window beyond which the answer stops changing, or `None`.
    pub ceiling_tokens: Option<u64>,
    /// The conversations that were replayed.
    pub sampled: Vec<String>,
    /// The conversations that were not, and why.
    pub skipped: Vec<(String, SkipReason)>,
}

impl WindowMeasurement {
    /// How many conversations the numbers rest on.
    #[must_use]
    pub fn sample_size(&self) -> usize {
        self.sampled.len()
    }

    /// The `[context.models."<id>"]` fragment this measurement implies, or
    /// `None` when nothing was measured.
    ///
    /// The ceiling is what [`crate::verbatim_window`] reads. The trust setting
    /// is reported beside it as a comment rather than as a key, because
    /// nothing consumes it yet and a key nothing reads is a claim that
    /// something does.
    #[must_use]
    pub fn config_fragment(&self) -> Option<String> {
        let ceiling = self.ceiling_tokens?;
        let trust = match self.trust_tokens {
            Some(t) => format!("{t}"),
            None => "not reached on this ladder".to_string(),
        };
        Some(format!(
            "# measured over {} conversation(s); trust setting: {trust}\n\
             [context.models.\"{}\"]\n\
             verbatim_window_ceiling_tokens = {ceiling}\n",
            self.sample_size(),
            self.model
        ))
    }

    /// The report a person reads: both numbers, the sample size, and what was
    /// left out.
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = format!("model: {}\n", self.model);
        out.push_str(&format!(
            "sampled: {} conversation(s)\nskipped: {}\n",
            self.sample_size(),
            self.skipped.len()
        ));
        out.push_str(&format!(
            "trust setting: {}\nsufficiency ceiling: {}\n",
            self.trust_tokens.map_or_else(
                || "not reached on this ladder".to_string(),
                |t| t.to_string()
            ),
            self.ceiling_tokens.map_or_else(
                || "not reached on this ladder".to_string(),
                |t| t.to_string()
            ),
        ));
        if !self.sampled.is_empty() {
            out.push_str(&format!("sampled ids: {}\n", self.sampled.join(", ")));
        }
        for (id, why) in &self.skipped {
            out.push_str(&format!("skipped {id}: {}\n", why.as_key()));
        }
        out
    }
}

/// Measure both numbers from replayed conversations.
///
/// The reference answer for each conversation is the one from the largest rung.
/// A rung counts when enough conversations agree with their own reference, and
/// the answer is the smallest such rung - so both numbers are the point at
/// which the model stops needing more, rather than the point at which one
/// conversation happened to settle.
#[must_use]
pub fn measure(
    model: impl Into<String>,
    ladder: &[u64],
    replayed: Vec<ReplayedConversation>,
    skipped: Vec<(String, SkipReason)>,
) -> WindowMeasurement {
    let model = model.into();
    let sampled: Vec<String> = replayed.iter().map(|r| r.conversation_id.clone()).collect();

    // The ladder's order is the whole method: the answer is the SMALLEST rung
    // that agrees, and the largest is the reference. An unordered ladder
    // reports a rung that means nothing, so it measures nothing instead.
    let ascending = ladder.windows(2).all(|w| w[0] < w[1]);
    let rung = |threshold: f64| -> Option<u64> {
        if replayed.len() < MIN_SAMPLE || ladder.len() < 2 || !ascending {
            return None;
        }
        let needed = (replayed.len() as f64 * AGREEMENT_QUORUM).ceil() as usize;
        // The largest rung is the reference, so it cannot be its own evidence:
        // every answer agrees with itself, and reading the top rung as a
        // finding would report "the model needs exactly the ladder I chose" for
        // every model. A ladder whose rungs all disagree with the reference
        // measured nothing, and says so.
        for (i, tokens) in ladder.iter().enumerate().take(ladder.len() - 1) {
            let agreeing = replayed
                .iter()
                .filter(|r| {
                    let (Some(at), Some(reference)) = (r.answers.get(i), r.answers.last()) else {
                        return false;
                    };
                    similarity(at, reference) >= threshold
                })
                .count();
            if agreeing >= needed {
                return Some(*tokens);
            }
        }
        None
    };

    WindowMeasurement {
        model,
        ladder: ladder.to_vec(),
        trust_tokens: rung(TRUST_SIMILARITY),
        ceiling_tokens: rung(CEILING_SIMILARITY),
        sampled,
        skipped,
    }
}

/// How alike two answers are, on 0..=1.
///
/// Jaccard overlap of lower-cased word sets. Deterministic, needs no model and
/// no embedding backend, and it moves for the thing that matters here - an
/// answer that lost a fact says different words - while staying still for
/// wording a model varies between identical runs.
///
/// Two empty answers are identical, and an empty answer against a non-empty one
/// is not.
#[must_use]
pub fn similarity(a: &str, b: &str) -> f64 {
    let words = |s: &str| -> BTreeSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect()
    };
    let (a, b) = (words(a), words(b));
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let union = a.union(&b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(&b).count() as f64 / union as f64
}

/// The messages a replay sends for one conversation at one rung: the history
/// cut to `window_tokens`, then the prompt being replayed.
///
/// The cut is `verbatim_window::messages_within_tokens`, the same function the
/// live window uses.
///
/// **It is not handed the same projection, and that is a stated limit rather
/// than an oversight.** A live turn seeds its projection from the eviction
/// decisions earlier turns recorded, so a result already distilled costs a
/// pointer there and its full stored size here. On a conversation with prior
/// evictions this eval therefore cuts EARLIER than live would for the same
/// token target, and the ceiling it reports is conservative - it will hold at
/// least as much as it measured. Seeding it faithfully would mean reading the
/// scratchpad for every conversation at every rung, which is a per-rung store
/// read to make a number that is already erring in the safe direction.
///
/// `None` when the conversation holds no prompt to replay, or no history to
/// withhold - both are conversations this eval can say nothing about.
#[must_use]
pub fn replay_prompt(
    conversation: &Conversation,
    window_tokens: u64,
    estimate: &dyn Fn(&str) -> u64,
) -> Option<Vec<Message>> {
    // The prompt being replayed is the last thing the user said; everything
    // before it is the history the rung is allowed to keep.
    let last_prompt = conversation
        .messages
        .iter()
        .rposition(|m| m.role == Role::User)?;
    let history = &conversation.messages[..last_prompt];
    if !history.iter().any(|m| m.role == Role::User) {
        // One turn only: there is nothing to withhold, so replaying it
        // measures nothing about carrying things forward.
        return None;
    }
    let keep = crate::verbatim_window::messages_within_tokens(
        history,
        &ContextProjection::default(),
        estimate,
        window_tokens,
    );
    let mut out: Vec<Message> = history[history.len() - keep..].to_vec();
    out.push(conversation.messages[last_prompt].clone());
    Some(out)
}

/// Why one conversation cannot be replayed, or `None` when it can.
#[must_use]
pub fn unusable(conversation: &Conversation) -> Option<SkipReason> {
    let Some(last_prompt) = conversation
        .messages
        .iter()
        .rposition(|m| m.role == Role::User)
    else {
        return Some(SkipReason::NoPromptToReplay);
    };
    if !conversation.messages[..last_prompt]
        .iter()
        .any(|m| m.role == Role::User)
    {
        return Some(SkipReason::NoHistory);
    }
    None
}

/// Replay every usable conversation across the ladder and measure the result.
///
/// `replay` answers one rung of one conversation: it is handed the messages
/// [`replay_prompt`] built and returns what the model said. Passing it in
/// rather than taking an [`crate::ports::llm::LlmClient`] is what keeps this
/// testable without a model, and what keeps the eval's own dependencies to the
/// store.
///
/// A rung that fails takes its whole conversation out of the sample rather than
/// leaving a hole in its ladder: a measurement built from a conversation with a
/// missing rung would compare answers that are not the same conversation's.
pub async fn run_replay_eval<F, Fut>(
    model: impl Into<String>,
    conversations: Vec<Conversation>,
    ladder: &[u64],
    estimate: &dyn Fn(&str) -> u64,
    mut replay: F,
) -> WindowMeasurement
where
    F: FnMut(String, u64, Vec<Message>) -> Fut,
    Fut: std::future::Future<Output = Result<String, CoreError>>,
{
    let mut replayed = Vec::new();
    let mut skipped = Vec::new();

    for conversation in conversations {
        let id = conversation.id.0.clone();
        if let Some(why) = unusable(&conversation) {
            skipped.push((id, why));
            continue;
        }
        let mut answers = Vec::with_capacity(ladder.len());
        let mut failure = None;
        for rung in ladder {
            let Some(prompt) = replay_prompt(&conversation, *rung, estimate) else {
                failure = Some(SkipReason::NoPromptToReplay);
                break;
            };
            match replay(id.clone(), *rung, prompt).await {
                Ok(answer) => answers.push(answer),
                Err(e) => {
                    failure = Some(SkipReason::ReplayFailed(e.to_string()));
                    break;
                }
            }
        }
        match failure {
            Some(why) => skipped.push((id, why)),
            None => replayed.push(ReplayedConversation {
                conversation_id: id,
                answers,
            }),
        }
    }

    measure(model, ladder, replayed, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Message {
        Message::new(Role::User, text)
    }

    fn assistant(text: &str) -> Message {
        Message::new(Role::Assistant, text)
    }

    fn conversation(id: &str, turns: usize) -> Conversation {
        let mut conv = Conversation::new(id, "t");
        for i in 0..turns {
            conv.messages.push(user(&format!("q{i}")));
            conv.messages.push(assistant(&format!("a{i}")));
        }
        conv.messages.push(user("the prompt being replayed"));
        conv
    }

    fn replayed(id: &str, answers: &[&str]) -> ReplayedConversation {
        ReplayedConversation {
            conversation_id: id.to_string(),
            answers: answers.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// AC: the measurement itself calls nothing. Answers arrive as data, so
    /// the numbers are a pure function of what was replayed.
    #[test]
    fn the_measurement_is_a_pure_function_of_the_answers() {
        let ladder = [1_000, 4_000, 16_000];
        let one = measure(
            "m",
            &ladder,
            vec![replayed(
                "c1",
                &["wrong", "right answer here", "right answer here"],
            )],
            Vec::new(),
        );
        let two = measure(
            "m",
            &ladder,
            vec![replayed(
                "c1",
                &["wrong", "right answer here", "right answer here"],
            )],
            Vec::new(),
        );
        assert_eq!(one, two, "the same answers must measure the same");
    }

    /// AC: both numbers, per model, with the sample size.
    #[test]
    fn the_report_names_both_numbers_the_model_and_the_sample_size() {
        // Four conversations, all of which settle at the middle rung.
        let ladder = [1_000, 4_000, 16_000];
        let replayed_all: Vec<ReplayedConversation> = (0..4)
            .map(|i| {
                replayed(
                    &format!("c{i}"),
                    &[
                        "a completely different thing",
                        "the settled answer",
                        "the settled answer",
                    ],
                )
            })
            .collect();
        let m = measure("a-local-model", &ladder, replayed_all, Vec::new());

        assert_eq!(m.model, "a-local-model");
        assert_eq!(m.sample_size(), 4);
        assert_eq!(
            m.ceiling_tokens,
            Some(4_000),
            "the smallest rung beyond which the answer stops changing"
        );
        assert_eq!(
            m.trust_tokens,
            Some(4_000),
            "and the smallest at which it is still the same answer"
        );

        let report = m.report();
        assert!(report.contains("a-local-model"), "{report}");
        assert!(report.contains("sampled: 4"), "{report}");
        assert!(report.contains("trust setting: 4000"), "{report}");
        assert!(report.contains("sufficiency ceiling: 4000"), "{report}");
    }

    /// AC: it states what it skipped rather than presenting a sample as
    /// coverage.
    #[test]
    fn the_report_names_what_it_skipped_and_why() {
        let m = measure(
            "m",
            &[1_000, 4_000],
            vec![replayed("c1", &["an answer", "an answer"])],
            vec![
                ("c2".to_string(), SkipReason::NoHistory),
                (
                    "c3".to_string(),
                    SkipReason::ReplayFailed("the connector timed out".to_string()),
                ),
            ],
        );

        let report = m.report();
        assert!(report.contains("sampled: 1"), "{report}");
        assert!(report.contains("skipped: 2"), "{report}");
        assert!(report.contains("skipped c2: no_history"), "{report}");
        assert!(report.contains("skipped c3: replay_failed"), "{report}");
    }

    /// A model that never settles reports no number rather than the last rung
    /// it tried, so the conservative default stands.
    #[test]
    fn a_model_that_never_settles_reports_no_number() {
        let ladder = [1_000, 4_000];
        let m = measure(
            "m",
            &ladder,
            vec![
                replayed("c1", &["alpha beta", "gamma delta"]),
                replayed("c2", &["epsilon", "zeta"]),
            ],
            Vec::new(),
        );
        assert_eq!(m.ceiling_tokens, None);
        assert!(m.config_fragment().is_none(), "nothing to write");
        assert!(m.report().contains("not reached on this ladder"));
    }

    /// A single conversation that happens not to need its history says nothing
    /// about the model, so a rung needs a quorum.
    #[test]
    fn one_agreeing_conversation_does_not_set_the_number() {
        let ladder = [1_000, 4_000, 16_000];
        let m = measure(
            "m",
            &ladder,
            vec![
                replayed(
                    "c1",
                    &["the same answer", "the same answer", "the same answer"],
                ),
                replayed(
                    "c2",
                    &[
                        "something else entirely",
                        "the settled answer",
                        "the settled answer",
                    ],
                ),
                replayed(
                    "c3",
                    &["another thing", "the settled answer", "the settled answer"],
                ),
            ],
            Vec::new(),
        );
        assert_eq!(
            m.ceiling_tokens,
            Some(4_000),
            "one conversation out of three cannot carry the smaller rung"
        );
    }

    /// The top of the ladder is a bound the experiment chose, not a plateau it
    /// found, so agreeing only there is no measurement at all.
    #[test]
    fn agreeing_only_at_the_top_of_the_ladder_measures_nothing() {
        let m = measure(
            "m",
            &[1_000, 4_000],
            (0..4)
                .map(|i| replayed(&format!("c{i}"), &["nothing like it", "the settled answer"]))
                .collect(),
            Vec::new(),
        );
        assert_eq!(m.ceiling_tokens, None);
        assert_eq!(m.trust_tokens, None);
    }

    /// The quorum alone cannot rule out a sample of one - `ceil(1 * 0.8)` is 1,
    /// so one agreeing conversation would carry a rung by itself, which is
    /// exactly what the quorum exists to prevent.
    #[test]
    fn a_sample_smaller_than_the_floor_measures_nothing() {
        let ladder = [1_000, 4_000];
        for n in 1..MIN_SAMPLE {
            let m = measure(
                "m",
                &ladder,
                (0..n)
                    .map(|i| replayed(&format!("c{i}"), &["settled", "settled"]))
                    .collect(),
                Vec::new(),
            );
            assert_eq!(
                m.ceiling_tokens, None,
                "{n} conversation(s) cannot carry a rung"
            );
        }

        let enough = measure(
            "m",
            &ladder,
            (0..MIN_SAMPLE)
                .map(|i| replayed(&format!("c{i}"), &["settled", "settled"]))
                .collect(),
            Vec::new(),
        );
        assert_eq!(enough.ceiling_tokens, Some(1_000), "and the floor can");
    }

    /// The ladder's order is the whole method: the answer is the SMALLEST rung
    /// that agrees, and the largest is the reference.
    #[test]
    fn an_unordered_ladder_measures_nothing_rather_than_the_wrong_thing() {
        let replayed_all: Vec<ReplayedConversation> = (0..4)
            .map(|i| replayed(&format!("c{i}"), &["settled", "settled", "settled"]))
            .collect();

        let ascending = measure(
            "m",
            &[1_000, 4_000, 16_000],
            replayed_all.clone(),
            Vec::new(),
        );
        assert_eq!(ascending.ceiling_tokens, Some(1_000));

        for bad in [
            vec![16_000, 4_000, 1_000],
            vec![1_000, 16_000, 4_000],
            vec![1_000, 1_000, 4_000],
        ] {
            let m = measure("m", &bad, replayed_all.clone(), Vec::new());
            assert_eq!(
                m.ceiling_tokens, None,
                "a rung read off {bad:?} would mean nothing"
            );
        }
    }

    #[test]
    fn a_ladder_of_one_rung_can_measure_nothing() {
        let m = measure(
            "m",
            &[1_000],
            vec![replayed("c1", &["an answer"])],
            Vec::new(),
        );
        assert_eq!(m.ceiling_tokens, None);
    }

    /// AC: the result is readable from the per-model record the window already
    /// reads, not only from the eval's own output.
    #[test]
    fn the_measurement_renders_the_per_model_record_the_window_reads() {
        let m = measure(
            "a-local-model",
            &[1_000, 4_000],
            (0..4)
                .map(|i| replayed(&format!("c{i}"), &["settled here", "settled here"]))
                .collect(),
            Vec::new(),
        );
        let fragment = m.config_fragment().expect("a measured ceiling");
        assert!(
            fragment.contains("[context.models.\"a-local-model\"]"),
            "{fragment}"
        );
        assert!(
            fragment.contains("verbatim_window_ceiling_tokens = 1000"),
            "{fragment}"
        );
    }

    /// AC: the eval runs offline against stored conversations. The answers
    /// arrive through a closure, so this whole run touches no service - which
    /// is also what lets a test drive it with a scripted model.
    #[tokio::test]
    async fn the_runner_replays_every_usable_conversation_and_names_the_rest() {
        let estimate = |s: &str| (s.len() as u64).div_ceil(4);
        let mut one_turn = Conversation::new("c-short", "t");
        one_turn.messages.push(user("the only prompt"));

        let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = std::sync::Arc::clone(&asked);

        let m = run_replay_eval(
            "a-local-model",
            vec![conversation("c1", 4), one_turn, conversation("c2", 4)],
            &[1_000, 4_000],
            &estimate,
            move |id, rung, prompt| {
                let seen = std::sync::Arc::clone(&seen);
                async move {
                    seen.lock().unwrap().push((id, rung, prompt.len()));
                    Ok("the settled answer".to_string())
                }
            },
        )
        .await;

        assert_eq!(m.sample_size(), 2, "{:?}", m.sampled);
        assert_eq!(
            m.skipped,
            vec![("c-short".to_string(), SkipReason::NoHistory)],
            "a conversation with nothing to withhold is named, not counted"
        );
        assert_eq!(
            asked.lock().unwrap().len(),
            4,
            "two usable conversations, two rungs each"
        );
    }

    /// A rung that fails takes its whole conversation out of the sample: a
    /// ladder with a hole would compare answers that are not the same
    /// conversation's.
    #[tokio::test]
    async fn a_failed_rung_skips_its_conversation_and_says_what_went_wrong() {
        let estimate = |s: &str| (s.len() as u64).div_ceil(4);
        let m = run_replay_eval(
            "m",
            vec![conversation("c1", 4)],
            &[1_000, 4_000],
            &estimate,
            |_id, rung, _prompt| async move {
                if rung == 4_000 {
                    Err(CoreError::Llm("the connector timed out".to_string()))
                } else {
                    Ok("an answer".to_string())
                }
            },
        )
        .await;

        assert_eq!(m.sample_size(), 0);
        assert_eq!(m.skipped.len(), 1);
        assert_eq!(m.skipped[0].1.as_key(), "replay_failed");
        assert!(
            m.report().contains("skipped c1: replay_failed"),
            "{}",
            m.report()
        );
    }

    #[test]
    fn similarity_reads_the_same_answer_as_the_same() {
        assert_eq!(similarity("", ""), 1.0);
        assert_eq!(
            similarity("the deploy key is sealed", "the deploy key is sealed"),
            1.0
        );
        assert!(similarity("the deploy key is sealed", "The Deploy Key is sealed.") > 0.99);
        assert!(similarity("the deploy key is sealed", "no idea") < 0.2);
        assert_eq!(similarity("something", ""), 0.0);
    }

    // --- what the eval can and cannot use ---------------------------------

    #[test]
    fn a_conversation_with_one_turn_has_no_history_to_withhold() {
        let mut conv = Conversation::new("c1", "t");
        conv.messages.push(user("the only prompt"));
        assert_eq!(unusable(&conv), Some(SkipReason::NoHistory));
        assert!(replay_prompt(&conv, 1_000, &|s| s.len() as u64).is_none());
    }

    #[test]
    fn a_conversation_that_ends_without_a_prompt_cannot_be_replayed() {
        let mut conv = Conversation::new("c1", "t");
        conv.messages.push(assistant("a reply nobody asked for"));
        assert_eq!(unusable(&conv), Some(SkipReason::NoPromptToReplay));
    }

    /// The rung cuts the HISTORY and never the prompt being replayed: the
    /// question is what the model answers with less context, not what it
    /// answers to less of the question.
    #[test]
    fn the_rung_cuts_the_history_and_never_the_prompt() {
        let conv = conversation("c1", 8);
        let estimate = |s: &str| (s.len() as u64).div_ceil(4);

        let narrow = replay_prompt(&conv, 1, &estimate).expect("a usable conversation");
        let wide = replay_prompt(&conv, 1_000_000, &estimate).expect("a usable conversation");

        assert!(
            narrow.len() < wide.len(),
            "a smaller rung carries less history"
        );
        for prompt in [&narrow, &wide] {
            assert_eq!(
                prompt.last().map(|m| m.content.as_str()),
                Some("the prompt being replayed"),
                "every rung answers the same question"
            );
        }
    }
}
