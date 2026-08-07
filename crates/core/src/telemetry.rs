//! What a turn measures about itself, and the spans it hangs those
//! measurements from.
//!
//! ## The question this answers
//!
//! "Adele took four minutes to answer at 14:20." An operator has one
//! identifier from that report and needs to reach the provider call that was
//! slow, without turning anything on and without reproducing the turn.
//!
//! So a turn is one trace. Each iteration of the tool loop is a child span of
//! it, and the provider call and each tool dispatch are children of the round.
//! Four minutes then decomposes into provider time, tool time and the rest by
//! reading one trace, and the same decomposition is a set of histograms for
//! anybody asking about the trend rather than the incident.
//!
//! A conversation is deliberately **not** a trace. It lives for days and holds
//! an unbounded number of turns, which no backend renders usefully. The
//! conversation id is a span attribute instead, so one query still returns
//! every turn in a conversation.
//!
//! ## Every span here is at INFO, and closes with a line
//!
//! The daemon turns span-close events on, so each closing span writes how long
//! it was open. A turn may run up to `MAX_TOOL_ROUNDS` rounds, so a
//! pathological turn writes that many close lines and a few more.
//!
//! That is the intended trade and not an oversight. The close line *is* the
//! per-round duration, which is the single thing the report above needs; a
//! round already writes several lines of its own, so the close line is a small
//! addition to what is there; and demoting these spans to DEBUG would remove
//! the round from the trace in every shipped deployment, which is the opposite
//! of what this exists for.
//!
//! ## What may be recorded
//!
//! Ids, counts, durations, names of tools, models and providers, and token
//! counts. **Never content** - no prompt, no assembled context, no tool
//! argument, no search query, no model reply. A span field is the easiest
//! place to break that rule and the hardest place to notice it, because
//! nothing prints a span field unless an event fires inside the span. Every
//! span here is built by hand for that reason; there is no `#[instrument]`,
//! which would capture each argument by default.
//!
//! ## Label bounding
//!
//! The metrics registry caps a metric at 64 distinct label sets, first come,
//! with no eviction. Once burned, that dimension is dead until the process
//! restarts. So every outcome label is an enum rendering to `&'static str`,
//! which makes an unbounded value impossible to pass, and the two
//! caller-influenced axes - model and provider - come from operator
//! configuration rather than from a prompt. A conversation id, a user id or a
//! request id is never a label.

use std::fmt;
use std::time::Duration;

use adelie_telemetry::metrics::{self, Label};

use crate::ports::llm::TokenUsage;
use crate::ports::turn_telemetry::TurnRoute;

// ---------------------------------------------------------------------------
// Metric names.
// ---------------------------------------------------------------------------

/// How long a whole turn took, by outcome.
pub(crate) const TURN_DURATION: &str = "turn.duration";

/// How many rounds each turn spent. Read beside `turn.duration`'s count to get
/// the mean rounds per turn.
pub(crate) const TURN_ROUNDS: &str = "turn.rounds";

/// How long one iteration of the tool loop took, by outcome.
pub(crate) const ROUND_DURATION: &str = "turn.round.duration";

/// How long one provider call took, by provider, model and outcome.
pub(crate) const LLM_CALL_DURATION: &str = "llm.call.duration";

/// How long one tool dispatch took, by tool name, where it ran, and outcome.
pub(crate) const TOOL_CALL_DURATION: &str = "tool.call.duration";

/// Prompt tokens the provider reported, by provider and model.
pub(crate) const TOKENS_INPUT: &str = "llm.tokens.input";

/// Completion tokens the provider reported.
pub(crate) const TOKENS_OUTPUT: &str = "llm.tokens.output";

/// Tokens written into the provider's prompt cache.
pub(crate) const TOKENS_CACHE_WRITE: &str = "llm.tokens.cache_write";

/// Tokens served from the provider's prompt cache. On a caching provider this
/// is most of the cost story: a cache read costs a fraction of a fresh input
/// token, so input alone makes a well-cached turn look like a cold one.
pub(crate) const TOKENS_CACHE_READ: &str = "llm.tokens.cache_read";

/// Calls whose token count the provider did not report, by provider and by
/// which count was missing.
///
/// A count that is absent contributes nothing to the totals above, because
/// recording `0` would understate them with no way afterwards to tell a real
/// zero from a missing number. This counter is how a total that looks low gets
/// checked against how many calls did not report.
pub(crate) const TOKENS_UNREPORTED: &str = "llm.tokens.unreported";

// ---------------------------------------------------------------------------
// Outcomes. Each renders to a `&'static str`, so an unbounded value cannot
// reach a label: it has the wrong lifetime.
// ---------------------------------------------------------------------------

/// How a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnOutcome {
    /// The model produced the answer the user reads.
    Answered,
    /// The user cancelled, or the caller's token tripped.
    Cancelled,
    /// The turn ended on an error the user sees in place of an answer.
    Failed,
    /// The turn used every round it is allowed and wound down.
    RoundsExhausted,
}

impl TurnOutcome {
    pub(crate) fn as_label(self) -> &'static str {
        match self {
            Self::Answered => "answered",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::RoundsExhausted => "rounds_exhausted",
        }
    }
}

/// How one iteration of the tool loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoundOutcome {
    /// The model answered, so this was the last round.
    Answered,
    /// The model called tools and every one of them succeeded.
    ToolsCalled,
    /// The model called tools and at least one failed.
    ToolError,
    /// The provider call itself failed.
    LlmError,
    /// The round did not finish: the turn was cancelled inside it.
    Cancelled,
}

impl RoundOutcome {
    pub(crate) fn as_label(self) -> &'static str {
        match self {
            Self::Answered => "answered",
            Self::ToolsCalled => "tools_called",
            Self::ToolError => "tool_error",
            Self::LlmError => "llm_error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// How one tool dispatch ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolOutcome {
    Ok,
    Error,
}

impl ToolOutcome {
    pub(crate) fn as_label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
        }
    }
}

/// Where a tool ran. The turn loop routes a registered name to the caller's
/// own machine and everything else to the daemon, and the two have very
/// different latency, so they are separate series rather than one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolRunner {
    Client,
    Server,
}

impl ToolRunner {
    pub(crate) fn as_label(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::Server => "server",
        }
    }
}

// ---------------------------------------------------------------------------
// Spans. Built by hand, never by `#[instrument]`.
// ---------------------------------------------------------------------------

/// The turn's root span, carrying every id an operator can start from.
///
/// `rounds`, `outcome` and `duration_ms` are declared empty and filled in when
/// the turn ends, so one span answers both "which turn was this" and "what did
/// it cost" without a second lookup.
pub(crate) fn turn_span(conversation_id: &str, request_id: &str, user_id: &str) -> tracing::Span {
    let route = crate::ports::turn_telemetry::current_turn_route();
    tracing::info_span!(
        "turn",
        request_id = request_id,
        conversation_id = conversation_id,
        user_id = user_id,
        connection_id = route.connection_id(),
        model = route.model(),
        provider = route.provider(),
        rounds = tracing::field::Empty,
        outcome = tracing::field::Empty,
        duration_ms = tracing::field::Empty,
    )
}

/// One iteration of the tool loop.
///
/// The token fields stay empty when the provider reported no count, which is
/// how a reader tells "the provider did not say" from "it was zero".
pub(crate) fn round_span(round: usize) -> tracing::Span {
    tracing::info_span!(
        "turn.round",
        round = round,
        tools = tracing::field::Empty,
        outcome = tracing::field::Empty,
        input_tokens = tracing::field::Empty,
        output_tokens = tracing::field::Empty,
        cache_write_tokens = tracing::field::Empty,
        cache_read_tokens = tracing::field::Empty,
    )
}

/// One provider call, hung from its round.
///
/// The parent is explicit because the round span is never *entered*: entering
/// it would mean holding a span guard across an await, which attributes every
/// other task polled on that thread to this round. Naming the parent gives the
/// same tree with none of that risk.
pub(crate) fn llm_span(parent: &tracing::Span, round: usize, route: &TurnRoute) -> tracing::Span {
    tracing::info_span!(
        parent: parent,
        "llm.call",
        round = round,
        provider = route.provider(),
        model = route.model(),
        outcome = tracing::field::Empty,
    )
}

/// One tool dispatch, hung from its round. `tool` is the model-chosen name,
/// capped at the same boundary every other tool-name field uses.
pub(crate) fn tool_span(parent: &tracing::Span, tool: &str, runner: ToolRunner) -> tracing::Span {
    tracing::info_span!(
        parent: parent,
        "tool.call",
        tool = %crate::tools::summarize_tool_name(tool),
        runner = runner.as_label(),
        outcome = tracing::field::Empty,
    )
}

/// The turn's one pre-prompt recall lookup, which is where the embedding
/// round-trip happens.
pub(crate) fn recall_span(conversation_id: &str) -> tracing::Span {
    tracing::info_span!("recall.lookup", conversation_id = conversation_id)
}

// ---------------------------------------------------------------------------
// Measurements.
// ---------------------------------------------------------------------------

/// The `(provider, model)` label pair every per-call metric carries.
fn route_labels(route: &TurnRoute) -> [Label; 2] {
    [
        Label::new("provider", route.provider()),
        Label::new("model", route.model()),
    ]
}

/// Record a finished turn.
pub(crate) fn record_turn(elapsed: Duration, rounds: usize, outcome: TurnOutcome) {
    let labels = [Label::new("outcome", outcome.as_label())];
    metrics::record_duration(TURN_DURATION, elapsed, &labels);
    metrics::add(TURN_ROUNDS, rounds as u64, &labels);
}

/// Record a finished round.
pub(crate) fn record_round(elapsed: Duration, outcome: RoundOutcome) {
    metrics::record_duration(
        ROUND_DURATION,
        elapsed,
        &[Label::new("outcome", outcome.as_label())],
    );
}

/// Record a finished provider call.
pub(crate) fn record_llm_call(elapsed: Duration, route: &TurnRoute, ok: bool) {
    let [provider, model] = route_labels(route);
    metrics::record_duration(
        LLM_CALL_DURATION,
        elapsed,
        &[
            provider,
            model,
            Label::new("outcome", if ok { "ok" } else { "error" }),
        ],
    );
}

/// Record a finished tool dispatch.
pub(crate) fn record_tool_call(
    elapsed: Duration,
    tool: &str,
    runner: ToolRunner,
    outcome: ToolOutcome,
) {
    metrics::record_duration(
        TOOL_CALL_DURATION,
        elapsed,
        &[
            Label::new("tool", crate::tools::summarize_tool_name(tool)),
            Label::new("runner", runner.as_label()),
            Label::new("outcome", outcome.as_label()),
        ],
    );
}

/// One of the four token counts: what to call it in a label, what metric it
/// accumulates into, and how to read it off a provider's report.
type TokenCount = (&'static str, &'static str, fn(&TokenUsage) -> Option<u64>);

/// The four counts, and the name each is recorded under.
///
/// One list, read by both the recording below and the span fields, so a count
/// cannot be recorded to the facade and left off the span.
const COUNTS: [TokenCount; 4] = [
    ("input", TOKENS_INPUT, |u| u.input_tokens),
    ("output", TOKENS_OUTPUT, |u| u.output_tokens),
    ("cache_write", TOKENS_CACHE_WRITE, |u| {
        u.cache_creation_input_tokens
    }),
    ("cache_read", TOKENS_CACHE_READ, |u| {
        u.cache_read_input_tokens
    }),
];

/// Record one round's token usage, and count what the provider left out.
///
/// `None` is not zero. A count the provider did not report is skipped and
/// counted as unreported instead, so no total is silently understated. A
/// response with no usage at all counts every one of the four as unreported,
/// because that is what a connector that reports nothing looks like from here.
pub(crate) fn record_token_usage(usage: Option<&TokenUsage>, route: &TurnRoute) {
    let [provider, model] = route_labels(route);
    for (which, name, read) in COUNTS {
        match usage.and_then(read) {
            Some(value) => metrics::add(name, value, &[provider.clone(), model.clone()]),
            None => metrics::increment(
                TOKENS_UNREPORTED,
                &[provider.clone(), Label::new("count", which)],
            ),
        }
    }
}

/// A token count the provider may not have reported.
///
/// Renders as the number, or as `-` when the provider said nothing. A log line
/// that printed `0` for an absence would be indistinguishable from a real
/// zero, and there would be no way afterwards to tell which it was.
pub(crate) struct Count(pub(crate) Option<u64>);

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(value) => write!(f, "{value}"),
            None => f.write_str("-"),
        }
    }
}

/// A turn's token totals, summed from its rounds.
///
/// Each total stays `None` until some round reported that count, so a turn
/// whose provider reports nothing is visibly different from one that really
/// used no tokens. A round that did not report contributes nothing rather than
/// a zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TokenTotals {
    pub(crate) input: Option<u64>,
    pub(crate) output: Option<u64>,
    pub(crate) cache_write: Option<u64>,
    pub(crate) cache_read: Option<u64>,
}

impl TokenTotals {
    /// Add one round's counts.
    pub(crate) fn add(&mut self, usage: &TokenUsage) {
        fn accumulate(total: &mut Option<u64>, reported: Option<u64>) {
            if let Some(value) = reported {
                *total = Some(total.unwrap_or(0).saturating_add(value));
            }
        }
        accumulate(&mut self.input, usage.input_tokens);
        accumulate(&mut self.output, usage.output_tokens);
        accumulate(&mut self.cache_write, usage.cache_creation_input_tokens);
        accumulate(&mut self.cache_read, usage.cache_read_input_tokens);
    }
}

/// What a turn reports when it ends.
///
/// The turn body has several exits - an answer, a cancellation, a user-visible
/// error, an exhausted round budget - and one completion line has to cover all
/// of them. So the body fills this in as it runs and the caller reads it once,
/// rather than each exit remembering to write its own line.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TurnReport {
    /// How many rounds of the tool loop the turn ran.
    pub(crate) rounds: usize,
    /// How the turn ended. `Failed` until a path says otherwise, so an exit
    /// nobody classified reads as a problem rather than as a success.
    pub(crate) outcome: TurnOutcome,
    /// The turn's tokens, summed from its rounds rather than counted
    /// separately, so the two can never disagree.
    pub(crate) tokens: TokenTotals,
}

impl Default for TurnReport {
    fn default() -> Self {
        Self {
            rounds: 0,
            outcome: TurnOutcome::Failed,
            tokens: TokenTotals::default(),
        }
    }
}

/// One round of the tool loop, reported when it ends by any path.
///
/// A round has many exits: it answers, it dispatches tools and goes round
/// again, the provider rejects the prompt and the recovery ladder retries it,
/// the user cancels, an error ends the turn. A tokens-were-spent measurement
/// that each of those paths had to remember to write would be missing from
/// most of them, and the paths that forget are exactly the failures worth
/// measuring. So the reporting happens on drop, which every exit performs.
///
/// The default outcome is [`RoundOutcome::Cancelled`] because that is what an
/// exit nobody classified actually was: the round stopped part-way.
pub(crate) struct RoundGuard {
    span: tracing::Span,
    started: std::time::Instant,
    round: usize,
    outcome: RoundOutcome,
    /// Whether the provider call was issued at all. A round that stopped
    /// before dispatching has nothing to report and must not be counted as a
    /// call whose usage went missing.
    llm_called: bool,
    usage: Option<TokenUsage>,
    route: TurnRoute,
}

impl RoundGuard {
    /// Open a round. `round` is one-based, matching what the log line says.
    pub(crate) fn new(round: usize, route: TurnRoute) -> Self {
        Self {
            span: round_span(round),
            started: std::time::Instant::now(),
            round,
            outcome: RoundOutcome::Cancelled,
            llm_called: false,
            usage: None,
            route,
        }
    }

    /// The round's span, for hanging the provider call and each tool dispatch
    /// from.
    pub(crate) fn span(&self) -> &tracing::Span {
        &self.span
    }

    /// Note that the provider call was issued.
    pub(crate) fn llm_called(&mut self) {
        self.llm_called = true;
    }

    /// Note what the provider reported. `None` is a real answer here: it means
    /// the connector reported no usage at all.
    pub(crate) fn set_usage(&mut self, usage: Option<TokenUsage>) {
        if let Some(usage) = &usage {
            record_tokens_on_span(&self.span, usage);
        }
        self.usage = usage;
    }

    /// Note which tools this round called. Names only, each capped at the same
    /// boundary every other tool-name field uses.
    pub(crate) fn set_tools(&mut self, names: impl Iterator<Item = String>) {
        let names: Vec<String> = names
            .map(|n| crate::tools::summarize_tool_name(&n))
            .collect();
        self.span.record("tools", names.join(","));
    }

    /// Note how the round ended.
    pub(crate) fn set_outcome(&mut self, outcome: RoundOutcome) {
        self.outcome = outcome;
    }
}

impl Drop for RoundGuard {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        self.span.record("outcome", self.outcome.as_label());
        record_round(elapsed, self.outcome);
        if self.llm_called {
            record_token_usage(self.usage.as_ref(), &self.route);
        }
        let usage = self.usage.clone().unwrap_or_default();
        self.span.in_scope(|| {
            tracing::info!(
                round = self.round,
                duration_ms = elapsed.as_millis() as u64,
                outcome = self.outcome.as_label(),
                input_tokens = %Count(usage.input_tokens),
                output_tokens = %Count(usage.output_tokens),
                cache_write_tokens = %Count(usage.cache_creation_input_tokens),
                cache_read_tokens = %Count(usage.cache_read_input_tokens),
                "round finished"
            );
        });
    }
}

/// Put a round's token counts on its span, present ones only.
///
/// An absent count leaves its field empty rather than recording a zero, so a
/// trace shows the same distinction the metrics do.
pub(crate) fn record_tokens_on_span(span: &tracing::Span, usage: &TokenUsage) {
    if let Some(value) = usage.input_tokens {
        span.record("input_tokens", value);
    }
    if let Some(value) = usage.output_tokens {
        span.record("output_tokens", value);
    }
    if let Some(value) = usage.cache_creation_input_tokens {
        span.record("cache_write_tokens", value);
    }
    if let Some(value) = usage.cache_read_input_tokens {
        span.record("cache_read_tokens", value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_count_renders_as_absent_not_as_zero() {
        assert_eq!(Count(Some(0)).to_string(), "0");
        assert_eq!(Count(None).to_string(), "-");
    }

    #[test]
    fn totals_skip_what_a_provider_did_not_report() {
        let mut totals = TokenTotals::default();
        totals.add(&TokenUsage {
            input_tokens: Some(100),
            output_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        });
        totals.add(&TokenUsage {
            input_tokens: Some(200),
            output_tokens: Some(20),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        });

        assert_eq!(totals.input, Some(300));
        assert_eq!(
            totals.output,
            Some(20),
            "the round that reported an output count still contributes it"
        );
        assert_eq!(
            totals.cache_read, None,
            "a count no round reported stays absent rather than becoming zero"
        );
    }

    #[test]
    fn every_outcome_renders_to_a_bounded_label() {
        // A `&'static str` return is what makes an unbounded value impossible
        // to pass into a label: a model-chosen or user-supplied string has the
        // wrong lifetime and does not compile. The exhaustive matches with no
        // wildcard arm are the other half - a new variant forces someone to
        // name it rather than landing silently as `other`.
        let turn = [
            TurnOutcome::Answered,
            TurnOutcome::Cancelled,
            TurnOutcome::Failed,
            TurnOutcome::RoundsExhausted,
        ]
        .map(TurnOutcome::as_label);
        let round = [
            RoundOutcome::Answered,
            RoundOutcome::ToolsCalled,
            RoundOutcome::ToolError,
            RoundOutcome::LlmError,
            RoundOutcome::Cancelled,
        ]
        .map(RoundOutcome::as_label);

        for label in turn.iter().chain(round.iter()) {
            assert!(
                !label.is_empty() && label.is_ascii(),
                "an outcome label is a series key and must be a short stable \
                 token; got {label:?}"
            );
        }
        assert_eq!(
            turn.len(),
            turn.iter().collect::<std::collections::HashSet<_>>().len(),
            "two turn outcomes rendering to one label would merge two series"
        );
        assert_eq!(
            round.len(),
            round.iter().collect::<std::collections::HashSet<_>>().len(),
            "two round outcomes rendering to one label would merge two series"
        );
    }

    #[test]
    fn an_unresolved_route_still_produces_two_labels() {
        // The daemon falls through to its statically-configured primary client
        // when no concrete connection resolves, and then knows neither. The
        // metric must still be recorded, under a named sentinel, rather than
        // dropped or labelled with an empty string that renders as nothing.
        let [provider, model] = route_labels(&TurnRoute::default());
        assert_eq!(provider.value(), crate::ports::turn_telemetry::UNSET);
        assert_eq!(model.value(), crate::ports::turn_telemetry::UNSET);
    }

    #[test]
    fn the_four_counts_are_read_from_one_list() {
        // The span fields and the facade recording both walk `COUNTS`, so a
        // fifth count cannot be added to one and forgotten in the other.
        let usage = TokenUsage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            cache_creation_input_tokens: Some(3),
            cache_read_input_tokens: Some(4),
        };
        let read: Vec<Option<u64>> = COUNTS.iter().map(|(_, _, read)| read(&usage)).collect();
        assert_eq!(read, vec![Some(1), Some(2), Some(3), Some(4)]);
    }
}
