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
//! conversation id is a span attribute on every span here instead, so one query
//! still returns every turn in a conversation.
//!
//! ## The trace id is the request id
//!
//! A uuid is 16 bytes and a W3C trace id is 16 bytes, so the turn's own
//! correlation id becomes its trace id with no mapping table between them. The
//! client mints that id, the daemon adopts it, and
//! [`crate::ports::turn_telemetry`] owns both rules. One identifier then
//! appears in the client's event stream, in the pod log and in a backend.
//!
//! Binding it to an exported span needs the `otel` feature; carrying it does
//! not. A default build still puts the value on every line the turn writes.
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
//! counts - what a provider said a call cost, which [`tokens`] owns, and the
//! per-part breakdown of what filled the input, which [`prompt`] owns. **Never
//! content** - no prompt, no assembled context, no tool argument, no search
//! query, no model reply. A span field is the easiest
//! place to break that rule and the hardest place to notice it, because
//! nothing prints a span field unless an event fires inside the span. Every
//! span here is built by hand for that reason; there is no `#[instrument]`,
//! which would capture each argument by default.
//!
//! ## Label bounding
//!
//! The metrics registry caps a metric at 64 distinct label sets, first come,
//! with no eviction. Once burned, that dimension is dead until the process
//! restarts.
//!
//! So every outcome and purpose label is an enum rendering to `&'static str`,
//! which makes an unbounded value impossible to pass, and `provider` and
//! `model` come from operator configuration rather than from a prompt.
//!
//! Two values on a span come from outside and go through [`Safe`] for that
//! reason: the tool name, which the model writes, and the conversation id,
//! which arrives on the wire. Neither is bounded otherwise, and a `%` field
//! reaches the console line through `Display`, which does not escape what
//! `Debug` would.
//!
//! One label is not bounded by its type, and it is the one to keep an eye on:
//! `tool`, whose value the **model** writes. It is bounded at the call site
//! instead - a name the turn did not advertise is recorded as [`UNKNOWN_TOOL`].
//! A conversation id, a user id or a request id is never a label.

mod prompt;
mod tokens;

pub(crate) use prompt::{PromptBreakdown, PromptPart};
pub(crate) use tokens::{
    Count, TokenTotals, record_genai_tokens_on_span, record_token_usage, record_tokens_on_span,
};

use std::time::Duration;

use adelie_telemetry::Safe;
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

/// How long one tool dispatch took, by tool name and outcome.
///
/// **Not** by where it ran. `runner` is on the span, where a series key costs
/// nothing; on the metric it would double the label sets this one name spends,
/// and the tool axis is the one under pressure.
pub(crate) const TOOL_CALL_DURATION: &str = "tool.call.duration";

/// The `tool` label for a name this turn never advertised.
///
/// A model can emit any string as a tool name, and an invented one is
/// dispatched, fails, and is recorded like any other. Without this, sixty-four
/// invented names - about sixty-four rounds of one conversation - fill the
/// registry's per-metric label budget, which has no eviction, and every real
/// tool afterwards folds into `cardinality=other` until the process restarts.
/// So a name the daemon does not know is not a name; it is this.
///
/// The caller answers by asking what it offered this round and, failing that,
/// what the executor knows - the daemon's own tool list, which is the set that
/// bounds this label.
pub(crate) const UNKNOWN_TOOL: &str = "unknown";

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
    /// The round produced nothing the turn could use and it went round again
    /// with different settings - the hosted-search demotion. Distinct from a
    /// cancellation, which is what it read as before, and distinct from an
    /// error, because nothing failed.
    Retried,
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
            Self::Retried => "retried",
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
pub(crate) fn turn_span(
    conversation_id: &str,
    request_id: &str,
    user_id: &str,
    trace: &crate::ports::turn_telemetry::TurnTrace,
) -> tracing::Span {
    let route = crate::ports::turn_telemetry::current_turn_route();
    let span = tracing::info_span!(
        "turn",
        request_id = request_id,
        conversation_id = %Safe::name(conversation_id),
        user_id = user_id,
        connection_id = route.connection_id(),
        model = route.model(),
        provider = route.provider(),
        trace_id = %trace.trace.trace_id(),
        // The security level this turn ran at, so a backend can find every
        // turn that ran permissively without reading any of them.
        tool_policy = crate::ports::llm::current_tool_policy().as_str(),
        rounds = tracing::field::Empty,
        outcome = tracing::field::Empty,
        duration_ms = tracing::field::Empty,
        // What filled the input (#1203). A span fixes its field set when it
        // opens, and a `record` against a field it never declared is dropped
        // silently, so every name `PromptPart::as_span_field` can return is
        // spelled out here. `tests/turn_telemetry.rs` is what holds the two
        // lists together.
        prompt.system_tokens = tracing::field::Empty,
        prompt.summary_tokens = tracing::field::Empty,
        prompt.current_task_tokens = tracing::field::Empty,
        prompt.working_state_tokens = tracing::field::Empty,
        prompt.plan_tokens = tracing::field::Empty,
        prompt.pinned_tokens = tracing::field::Empty,
        prompt.scratchpad_tokens = tracing::field::Empty,
        prompt.recall_tokens = tracing::field::Empty,
        prompt.transcript_tokens = tracing::field::Empty,
        prompt.tool_schema_tokens = tracing::field::Empty,
        prompt.total_tokens = tracing::field::Empty,
        prompt.tool_count = tracing::field::Empty,
    );
    // The turn is the root of its trace, so this is where the trace id the
    // client already knows becomes the one a backend indexes by. Everything
    // below inherits it.
    crate::otel_bridge::bind_parent(&span, trace);
    span
}

/// One iteration of the tool loop.
///
/// The token fields stay empty when the provider reported no count, which is
/// how a reader tells "the provider did not say" from "it was zero".
pub(crate) fn round_span(round: usize) -> tracing::Span {
    tracing::info_span!(
        "turn.round",
        round = round,
        conversation_id = %Safe::name(crate::ports::turn_telemetry::current_conversation_id()),
        tools = tracing::field::Empty,
        outcome = tracing::field::Empty,
        input_tokens = tracing::field::Empty,
        output_tokens = tracing::field::Empty,
        cache_write_tokens = tracing::field::Empty,
        cache_read_tokens = tracing::field::Empty,
    )
}

/// Which of a turn's provider calls this is.
///
/// A turn spends provider time outside its rounds - a title on the first
/// message, a compaction summary, the recovery ladder after an overflow, the
/// wind-down when the round budget runs out. Without this axis those calls are
/// a gap in the trace, and a turn whose four minutes went into compaction
/// decomposes into nothing an operator can act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LlmPurpose {
    /// A round of the tool loop: the model answering the user.
    Turn,
    /// Naming a new conversation from its first message.
    Title,
    /// Summarising the transcript to fit the window. Every path that folds
    /// the transcript reaches the provider through one function, so they share
    /// this one purpose: turn-entry compaction, the token-pressure fold, the
    /// assembler's pre-flight shrink, and the recovery ladder's last step.
    Compaction,
    /// Sorting a large tool fleet into namespaces the provider can search.
    /// The ladder that runs after the provider rejects an oversized prompt.
    Categorization,
    /// The closing reply when the round budget is spent.
    WindDown,
}

impl LlmPurpose {
    pub(crate) fn as_label(self) -> &'static str {
        match self {
            Self::Turn => "turn",
            Self::Title => "title",
            Self::Compaction => "compaction",
            Self::Categorization => "categorization",
            Self::WindDown => "wind_down",
        }
    }
}

/// A provider call a turn makes outside its rounds, hung from the turn itself.
///
/// The turn span is current wherever these are built, so the parent is
/// contextual rather than named. Unlike a round's call they have no round to
/// hang from - they are the turn's own overheads.
///
/// The token attributes are declared empty for the reason [`llm_span`] gives.
pub(crate) fn aux_llm_span(purpose: LlmPurpose) -> tracing::Span {
    let route = crate::ports::turn_telemetry::current_turn_route();
    tracing::info_span!(
        "llm.call",
        purpose = purpose.as_label(),
        provider = route.provider(),
        model = route.model(),
        conversation_id = %Safe::name(crate::ports::turn_telemetry::current_conversation_id()),
        provider_request_id = tracing::field::Empty,
        gen_ai.usage.input_tokens = tracing::field::Empty,
        gen_ai.usage.output_tokens = tracing::field::Empty,
        gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
        gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
    )
}

/// Measure a provider call a turn makes outside its rounds.
///
/// Returns what the call returned, so a call site wraps rather than
/// restructures. The measurement lands on the same histogram as a round's
/// call, separated by the `purpose` label, so one query answers "where did the
/// provider time go" for the whole turn.
///
/// The response type is named rather than generic, because this reads the
/// token counts off it and puts them on the span. Every call this wraps is a
/// completion, so there is nothing else for it to carry.
pub(crate) async fn measured_aux_call<F>(
    purpose: LlmPurpose,
    call: F,
) -> Result<crate::ports::llm::LlmResponse, crate::CoreError>
where
    F: std::future::Future<Output = Result<crate::ports::llm::LlmResponse, crate::CoreError>>,
{
    use tracing::Instrument;
    // A handle of this function's own, so the span is still open when the call
    // returns and the counts it reported can be recorded onto it. The
    // instrumented future holds the only other handle and drops it as the await
    // ends.
    let span = aux_llm_span(purpose);
    let started = std::time::Instant::now();
    let outcome = call.instrument(span.clone()).await;
    if let Ok(response) = &outcome
        && let Some(usage) = &response.usage
    {
        record_genai_tokens_on_span(&span, usage);
    }
    // Closed here, by name. Left alone this handle would live to the end of the
    // function, and an exported trace would draw the provider call across work
    // that happened after it - the same trap the round's call site names.
    drop(span);
    let route = crate::ports::turn_telemetry::current_turn_route();
    let [provider, model] = route_labels(&route);
    // These helpers absorb their own failures and answer with a fallback, so
    // there is no outcome to report beyond "a call happened and took this
    // long". A failure shows as a duration in the bucket the timeout lands in,
    // and on the WARN line the helper itself writes.
    metrics::record_duration(
        LLM_CALL_DURATION,
        started.elapsed(),
        &[provider, model, Label::new("purpose", purpose.as_label())],
    );
    outcome
}

/// One provider call, hung from its round.
///
/// The parent is explicit because the round span is never *entered*: entering
/// it would mean holding a span guard across an await, which attributes every
/// other task polled on that thread to this round. Naming the parent gives the
/// same tree with none of that risk.
///
/// The four token attributes are declared empty and filled in by
/// [`record_genai_tokens_on_span`] when the response arrives, because a span
/// fixes its field set when it opens and a count is known only after the call
/// returns. A `record` against a field the span never declared is dropped
/// silently and the span exports without it, so the names here and the names in
/// [`GEN_AI_COUNTS`] have to agree; the tests in `tests/turn_telemetry.rs` are
/// what holds them together, because nothing else would report the drift.
pub(crate) fn llm_span(parent: &tracing::Span, round: usize, route: &TurnRoute) -> tracing::Span {
    tracing::info_span!(
        parent: parent,
        "llm.call",
        purpose = LlmPurpose::Turn.as_label(),
        round = round,
        provider = route.provider(),
        model = route.model(),
        conversation_id = %Safe::name(crate::ports::turn_telemetry::current_conversation_id()),
        outcome = tracing::field::Empty,
        provider_request_id = tracing::field::Empty,
        gen_ai.usage.input_tokens = tracing::field::Empty,
        gen_ai.usage.output_tokens = tracing::field::Empty,
        gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
        gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
    )
}

/// One tool dispatch, hung from its round.
///
/// `tool` is chosen by the model, so it is rendered through [`Safe`] - the one
/// way a caller-influenced value reaches a log field. Nothing else bounds it:
/// a newline in a name produces what reads as a second genuine log line, an
/// ANSI escape survives with colour off, and a bidi control shows a name as
/// something it is not.
pub(crate) fn tool_span(parent: &tracing::Span, tool: &str, runner: ToolRunner) -> tracing::Span {
    tracing::info_span!(
        parent: parent,
        "tool.call",
        tool = %Safe::name(tool),
        runner = runner.as_label(),
        conversation_id = %Safe::name(crate::ports::turn_telemetry::current_conversation_id()),
        outcome = tracing::field::Empty,
    )
}

/// The turn's one pre-prompt recall lookup, which is where the embedding
/// round-trip happens.
pub(crate) fn recall_span(conversation_id: &str) -> tracing::Span {
    tracing::info_span!("recall.lookup", conversation_id = %Safe::name(conversation_id))
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
            Label::new("purpose", LlmPurpose::Turn.as_label()),
            Label::new("outcome", if ok { "ok" } else { "error" }),
        ],
    );
}

/// Record a finished tool dispatch.
///
/// `known` is whether the name belongs to a set the daemon controls: the turn
/// offered it, or the executor knows it. It is the caller's answer rather than
/// something read here, because only the turn holds both facts. A name that is
/// neither is recorded as [`UNKNOWN_TOOL`]: see that constant for what it
/// prevents.
pub(crate) fn record_tool_call(elapsed: Duration, tool: &str, known: bool, outcome: ToolOutcome) {
    let tool = if known {
        Safe::name(tool).to_string()
    } else {
        UNKNOWN_TOOL.to_string()
    };
    metrics::record_duration(
        TOOL_CALL_DURATION,
        elapsed,
        &[
            Label::new("tool", tool),
            Label::new("outcome", outcome.as_label()),
        ],
    );
}

/// The most tool names one round's span attribute lists by name.
const MAX_TOOLS_ON_SPAN: usize = 16;

/// One turn, reported when it ends by any path.
///
/// The turn body has several exits - an answer, a cancellation, a user-visible
/// error, an exhausted round budget - and one completion line has to cover all
/// of them. So the body fills this in as it runs and the reporting happens on
/// drop, for the same reason [`RoundGuard`] does: an exit that has to remember
/// to write its own line is an exit that will not, and a turn that ends by
/// panicking is one worth having a line for.
///
/// The default outcome is [`TurnOutcome::Failed`], so an exit nobody
/// classified reads as a problem rather than as a success.
pub(crate) struct TurnGuard {
    span: tracing::Span,
    started: std::time::Instant,
    /// How many rounds of the tool loop the turn ran.
    pub(crate) rounds: usize,
    /// How the turn ended.
    pub(crate) outcome: TurnOutcome,
    /// The turn's tokens, summed from its rounds rather than counted
    /// separately, so the two can never disagree.
    pub(crate) tokens: TokenTotals,
    /// What filled the input the turn opened with (#1203). `None` until the
    /// turn assembles a prompt, which a turn cancelled before its first round
    /// never does - and an unrecorded part is exactly what that is.
    prompt: Option<PromptBreakdown>,
}

impl TurnGuard {
    /// Open a turn against the span it runs inside.
    pub(crate) fn new(span: tracing::Span) -> Self {
        Self {
            span,
            started: std::time::Instant::now(),
            rounds: 0,
            outcome: TurnOutcome::Failed,
            tokens: TokenTotals::default(),
            prompt: None,
        }
    }

    /// Note what filled the input, the first time the turn assembles a prompt.
    ///
    /// Later calls are ignored, and that is the measurement's definition
    /// rather than defensiveness: a turn assembles a prompt per round, each
    /// one carrying the tool traffic the rounds before it produced, so a
    /// last-writer-wins field would report the tail of a tool loop under a
    /// name that reads as the turn's own. What this answers is what the turn
    /// cost before it did anything - the standing bill for the system prompt,
    /// the pinned notes, the recall offer and the tool fleet. What the rounds
    /// then add to it is a separate measurement.
    pub(crate) fn set_prompt_breakdown(&mut self, breakdown: PromptBreakdown) {
        self.prompt.get_or_insert(breakdown);
    }
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        record_turn(elapsed, self.rounds, self.outcome);
        if let Some(breakdown) = &self.prompt {
            prompt::record_on_span(&self.span, breakdown);
            prompt::record_metrics(breakdown);
        }
        self.span.record("rounds", self.rounds);
        self.span.record("outcome", self.outcome.as_label());
        self.span.record("duration_ms", elapsed.as_millis() as u64);
        let tokens = self.tokens;
        let outcome = self.outcome;
        let rounds = self.rounds;
        // The one line an operator greps. Emitted inside the turn span so it
        // carries the ids too.
        self.span.in_scope(|| {
            tracing::info!(
                duration_ms = elapsed.as_millis() as u64,
                model = crate::ports::turn_telemetry::current_turn_route().model(),
                rounds = rounds,
                input_tokens = %Count(tokens.input),
                output_tokens = %Count(tokens.output),
                cache_write_tokens = %Count(tokens.cache_write),
                cache_read_tokens = %Count(tokens.cache_read),
                outcome = outcome.as_label(),
                "turn finished"
            );
        });
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

    /// Note which tools this round called. Names only, each rendered through
    /// [`Safe`] because the model chooses them.
    ///
    /// The list is capped as well as each name, because nothing bounds how
    /// many calls a provider returns in one response and the whole attribute
    /// is exported when the span closes. Past the cap the count is kept, which
    /// is the part worth reading anyway.
    pub(crate) fn set_tools(&mut self, names: impl Iterator<Item = String>) {
        let names: Vec<String> = names.map(|n| Safe::name(n).to_string()).collect();
        let rendered = if names.len() <= MAX_TOOLS_ON_SPAN {
            names.join(",")
        } else {
            format!(
                "{}, and {} more",
                names[..MAX_TOOLS_ON_SPAN].join(","),
                names.len() - MAX_TOOLS_ON_SPAN
            )
        };
        self.span.record("tools", rendered);
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

#[cfg(test)]
mod tests {
    use super::*;

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
            RoundOutcome::Retried,
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
}
