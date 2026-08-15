//! What filled a turn's input, part by part.
//!
//! ## The question this answers
//!
//! `llm.tokens.input` says a round cost 40k. It cannot say whether that was
//! the transcript, the pinned notes or eighty tool schemas, and each of those
//! has a different fix: compact the transcript, prune the notes, drop a
//! server's tools, narrow the recall. So the breakdown is the number an
//! operator acts on, and the parts have to be separable in the way the fix is,
//! not merely add up.
//!
//! One real turn measured while this was being built had already spent 34,324
//! tokens on a 254-character prompt, about 23.7k of it tool schemas for 99
//! tools, before the turn did anything at all. Nothing showed that without
//! reading a log by hand.
//!
//! ## The unit is a claim, so it is in every name
//!
//! Every figure here is **estimated tokens**, counted with the same estimator
//! the context budget uses - the turn passes one closure to assembly and it
//! serves both the budget check and this breakdown, so the two can never
//! disagree about what a block costs. Each field name ends in `_tokens`, and
//! the one figure that is not a token count is [`TOOL_COUNT_FIELD`], named as
//! the count it is. A character count and a token count for the same block
//! look equally plausible side by side, which is why neither is left to the
//! reader to infer.
//!
//! **They are estimates, and they do not sum to what the provider bills.** The
//! provider tokenises its own way and its reported input count stays the
//! authority; these are a breakdown of where that number went, accurate to the
//! estimator's own precision.
//!
//! ## Zero is a measurement here, unlike a provider's count
//!
//! A turn with nothing pinned records `0` for the pinned part rather than
//! leaving the field off. That is the opposite of the convention the
//! provider-reported counts follow, and deliberately: a provider can decline to
//! say, so an absent count there is unknowable and a `0` would invent a
//! measurement. The assembler always knows whether it emitted a block, so an
//! absent field here could only mean the part went unmeasured - which is the
//! one thing a reader must be able to tell apart from an empty block.
//!
//! ## Counters, not histograms
//!
//! The metrics facade offers one histogram and it is a *duration*
//! histogram: fixed millisecond buckets, a millisecond sum, and an OTLP export
//! that names its values `ms`. Token counts put through it would be labelled
//! as milliseconds everywhere they surfaced, which is exactly the units claim
//! this module exists to keep. So the per-part figures accumulate as counters,
//! the way `llm.tokens.input` already does, and [`PROMPT_MEASURED`] is the
//! denominator that turns them back into a per-turn mean.
//!
//! ## Label bounding
//!
//! One metric name carries one label, `part`, whose value comes from
//! [`PromptPart::as_label`] and is therefore a `&'static str` from a closed
//! set of ten. No conversation id, user id, model or provider reaches any
//! metric here: the registry caps a metric at 64 label sets with no eviction,
//! so an unbounded label is an unbounded leak in a process that runs for
//! weeks. Per-conversation stays a trace question, answered by the span fields
//! below.

use adelie_telemetry::metrics::{self, Label};

// ---------------------------------------------------------------------------
// Metric names.
// ---------------------------------------------------------------------------

/// Estimated prompt tokens contributed by one part of an assembled prompt, by
/// part name.
///
/// A counter rather than a histogram for the reason the module header gives.
/// Read against [`PROMPT_MEASURED`] for a per-turn mean, or against each other
/// for the fraction of the input one part is spending.
pub(crate) const PROMPT_PART_TOKENS: &str = "llm.prompt.part.tokens";

/// Tool schemas advertised to the model, summed over measured prompts.
pub(crate) const PROMPT_TOOLS: &str = "llm.prompt.tools";

/// Prompts whose breakdown was recorded: the denominator for the two counters
/// above.
pub(crate) const PROMPT_MEASURED: &str = "llm.prompt.measured";

/// Estimated tokens spent on the tool schemas one connection put in a round's
/// block, by that connection (#1212).
///
/// The `part` axis above reports one aggregate, and an operator reading 23.7k
/// on tools cannot tell which server to drop - which is the remedy the
/// measurement exists to support. The `server` label is the connection's own
/// label: the daemon's built-ins, the client's, and one value per configured
/// MCP server. Bounded by the operator's configuration, which is what makes it
/// safe where a conversation id would not be.
pub(crate) const PROMPT_TOOL_SERVER_TOKENS: &str = "llm.prompt.tool.tokens";

/// Tool schemas advertised, summed over rounds rather than over turns.
///
/// [`PROMPT_TOOLS`] counts a turn's opening block once. Within a turn the set
/// only grows, so that figure is the floor of exactly the growth this counter
/// exists to show. Read against [`PROMPT_ROUND_MEASURED`].
pub(crate) const PROMPT_ROUND_TOOLS: &str = "llm.prompt.round.tools";

/// Rounds whose tool block was measured: the denominator for the two counters
/// above.
pub(crate) const PROMPT_ROUND_MEASURED: &str = "llm.prompt.round.measured";

// ---------------------------------------------------------------------------
// Span field names. Each is a literal in the `turn` span's declaration too,
// because a span fixes its field set when it opens and a `record` against a
// field the span never declared is dropped silently. `PromptPart::ALL` and
// that declaration have to agree; `tests/turn_telemetry.rs` is what holds
// them together, because nothing else would report the drift.
// ---------------------------------------------------------------------------

/// What every part adds up to.
pub(crate) const TOTAL_FIELD: &str = "prompt.total_tokens";

/// How many tools this prompt advertised. A count, not a token figure.
pub(crate) const TOOL_COUNT_FIELD: &str = "prompt.tool_count";

/// The tool-schema cost of the largest tool block any round of the turn sent.
///
/// The field above is the turn's opening figure. Within a turn the advertised
/// set only ever grows, so the opening figure is the floor and this is the
/// ceiling; a turn whose tool loop doubled its own tool block shows it here and
/// nowhere else.
pub(crate) const TOOL_TOKENS_PEAK_FIELD: &str = "prompt.tool_schema_tokens_max";

/// How many tools that largest block carried.
pub(crate) const TOOL_COUNT_PEAK_FIELD: &str = "prompt.tool_count_max";

/// Stored bytes of the turn's tool results, by what eviction did with them
/// (#1205).
///
/// Four label values and no more: `carried`, `evicted`, `reduced`,
/// `shrunk_elsewhere`. They sum to the stored total.
///
/// The three savings are separate because they are separate things: an evicted
/// result left the turn's view, a reduced one lost only its envelope, and the
/// third is work by a mechanism this module does not own. Reading them as one
/// number would say a turn had freed context it is still carrying, or credit a
/// bucket that did nothing.
pub(crate) const CONTEXT_TOOL_BYTES: &str = "llm.context.tool.bytes";

/// Turns whose tool bytes were censused: the denominator for the counter
/// above.
pub(crate) const CONTEXT_MEASURED: &str = "llm.context.measured";

/// Stored bytes of every tool result the turn held.
pub(crate) const TOOL_BYTES_FIELD: &str = "context.tool_bytes";

/// Of those, the bytes behind results the turn reads as a pointer.
pub(crate) const TOOL_BYTES_EVICTED_FIELD: &str = "context.tool_bytes_evicted";

/// Of those, the bytes behind results the turn reads without their envelope.
pub(crate) const TOOL_BYTES_REDUCED_FIELD: &str = "context.tool_bytes_reduced";

/// What the round actually reads for those same results.
pub(crate) const TOOL_BYTES_CARRIED_FIELD: &str = "context.tool_bytes_carried";

/// Bytes some other mechanism shrunk away - the oversized-head notice, or
/// overflow recovery. Reported rather than folded into either bucket above.
pub(crate) const TOOL_BYTES_ELSEWHERE_FIELD: &str = "context.tool_bytes_shrunk_elsewhere";

/// What fraction of the tool bytes the prompt carries is still there, as whole
/// percent.
///
/// The figure the epic is measured against, and the one that says which model
/// leaks most - the turn span carries `model`, so per-model needs no label
/// here. A percent rather than a ratio because the unit is a claim and a bare
/// `0.34` beside a byte count reads as either.
///
/// Deliberately NOT named `..._bytes_..._pct`: a name carrying `_bytes` passes
/// a unit check by the substring alone, whatever it actually reports.
pub(crate) const TOOL_CARRIED_PCT_FIELD: &str = "context.tool_carried_pct";

// ---------------------------------------------------------------------------
// The parts.
// ---------------------------------------------------------------------------

/// One part of an assembled prompt, as an operator would act on it.
///
/// The set is closed and each variant renders to a `&'static str`, which is
/// what makes an unbounded value impossible to pass into the `part` label: it
/// has the wrong lifetime.
///
/// Every block the assembler emits belongs to exactly one of these, so the
/// parts sum to the whole prompt and nothing hides in an unmeasured
/// remainder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptPart {
    /// The cached system instruction - standing guidance, the personality
    /// blurb, the client-context block, the machine topology, the tool-listing
    /// note and any one-turn refinement - plus the ambient `[Now]` line.
    ///
    /// `[Now]` is a separate per-turn message so that a volatile timestamp
    /// never busts the prompt-prefix cache, but it is one line and it is part
    /// of the standing frame an operator reads, so it is reported here rather
    /// than as a part of its own.
    System,
    /// The `[Summary of earlier conversation]` block: what compaction left
    /// behind of the history the window dropped.
    Summary,
    /// The `[Earlier turns]` index: one line per turn before this one, so a
    /// turn the window dropped is distinguishable from a turn that never
    /// happened (#1206).
    TurnIndex,
    /// The `[Current task]` anchor, re-surfaced when the goal has drifted out
    /// of view.
    CurrentTask,
    /// The `[Working state]` line: a count of notes and open to-dos.
    WorkingState,
    /// The `[Plan]` block: the open todo tree.
    Plan,
    /// The `[Pinned]` block: notes the model pinned, and the live content of
    /// any knowledge entry they attach.
    Pinned,
    /// The `[Scratchpad]` index: the free-form note keys.
    Scratchpad,
    /// The `[Recall]` block: candidate memory for the user's prompt.
    Recall,
    /// The conversation transcript, as this prompt carries it - the window,
    /// with the round's projected content and any collapsed-run markers.
    Transcript,
    /// The tool schemas advertised alongside the messages. Sent out of band,
    /// in the request's `tools` array, so no message body shows what they
    /// cost.
    ToolSchemas,
}

impl PromptPart {
    /// Every part, in the order a prompt renders them.
    pub(crate) const ALL: [PromptPart; 11] = [
        Self::System,
        Self::Summary,
        Self::TurnIndex,
        Self::CurrentTask,
        Self::WorkingState,
        Self::Plan,
        Self::Pinned,
        Self::Scratchpad,
        Self::Recall,
        Self::Transcript,
        Self::ToolSchemas,
    ];

    /// The `part` label value. Bounded by its type.
    pub(crate) fn as_label(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Summary => "summary",
            Self::TurnIndex => "turn_index",
            Self::CurrentTask => "current_task",
            Self::WorkingState => "working_state",
            Self::Plan => "plan",
            Self::Pinned => "pinned",
            Self::Scratchpad => "scratchpad",
            Self::Recall => "recall",
            Self::Transcript => "transcript",
            Self::ToolSchemas => "tool_schemas",
        }
    }

    /// The turn-span field this part is recorded under. Always ends in
    /// `_tokens`, because the unit is a claim and a reader must not have to
    /// infer it.
    pub(crate) fn as_span_field(self) -> &'static str {
        match self {
            Self::System => "prompt.system_tokens",
            Self::Summary => "prompt.summary_tokens",
            Self::TurnIndex => "prompt.turn_index_tokens",
            Self::CurrentTask => "prompt.current_task_tokens",
            Self::WorkingState => "prompt.working_state_tokens",
            Self::Plan => "prompt.plan_tokens",
            Self::Pinned => "prompt.pinned_tokens",
            Self::Scratchpad => "prompt.scratchpad_tokens",
            Self::Recall => "prompt.recall_tokens",
            Self::Transcript => "prompt.transcript_tokens",
            Self::ToolSchemas => "prompt.tool_schema_tokens",
        }
    }

    /// Where this part sits in [`PromptBreakdown`]'s array.
    const fn index(self) -> usize {
        match self {
            Self::System => 0,
            Self::Summary => 1,
            Self::TurnIndex => 2,
            Self::CurrentTask => 3,
            Self::WorkingState => 4,
            Self::Plan => 5,
            Self::Pinned => 6,
            Self::Scratchpad => 7,
            Self::Recall => 8,
            Self::Transcript => 9,
            Self::ToolSchemas => 10,
        }
    }
}

/// What each part of one assembled prompt cost, in estimated tokens.
///
/// Built by the assembler as it lays the prompt out, so a block's cost is
/// attributed where it is produced rather than recovered afterwards by reading
/// the prompt back. Recovering it would mean matching on the `[..]` tag the
/// block happens to open with, which nothing holds stable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PromptBreakdown {
    tokens: [u64; PromptPart::ALL.len()],
    tool_count: usize,
}

impl PromptBreakdown {
    /// Add what one block of `part` cost.
    pub(crate) fn add(&mut self, part: PromptPart, tokens: u64) {
        let slot = &mut self.tokens[part.index()];
        *slot = slot.saturating_add(tokens);
    }

    /// Record the tool schemas: how many tools were advertised, and what their
    /// schemas cost.
    ///
    /// One call for both, because the pair is the whole point - a schema bill
    /// without a tool count says nothing about whether to drop a server.
    pub(crate) fn set_tools(&mut self, count: usize, schema_tokens: u64) {
        self.tool_count = count;
        self.tokens[PromptPart::ToolSchemas.index()] = schema_tokens;
    }

    /// What `part` cost. Zero for a block that did not render, which is a
    /// measurement and not an absence.
    pub(crate) fn tokens(&self, part: PromptPart) -> u64 {
        self.tokens[part.index()]
    }

    /// How many tools this prompt advertised.
    pub(crate) fn tool_count(&self) -> usize {
        self.tool_count
    }

    /// What those schemas cost, in estimated tokens.
    pub(crate) fn tool_schema_tokens(&self) -> u64 {
        self.tokens(PromptPart::ToolSchemas)
    }

    /// Every part summed: the whole prompt, plus its out-of-band schemas.
    pub(crate) fn total_tokens(&self) -> u64 {
        self.tokens
            .iter()
            .fold(0u64, |sum, part| sum.saturating_add(*part))
    }
}

/// The largest tool block any round of one turn sent.
///
/// The pair travels together, from the round whose schemas cost the most: a
/// schema bill without its tool count says nothing about whether to drop a
/// server, and two independent maxima would report a count and a cost that no
/// single round ever sent together.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ToolBlockPeak {
    count: usize,
    tokens: u64,
}

impl ToolBlockPeak {
    /// Take in what one round's block cost.
    ///
    /// The costliest round wins, and its count travels with its cost. Keeping
    /// the last round's figure instead would under-report every turn whose
    /// bound retired an activation, and keeping two independent maxima would
    /// report a pair no round ever sent.
    pub(crate) fn observe(&mut self, count: usize, tokens: u64) {
        if tokens > self.tokens {
            self.count = count;
            self.tokens = tokens;
        }
    }

    /// How many tools the costliest round advertised.
    pub(crate) fn count(self) -> usize {
        self.count
    }

    /// What that round's schemas cost.
    pub(crate) fn tokens(self) -> u64 {
        self.tokens
    }
}

/// Put one prompt's breakdown on the turn span.
///
/// Every part is recorded, including the zeros: see the module header for why
/// this differs from how a provider's own counts are handled.
pub(crate) fn record_on_span(span: &tracing::Span, breakdown: &PromptBreakdown) {
    for part in PromptPart::ALL {
        span.record(part.as_span_field(), breakdown.tokens(part));
    }
    span.record(TOTAL_FIELD, breakdown.total_tokens());
    span.record(TOOL_COUNT_FIELD, breakdown.tool_count() as u64);
}

/// Put the turn's largest tool block on its span, beside the opening figure it
/// is the ceiling of.
pub(crate) fn record_peak_on_span(span: &tracing::Span, peak: ToolBlockPeak) {
    span.record(TOOL_COUNT_PEAK_FIELD, peak.count() as u64);
    span.record(TOOL_TOKENS_PEAK_FIELD, peak.tokens());
}

/// Accumulate one round's tool block into the metrics facade: what it cost per
/// connection, how many tools it carried, and that a round was measured.
pub(crate) fn record_round_tool_cost(count: usize, by_server: &[(String, u64)]) {
    for (server, tokens) in by_server {
        metrics::add(
            PROMPT_TOOL_SERVER_TOKENS,
            *tokens,
            &[Label::new("server", server.clone())],
        );
    }
    metrics::add(PROMPT_ROUND_TOOLS, count as u64, &[]);
    metrics::increment(PROMPT_ROUND_MEASURED, &[]);
}

/// Put the turn's tool-byte census on its span, and accumulate it (#1205).
///
/// Without this there is no way to see whether the sweep is doing its job, or
/// which model leaks most - which was the argument for taking eviction off step
/// discipline in the first place.
pub(crate) fn record_tool_bytes(span: &tracing::Span, census: &crate::planning::ToolByteCensus) {
    span.record(TOOL_BYTES_FIELD, census.total as u64);
    span.record(TOOL_BYTES_EVICTED_FIELD, census.evicted as u64);
    span.record(TOOL_BYTES_REDUCED_FIELD, census.reduced as u64);
    span.record(TOOL_BYTES_ELSEWHERE_FIELD, census.shrunk_elsewhere() as u64);
    span.record(TOOL_BYTES_CARRIED_FIELD, census.carried as u64);
    span.record(TOOL_CARRIED_PCT_FIELD, census.carried_percent());

    for (state, bytes) in [
        ("carried", census.carried),
        ("evicted", census.evicted),
        ("reduced", census.reduced),
        ("shrunk_elsewhere", census.shrunk_elsewhere()),
    ] {
        metrics::add(
            CONTEXT_TOOL_BYTES,
            bytes as u64,
            &[Label::new("state", state)],
        );
    }
    metrics::increment(CONTEXT_MEASURED, &[]);
}

/// Accumulate one prompt's breakdown into the metrics facade.
pub(crate) fn record_metrics(breakdown: &PromptBreakdown) {
    for part in PromptPart::ALL {
        metrics::add(
            PROMPT_PART_TOKENS,
            breakdown.tokens(part),
            &[Label::new("part", part.as_label())],
        );
    }
    metrics::add(PROMPT_TOOLS, breakdown.tool_count() as u64, &[]);
    metrics::increment(PROMPT_MEASURED, &[]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_part_has_its_own_slot_label_and_field() {
        // One mis-copied index would make two parts share a slot, and the
        // breakdown would report one of them as the other with nothing else
        // amiss - the total would still be right.
        let indices: Vec<usize> = PromptPart::ALL.iter().map(|p| p.index()).collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            PromptPart::ALL.len(),
            "two parts share a slot, so one is recorded as the other: {indices:?}"
        );
        assert_eq!(
            sorted.last(),
            Some(&(PromptPart::ALL.len() - 1)),
            "the slots must fill the array exactly: {sorted:?}"
        );

        let labels: std::collections::HashSet<&str> =
            PromptPart::ALL.iter().map(|p| p.as_label()).collect();
        assert_eq!(
            labels.len(),
            PromptPart::ALL.len(),
            "two parts rendering to one label would merge two series"
        );
        let fields: std::collections::HashSet<&str> =
            PromptPart::ALL.iter().map(|p| p.as_span_field()).collect();
        assert_eq!(
            fields.len(),
            PromptPart::ALL.len(),
            "two parts sharing a span field would overwrite each other"
        );
    }

    #[test]
    fn every_span_field_states_tokens_as_its_unit() {
        for part in PromptPart::ALL {
            let field = part.as_span_field();
            assert!(
                field.ends_with("_tokens"),
                "`{field}` leaves its unit to be inferred, and a character \
                 count reads exactly like a token count"
            );
        }
        assert!(TOTAL_FIELD.ends_with("_tokens"));
        assert!(
            !TOOL_COUNT_FIELD.ends_with("_tokens"),
            "the tool count is not a token figure and must not be named as one"
        );
        // The per-turn peaks are the same two units, and the same claim.
        assert!(
            TOOL_TOKENS_PEAK_FIELD.contains("_tokens"),
            "`{TOOL_TOKENS_PEAK_FIELD}` leaves its unit to be inferred"
        );
        assert!(
            !TOOL_COUNT_PEAK_FIELD.contains("_tokens"),
            "`{TOOL_COUNT_PEAK_FIELD}` is a count and must not read as tokens"
        );
    }

    #[test]
    fn the_peak_is_the_largest_block_a_round_sent_not_the_last_one() {
        // A turn's advertised set grows as its tool loop activates tools, and
        // the last round is not the largest when the bound retires one. The
        // figure an operator reads has to be the worst the turn actually sent.
        let mut peak = ToolBlockPeak::default();
        peak.observe(10, 1_000);
        peak.observe(4, 400);
        assert_eq!(peak.tokens(), 1_000);
        assert_eq!(peak.count(), 10);
    }

    #[test]
    fn the_peak_pair_comes_from_one_round_rather_than_two_separate_maxima() {
        // Reporting the largest count beside the largest cost would describe a
        // round that never happened, and the pair is the whole point: a bill
        // without its count names no server to drop.
        let mut peak = ToolBlockPeak::default();
        peak.observe(2, 900);
        peak.observe(40, 100);
        assert_eq!(
            (peak.count(), peak.tokens()),
            (2, 900),
            "the costliest round's own pair, not the largest of each"
        );
    }

    #[test]
    fn a_part_that_did_not_render_reports_zero_rather_than_nothing() {
        let mut breakdown = PromptBreakdown::default();
        breakdown.add(PromptPart::System, 120);
        assert_eq!(breakdown.tokens(PromptPart::Pinned), 0);
        assert_eq!(breakdown.total_tokens(), 120);
    }

    #[test]
    fn the_total_is_the_sum_of_the_parts() {
        let mut breakdown = PromptBreakdown::default();
        for (i, part) in PromptPart::ALL.iter().enumerate() {
            breakdown.add(*part, i as u64 + 1);
        }
        // `set_tools` assigns rather than adds, so the tool part's `add` above
        // is replaced and not doubled.
        breakdown.set_tools(3, 40);
        // Every part but the last carries its own `i + 1`; the last is
        // `ToolSchemas`, whose figure `set_tools` replaced.
        let parts = PromptPart::ALL.len() as u64;
        let expected: u64 = (1..parts).sum::<u64>() + 40;
        assert_eq!(breakdown.total_tokens(), expected);
        assert_eq!(breakdown.tool_count(), 3);
        assert_eq!(breakdown.tokens(PromptPart::ToolSchemas), 40);
    }
}
