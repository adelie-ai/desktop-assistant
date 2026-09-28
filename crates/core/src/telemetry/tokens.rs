//! What a provider said one call cost, and where those counts are recorded.
//!
//! Four numbers arrive with a completion - prompt tokens, completion tokens,
//! and the two prompt-cache counts - and each lands in four places: the
//! per-request [`TOKEN_USAGE_HISTOGRAM`] for the distribution, a
//! `gen_ai.usage.*` attribute on the call's own span for the incident, a
//! round's own span fields for the log line an operator greps, and - for the
//! two counts this module still totals as plain running sums,
//! [`TOKENS_CACHE_WRITE`] and [`TOKENS_CACHE_READ`] - a counter for the trend.
//! `llm.tokens.input` and `llm.tokens.output` were the same shape until this
//! histogram replaced them: its sum and count already give the same totals
//! and rates, so keeping a separate running sum for those two would be one
//! number computed twice.
//!
//! ## `None` is not zero
//!
//! A provider may report nothing, or report only some of the four. Recording
//! `0` for a count nobody gave would sum into a total that reads as a real
//! measurement, with no way afterwards to tell it from one. So an absent count
//! is skipped everywhere here: no record is added to [`TOKEN_USAGE_HISTOGRAM`], the span
//! attribute is left unrecorded, [`Count`] renders it as `-`, and - for a turn's rounds,
//! which is the population [`TOKENS_UNREPORTED`] has always covered -
//! [`TOKENS_UNREPORTED`] counts it separately on the metrics side.
//!
//! This is the opposite of the rule [`super::prompt`] follows, and the
//! difference is who knows the answer: a provider can decline to say, while
//! the assembler always knows whether it emitted a block.

use std::fmt;

use adelie_telemetry::metrics::{self, Label};

use crate::ports::llm::TokenUsage;
use crate::ports::turn_telemetry::TurnRoute;

use super::{LlmPurpose, route_labels};

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
// Token counts on the provider-call span.
//
// The metrics above answer "how many tokens did this model burn today". They
// cannot answer "what did this turn cost", because that needs a conversation
// id, which is unbounded and would burn the 64-value cap described at the top
// of this module on first contact. A span attribute has no cardinality budget,
// and the provider-call span already carries the conversation id and the round,
// so the counts go there as well.
//
// The four names below are the **OpenTelemetry GenAI semantic convention's**
// own, not this project's, so a backend that special-cases GenAI attributes
// renders a provider call natively instead of showing four fields it has no
// meaning for.
//
// The convention is followed as a written specification rather than through a
// crate: `opentelemetry-semantic-conventions` is not a dependency of this
// workspace, and the GenAI registry has moved out of it into a repository of
// its own. Followed here: the `gen_ai.usage.*` group of the OpenTelemetry
// GenAI semantic conventions, read on 2026-08-08, which the main registry at
// semconv 1.41.0 defers to for every GenAI attribute. That group is at
// Development stability, so the names can still move; each is written once,
// here, so a move is one edit.
//
// The metric names above are deliberately left alone. They are a separate
// signal with separate consumers, and renaming a metric breaks the queries
// already reading it - so the convention is adopted where it is new and free.
// ---------------------------------------------------------------------------

/// Prompt tokens the provider reported for one call.
const GEN_AI_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";

/// Completion tokens the provider reported for one call.
const GEN_AI_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";

/// Input tokens written into the provider's prompt cache.
const GEN_AI_CACHE_CREATION_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation.input_tokens";

/// Input tokens served from the provider's prompt cache.
const GEN_AI_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";

/// One token count on a span: the attribute it is recorded under, and how to
/// read it off a provider's report.
type GenAiCount = (&'static str, fn(&TokenUsage) -> Option<u64>);

/// Each count a provider reports, and the attribute it is recorded under.
///
/// One list, read by the recording below, so a count cannot be read off the
/// provider's report and written under another count's name.
const GEN_AI_COUNTS: [GenAiCount; 4] = [
    (GEN_AI_INPUT_TOKENS, |u| u.input_tokens),
    (GEN_AI_OUTPUT_TOKENS, |u| u.output_tokens),
    (GEN_AI_CACHE_CREATION_INPUT_TOKENS, |u| {
        u.cache_creation_input_tokens
    }),
    (GEN_AI_CACHE_READ_INPUT_TOKENS, |u| {
        u.cache_read_input_tokens
    }),
];

/// Put one provider call's token counts on its `llm.call` span.
///
/// Only the counts the provider actually reported. An absent count leaves its
/// attribute unrecorded rather than recording a zero, because a zero sums into
/// a total that reads as a real measurement and there is no way afterwards to
/// tell it from one. `llm.tokens.unreported` draws the same distinction on the
/// metrics side.
///
/// The caller passes the span rather than this reading the current one: the
/// counts are known only after the call returns, and by then the span is no
/// longer the one the connector ran inside.
pub(crate) fn record_genai_tokens_on_span(span: &tracing::Span, usage: &TokenUsage) {
    for (attribute, read) in GEN_AI_COUNTS {
        if let Some(value) = read(usage) {
            span.record(attribute, value);
        }
    }
}

/// One of the four token counts a provider may report: the label
/// `llm.tokens.unreported` uses for it, the running-sum counter it still
/// accumulates into (`None` for `input` and `output`, which
/// [`TOKEN_USAGE_HISTOGRAM`] covers instead), and how to read it off a
/// provider's report.
type TokenCount = (
    &'static str,
    Option<&'static str>,
    fn(&TokenUsage) -> Option<u64>,
);

/// The four counts. One list, read by [`record_token_usage`], so a count
/// cannot be given a counter here and left off `llm.tokens.unreported`, or
/// the reverse.
const COUNTS: [TokenCount; 4] = [
    ("input", None, |u| u.input_tokens),
    ("output", None, |u| u.output_tokens),
    ("cache_write", Some(TOKENS_CACHE_WRITE), |u| {
        u.cache_creation_input_tokens
    }),
    ("cache_read", Some(TOKENS_CACHE_READ), |u| {
        u.cache_read_input_tokens
    }),
];

/// Record one round's token usage as running-sum counters, and count what the
/// provider left out.
///
/// Only [`TOKENS_CACHE_WRITE`] and [`TOKENS_CACHE_READ`] still accumulate a
/// counter here - `llm.tokens.input` and `llm.tokens.output` were removed
/// once [`TOKEN_USAGE_HISTOGRAM`] replaced them, since its sum and count
/// already give the same totals and rates. `None` is not zero for any of the
/// four regardless: a count the provider did not report is skipped and
/// counted as unreported instead, so no total is silently understated. A
/// response with no usage at all counts every one of the four as unreported,
/// because that is what a connector that reports nothing looks like from
/// here.
///
/// Scoped to a turn's rounds, the way it always was - not the aux calls
/// [`record_token_histogram`] additionally covers. Widening `llm.tokens.unreported`'s
/// population to those as well is a separate decision; a title or a compaction call that
/// reports nothing is simply not counted anywhere today, which is no worse than before
/// this histogram existed.
pub(crate) fn record_token_usage(usage: Option<&TokenUsage>, route: &TurnRoute) {
    let [provider, model] = route_labels(route);
    for (which, counter, read) in COUNTS {
        match usage.and_then(read) {
            Some(value) => {
                if let Some(counter) = counter {
                    metrics::add(counter, value, &[provider.clone(), model.clone()]);
                }
            }
            None => metrics::increment(
                TOKENS_UNREPORTED,
                &[provider.clone(), Label::new("count", which)],
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The per-request token-usage histogram.
//
// Follows the OpenTelemetry GenAI semantic conventions' `gen_ai.client.token.usage`
// histogram, read on 2026-09-28 - the same convention family `GEN_AI_COUNTS` above
// follows for the span attributes, at the same Development stability. The wider GenAI
// metrics conventions have since moved toward a different shape (separate
// `gen_ai.client.inference.operation.input_tokens` / `.output_tokens` histograms with no
// token-type attribute, plus per-modality counters); this module keeps the single
// `gen_ai.client.token.usage` histogram with a `gen_ai.token.type` attribute, which is the
// shape this metric was specified with, and does not follow the newer split.
// ---------------------------------------------------------------------------

/// The per-request token-usage histogram: one record per model call per token type the
/// provider reported.
pub(crate) const TOKEN_USAGE_HISTOGRAM: &str = "gen_ai.client.token.usage";

/// The unit [`TOKEN_USAGE_HISTOGRAM`] reports in, in UCUM notation.
pub(crate) const TOKEN_USAGE_UNIT: &str = "{token}";

/// Bucket boundaries for [`TOKEN_USAGE_HISTOGRAM`], bracketing the range from a trivial
/// call to a large context window. Includes exactly `25_000.0`: the input-size threshold
/// an operator asks "how many calls sent more than this" about.
pub(crate) const TOKEN_USAGE_BUCKETS: &[f64] = &[
    0.0,
    64.0,
    256.0,
    1_024.0,
    2_048.0,
    4_096.0,
    8_192.0,
    16_384.0,
    25_000.0,
    32_768.0,
    50_000.0,
    65_536.0,
    100_000.0,
    131_072.0,
    200_000.0,
    262_144.0,
    524_288.0,
    1_048_576.0,
];

/// The attribute [`TOKEN_USAGE_HISTOGRAM`] records the token kind under.
const GEN_AI_TOKEN_TYPE: &str = "gen_ai.token.type";

/// The attribute [`TOKEN_USAGE_HISTOGRAM`] records the provider under.
const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";

/// The attribute [`TOKEN_USAGE_HISTOGRAM`] records the model under.
const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";

/// A `gen_ai.token.type` attribute value.
///
/// The OpenTelemetry GenAI semantic conventions name only `input` and `output`.
/// `cache_read` and `cache_creation` extend that vocabulary the same way the
/// `gen_ai.usage.cache_*` span attributes already do above - this crate's own read of what
/// a caching provider's four counts are, not a convention it invented independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenType {
    Input,
    Output,
    CacheRead,
    CacheCreation,
}

impl TokenType {
    fn as_label(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::CacheRead => "cache_read",
            Self::CacheCreation => "cache_creation",
        }
    }
}

/// One token type, and how to read it off a provider's report.
type HistogramCount = (TokenType, fn(&TokenUsage) -> Option<u64>);

/// The four counts [`TOKEN_USAGE_HISTOGRAM`] may record, one record per type the provider
/// actually reported.
const HISTOGRAM_COUNTS: [HistogramCount; 4] = [
    (TokenType::Input, |u| u.input_tokens),
    (TokenType::Output, |u| u.output_tokens),
    (TokenType::CacheCreation, |u| u.cache_creation_input_tokens),
    (TokenType::CacheRead, |u| u.cache_read_input_tokens),
];

/// Record one provider call's tokens into [`TOKEN_USAGE_HISTOGRAM`], one record per token
/// type the provider actually reported.
///
/// `None` is not zero here either: a type the provider left out produces no record, so the
/// histogram's sum and count are never inflated by an absence read as a zero. Unlike
/// [`record_token_usage`], a missing type here is not separately counted -
/// `llm.tokens.unreported` stays scoped to a turn's rounds, the population it always had.
///
/// Called for every provider call a turn makes, not only its rounds: `purpose`
/// distinguishes a round from a title, a compaction, a categorization pass and a
/// wind-down, which is what lets a query ask "which purpose is expensive" rather than only
/// "how expensive was this turn".
pub(crate) fn record_token_histogram(
    usage: Option<&TokenUsage>,
    route: &TurnRoute,
    purpose: LlmPurpose,
) {
    let gen_ai_provider = Label::new(GEN_AI_PROVIDER_NAME, route.provider());
    let gen_ai_model = Label::new(GEN_AI_REQUEST_MODEL, route.model());
    let purpose = Label::new("purpose", purpose.as_label());

    for (token_type, read) in HISTOGRAM_COUNTS {
        if let Some(value) = usage.and_then(read) {
            metrics::record_value(
                TOKEN_USAGE_HISTOGRAM,
                value as f64,
                TOKEN_USAGE_UNIT,
                TOKEN_USAGE_BUCKETS,
                &[
                    Label::new(GEN_AI_TOKEN_TYPE, token_type.as_label()),
                    gen_ai_provider.clone(),
                    gen_ai_model.clone(),
                    purpose.clone(),
                ],
            );
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
    fn the_four_counts_are_read_from_one_list() {
        // `record_token_usage` walks `COUNTS` for both its counters and its
        // unreported accounting, so a fifth count cannot be added to one and
        // forgotten in the other.
        let usage = TokenUsage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            cache_creation_input_tokens: Some(3),
            cache_read_input_tokens: Some(4),
        };
        let read: Vec<Option<u64>> = COUNTS.iter().map(|(_, _, read)| read(&usage)).collect();
        assert_eq!(read, vec![Some(1), Some(2), Some(3), Some(4)]);
    }

    fn route(provider: &str, model: &str) -> TurnRoute {
        TurnRoute {
            connection_id: None,
            provider: Some(provider.to_owned()),
            model: Some(model.to_owned()),
        }
    }

    fn label<'a>(labels: &'a [Label], key: &str) -> &'a str {
        labels
            .iter()
            .find(|label| label.key() == key)
            .unwrap_or_else(|| panic!("no {key} label among {labels:?}"))
            .value()
    }

    /// Named for the acceptance criterion in adelie-ai/desktop-assistant#1359: a call
    /// whose provider reports every token type produces one histogram record per type,
    /// each with the right attributes and the right value.
    #[test]
    fn a_call_reporting_every_token_type_produces_one_histogram_record_per_type() {
        let scope = adelie_telemetry::metrics::TestScope::new();
        let usage = TokenUsage {
            input_tokens: Some(30_000),
            output_tokens: Some(512),
            cache_creation_input_tokens: Some(1_024),
            cache_read_input_tokens: Some(2_048),
        };

        record_token_histogram(
            Some(&usage),
            &route("anthropic", "claude-x"),
            LlmPurpose::Compaction,
        );

        let summary = scope.snapshot();
        let histograms: Vec<_> = summary
            .value_histograms
            .iter()
            .filter(|histogram| histogram.name == TOKEN_USAGE_HISTOGRAM)
            .collect();
        assert_eq!(
            histograms.len(),
            4,
            "one record per token type the provider reported"
        );

        for histogram in &histograms {
            assert_eq!(histogram.unit, TOKEN_USAGE_UNIT);
            assert_eq!(histogram.total.count, 1);
            assert_eq!(
                label(&histogram.labels, "gen_ai.provider.name"),
                "anthropic"
            );
            assert_eq!(label(&histogram.labels, "gen_ai.request.model"), "claude-x");
            assert_eq!(label(&histogram.labels, "purpose"), "compaction");
        }

        let value_for = |token_type: &str| {
            histograms
                .iter()
                .find(|histogram| label(&histogram.labels, "gen_ai.token.type") == token_type)
                .unwrap_or_else(|| panic!("no record for gen_ai.token.type={token_type}"))
                .total
                .sum
        };
        assert_eq!(value_for("input"), 30_000.0);
        assert_eq!(value_for("output"), 512.0);
        assert_eq!(value_for("cache_creation"), 1_024.0);
        assert_eq!(value_for("cache_read"), 2_048.0);
    }

    /// Named for the acceptance criterion in adelie-ai/desktop-assistant#1359: a call
    /// whose provider does not report a given type produces no record for that type, and
    /// `llm.tokens.unreported` still counts it.
    ///
    /// `llm.tokens.unreported` is `record_token_usage`'s counter, not the histogram's own -
    /// a round calls both (see `RoundGuard::drop` in `telemetry/mod.rs`), so this test
    /// drives them together the way the real call site does.
    #[test]
    fn a_missing_token_type_produces_no_record_and_counts_as_unreported() {
        let scope = adelie_telemetry::metrics::TestScope::new();
        let usage = TokenUsage {
            input_tokens: Some(100),
            output_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };
        let route = route("openai", "gpt-x");

        record_token_usage(Some(&usage), &route);
        record_token_histogram(Some(&usage), &route, LlmPurpose::Turn);

        let summary = scope.snapshot();
        assert_eq!(
            summary
                .value_histograms
                .iter()
                .filter(|histogram| histogram.name == TOKEN_USAGE_HISTOGRAM)
                .count(),
            1,
            "only the reported type gets a histogram record"
        );

        let unreported: u64 = summary
            .counters
            .iter()
            .filter(|counter| counter.name == TOKENS_UNREPORTED)
            .map(|counter| counter.total)
            .sum();
        assert_eq!(
            unreported, 3,
            "output, cache_creation and cache_read were all left out"
        );
    }

    /// Named for the acceptance criterion in adelie-ai/desktop-assistant#1359: the
    /// histogram's bucket boundaries include exactly 25000, read back off the recorded
    /// histogram rather than compared against the constant it was built from.
    #[test]
    fn the_recorded_histogram_bucket_boundaries_include_exactly_25000() {
        let scope = adelie_telemetry::metrics::TestScope::new();
        let usage = TokenUsage {
            input_tokens: Some(1),
            output_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };

        record_token_histogram(
            Some(&usage),
            &route("anthropic", "claude-x"),
            LlmPurpose::Turn,
        );

        let summary = scope.snapshot();
        let histogram = summary
            .value_histograms
            .iter()
            .find(|histogram| histogram.name == TOKEN_USAGE_HISTOGRAM)
            .expect("the histogram must have recorded");

        assert!(
            histogram.total.bounds().contains(&25_000.0),
            "25000 must be one of the recorded bucket boundaries, found {:?}",
            histogram.total.bounds()
        );
    }

    /// Named for the acceptance criterion in adelie-ai/desktop-assistant#1359: neither of
    /// the removed counters is emitted, whatever this module still does record.
    #[test]
    fn the_old_input_and_output_counters_are_never_emitted() {
        let scope = adelie_telemetry::metrics::TestScope::new();
        let usage = TokenUsage {
            input_tokens: Some(100),
            output_tokens: Some(20),
            cache_creation_input_tokens: Some(5),
            cache_read_input_tokens: Some(3),
        };
        let route = route("anthropic", "claude-x");

        record_token_usage(Some(&usage), &route);
        record_token_histogram(Some(&usage), &route, LlmPurpose::Turn);

        let summary = scope.snapshot();
        for name in ["llm.tokens.input", "llm.tokens.output"] {
            assert!(
                !summary.counters.iter().any(|counter| counter.name == name),
                "{name} must not be emitted; the histogram replaced it"
            );
        }
    }

    /// A call recorded under two different purposes is two distinct series, so a query
    /// grouping by purpose does not merge a round into a title call.
    #[test]
    fn different_purposes_are_distinct_series() {
        let scope = adelie_telemetry::metrics::TestScope::new();
        let usage = TokenUsage {
            input_tokens: Some(10),
            output_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };
        let route = route("anthropic", "claude-x");

        record_token_histogram(Some(&usage), &route, LlmPurpose::Turn);
        record_token_histogram(Some(&usage), &route, LlmPurpose::Title);

        let summary = scope.snapshot();
        let purposes: std::collections::BTreeSet<&str> = summary
            .value_histograms
            .iter()
            .filter(|histogram| histogram.name == TOKEN_USAGE_HISTOGRAM)
            .map(|histogram| label(&histogram.labels, "purpose"))
            .collect();
        assert_eq!(
            purposes,
            std::collections::BTreeSet::from(["turn", "title"])
        );
    }
}
