//! What a provider said one call cost, and where those counts are recorded.
//!
//! Four numbers arrive with a completion - prompt tokens, completion tokens,
//! and the two prompt-cache counts - and each lands in three places: a metric
//! for the trend, a `gen_ai.usage.*` attribute on the call's own span for the
//! incident, and a round's own span fields for the log line an operator greps.
//!
//! ## `None` is not zero
//!
//! A provider may report nothing, or report only some of the four. Recording
//! `0` for a count nobody gave would sum into a total that reads as a real
//! measurement, with no way afterwards to tell it from one. So an absent count
//! is skipped everywhere here: [`TOKENS_UNREPORTED`] counts it on the metrics
//! side, the span attribute is left unrecorded, and [`Count`] renders it as
//! `-`.
//!
//! This is the opposite of the rule [`super::prompt`] follows, and the
//! difference is who knows the answer: a provider can decline to say, while
//! the assembler always knows whether it emitted a block.

use std::fmt;

use adelie_telemetry::metrics::{self, Label};

use crate::ports::llm::TokenUsage;
use crate::ports::turn_telemetry::TurnRoute;

use super::route_labels;

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
