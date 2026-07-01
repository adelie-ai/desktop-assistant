//! Connector-agnostic classification of opaque backend LLM errors (epic #178).
//!
//! Connectors surface unrecognized provider failures as
//! [`crate::CoreError::Llm`] — a bare string the dispatch loop can't act on.
//! This module normalizes such an error into a small **closed** set of causes
//! ([`NormalizedCause`]) that map back onto the structured `CoreError`
//! variants which already have recovery/handling (overflow → recovery ladder,
//! rate-limit → backoff, …). Routing more errors into the right existing
//! variant is the whole value: no new recovery code, just better detection.
//!
//! This file is **tier 1** of the planned three-tier classifier: deterministic
//! built-in matchers, refactored from per-connector hardcoded knowledge into
//! one connector-agnostic table. Tier 2 (a persisted learned cache) and tier 3
//! (cheap-LLM classification of genuinely novel errors) build on top of this
//! in later slices of #178; both fall back to [`classify_builtin`] first.
//!
//! ## Safety contract
//! - The cause set is **closed**. Anything unrecognized is
//!   [`NormalizedCause::Unknown`], which maps to `None` in
//!   [`cause_to_core_error`] so the caller keeps the original error and
//!   behavior is unchanged on a miss.
//! - Terminal causes ([`NormalizedCause::is_terminal`]) — billing/auth — must
//!   never drive an automatic retry. The classifier only *labels*; enforcing
//!   "never retry terminal" is the caller's job (the future decorator).

use crate::CoreError;

/// A normalized, connector-agnostic cause for a backend LLM error.
///
/// Each variant either maps onto an existing structured [`CoreError`] (see
/// [`cause_to_core_error`]) or is intentionally left unmapped so the original
/// error surfaces unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizedCause {
    /// The prompt exceeded the model's context window. Token counts are
    /// best-effort — present only when the provider stated them.
    ContextOverflow {
        prompt_tokens: Option<u64>,
        max_tokens: Option<u64>,
    },
    /// Provider rate limit / throttling. Retryable with backoff.
    RateLimited,
    /// Billing or quota exhausted. **Terminal** — never auto-retry.
    BillingFatal,
    /// Authentication / authorization failure. **Terminal** — never auto-retry.
    Auth,
    /// The model is loading or warming up. Retryable.
    ModelLoading,
    /// The model/endpoint does not support the requested tool use.
    ToolsUnsupported,
    /// A transient server-side error (5xx / overloaded). Retryable.
    Transient,
    /// Unrecognized — surface the original error unchanged.
    Unknown,
}

impl NormalizedCause {
    /// Whether this cause is terminal: a retry can never succeed and may rack
    /// up cost (billing) or is futile (auth). The caller must not auto-retry.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::BillingFatal | Self::Auth)
    }

    /// Stable string key for persisting a learned classification (tier 2).
    /// Overflow token counts are intentionally dropped — the learned cache
    /// keys on the cause *kind*; tier 1 recovers counts from the live message
    /// when it carries them.
    pub fn as_key(self) -> &'static str {
        match self {
            Self::ContextOverflow { .. } => "context_overflow",
            Self::RateLimited => "rate_limited",
            Self::BillingFatal => "billing_fatal",
            Self::Auth => "auth",
            Self::ModelLoading => "model_loading",
            Self::ToolsUnsupported => "tools_unsupported",
            Self::Transient => "transient",
            Self::Unknown => "unknown",
        }
    }

    /// Inverse of [`Self::as_key`]. `ContextOverflow` rehydrates with no
    /// counts. Returns `None` for an unrecognized key, which the caller
    /// treats as a cache miss (and never as a behavior change).
    pub fn from_key(key: &str) -> Option<Self> {
        Some(match key {
            "context_overflow" => Self::ContextOverflow {
                prompt_tokens: None,
                max_tokens: None,
            },
            "rate_limited" => Self::RateLimited,
            "billing_fatal" => Self::BillingFatal,
            "auth" => Self::Auth,
            "model_loading" => Self::ModelLoading,
            "tools_unsupported" => Self::ToolsUnsupported,
            "transient" => Self::Transient,
            "unknown" => Self::Unknown,
            _ => return None,
        })
    }
}

/// The signals available about a backend error, gathered at the connector
/// boundary where the provider's HTTP status and error code are still visible
/// (a bare `CoreError::Llm` string has already lost them).
#[derive(Debug, Clone, Copy)]
pub struct ErrorContext<'a> {
    /// Connector id, e.g. `"bedrock"`, `"anthropic"`, `"openai"`.
    pub connector: &'a str,
    /// HTTP status, when the transport exposed one.
    pub http_status: Option<u16>,
    /// Provider-specific error code, e.g. `"ValidationException"`.
    pub provider_code: Option<&'a str>,
    /// The raw, human-readable error message.
    pub message: &'a str,
}

/// Map a [`NormalizedCause`] to the structured [`CoreError`] that carries the
/// right recovery/handling. Returns `None` when there is no dedicated
/// variant (or the cause is `Unknown`): the caller keeps the original error,
/// so an unclassified failure behaves exactly as it does today.
pub fn cause_to_core_error(cause: NormalizedCause, detail: String) -> Option<CoreError> {
    match cause {
        NormalizedCause::ContextOverflow {
            prompt_tokens,
            max_tokens,
        } => Some(CoreError::ContextOverflow {
            prompt_tokens,
            max_tokens,
            detail,
        }),
        NormalizedCause::RateLimited => Some(CoreError::RateLimited {
            retry_after: None,
            detail,
        }),
        NormalizedCause::BillingFatal => Some(CoreError::QuotaExceeded { detail }),
        NormalizedCause::ModelLoading => Some(CoreError::ModelLoading { detail }),
        NormalizedCause::ToolsUnsupported => Some(CoreError::ToolsUnsupported { detail }),
        // No dedicated CoreError variant: surface the original error unchanged.
        NormalizedCause::Auth | NormalizedCause::Transient | NormalizedCause::Unknown => None,
    }
}

/// Tier-1 deterministic classification from built-in matchers. Returns
/// [`NormalizedCause::Unknown`] when nothing matches — the signal for higher
/// tiers (learned cache, LLM) to take over, and the safe default that leaves
/// behavior unchanged.
pub fn classify_builtin(ctx: &ErrorContext) -> NormalizedCause {
    let lower = ctx.message.to_ascii_lowercase();

    // Context overflow — connector-agnostic phrasing across providers.
    if lower.contains("prompt is too long")
        || lower.contains("input is too long")
        || lower.contains("maximum context length")
        || (lower.contains("context length") && lower.contains("exceed"))
        || lower.contains("too many tokens")
    {
        let fields = extract_overflow_fields(ctx.message);
        return NormalizedCause::ContextOverflow {
            prompt_tokens: fields.prompt_tokens,
            max_tokens: fields.max_context_tokens,
        };
    }

    // Billing / quota — TERMINAL, and checked *before* rate-limit because some
    // providers return quota exhaustion with HTTP 429 (e.g. OpenAI
    // insufficient_quota). Misreading that as a retryable rate-limit would
    // loop and burn money.
    if lower.contains("quota")
        || lower.contains("billing")
        || lower.contains("insufficient_quota")
        || lower.contains("payment")
        || lower.contains("credit balance")
    {
        return NormalizedCause::BillingFatal;
    }

    // Auth — TERMINAL.
    if ctx.http_status == Some(401)
        || ctx.http_status == Some(403)
        || lower.contains("unauthorized")
        || lower.contains("forbidden")
        || lower.contains("invalid api key")
        || lower.contains("access denied")
        || lower.contains("not authorized")
    {
        return NormalizedCause::Auth;
    }

    // Rate limit / throttling.
    if ctx.http_status == Some(429)
        || lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("throttl")
    {
        return NormalizedCause::RateLimited;
    }

    // Model loading / warming up.
    if lower.contains("loading") || lower.contains("warming up") || lower.contains("not ready") {
        return NormalizedCause::ModelLoading;
    }

    // Tool use unsupported by the model/endpoint.
    if lower.contains("tools are not supported")
        || (lower.contains("tool use")
            && (lower.contains("not support")
                || lower.contains("doesn't support")
                || lower.contains("does not support")))
    {
        return NormalizedCause::ToolsUnsupported;
    }

    // Transient server-side failures.
    if matches!(ctx.http_status, Some(500..=599))
        || lower.contains("internal server error")
        || lower.contains("service unavailable")
        || lower.contains("overloaded")
    {
        return NormalizedCause::Transient;
    }

    NormalizedCause::Unknown
}

/// Default output-token reservation assumed when a `max_context` overflow error
/// doesn't state how many output tokens the request asked for. Providers report
/// a *total* window (input + output); to turn that into an input-token ceiling
/// we must subtract the output reservation, or the very next turn re-overflows
/// by exactly the output headroom (issue #425 was over by its 8192-token
/// reservation). A modest default keeps us safely under when the number is
/// absent; snapping (`context_window::snap_down_to_common`) adds further margin.
pub const DEFAULT_OUTPUT_RESERVE_TOKENS: u64 = 8_192;

/// Structured numbers parsed from a context-overflow error message.
///
/// Every field is best-effort (`None` when the provider didn't state it, or we
/// couldn't anchor it). Provider phrasings diverge wildly and are wrapped in
/// noise — a Bedrock/Mantle error prefixes a random `requestId` UUID
/// (`f2e534ff-…`) and an HTTP status code *before* the real token counts, and
/// the clause order differs (`prompt … N > M maximum` vs `maximum … is M …
/// prompt … N input`). So we never read numbers positionally; each field is
/// anchored on a nearby keyword. Reading positionally is exactly what poisoned
/// the learned window at 534 tokens in issue #425 (`f2e534ff` → `2`, `534`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverflowFields {
    /// The model's stated context ceiling. Usually the *total* window
    /// (input + output); [`derive_input_ceiling`] subtracts output to get an
    /// input budget.
    pub max_context_tokens: Option<u64>,
    /// The prompt/input token count the provider reported for the rejected
    /// request.
    pub prompt_tokens: Option<u64>,
    /// Output tokens the request reserved/requested, when the error states it.
    pub requested_output_tokens: Option<u64>,
}

impl OverflowFields {
    /// Merge two extractions, preferring `self`'s populated fields and filling
    /// its gaps from `other`. Used to backfill a deterministic parse with an
    /// LLM extraction (issue #425) without letting the LLM overwrite a value we
    /// already anchored confidently.
    pub fn or(self, other: OverflowFields) -> OverflowFields {
        OverflowFields {
            max_context_tokens: self.max_context_tokens.or(other.max_context_tokens),
            prompt_tokens: self.prompt_tokens.or(other.prompt_tokens),
            requested_output_tokens: self
                .requested_output_tokens
                .or(other.requested_output_tokens),
        }
    }
}

/// Keyword-anchored extraction of the token numbers from an overflow message.
///
/// Anchoring (rather than positional order) makes this robust to the requestId
/// UUID / status-code noise and to differing clause order across providers.
pub fn extract_overflow_fields(message: &str) -> OverflowFields {
    // Digits are case-independent; lowercase once so keyword anchors match
    // regardless of provider capitalization. ASCII-lowercasing preserves byte
    // length, so offsets are stable.
    let s = message.to_ascii_lowercase();
    let max_context_tokens = number_after(&s, "maximum context length")
        .or_else(|| number_after(&s, "context length is"))
        .or_else(|| number_before(&s, "maximum"));
    let prompt_tokens = number_after(&s, "prompt contains")
        .or_else(|| number_after(&s, "prompt is too long"))
        .or_else(|| number_after(&s, "input length"))
        .or_else(|| number_before(&s, "input tokens"));
    let requested_output_tokens =
        number_before(&s, "output tokens").or_else(|| number_after(&s, "requested"));
    OverflowFields {
        max_context_tokens,
        prompt_tokens,
        requested_output_tokens,
    }
}

/// Turn extracted overflow numbers into a safe **input-token** ceiling — the
/// value the learned cap and budget resolution key on.
///
/// Precedence:
///   1. `max_context − output_reserve` when the total window is known (the
///      output reservation is what pushed issue #425 over by one token, so it
///      must come off the top);
///   2. otherwise the rejected `prompt_tokens` as-is — it was too big *with*
///      output headroom, and snapping down provides that headroom;
///   3. `None` when the message carried no usable number (nothing to learn).
///
/// The returned value is deliberately un-snapped; snapping to a common size
/// happens at apply time so the ladder stays tunable without rewriting stored
/// observations.
pub fn derive_input_ceiling(fields: &OverflowFields) -> Option<u64> {
    if let Some(max_context) = fields.max_context_tokens {
        let reserve = fields
            .requested_output_tokens
            .unwrap_or(DEFAULT_OUTPUT_RESERVE_TOKENS)
            // Never reserve so much output that no input fits — cap at half the
            // window. Guards tiny windows (where an 8192 default would exceed
            // the model) and an implausibly large stated output.
            .min(max_context / 2);
        return Some(max_context - reserve);
    }
    fields.prompt_tokens
}

/// First base-10 integer at or after the first occurrence of `keyword`.
fn number_after(haystack: &str, keyword: &str) -> Option<u64> {
    let idx = haystack.find(keyword)?;
    haystack[idx + keyword.len()..]
        .split(|c: char| !c.is_ascii_digit())
        .find(|t| !t.is_empty())
        .and_then(|t| t.parse().ok())
}

/// Last base-10 integer appearing before the first occurrence of `keyword`.
fn number_before(haystack: &str, keyword: &str) -> Option<u64> {
    let idx = haystack.find(keyword)?;
    haystack[..idx]
        .split(|c: char| !c.is_ascii_digit())
        .rfind(|t| !t.is_empty())
        .and_then(|t| t.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(message: &str) -> ErrorContext<'_> {
        ErrorContext {
            connector: "bedrock",
            http_status: None,
            provider_code: None,
            message,
        }
    }

    fn ctx_status(message: &str, status: u16) -> ErrorContext<'_> {
        ErrorContext {
            connector: "openai",
            http_status: Some(status),
            provider_code: None,
            message,
        }
    }

    #[test]
    fn classifies_context_overflow_without_counts() {
        assert_eq!(
            classify_builtin(&ctx("Input is too long for requested model.")),
            NormalizedCause::ContextOverflow {
                prompt_tokens: None,
                max_tokens: None
            }
        );
    }

    #[test]
    fn classifies_context_overflow_with_counts() {
        assert_eq!(
            classify_builtin(&ctx(
                "Input length (479258) exceeds model's maximum context length (131072)."
            )),
            NormalizedCause::ContextOverflow {
                prompt_tokens: Some(479_258),
                max_tokens: Some(131_072)
            }
        );
    }

    /// Issue #425 regression: the real Bedrock/Mantle error that bricked the
    /// budget. A positional parse grabbed `2` and `534` from the requestId
    /// UUID (`f2e534ff`) instead of the real counts. Keyword anchoring must
    /// ignore the UUID and read the true numbers.
    #[test]
    fn extracts_overflow_fields_ignoring_request_id_uuid() {
        let msg = "validation error: The model returned the following errors: Mantle \
                   streaming error for requestId f2e534ff-436e-461b-8d93-906629545d84: \
                   ErrorEvent { error: APIError { type: \"BadRequestError\", code: \
                   Some(400), message: \"This model's maximum context length is 202752 \
                   tokens. However, you requested 8192 output tokens and your prompt \
                   contains at least 194561 input tokens, for a total of at least 202753 \
                   tokens. Please reduce the length of the input prompt or the number of \
                   requested output tokens. (parameter=input_tokens, value=194561)\", \
                   param: None } }";
        let f = extract_overflow_fields(msg);
        assert_eq!(f.max_context_tokens, Some(202_752));
        assert_eq!(f.prompt_tokens, Some(194_561));
        assert_eq!(f.requested_output_tokens, Some(8_192));
        // And the derived INPUT ceiling reserves output off the total window —
        // NOT the poisoned 534.
        assert_eq!(derive_input_ceiling(&f), Some(202_752 - 8_192));

        // The full classifier path must agree (no more 2 / 534).
        assert_eq!(
            classify_builtin(&ctx(msg)),
            NormalizedCause::ContextOverflow {
                prompt_tokens: Some(194_561),
                max_tokens: Some(202_752),
            }
        );
    }

    /// The Anthropic ordering puts the max *after* the prompt
    /// (`prompt is too long: N tokens > M maximum`) — the opposite of the
    /// OpenAI/GLM phrasing. Anchoring reads both correctly.
    #[test]
    fn extracts_overflow_fields_anthropic_ordering() {
        let f = extract_overflow_fields("prompt is too long: 219473 tokens > 200000 maximum");
        assert_eq!(f.prompt_tokens, Some(219_473));
        assert_eq!(f.max_context_tokens, Some(200_000));
        assert_eq!(f.requested_output_tokens, None);
    }

    #[test]
    fn extracts_no_overflow_fields_when_absent() {
        let f = extract_overflow_fields("Input is too long for requested model.");
        assert_eq!(f, OverflowFields::default());
        assert_eq!(derive_input_ceiling(&f), None);
    }

    #[test]
    fn overflow_fields_or_backfills_gaps_only() {
        let anchored = OverflowFields {
            max_context_tokens: Some(202_752),
            prompt_tokens: None,
            requested_output_tokens: None,
        };
        let llm = OverflowFields {
            max_context_tokens: Some(999), // must NOT overwrite the anchored value
            prompt_tokens: Some(194_561),
            requested_output_tokens: Some(8_192),
        };
        let merged = anchored.or(llm);
        assert_eq!(merged.max_context_tokens, Some(202_752));
        assert_eq!(merged.prompt_tokens, Some(194_561));
        assert_eq!(merged.requested_output_tokens, Some(8_192));
    }

    #[test]
    fn classifies_billing_as_terminal_even_with_429() {
        // OpenAI returns insufficient-quota with HTTP 429; it must be billing
        // (terminal), not a retryable rate-limit.
        let c = classify_builtin(&ctx_status(
            "You exceeded your current quota, please check your plan and billing details.",
            429,
        ));
        assert_eq!(c, NormalizedCause::BillingFatal);
        assert!(c.is_terminal());
    }

    #[test]
    fn classifies_auth_from_status_and_message() {
        assert_eq!(
            classify_builtin(&ctx_status("Unauthorized", 401)),
            NormalizedCause::Auth
        );
        assert_eq!(
            classify_builtin(&ctx("Invalid API key provided")),
            NormalizedCause::Auth
        );
        assert!(NormalizedCause::Auth.is_terminal());
    }

    #[test]
    fn classifies_rate_limit_without_quota_words() {
        assert_eq!(
            classify_builtin(&ctx_status("Too Many Requests", 429)),
            NormalizedCause::RateLimited
        );
        assert_eq!(
            classify_builtin(&ctx("ThrottlingException: rate of requests exceeded")),
            NormalizedCause::RateLimited
        );
        assert!(!NormalizedCause::RateLimited.is_terminal());
    }

    #[test]
    fn classifies_model_loading_and_tools_unsupported() {
        assert_eq!(
            classify_builtin(&ctx("The model is currently loading")),
            NormalizedCause::ModelLoading
        );
        assert_eq!(
            classify_builtin(&ctx("This model doesn't support tool use in streaming")),
            NormalizedCause::ToolsUnsupported
        );
    }

    #[test]
    fn classifies_transient_5xx() {
        assert_eq!(
            classify_builtin(&ctx_status("internal server error", 500)),
            NormalizedCause::Transient
        );
    }

    #[test]
    fn unknown_error_stays_unknown() {
        assert_eq!(
            classify_builtin(&ctx("a wild and unfamiliar failure appeared")),
            NormalizedCause::Unknown
        );
    }

    #[test]
    fn cause_key_round_trips() {
        for cause in [
            NormalizedCause::ContextOverflow {
                prompt_tokens: None,
                max_tokens: None,
            },
            NormalizedCause::RateLimited,
            NormalizedCause::BillingFatal,
            NormalizedCause::Auth,
            NormalizedCause::ModelLoading,
            NormalizedCause::ToolsUnsupported,
            NormalizedCause::Transient,
            NormalizedCause::Unknown,
        ] {
            assert_eq!(NormalizedCause::from_key(cause.as_key()), Some(cause));
        }
        assert_eq!(NormalizedCause::from_key("nonsense"), None);
    }

    #[test]
    fn cause_maps_to_core_error_variants() {
        assert!(matches!(
            cause_to_core_error(
                NormalizedCause::ContextOverflow {
                    prompt_tokens: Some(9),
                    max_tokens: Some(8)
                },
                "d".into()
            ),
            Some(CoreError::ContextOverflow {
                prompt_tokens: Some(9),
                max_tokens: Some(8),
                ..
            })
        ));
        assert!(matches!(
            cause_to_core_error(NormalizedCause::BillingFatal, "d".into()),
            Some(CoreError::QuotaExceeded { .. })
        ));
        assert!(matches!(
            cause_to_core_error(NormalizedCause::RateLimited, "d".into()),
            Some(CoreError::RateLimited { .. })
        ));
        assert!(matches!(
            cause_to_core_error(NormalizedCause::ModelLoading, "d".into()),
            Some(CoreError::ModelLoading { .. })
        ));
        assert!(matches!(
            cause_to_core_error(NormalizedCause::ToolsUnsupported, "d".into()),
            Some(CoreError::ToolsUnsupported { .. })
        ));
        // No dedicated variant -> original error surfaces unchanged.
        assert!(cause_to_core_error(NormalizedCause::Auth, "d".into()).is_none());
        assert!(cause_to_core_error(NormalizedCause::Unknown, "d".into()).is_none());
    }
}
