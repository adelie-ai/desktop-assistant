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
        let (prompt_tokens, max_tokens) = first_two_numbers(ctx.message);
        return NormalizedCause::ContextOverflow {
            prompt_tokens,
            max_tokens,
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

/// Extract the first two base-10 integers from `s`, in order. Across the
/// recognized overflow phrasings the counts appear as `(prompt, max)`; fewer
/// than two means the provider stated the overflow without numbers.
fn first_two_numbers(s: &str) -> (Option<u64>, Option<u64>) {
    let nums: Vec<u64> = s
        .split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse::<u64>().ok())
        .collect();
    match nums.as_slice() {
        [prompt, max, ..] => (Some(*prompt), Some(*max)),
        _ => (None, None),
    }
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
