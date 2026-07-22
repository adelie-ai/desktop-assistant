//! Error classification: the base OpenAI-compatible mapping of an HTTP error
//! response to a [`CoreError`], plus the sub-detectors connectors extend.

use serde::Deserialize;

use desktop_assistant_core::CoreError;
use desktop_assistant_llm_http::parse_retry_after_header;
use reqwest::StatusCode;
use reqwest::header::HeaderMap;

/// The OpenAI-shaped error envelope `{ "error": { code, message, type } }`,
/// shared by the Chat Completions and Responses surfaces and echoed by the
/// OpenAI-compatible aggregators. Only the inner `error` shape is inspected.
#[derive(Deserialize, Default)]
struct ErrorEnvelope {
    #[serde(default)]
    error: ErrorBody,
}

#[derive(Deserialize, Default)]
struct ErrorBody {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: String,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
}

/// Parsed token counts from a context-overflow error, in domain order.
///
/// Both fields are optional: the provider's wording does not always carry
/// numbers, and the core's overflow-recovery path tolerates absent
/// measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextOverflowInfo {
    /// The prompt/input tokens the request actually used, when stated.
    pub prompt_tokens: Option<u64>,
    /// The model's maximum context window, when stated.
    pub max_tokens: Option<u64>,
}

/// Classify an OpenAI-compatible HTTP error response into a [`CoreError`].
///
/// This is the **base** mapping shared by OpenRouter and Azure; each wraps it
/// to add provider-specific cases (OpenRouter's 402 out-of-credits, Azure's
/// `content_filter` decline) before delegating here. Order of checks:
///
/// 1. context-overflow body ([`detect_context_overflow`]) ->
///    [`CoreError::ContextOverflow`] with parsed counts;
/// 2. `insufficient_quota` body ([`detect_insufficient_quota`]) ->
///    [`CoreError::QuotaExceeded`] (permanent billing; not retried), regardless
///    of status -- some providers signal it with HTTP 429, which would
///    otherwise look retryable;
/// 3. HTTP 429 -> [`CoreError::RateLimited`] carrying the `Retry-After` hint;
/// 4. HTTP 5xx (incl. 503) -> [`CoreError::RateLimited`] (transient overload);
/// 5. anything else -> a clear [`CoreError::Llm`] message.
///
/// The raw `body` is included in the detail so the failure is diagnosable; a
/// connector that must scrub a decline body (e.g. `content_filter`, which
/// echoes flagged user content) handles that case before calling this and does
/// not fall through here.
pub fn classify_error(status: StatusCode, headers: &HeaderMap, body: &str) -> CoreError {
    let detail = format!("OpenAI-compatible API error (HTTP {status}): {body}");

    if let Some(info) = detect_context_overflow(body) {
        return CoreError::ContextOverflow {
            prompt_tokens: info.prompt_tokens,
            max_tokens: info.max_tokens,
            detail,
        };
    }

    if detect_insufficient_quota(body) {
        return CoreError::QuotaExceeded { detail };
    }

    if status.as_u16() == 429 {
        return CoreError::RateLimited {
            retry_after: parse_retry_after_header(headers),
            detail,
        };
    }

    if status.is_server_error() {
        return CoreError::RateLimited {
            retry_after: parse_retry_after_header(headers),
            detail,
        };
    }

    CoreError::Llm(detail)
}

/// Detect a context-window-overflow rejection in an HTTP error body.
///
/// Returns `Some` when the body parses as the OpenAI error envelope and either
/// `error.code == "context_length_exceeded"` or the message mentions "maximum
/// context" (the wording varies across the aggregators' upstreams). The counts
/// are parsed from the message when present.
///
/// Exposed so a connector can extend or reuse the base detection. Pattern-
/// matching on an error body is normally banned (`AGENTS.md`), but this is the
/// sanctioned connector-boundary carve-out: it is the only signal these
/// providers give for a context rejection, and converting it to a structured
/// [`CoreError::ContextOverflow`] here means downstream code never has to.
pub fn detect_context_overflow(body: &str) -> Option<ContextOverflowInfo> {
    let envelope: ErrorEnvelope = serde_json::from_str(body).ok()?;
    let code_matches = envelope.error.code.as_deref() == Some("context_length_exceeded");
    let message_matches = envelope
        .error
        .message
        .to_ascii_lowercase()
        .contains("maximum context");
    if !code_matches && !message_matches {
        return None;
    }
    let (prompt_tokens, max_tokens) = parse_context_length_message(&envelope.error.message);
    Some(ContextOverflowInfo {
        prompt_tokens,
        max_tokens,
    })
}

/// Detect the permanent `insufficient_quota` billing error in an HTTP error
/// body. True when the envelope's `error.code` or `error.type` is
/// `insufficient_quota`.
///
/// Exposed so a connector can OR in its own permanent-billing signals (e.g.
/// OpenRouter's out-of-credits shape) before mapping to
/// [`CoreError::QuotaExceeded`]. OpenAI overloads HTTP 429 for both transient
/// throttling (retryable) and this permanent case (not retryable);
/// distinguishing them here keeps `is_retryable_error` a flat variant match
/// downstream.
pub fn detect_insufficient_quota(body: &str) -> bool {
    let Ok(envelope): Result<ErrorEnvelope, _> = serde_json::from_str(body) else {
        return false;
    };
    envelope.error.code.as_deref() == Some("insufficient_quota")
        || envelope.error.error_type.as_deref() == Some("insufficient_quota")
}

/// Parse OpenAI's `"This model's maximum context length is 128000 tokens.
/// However, your messages resulted in 153827 tokens. ..."` wording into
/// `(prompt_tokens, max_tokens)`.
///
/// OpenAI lists the numbers in `(max, prompt)` order -- opposite to the domain
/// [`ContextOverflowInfo`] order -- so this swaps them. A missing value yields
/// `None` in that slot.
fn parse_context_length_message(message: &str) -> (Option<u64>, Option<u64>) {
    let nums: Vec<u64> = message
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    match nums.as_slice() {
        [max, prompt, ..] => (Some(*prompt), Some(*max)),
        [max] => (None, Some(*max)),
        _ => (None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with_retry_after(secs: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_str(secs).expect("valid header"),
        );
        h
    }

    // --- detect_context_overflow ----------------------------------------

    #[test]
    fn detect_context_overflow_extracts_counts_in_domain_order() {
        let body = r#"{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 153827 tokens. Please reduce the length of the messages."}}"#;
        let info = detect_context_overflow(body).expect("overflow detected");
        // OpenAI lists (max, prompt); we return (prompt, max).
        assert_eq!(info.prompt_tokens, Some(153_827));
        assert_eq!(info.max_tokens, Some(128_000));
    }

    #[test]
    fn detect_context_overflow_none_for_other_codes() {
        let body = r#"{"error":{"code":"invalid_api_key","type":"invalid_request_error","message":"bad key"}}"#;
        assert!(detect_context_overflow(body).is_none());
    }

    #[test]
    fn detect_context_overflow_tolerates_missing_numbers() {
        let body = r#"{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"context length exceeded"}}"#;
        let info = detect_context_overflow(body).expect("still detected by code");
        assert_eq!(info.prompt_tokens, None);
        assert_eq!(info.max_tokens, None);
    }

    #[test]
    fn detect_context_overflow_matches_maximum_context_message() {
        // Some upstreams omit the code but say "maximum context" in the text.
        let body = r#"{"error":{"message":"Requested tokens exceed the maximum context length of 8192 tokens.","type":"invalid_request_error"}}"#;
        let info = detect_context_overflow(body).expect("detected by message");
        assert_eq!(info.max_tokens, Some(8192));
    }

    #[test]
    fn detect_context_overflow_none_for_non_json() {
        assert!(detect_context_overflow("Bad Gateway").is_none());
    }

    // --- detect_insufficient_quota --------------------------------------

    #[test]
    fn detect_insufficient_quota_matches_code() {
        let body = r#"{"error":{"code":"insufficient_quota","type":"invalid_request_error","message":"You exceeded your current quota"}}"#;
        assert!(detect_insufficient_quota(body));
    }

    #[test]
    fn detect_insufficient_quota_matches_type() {
        let body = r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#;
        assert!(detect_insufficient_quota(body));
    }

    #[test]
    fn detect_insufficient_quota_false_for_other_bodies() {
        let body = r#"{"error":{"code":"rate_limit_exceeded","type":"rate_limit_error","message":"slow down"}}"#;
        assert!(!detect_insufficient_quota(body));
        assert!(!detect_insufficient_quota("not json"));
    }

    // --- classify_error -------------------------------------------------

    #[test]
    fn classify_429_rate_limit_maps_to_rate_limited_with_retry_after() {
        let body = r#"{"error":{"code":"rate_limit_exceeded","type":"rate_limit_error","message":"Rate limit reached"}}"#;
        let err = classify_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers_with_retry_after("20"),
            body,
        );
        match err {
            CoreError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(20)));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn classify_429_insufficient_quota_maps_to_quota_exceeded() {
        let body = r#"{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"You exceeded your current quota, please check your plan and billing details."}}"#;
        let err = classify_error(StatusCode::TOO_MANY_REQUESTS, &HeaderMap::new(), body);
        match err {
            CoreError::QuotaExceeded { detail } => {
                assert!(detail.contains("insufficient_quota"), "detail: {detail}");
            }
            other => panic!("expected QuotaExceeded, got {other:?}"),
        }
    }

    #[test]
    fn classify_400_context_length_exceeded_maps_to_context_overflow() {
        let body = r#"{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 153827 tokens."}}"#;
        let err = classify_error(StatusCode::BAD_REQUEST, &HeaderMap::new(), body);
        match err {
            CoreError::ContextOverflow {
                prompt_tokens,
                max_tokens,
                ..
            } => {
                assert_eq!(prompt_tokens, Some(153_827), "prompt first in domain order");
                assert_eq!(max_tokens, Some(128_000));
            }
            other => panic!("expected ContextOverflow, got {other:?}"),
        }
    }

    #[test]
    fn classify_503_maps_to_rate_limited() {
        let err = classify_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &HeaderMap::new(),
            "upstream overloaded",
        );
        assert!(matches!(err, CoreError::RateLimited { .. }));
    }

    #[test]
    fn classify_500_maps_to_rate_limited() {
        let err = classify_error(StatusCode::INTERNAL_SERVER_ERROR, &HeaderMap::new(), "boom");
        assert!(matches!(err, CoreError::RateLimited { .. }));
    }

    #[test]
    fn classify_401_maps_to_generic_llm() {
        let body = r#"{"error":{"code":"invalid_api_key","type":"invalid_request_error","message":"Incorrect API key provided"}}"#;
        let err = classify_error(StatusCode::UNAUTHORIZED, &HeaderMap::new(), body);
        match err {
            CoreError::Llm(detail) => assert!(detail.contains("401"), "detail: {detail}"),
            other => panic!("expected Llm, got {other:?}"),
        }
    }

    #[test]
    fn classify_400_generic_bad_request_maps_to_llm() {
        // A 400 that is NOT a context overflow falls through to Llm.
        let body = r#"{"error":{"code":"invalid_request_error","type":"invalid_request_error","message":"unknown field"}}"#;
        let err = classify_error(StatusCode::BAD_REQUEST, &HeaderMap::new(), body);
        assert!(matches!(err, CoreError::Llm(_)));
    }
}
