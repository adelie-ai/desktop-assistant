//! Error classification and safety-decline mapping for the Gemini surfaces.
//!
//! This is the sanctioned place for string-matching on provider error bodies
//! (a documented carve-out kept at the connector boundary). The Google error
//! envelope is `{ "error": { "code", "message", "status" } }`, where `status`
//! is a canonical code like `RESOURCE_EXHAUSTED` / `INVALID_ARGUMENT` /
//! `UNAVAILABLE`.
//!
//! Declines are not errors: a `SAFETY` block is a business decline, mapped to a
//! specific, informative `CoreError::Llm` reason (which category, that the
//! request was refused) and logged at info/debug — never dumping the flagged
//! body. There is no dedicated decline variant on `CoreError` today.
//
// NOTE: a dedicated `CoreError::Declined { reason, retryable: false }` variant
// is a possible follow-up so the dispatch layer can distinguish a safety
// refusal from a genuine backend failure without string inspection.

use desktop_assistant_core::CoreError;
use desktop_assistant_llm_http::parse_retry_after_header;

use crate::wire::{ErrorEnvelope, GenerateContentResponse, SafetyRating};

/// Finish reasons (and prompt block reasons) that mean the safety filter
/// refused the request/response.
const SAFETY_FINISH_REASONS: &[&str] = &["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"];

/// Parsed token counts from a Gemini context-overflow rejection. Either count
/// may be absent when the message states the overflow without numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextOverflowInfo {
    pub prompt_tokens: Option<u64>,
    pub max_tokens: Option<u64>,
}

/// Classify a non-success HTTP response from a Gemini surface into a
/// `CoreError`. `body` is the response body text; the `Retry-After` header (if
/// any) is folded into `RateLimited`/`QuotaExceeded` hints.
///
/// Never echoes a bearer token or service-account material: those never appear
/// in a Google error body, and auth failures are surfaced with a fixed
/// guidance string plus the provider's own (safe) message.
pub fn classify_http_error(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> CoreError {
    let (status_code, message) = parse_error_envelope(body);
    let http = status.as_u16();
    let status_str = status_code.unwrap_or_default();
    // Prefer the provider's structured message; fall back to a trimmed body.
    // Neither can carry a bearer token or SA key (Google never echoes them),
    // but we cap the length so an unexpected large body can't spam logs.
    let msg = message.unwrap_or_else(|| body.trim().chars().take(500).collect::<String>());
    let retry_after = parse_retry_after_header(headers);

    if http == 429 || status_str == "RESOURCE_EXHAUSTED" {
        if is_hard_quota(&msg) {
            return CoreError::QuotaExceeded {
                detail: format!("Google Vertex quota exceeded: {msg}"),
            };
        }
        return CoreError::RateLimited {
            retry_after,
            detail: format!("Google Vertex rate limited (HTTP {http}): {msg}"),
        };
    }

    if (http == 400 || status_str == "INVALID_ARGUMENT")
        && let Some(info) = parse_context_overflow(&msg)
    {
        return CoreError::ContextOverflow {
            prompt_tokens: info.prompt_tokens,
            max_tokens: info.max_tokens,
            detail: format!("Google Vertex context overflow: {msg}"),
        };
    }

    if http == 401
        || http == 403
        || status_str == "UNAUTHENTICATED"
        || status_str == "PERMISSION_DENIED"
    {
        return CoreError::Llm(format!(
            "Google Vertex authentication/authorization failed (HTTP {http}): {msg}. \
             Check the service-account credentials, that the project has the Vertex AI \
             API enabled, and that the account holds the aiplatform.user role."
        ));
    }

    if status.is_server_error() || status_str == "UNAVAILABLE" || status_str == "INTERNAL" {
        return CoreError::RateLimited {
            retry_after,
            detail: format!("Google Vertex service error (HTTP {http}): {msg}"),
        };
    }

    CoreError::Llm(format!("Google Vertex API error (HTTP {http}): {msg}"))
}

/// Detect a Gemini context-overflow rejection, extracting `(prompt, max)`
/// token counts when the message carries them.
pub fn parse_context_overflow(message: &str) -> Option<ContextOverflowInfo> {
    let lower = message.to_ascii_lowercase();
    let is_overflow = (lower.contains("token count") && lower.contains("exceed"))
        || lower.contains("maximum number of tokens")
        || (lower.contains("input") && lower.contains("too long"))
        || (lower.contains("exceeds") && lower.contains("context"));
    if !is_overflow {
        return None;
    }
    // Across the recognized shapes the counts appear as (prompt, max).
    let nums: Vec<u64> = message
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    let (prompt_tokens, max_tokens) = match nums.as_slice() {
        [prompt, max, ..] => (Some(*prompt), Some(*max)),
        _ => (None, None),
    };
    Some(ContextOverflowInfo {
        prompt_tokens,
        max_tokens,
    })
}

/// Whether a `RESOURCE_EXHAUSTED` message is a hard billing/quota failure
/// (permanent, -> `QuotaExceeded`) rather than a transient per-minute rate
/// limit (-> `RateLimited`).
pub fn is_hard_quota(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("billing") || lower.contains("check your plan")
}

/// Inspect a streamed response frame for a safety block. Returns the offending
/// safety category (or the raw block/finish reason when no category is
/// itemized) when the prompt or a candidate was refused; `None` otherwise.
pub fn safety_block(resp: &GenerateContentResponse) -> Option<String> {
    if let Some(pf) = &resp.prompt_feedback
        && let Some(reason) = pf.block_reason.as_deref()
        && !reason.is_empty()
        && reason != "BLOCK_REASON_UNSPECIFIED"
    {
        return Some(
            category_from_ratings(&pf.safety_ratings).unwrap_or_else(|| reason.to_string()),
        );
    }
    for cand in &resp.candidates {
        if let Some(fr) = cand.finish_reason.as_deref()
            && SAFETY_FINISH_REASONS.contains(&fr)
        {
            return Some(
                category_from_ratings(&cand.safety_ratings).unwrap_or_else(|| fr.to_string()),
            );
        }
    }
    None
}

/// The category of the first `blocked` safety rating, if any.
fn category_from_ratings(ratings: &[SafetyRating]) -> Option<String> {
    ratings
        .iter()
        .find(|r| r.blocked)
        .and_then(|r| r.category.clone())
}

/// Build the user-facing decline error for a safety block. Names the category
/// and states the request was refused; never includes the flagged content.
pub fn safety_decline_error(category: &str) -> CoreError {
    CoreError::Llm(format!(
        "Gemini declined this request via its safety filter (category: {category}). \
         The request was refused; rephrasing may help."
    ))
}

/// Parse the Google error envelope, returning `(status, message)` where each
/// may be absent. Used by [`classify_http_error`] and testable on its own.
pub fn parse_error_envelope(body: &str) -> (Option<String>, Option<String>) {
    match serde_json::from_str::<ErrorEnvelope>(body) {
        Ok(env) => (env.error.status, env.error.message),
        Err(_) => (None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use reqwest::header::HeaderMap;

    fn body_for(status: &str, message: &str, code: u16) -> String {
        format!(r#"{{"error":{{"code":{code},"message":{message:?},"status":{status:?}}}}}"#)
    }

    #[test]
    fn resource_exhausted_rate_limit_is_rate_limited_with_retry_after() {
        let mut headers = HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "17".parse().unwrap());
        let body = body_for(
            "RESOURCE_EXHAUSTED",
            "Quota exceeded for quota metric 'Generate Content requests per minute'",
            429,
        );
        let err = classify_http_error(StatusCode::TOO_MANY_REQUESTS, &headers, &body);
        match err {
            CoreError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(17)));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn resource_exhausted_billing_is_quota_exceeded() {
        let body = body_for(
            "RESOURCE_EXHAUSTED",
            "You exceeded your current quota, please check your plan and billing details.",
            429,
        );
        let err = classify_http_error(StatusCode::TOO_MANY_REQUESTS, &HeaderMap::new(), &body);
        assert!(
            matches!(err, CoreError::QuotaExceeded { .. }),
            "billing quota must be permanent, got {err:?}"
        );
    }

    #[test]
    fn invalid_argument_context_overflow_is_context_overflow() {
        let body = body_for(
            "INVALID_ARGUMENT",
            "The input token count (1290000) exceeds the maximum number of tokens allowed (1048576).",
            400,
        );
        let err = classify_http_error(StatusCode::BAD_REQUEST, &HeaderMap::new(), &body);
        match err {
            CoreError::ContextOverflow {
                prompt_tokens,
                max_tokens,
                ..
            } => {
                assert_eq!(prompt_tokens, Some(1_290_000));
                assert_eq!(max_tokens, Some(1_048_576));
            }
            other => panic!("expected ContextOverflow, got {other:?}"),
        }
    }

    #[test]
    fn invalid_argument_other_is_generic_llm() {
        let body = body_for("INVALID_ARGUMENT", "contents: must not be empty", 400);
        let err = classify_http_error(StatusCode::BAD_REQUEST, &HeaderMap::new(), &body);
        assert!(matches!(err, CoreError::Llm(_)), "got {err:?}");
    }

    #[test]
    fn unauthenticated_401_is_llm_naming_the_fix_without_token() {
        let body = body_for(
            "UNAUTHENTICATED",
            "Request had invalid authentication credentials.",
            401,
        );
        let err = classify_http_error(StatusCode::UNAUTHORIZED, &HeaderMap::new(), &body);
        let CoreError::Llm(detail) = err else {
            panic!("expected Llm, got {err:?}");
        };
        let lower = detail.to_ascii_lowercase();
        assert!(
            lower.contains("credential") || lower.contains("auth"),
            "auth error should name the fix: {detail}"
        );
    }

    #[test]
    fn permission_denied_403_is_llm() {
        let body = body_for(
            "PERMISSION_DENIED",
            "Permission denied on resource project.",
            403,
        );
        let err = classify_http_error(StatusCode::FORBIDDEN, &HeaderMap::new(), &body);
        assert!(matches!(err, CoreError::Llm(_)), "got {err:?}");
    }

    #[test]
    fn unavailable_503_is_rate_limited() {
        let body = body_for("UNAVAILABLE", "The service is currently unavailable.", 503);
        let err = classify_http_error(StatusCode::SERVICE_UNAVAILABLE, &HeaderMap::new(), &body);
        assert!(matches!(err, CoreError::RateLimited { .. }), "got {err:?}");
    }

    #[test]
    fn internal_500_is_rate_limited() {
        let body = body_for("INTERNAL", "Internal error.", 500);
        let err = classify_http_error(StatusCode::INTERNAL_SERVER_ERROR, &HeaderMap::new(), &body);
        assert!(matches!(err, CoreError::RateLimited { .. }), "got {err:?}");
    }

    #[test]
    fn unparseable_body_still_classifies_by_status_code() {
        // A 500 with a non-JSON body is still transient.
        let err = classify_http_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &HeaderMap::new(),
            "upstream proxy exploded",
        );
        assert!(matches!(err, CoreError::RateLimited { .. }), "got {err:?}");
    }

    #[test]
    fn parse_context_overflow_extracts_counts() {
        let info = parse_context_overflow(
            "The input token count (1290000) exceeds the maximum number of tokens allowed (1048576).",
        )
        .expect("overflow");
        assert_eq!(info.prompt_tokens, Some(1_290_000));
        assert_eq!(info.max_tokens, Some(1_048_576));
    }

    #[test]
    fn parse_context_overflow_tolerates_missing_counts() {
        let info =
            parse_context_overflow("The input is too long for this model.").expect("overflow");
        assert_eq!(info.prompt_tokens, None);
        assert_eq!(info.max_tokens, None);
    }

    #[test]
    fn parse_context_overflow_rejects_unrelated() {
        assert!(parse_context_overflow("permission denied on resource").is_none());
        assert!(parse_context_overflow("invalid value 123 for field").is_none());
    }

    #[test]
    fn is_hard_quota_distinguishes_billing_from_rate() {
        assert!(is_hard_quota(
            "You exceeded your current quota, please check your plan and billing details."
        ));
        assert!(!is_hard_quota(
            "Quota exceeded for quota metric 'requests per minute'"
        ));
    }

    #[test]
    fn safety_block_detected_from_prompt_feedback() {
        let resp: GenerateContentResponse = serde_json::from_str(
            r#"{"promptFeedback":{"blockReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH","blocked":true}]}}"#,
        )
        .unwrap();
        let category = safety_block(&resp).expect("safety block detected");
        assert_eq!(category, "HARM_CATEGORY_DANGEROUS_CONTENT");
    }

    #[test]
    fn safety_block_detected_from_candidate_finish_reason() {
        let resp: GenerateContentResponse = serde_json::from_str(
            r#"{"candidates":[{"finishReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_HARASSMENT","blocked":true}]}]}"#,
        )
        .unwrap();
        let category = safety_block(&resp).expect("safety block detected");
        assert_eq!(category, "HARM_CATEGORY_HARASSMENT");
    }

    #[test]
    fn safety_block_absent_for_normal_stop() {
        let resp: GenerateContentResponse = serde_json::from_str(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
        assert!(safety_block(&resp).is_none());
    }

    #[test]
    fn safety_decline_error_names_category_and_refusal() {
        let err = safety_decline_error("HARM_CATEGORY_HATE_SPEECH");
        let CoreError::Llm(detail) = err else {
            panic!("expected Llm");
        };
        assert!(detail.contains("HARM_CATEGORY_HATE_SPEECH"));
        assert!(detail.to_ascii_lowercase().contains("refused"));
    }

    #[test]
    fn parse_error_envelope_extracts_status_and_message() {
        let (status, message) = parse_error_envelope(&body_for("UNAVAILABLE", "down", 503));
        assert_eq!(status.as_deref(), Some("UNAVAILABLE"));
        assert_eq!(message.as_deref(), Some("down"));
    }
}
