//! Shared HTTP error-handling helpers for the LLM connector crates.
//!
//! Issue #46. Every reqwest-based connector (`llm-anthropic`,
//! `llm-openai`, `llm-ollama`) repeated the same "if non-success, read
//! the body and bail with a formatted message" idiom at every endpoint.
//! This crate centralises that pattern.
//!
//! Out of scope: the streaming dispatch sites that need
//! provider-specific routing on the error (context-overflow detection,
//! rate-limit / quota classification, etc.) keep their bespoke arms —
//! those produce structured `CoreError` variants
//! (`ContextOverflow`, `RateLimited`, `QuotaExceeded`, `ModelLoading`,
//! `ToolsUnsupported`) and aren't worth shoehorning into one helper.

use desktop_assistant_core::CoreError;
use reqwest::Response;

/// Return `Ok(response)` when the status is 2xx; otherwise consume the
/// body and return `CoreError::Llm` with a uniform
/// `"{context} (HTTP {status}): {body}"` detail string.
///
/// `context` is a short label like `"OpenAI embeddings API error"` or
/// `"Ollama model pull API error for 'llama3'"` — i.e. include the
/// provider name and (when relevant) the endpoint or resource. Body
/// read failures are surfaced as the literal `"unable to read body"`
/// so the error message still names the status code even when the
/// transport drops mid-frame.
pub async fn bail_for_status(response: Response, context: &str) -> Result<Response, CoreError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "unable to read body".into());
    Err(CoreError::Llm(format!("{context} (HTTP {status}): {body}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::Method::GET;
    use httpmock::MockServer;

    #[tokio::test]
    async fn bail_for_status_passes_through_2xx() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/ok");
            then.status(200).body("hello");
        });

        let response = reqwest::get(format!("{}/ok", server.base_url()))
            .await
            .unwrap();
        let response = bail_for_status(response, "Test API error").await.unwrap();
        assert_eq!(response.text().await.unwrap(), "hello");
    }

    #[tokio::test]
    async fn bail_for_status_formats_non_2xx_with_context() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/bad");
            then.status(503).body("upstream gave up");
        });

        let response = reqwest::get(format!("{}/bad", server.base_url()))
            .await
            .unwrap();
        let err = bail_for_status(response, "Sample API error")
            .await
            .unwrap_err();
        let CoreError::Llm(detail) = err else {
            panic!("expected CoreError::Llm, got {err:?}");
        };
        assert!(
            detail.contains("Sample API error"),
            "context label missing: {detail}"
        );
        assert!(detail.contains("503"), "status missing: {detail}");
        assert!(
            detail.contains("upstream gave up"),
            "body missing: {detail}"
        );
    }
}
