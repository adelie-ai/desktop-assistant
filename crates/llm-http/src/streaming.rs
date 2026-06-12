//! Shared streaming scaffold for the LLM connector crates.
//!
//! Issue #302. The reqwest-based connectors (`llm-anthropic`, `llm-openai`,
//! `llm-ollama`) — and, for the constants, `llm-bedrock` — each carried a
//! byte-identical copy of the streaming plumbing introduced by #214/#220:
//! the connect/stall timeout constants, the [`StreamStep`] / [`next_step`]
//! stall-loop primitive, the `Retry-After` header parser, the response
//! envelope builder, and a `StallingStream` test harness. Copy-pasted fixes
//! are a demonstrated failure mode in this codebase, so this module is the
//! single home for all of it.
//!
//! What stays provider-specific: the request construction, the connect race
//! (`tokio::select!` of `send()` against cancellation — the future and the
//! error mapping differ per provider), the SSE/NDJSON event parsing, and
//! Bedrock's AWS-SDK stream which can't share the `tokio_stream`-typed
//! [`next_step`] but does share the timeout constants.

use desktop_assistant_core::domain::ToolCall;
use desktop_assistant_core::ports::llm::{LlmResponse, TokenUsage};
use std::time::Duration;
use tokio_stream::StreamExt;

/// Maximum time to wait for the HTTP connection handshake (response headers)
/// before failing the turn (#214/#220).
pub const STREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum gap allowed between two streamed events/chunks. Each received item
/// resets the clock (the heartbeat), so this only fires when the stream goes
/// silent mid-response (#214/#220).
pub const STREAM_EVENT_TIMEOUT: Duration = Duration::from_secs(60);

/// Outcome of racing a stream's next item against cancellation and a stall
/// timeout. Extracted so the timeout behaviour is unit-testable with a short
/// duration instead of the production [`STREAM_EVENT_TIMEOUT`] (#220).
pub enum StreamStep<T> {
    /// The stream yielded an item.
    Item(T),
    /// The stream ended (no more items).
    Done,
    /// The cancellation token tripped.
    Cancelled,
    /// No item arrived within the stall timeout.
    Stalled,
}

/// Await the next item from `stream`, racing it against `cancellation` and a
/// `timeout`. A fresh `tokio::time::sleep(timeout)` is created on every call,
/// so the stall window resets each time a caller consumes an item.
pub async fn next_step<S>(
    stream: &mut S,
    cancellation: &tokio_util::sync::CancellationToken,
    timeout: Duration,
) -> StreamStep<S::Item>
where
    S: tokio_stream::Stream + Unpin,
{
    tokio::select! {
        _ = cancellation.cancelled() => StreamStep::Cancelled,
        _ = tokio::time::sleep(timeout) => StreamStep::Stalled,
        next = stream.next() => match next {
            Some(item) => StreamStep::Item(item),
            None => StreamStep::Done,
        },
    }
}

/// Parse a `Retry-After` response header expressed as an integer number of
/// seconds. The HTTP-date form is uncommon on JSON APIs and intentionally
/// unsupported — it returns `None` ("no hint") rather than guessing.
pub fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    raw.trim().parse::<u64>().ok().map(Duration::from_secs)
}

/// Assemble an [`LlmResponse`] from accumulated streaming output: text, any
/// tool calls, and optional token usage. Every provider built this envelope
/// the same way at the end of its stream loop.
pub fn build_response(
    text: String,
    tool_calls: Vec<ToolCall>,
    usage: Option<TokenUsage>,
) -> LlmResponse {
    let response = if tool_calls.is_empty() {
        LlmResponse::text(text)
    } else {
        LlmResponse::with_tool_calls(text, tool_calls)
    };
    match usage {
        Some(u) => response.with_usage(u),
        None => response,
    }
}

/// Test-only stream harnesses shared across the connector crates' stall-loop
/// tests. Gated behind the `test-util` feature so it ships only as a
/// dev-dependency of the providers.
#[cfg(any(test, feature = "test-util"))]
pub mod test_util {
    /// Stream that yields `n` items then stays `Pending` forever, simulating a
    /// mid-stream stall (#220). Goes silent: never wakes, never ends.
    pub struct StallingStream {
        /// Number of items to yield before stalling.
        pub remaining: usize,
    }

    impl tokio_stream::Stream for StallingStream {
        type Item = u32;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            if self.remaining > 0 {
                self.remaining -= 1;
                std::task::Poll::Ready(Some(0))
            } else {
                std::task::Poll::Pending
            }
        }
    }

    /// Short label for a [`super::StreamStep`] variant, for test panic
    /// messages.
    pub fn step_name<T>(step: &super::StreamStep<T>) -> &'static str {
        match step {
            super::StreamStep::Item(_) => "Item",
            super::StreamStep::Done => "Done",
            super::StreamStep::Cancelled => "Cancelled",
            super::StreamStep::Stalled => "Stalled",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::{StallingStream, step_name};
    use super::*;
    use desktop_assistant_core::domain::ToolCall;

    #[tokio::test(start_paused = true)]
    async fn next_step_fires_stall_timeout_after_silence() {
        let cancellation = tokio_util::sync::CancellationToken::new();
        let mut stream = StallingStream { remaining: 1 };
        let timeout = Duration::from_millis(50);

        // First item arrives immediately (the heartbeat).
        match next_step(&mut stream, &cancellation, timeout).await {
            StreamStep::Item(_) => {}
            other => panic!("expected first item, got {}", step_name(&other)),
        }

        // Stream now silent: the per-event timeout must fire rather than hang.
        match next_step(&mut stream, &cancellation, timeout).await {
            StreamStep::Stalled => {}
            other => panic!("expected stall, got {}", step_name(&other)),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn next_step_prefers_cancellation_over_stall() {
        let cancellation = tokio_util::sync::CancellationToken::new();
        cancellation.cancel();
        let mut stream = StallingStream { remaining: 0 };
        let timeout = Duration::from_millis(50);
        match next_step(&mut stream, &cancellation, timeout).await {
            StreamStep::Cancelled => {}
            other => panic!("expected cancelled, got {}", step_name(&other)),
        }
    }

    #[test]
    fn parse_retry_after_integer_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("45"),
        );
        assert_eq!(
            parse_retry_after_header(&headers),
            Some(Duration::from_secs(45))
        );
    }

    #[test]
    fn parse_retry_after_missing() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after_header(&headers), None);
    }

    #[test]
    fn parse_retry_after_http_date_unparseable() {
        // HTTP-date form is uncommon on JSON APIs and intentionally
        // unsupported; treat it as "no hint" rather than guessing.
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after_header(&headers), None);
    }

    #[test]
    fn build_response_text_only() {
        let resp = build_response("hello".into(), vec![], None);
        assert_eq!(resp.text, "hello");
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn build_response_with_tool_calls() {
        let calls = vec![ToolCall::new("call_1", "do_thing", "{}")];
        let resp = build_response(String::new(), calls, None);
        assert_eq!(resp.tool_calls.len(), 1);
    }

    #[test]
    fn build_response_attaches_usage() {
        let usage = TokenUsage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };
        let resp = build_response("hi".into(), vec![], Some(usage));
        assert_eq!(resp.usage.as_ref().and_then(|u| u.input_tokens), Some(10));
    }

    #[test]
    fn shared_timeout_constants_match_legacy_values() {
        assert_eq!(STREAM_CONNECT_TIMEOUT, Duration::from_secs(30));
        assert_eq!(STREAM_EVENT_TIMEOUT, Duration::from_secs(60));
    }
}
