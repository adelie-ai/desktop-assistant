//! SSE streaming: parse `chat/completions` `choices[].delta` frames into an
//! [`LlmResponse`], accumulating indexed tool calls and the final `usage`.

use std::time::Duration;

use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmResponse, TokenUsage, ToolCallAccumulator,
};
use desktop_assistant_llm_http::{StreamStep, build_response, next_step};
use eventsource_stream::Eventsource;

use crate::usage::parse_usage;

/// One streamed `chat/completions` chunk (the JSON payload of a `data:` SSE
/// frame, excluding the `[DONE]` sentinel).
///
/// `usage` is kept as a raw [`serde_json::Value`] so it can be handed to
/// [`parse_usage`], which reconciles the plain and cache-augmented shapes; a
/// missing or `null` `usage` deserializes to `None`.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ChatChunk {
    /// The choices array; typically length 1 for `n=1`, and empty on a
    /// usage-only final chunk.
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    /// The token-usage object, present on the final chunk when usage
    /// accounting is requested.
    #[serde(default)]
    pub usage: Option<serde_json::Value>,
}

/// One element of a [`ChatChunk`]'s `choices` array.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ChatChoice {
    /// The choice index (0 for `n=1`).
    #[serde(default)]
    pub index: u32,
    /// The incremental delta for this choice.
    pub delta: ChatDelta,
    /// The finish reason on the terminating chunk, if any.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// The incremental `delta` of a [`ChatChoice`]: a text fragment and/or
/// partial tool calls.
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ChatDelta {
    /// A fragment of assistant text, if this delta carries content.
    #[serde(default)]
    pub content: Option<String>,
    /// Partial tool calls, keyed by their `index` across frames.
    #[serde(default)]
    pub tool_calls: Vec<ChatToolCallDelta>,
}

/// A partial tool call inside a streamed [`ChatDelta`]. The first frame for a
/// call carries `id` and `function.name`; later frames carry only
/// `function.arguments` fragments at the same `index`.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ChatToolCallDelta {
    /// The stable per-stream index that ties fragments of the same call
    /// together.
    #[serde(default)]
    pub index: usize,
    /// The tool-call id, present on the first fragment.
    #[serde(default)]
    pub id: Option<String>,
    /// The function name and/or an arguments fragment.
    #[serde(default)]
    pub function: Option<ChatFunctionDelta>,
}

/// The `function` object of a [`ChatToolCallDelta`].
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ChatFunctionDelta {
    /// The function name, present on the first fragment.
    #[serde(default)]
    pub name: Option<String>,
    /// A fragment of the JSON arguments string (concatenated across frames).
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Parse a single `data:` frame payload into a [`ChatChunk`].
///
/// The caller is responsible for detecting the `[DONE]` sentinel *before*
/// calling this (it is not valid JSON). Exposed so the frame-parsing logic is
/// unit-testable without a live stream; [`consume_chat_stream`] uses it per
/// frame and warns-and-skips on error.
pub fn parse_chat_chunk(data: &str) -> Result<ChatChunk, serde_json::Error> {
    serde_json::from_str(data)
}

/// Consume an SSE byte stream of `chat/completions` frames into an
/// [`LlmResponse`].
///
/// Wraps `byte_stream` with SSE framing, then for each `data:` frame:
/// - `[DONE]` terminates the stream;
/// - a well-formed [`ChatChunk`] contributes its `choices[0].delta.content`
///   text (each fragment passed to `on_chunk`) and accumulates
///   `choices[0].delta.tool_calls[]` into a [`ToolCallAccumulator`] keyed by
///   each call's `index`; a `usage` object updates the reported token usage;
/// - a malformed frame is warned-and-skipped, never fatal.
///
/// When `on_chunk` returns `false` the loop **breaks** rather than returning,
/// so tool calls and usage accumulated before the abort are preserved in the
/// returned response. Cancellation (via `cancellation`) and a per-event stall
/// `event_timeout` are enforced through the shared `llm-http`
/// [`next_step`](desktop_assistant_llm_http::next_step) primitive.
pub async fn consume_chat_stream<S, B, E>(
    byte_stream: S,
    cancellation: &CancellationToken,
    event_timeout: Duration,
    mut on_chunk: ChunkCallback,
) -> Result<LlmResponse, CoreError>
where
    S: tokio_stream::Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    let mut events = byte_stream.eventsource();

    let mut text = String::new();
    let mut tool_acc: ToolCallAccumulator<usize> = ToolCallAccumulator::new();
    let mut token_usage: Option<TokenUsage> = None;

    'stream: loop {
        let event = match next_step(&mut events, cancellation, event_timeout).await {
            StreamStep::Item(Ok(ev)) => ev,
            StreamStep::Item(Err(e)) => {
                // A frame-level SSE/transport hiccup: warn and skip rather
                // than fail the whole turn.
                tracing::warn!("skipping unreadable SSE frame: {e}");
                continue;
            }
            StreamStep::Done => break,
            StreamStep::Cancelled => {
                tracing::debug!("chat stream cancelled by token");
                return Err(CoreError::Cancelled);
            }
            StreamStep::Stalled => {
                tracing::error!(
                    timeout_s = event_timeout.as_secs(),
                    "chat stream stalled -- no further event"
                );
                return Err(CoreError::Llm("chat stream stalled".to_string()));
            }
        };

        let data = event.data.as_str();
        if data.trim() == "[DONE]" {
            break;
        }

        let chunk = match parse_chat_chunk(data) {
            Ok(c) => c,
            Err(e) => {
                // Tolerate a malformed frame; a single bad chunk must not
                // sink an otherwise good stream.
                tracing::warn!("skipping malformed chat chunk: {e}");
                continue;
            }
        };

        if let Some(usage_val) = &chunk.usage
            && let Some(u) = parse_usage(usage_val)
        {
            token_usage = Some(u);
        }

        let Some(choice) = chunk.choices.into_iter().next() else {
            // No choices (e.g. a usage-only final chunk): nothing more to do.
            continue;
        };

        // Tool calls first, so a same-frame content abort still preserves any
        // tool-call fragments carried alongside it.
        for tc in &choice.delta.tool_calls {
            let name = tc.function.as_ref().and_then(|f| f.name.clone());
            if tc.id.is_some() || name.is_some() {
                tool_acc.start(
                    tc.index,
                    tc.id.clone().unwrap_or_default(),
                    name.unwrap_or_default(),
                );
            }
            if let Some(f) = &tc.function
                && let Some(args) = &f.arguments
            {
                tool_acc.append(tc.index, args);
            }
        }

        if let Some(content) = choice.delta.content
            && !content.is_empty()
        {
            text.push_str(&content);
            if !on_chunk(content) {
                tracing::debug!("chat streaming aborted by callback");
                break 'stream;
            }
        }
    }

    Ok(build_response(
        text,
        tool_acc.into_tool_calls(),
        token_usage,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::Method::GET;
    use httpmock::MockServer;
    use std::sync::{Arc, Mutex};

    /// Assemble an SSE body from `data:` frame payloads (each terminated by the
    /// blank line SSE requires).
    fn sse_body(frames: &[&str]) -> String {
        let mut s = String::new();
        for f in frames {
            s.push_str("data: ");
            s.push_str(f);
            s.push_str("\n\n");
        }
        s
    }

    /// A single-chunk in-memory byte stream carrying the whole SSE body.
    fn sse_stream(
        body: String,
    ) -> impl tokio_stream::Stream<Item = Result<Vec<u8>, std::io::Error>> + Unpin {
        tokio_stream::iter(vec![Ok(body.into_bytes())])
    }

    fn no_cancel() -> CancellationToken {
        CancellationToken::new()
    }

    // --- parse_chat_chunk -----------------------------------------------

    #[test]
    fn parse_chunk_text_delta() {
        let chunk = parse_chat_chunk(r#"{"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#)
            .expect("ok");
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn parse_chunk_tool_call_delta() {
        let chunk = parse_chat_chunk(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}"#,
        )
        .expect("ok");
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("call_1"));
        let f = tc.function.as_ref().expect("function present");
        assert_eq!(f.name.as_deref(), Some("lookup"));
        assert_eq!(f.arguments.as_deref(), Some(r#"{"q":"#));
    }

    #[test]
    fn parse_chunk_with_usage() {
        let chunk = parse_chat_chunk(
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}"#,
        )
        .expect("ok");
        assert!(chunk.choices.is_empty());
        assert!(chunk.usage.is_some());
    }

    #[test]
    fn parse_chunk_malformed_is_err() {
        assert!(parse_chat_chunk("not valid json").is_err());
    }

    // --- consume_chat_stream --------------------------------------------

    #[tokio::test]
    async fn consume_accumulates_text_and_terminates_on_done() {
        let body = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":", world"}}]}"#,
            "[DONE]",
        ]);
        let received = Arc::new(Mutex::new(Vec::new()));
        let received_cl = Arc::clone(&received);
        let result = consume_chat_stream(
            sse_stream(body),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(move |c| {
                received_cl.lock().expect("lock").push(c);
                true
            }),
        )
        .await
        .expect("stream ok");
        assert_eq!(result.text, "Hello, world");
        assert!(result.tool_calls.is_empty());
        assert_eq!(
            *received.lock().expect("lock"),
            vec!["Hello".to_string(), ", world".to_string()]
        );
    }

    #[tokio::test]
    async fn consume_accumulates_indexed_tool_call() {
        // id + name in the first frame; arguments split across two more.
        let body = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":\"ru"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"st\"}"}}]}}]}"#,
            "[DONE]",
        ]);
        let result = consume_chat_stream(
            sse_stream(body),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(|_| true),
        )
        .await
        .expect("stream ok");
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_1");
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[0].arguments, r#"{"q":"rust"}"#);
    }

    #[tokio::test]
    async fn consume_accumulates_multiple_indexed_tool_calls() {
        let body = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"first","arguments":"{}"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"second","arguments":"{}"}}]}}]}"#,
            "[DONE]",
        ]);
        let result = consume_chat_stream(
            sse_stream(body),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(|_| true),
        )
        .await
        .expect("stream ok");
        assert_eq!(result.tool_calls.len(), 2);
        // Ascending key order guaranteed by the accumulator.
        assert_eq!(result.tool_calls[0].name, "first");
        assert_eq!(result.tool_calls[1].name, "second");
    }

    #[tokio::test]
    async fn consume_tolerates_malformed_frame() {
        let body = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"content":"good "}}]}"#,
            "this is not json",
            r#"{"choices":[{"index":0,"delta":{"content":"stuff"}}]}"#,
            "[DONE]",
        ]);
        let result = consume_chat_stream(
            sse_stream(body),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(|_| true),
        )
        .await
        .expect("stream ok despite bad frame");
        assert_eq!(result.text, "good stuff", "good frames still processed");
    }

    #[tokio::test]
    async fn consume_tolerates_unknown_shape_frame() {
        // A valid-JSON frame with no recognizable choices/usage is a no-op.
        let body = sse_body(&[
            r#"{"object":"chat.completion.chunk","choices":[]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
            "[DONE]",
        ]);
        let result = consume_chat_stream(
            sse_stream(body),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(|_| true),
        )
        .await
        .expect("stream ok");
        assert_eq!(result.text, "hi");
    }

    #[tokio::test]
    async fn consume_callback_abort_preserves_accumulated_state() {
        // First frame carries a full tool call; the content frame aborts.
        let body = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"do_it","arguments":"{}"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"stop here"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":" never seen"}}]}"#,
            "[DONE]",
        ]);
        let result = consume_chat_stream(
            sse_stream(body),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(|_| false), // abort on the first content fragment
        )
        .await
        .expect("stream ok");
        // Break (not early return) preserves the accumulated tool call...
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "do_it");
        // ...and the text emitted up to the abort point (but not after).
        assert_eq!(result.text, "stop here");
    }

    #[tokio::test]
    async fn consume_reads_usage_from_final_chunk() {
        let body = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":8}}}"#,
            "[DONE]",
        ]);
        let result = consume_chat_stream(
            sse_stream(body),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(|_| true),
        )
        .await
        .expect("stream ok");
        let usage = result.usage.expect("usage present");
        assert_eq!(usage.input_tokens, Some(42));
        assert_eq!(usage.output_tokens, Some(7));
        assert_eq!(usage.cache_read_input_tokens, Some(8));
    }

    #[tokio::test]
    async fn consume_missing_usage_leaves_response_usage_none() {
        let body = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
            "[DONE]",
        ]);
        let result = consume_chat_stream(
            sse_stream(body),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(|_| true),
        )
        .await
        .expect("stream ok");
        assert!(result.usage.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn consume_returns_cancelled_when_token_tripped() {
        // A never-yielding stream + a pre-cancelled token: next_step must pick
        // the cancellation branch deterministically.
        let token = CancellationToken::new();
        token.cancel();
        let pending = tokio_stream::pending::<Result<Vec<u8>, std::io::Error>>();
        let result =
            consume_chat_stream(pending, &token, Duration::from_secs(5), Box::new(|_| true)).await;
        assert!(matches!(result, Err(CoreError::Cancelled)));
    }

    #[tokio::test]
    async fn consume_from_real_reqwest_byte_stream() {
        // Proves the generic signature accepts reqwest's own `bytes_stream()`
        // (the shape the connectors feed it) end to end.
        let server = MockServer::start();
        let body = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#,
            "[DONE]",
        ]);
        server.mock(|when, then| {
            when.method(GET).path("/stream");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(&body);
        });

        let resp = reqwest::get(format!("{}/stream", server.base_url()))
            .await
            .expect("request ok");
        let result = consume_chat_stream(
            resp.bytes_stream(),
            &no_cancel(),
            Duration::from_secs(5),
            Box::new(|_| true),
        )
        .await
        .expect("stream ok");
        assert_eq!(result.text, "Hi");
    }
}
