//! Non-streaming `chat/completions` dispatch: the fallback used when a routed
//! backend rejects tool use in streaming mode (#619).
//!
//! Some OpenRouter-routed (and, in principle, Azure) backends accept tools only
//! on a non-streaming request. When the streaming attempt is rejected with the
//! provider's "tools unsupported in streaming" error (detected at the connector
//! boundary by [`detect_streaming_tools_unsupported`](crate::detect_streaming_tools_unsupported)
//! and classified to [`CoreError::ToolsUnsupported`]), the connector retries via
//! this module: POST `/chat/completions` with `stream: false`, parse the single
//! JSON response into an [`LlmResponse`], and emit the assistant text once
//! through the chunk callback so a text-consuming UI still receives it.
//!
//! The request-envelope shaping stays with each connector (they differ); what
//! lives here is the shared mechanism: the response wire types, the JSON ->
//! [`LlmResponse`] conversion ([`parse_chat_completion`]), the connect-race send
//! ([`send_chat_request`]), and the callback-emitting body read
//! ([`dispatch_non_streaming`]).

use tokio_util::sync::CancellationToken;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmResponse};

// NOTE: stub bodies — the real implementation lands in the follow-up commit
// (this commit is the failing spec). The signatures are the contract the tests
// pin.

/// Parse a non-streaming `chat/completions` JSON response body into an
/// [`LlmResponse`] (text + tool calls + usage).
pub fn parse_chat_completion(_body: &str) -> Result<LlmResponse, CoreError> {
    Ok(LlmResponse::text("__stub__"))
}

/// Consume a successful non-streaming `chat/completions` response into an
/// [`LlmResponse`], emitting the full assistant text through `on_chunk` once.
pub async fn dispatch_non_streaming(
    _response: reqwest::Response,
    _cancellation: &CancellationToken,
    _on_chunk: ChunkCallback,
) -> Result<LlmResponse, CoreError> {
    Ok(LlmResponse::text("__stub__"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::Method::POST;
    use httpmock::MockServer;
    use std::sync::{Arc, Mutex};

    /// A non-streaming response body: assistant text, one complete tool call,
    /// and a usage object with cache activity.
    const NS_BODY: &str = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hello","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"rust\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":11,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}"#;

    // --- parse_chat_completion -------------------------------------------

    #[test]
    fn parse_text_tool_calls_and_usage() {
        let resp = parse_chat_completion(NS_BODY).expect("parse ok");
        assert_eq!(resp.text, "hello");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_1");
        assert_eq!(resp.tool_calls[0].name, "lookup");
        assert_eq!(resp.tool_calls[0].arguments, r#"{"q":"rust"}"#);
        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.input_tokens, Some(11));
        assert_eq!(usage.output_tokens, Some(4));
        assert_eq!(usage.cache_read_input_tokens, Some(2));
    }

    #[test]
    fn parse_text_only_has_no_tool_calls_or_usage() {
        let body = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"just text"}}]}"#;
        let resp = parse_chat_completion(body).expect("ok");
        assert_eq!(resp.text, "just text");
        assert!(resp.tool_calls.is_empty());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn parse_tool_call_missing_arguments_defaults_to_empty_object() {
        let body = r#"{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"noargs"}}]}}]}"#;
        let resp = parse_chat_completion(body).expect("ok");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].arguments, "{}");
    }

    #[test]
    fn parse_tool_call_only_turn_has_empty_text() {
        let body = r#"{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"go","arguments":"{}"}}]}}]}"#;
        let resp = parse_chat_completion(body).expect("ok");
        assert_eq!(resp.text, "");
        assert_eq!(resp.tool_calls.len(), 1);
    }

    #[test]
    fn parse_empty_choices_yields_empty_response() {
        let resp = parse_chat_completion(r#"{"choices":[]}"#).expect("ok");
        assert_eq!(resp.text, "");
        assert!(resp.tool_calls.is_empty());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn parse_malformed_body_is_err() {
        assert!(parse_chat_completion("not json at all").is_err());
    }

    // --- dispatch_non_streaming ------------------------------------------

    /// Make a real `reqwest::Response` against a mock returning `body`.
    async fn response_for(body: &str) -> reqwest::Response {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body(body);
        });
        // Keep the server alive by leaking it for the duration of the test; the
        // request completes before we return the response.
        let resp = reqwest::Client::new()
            .post(format!("{}/chat/completions", server.base_url()))
            .body("{}")
            .send()
            .await
            .expect("send ok");
        // The body is buffered by reqwest lazily, but httpmock has already sent
        // it; the server can drop after `send()` returns headers+body.
        std::mem::forget(server);
        resp
    }

    #[tokio::test]
    async fn dispatch_emits_full_text_and_parses_tool_calls() {
        let response = response_for(NS_BODY).await;
        let received = Arc::new(Mutex::new(String::new()));
        let rc = Arc::clone(&received);
        let out = dispatch_non_streaming(
            response,
            &CancellationToken::new(),
            Box::new(move |c| {
                rc.lock().expect("lock").push_str(&c);
                true
            }),
        )
        .await
        .expect("dispatch ok");
        assert_eq!(out.text, "hello");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "lookup");
        // The consumer/UI still received the full assistant text via on_chunk.
        assert_eq!(*received.lock().expect("lock"), "hello");
    }

    #[tokio::test]
    async fn dispatch_precancelled_returns_cancelled() {
        let response = response_for(NS_BODY).await;
        let token = CancellationToken::new();
        token.cancel();
        let out = dispatch_non_streaming(response, &token, Box::new(|_| true)).await;
        assert!(
            matches!(out, Err(CoreError::Cancelled)),
            "a tripped token must short-circuit the non-streaming body read"
        );
    }
}
