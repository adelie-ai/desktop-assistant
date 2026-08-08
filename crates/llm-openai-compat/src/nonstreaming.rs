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
//! lives here is the shared mechanism:
//!
//! - the response wire types ([`ChatCompletion`] and friends) and the JSON ->
//!   [`LlmResponse`] conversion ([`parse_chat_completion`], reusing
//!   [`parse_usage`] and [`build_response`](desktop_assistant_llm_http::build_response));
//! - the connect-race send ([`send_chat_request`]), shared by the streaming and
//!   non-streaming paths so the cancellation / connect-timeout handling and the
//!   error classification live in one place;
//! - the callback-emitting body read ([`dispatch_non_streaming`]);
//! - [`StreamingDispatchError`], which hands the (unconsumed) callback back on
//!   the tools-unsupported arm so the connector can retry without rebuilding it.

use std::time::Duration;

use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolCall;
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmResponse, record_provider_request_id};
use desktop_assistant_llm_http::build_response;

use crate::usage::parse_usage;

/// Outcome of a streaming dispatch attempt.
///
/// The tools-unsupported arm carries the *unconsumed* [`ChunkCallback`] back so
/// the connector can retry non-streaming without forcing a `Clone` bound on the
/// callback (it is a boxed `dyn FnMut`). Mirrors the Bedrock connector's
/// `StreamingDispatchError` (#67).
pub enum StreamingDispatchError {
    /// The routed backend rejected tools-in-streaming; retry non-streaming.
    ToolsUnsupported {
        /// The callback the streaming attempt never consumed (the error is
        /// detected at the HTTP-status check, before the stream body is read).
        on_chunk: ChunkCallback,
        /// The provider's raw detail, for logging.
        detail: String,
    },
    /// Any other error; surfaced as-is.
    Other(CoreError),
}

/// A non-streaming `chat/completions` response envelope (only the fields this
/// crate reads).
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletion {
    /// The choices array; typically length 1 for `n=1`.
    #[serde(default)]
    pub choices: Vec<ChatCompletionChoice>,
    /// The token-usage object, handed to [`parse_usage`]; `None` when absent.
    #[serde(default)]
    pub usage: Option<serde_json::Value>,
}

/// One element of a [`ChatCompletion`]'s `choices` array.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionChoice {
    /// The full (non-delta) assistant message for this choice.
    pub message: ChatResponseMessage,
    /// The finish reason, if reported. Read for completeness; not acted on.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// The full assistant `message` of a non-streaming choice.
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ChatResponseMessage {
    /// Assistant text; absent on a tool-call-only turn.
    #[serde(default)]
    pub content: Option<String>,
    /// The complete tool calls requested by this turn (each carries whole
    /// `arguments`, unlike the streamed fragments).
    #[serde(default)]
    pub tool_calls: Vec<ChatResponseToolCall>,
}

/// A complete tool call inside a [`ChatResponseMessage`].
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ChatResponseToolCall {
    /// The provider-assigned tool-call id.
    #[serde(default)]
    pub id: Option<String>,
    /// The called function's name and (whole) JSON arguments string.
    pub function: ChatResponseFunction,
}

/// The `function` object of a [`ChatResponseToolCall`].
#[derive(Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ChatResponseFunction {
    /// The tool/function name.
    #[serde(default)]
    pub name: Option<String>,
    /// The call arguments as a JSON string.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Parse a non-streaming `chat/completions` JSON response body into an
/// [`LlmResponse`] (text + tool calls + usage).
///
/// Reads `choices[0].message`: `content` -> text, `tool_calls[]` -> domain
/// [`ToolCall`]s (each with its whole `arguments`; a call with no `arguments`
/// defaults to `"{}"`), and the top-level `usage` object via [`parse_usage`].
/// An empty `choices` array yields an empty text-only response. Mirrors what
/// the streaming path builds so the two dispatch paths are interchangeable.
pub fn parse_chat_completion(body: &str) -> Result<LlmResponse, CoreError> {
    let parsed: ChatCompletion = serde_json::from_str(body)
        .map_err(|e| CoreError::Llm(format!("failed to parse chat completion response: {e}")))?;

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    if let Some(choice) = parsed.choices.into_iter().next() {
        if let Some(content) = choice.message.content {
            text.push_str(&content);
        }
        for tc in choice.message.tool_calls {
            tool_calls.push(ToolCall::new(
                tc.id.unwrap_or_default(),
                tc.function.name.unwrap_or_default(),
                tc.function.arguments.unwrap_or_else(|| "{}".to_string()),
            ));
        }
    }

    let usage = parsed.usage.as_ref().and_then(parse_usage);
    Ok(build_response(text, tool_calls, usage))
}

/// Send a prepared `chat/completions` request, racing the connect handshake
/// against cancellation and `connect_timeout`, and classifying any non-2xx via
/// `classify`.
///
/// On success returns the [`reqwest::Response`] for the caller to consume --
/// `bytes_stream()` on the streaming path, or [`dispatch_non_streaming`] on the
/// non-streaming path. Shared by both so cancellation, the stall budget, and
/// the error-body classification are not duplicated. `stall_detail` is the
/// message for a connect timeout.
pub async fn send_chat_request(
    request: reqwest::RequestBuilder,
    cancellation: &CancellationToken,
    connect_timeout: Duration,
    stall_detail: &str,
    classify: impl FnOnce(StatusCode, &HeaderMap, &str) -> CoreError,
) -> Result<reqwest::Response, CoreError> {
    // Cooperative cancellation: bail before dialing out.
    if cancellation.is_cancelled() {
        return Err(CoreError::Cancelled);
    }

    let send_fut = request.send();
    let response = tokio::select! {
        _ = cancellation.cancelled() => return Err(CoreError::Cancelled),
        _ = tokio::time::sleep(connect_timeout) => {
            tracing::error!(
                timeout_s = connect_timeout.as_secs(),
                detail = stall_detail,
                "chat request send() timed out (no response headers)"
            );
            return Err(CoreError::Llm(stall_detail.to_string()));
        }
        r = send_fut => r.map_err(|e| CoreError::Llm(format!("HTTP request failed: {e}")))?,
    };

    if !response.status().is_success() {
        let status = response.status();
        let headers = response.headers().clone();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "unable to read body".into());
        return Err(classify(status, &headers, &body));
    }

    // Capture the routed backend's own request id onto the open `llm.call`
    // span (#1152). This one site covers every caller of `send_chat_request`
    // - Azure and OpenRouter, each streaming and non-streaming - so a single
    // header read here reaches all four dispatch paths.
    if let Some(request_id) = response
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
    {
        record_provider_request_id(request_id);
    }

    Ok(response)
}

/// Consume a successful non-streaming `chat/completions` response into an
/// [`LlmResponse`], emitting the full assistant text through `on_chunk` once.
///
/// The non-streaming path has no deltas, so the whole assistant text is emitted
/// in a single `on_chunk` call -- the consumer/UI still receives the prose. The
/// response is fully built regardless of the callback's abort return (there is
/// nothing further to suppress). The body read is raced against `cancellation`,
/// and a token already tripped on entry short-circuits before the read.
pub async fn dispatch_non_streaming(
    response: reqwest::Response,
    cancellation: &CancellationToken,
    mut on_chunk: ChunkCallback,
) -> Result<LlmResponse, CoreError> {
    if cancellation.is_cancelled() {
        return Err(CoreError::Cancelled);
    }

    let body = tokio::select! {
        _ = cancellation.cancelled() => return Err(CoreError::Cancelled),
        b = response.text() => b.map_err(|e| {
            CoreError::Llm(format!("failed to read chat completion body: {e}"))
        })?,
    };

    let llm_response = parse_chat_completion(&body)?;
    if !llm_response.text.is_empty() {
        // Emit the full text once so a text-consuming UI still receives it; the
        // response is already fully built, so the abort return is moot here.
        let _ = on_chunk(llm_response.text.clone());
    }
    Ok(llm_response)
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
        let body =
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"just text"}}]}"#;
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
        let resp = reqwest::Client::new()
            .post(format!("{}/chat/completions", server.base_url()))
            .body("{}")
            .send()
            .await
            .expect("send ok");
        // The mock has already sent headers+body; keep the server alive so the
        // socket is not torn down before the response body is read.
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

    // --- Provider request id capture (#1152) ------------------------------
    //
    // These drive `send_chat_request` directly: it is the one function this
    // module exports that reads the response, and Azure and OpenRouter each
    // reach it from both their streaming and non-streaming dispatch (proven
    // by reading both crates' call sites - see the report for #1152), so a
    // single header read there covers all four paths without editing either
    // connector crate.
    //
    // A minimal `tracing` layer that reads a named span's fields back in
    // process, modelled on `SpanCapture` in
    // `crates/core/tests/turn_telemetry.rs` but pared down to what these
    // tests need: one span, read once the call under test has finished.

    #[derive(Clone, Default)]
    struct RequestIdSpanCapture(
        Arc<
            std::sync::Mutex<
                std::collections::HashMap<String, std::collections::HashMap<String, String>>,
            >,
        >,
    );

    struct RequestIdFieldVisitor<'a>(&'a mut std::collections::HashMap<String, String>);

    impl tracing::field::Visit for RequestIdFieldVisitor<'_> {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    impl<S> tracing_subscriber::layer::Layer<S> for RequestIdSpanCapture
    where
        S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut fields = std::collections::HashMap::new();
            attrs.record(&mut RequestIdFieldVisitor(&mut fields));
            self.0
                .lock()
                .expect("capture lock")
                .insert(attrs.metadata().name().to_string(), fields);
        }

        fn on_record(
            &self,
            id: &tracing::span::Id,
            values: &tracing::span::Record<'_>,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let Some(span) = ctx.span(id) else {
                return;
            };
            let mut all = self.0.lock().expect("capture lock");
            let fields = all.entry(span.name().to_string()).or_default();
            values.record(&mut RequestIdFieldVisitor(fields));
        }
    }

    impl RequestIdSpanCapture {
        /// The fields recorded on the span named `name`, or empty if that
        /// span was never opened.
        fn fields_of(&self, name: &str) -> std::collections::HashMap<String, String> {
            self.0
                .lock()
                .expect("capture lock")
                .get(name)
                .cloned()
                .unwrap_or_default()
        }
    }

    /// Build a capturing subscriber and an `llm.call` span carrying the one
    /// field this test needs, declared the way core's `llm_span` and
    /// `aux_llm_span` declare it: `provider_request_id` as
    /// `tracing::field::Empty`. Core's spans carry more than this - the
    /// token counts among them - and none of it is what a connector records. A connector's `record_provider_request_id`
    /// call needs that declaration to have a field to land on - `tracing`
    /// silently drops a `record` for a field the span never declared, which
    /// is the trap this mirrors core's shape to avoid.
    ///
    /// The `DefaultGuard` must stay alive for the whole call under test; drop
    /// it only after the fields have been read back.
    fn request_id_capture() -> (
        RequestIdSpanCapture,
        tracing::subscriber::DefaultGuard,
        tracing::Span,
    ) {
        use tracing_subscriber::layer::SubscriberExt;

        let capture = RequestIdSpanCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());
        let guard = tracing::subscriber::set_default(subscriber);
        let span = tracing::info_span!("llm.call", provider_request_id = tracing::field::Empty);
        (capture, guard, span)
    }

    /// A `send_chat_request` call against `server`, driven inside a captured
    /// `llm.call` span, returning the fields recorded on it. `server` must
    /// outlive this call - the caller keeps its own `MockServer` binding in
    /// scope, the same way `response_for` above requires.
    async fn send_and_capture(server: &MockServer) -> std::collections::HashMap<String, String> {
        use tracing::Instrument;

        let request = reqwest::Client::new()
            .post(format!("{}/chat/completions", server.base_url()))
            .body("{}");
        let (capture, _guard, span) = request_id_capture();

        let _ = async {
            send_chat_request(
                request,
                &CancellationToken::new(),
                Duration::from_secs(5),
                "stalled",
                |status, _headers, body| CoreError::Llm(format!("HTTP {status}: {body}")),
            )
            .await
        }
        .instrument(span)
        .await;

        capture.fields_of("llm.call")
    }

    #[tokio::test]
    async fn openai_compat_call_records_the_provider_request_id() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .header("x-request-id", "req_test_compat_321")
                .body(NS_BODY);
        });

        let fields = send_and_capture(&server).await;
        assert_eq!(
            fields.get("provider_request_id").map(String::as_str),
            Some("req_test_compat_321"),
            "the shared dispatch must record the routed backend's x-request-id onto the span"
        );
    }

    #[tokio::test]
    async fn openai_compat_call_without_a_request_id_header_records_nothing() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body(NS_BODY);
        });

        let fields = send_and_capture(&server).await;
        assert!(
            !fields.contains_key("provider_request_id"),
            "an absent header must leave the field empty, not record an empty string: {fields:?}"
        );
    }

    #[tokio::test]
    async fn openai_compat_call_neutralises_a_control_character_in_the_request_id() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                // A literal newline cannot ride in an HTTP header value at
                // all - `http::HeaderValue` rejects the byte outright, so
                // that injection is stopped by the transport itself, not by
                // this module. A tab is the one C0 control character the
                // transport does carry (`HeaderValue` allows byte 9), so it
                // is what proves the dispatch routes the header through
                // `Safe::name` rather than passing it through raw.
                .header("x-request-id", "req\ttest")
                .body(NS_BODY);
        });

        let fields = send_and_capture(&server).await;
        let recorded = fields
            .get("provider_request_id")
            .expect("a present header, even a deceptive one, must still record a value");
        assert_eq!(
            recorded, "req\u{fffd}test",
            "Safe::name must replace the control character with U+FFFD, not pass it through"
        );
    }
}
