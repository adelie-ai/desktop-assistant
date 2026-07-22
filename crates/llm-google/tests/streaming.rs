//! Integration tests for `GoogleClient::stream_completion` against an
//! `httpmock` Gemini endpoint and a mock `TokenProvider` — no real network,
//! no real GCP. Covers SSE parsing (text / whole functionCall / usage incl.
//! cache), URL routing under `MODEL_OVERRIDE`, both auth surfaces, reasoning,
//! error classification, the safety decline, cancellation, malformed-frame
//! tolerance, and callback-abort state preservation.

use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolDefinition};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, ReasoningConfig, with_cancellation_token, with_model_override,
};
use desktop_assistant_llm_google::{AuthMode, GoogleClient, StaticTokenProvider};
use httpmock::prelude::*;
use tokio_util::sync::CancellationToken;

const STREAM_PATH: &str = "/v1/projects/test-proj/locations/us-central1/publishers/google/models/gemini-2.5-pro:streamGenerateContent";

/// A Vertex-mode client pointed at `server`, authenticated with a static
/// bearer so no real GCP token exchange runs.
fn vertex_client(server: &MockServer) -> GoogleClient {
    GoogleClient::new(String::new())
        .with_base_url(server.url(""))
        .with_project(Some("test-proj".into()))
        .with_location("us-central1")
        .with_token_provider(Arc::new(StaticTokenProvider::new("test-token")))
}

fn user(text: &str) -> Vec<Message> {
    vec![Message::new(Role::User, text)]
}

fn noop_chunk() -> ChunkCallback {
    Box::new(|_| true)
}

#[tokio::test]
async fn streams_text_and_reports_usage_including_cache() {
    let server = MockServer::start();
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\" world\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":2,\"cachedContentTokenCount\":4}}\n\n",
    );
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse);
    });

    let resp = vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect("stream ok");

    assert_eq!(resp.text, "Hello world");
    let usage = resp.usage.expect("usage present");
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(usage.cache_read_input_tokens, Some(4));
}

#[tokio::test]
async fn accumulates_whole_function_call() {
    let server = MockServer::start();
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"NYC\"}}}]}}],\"usageMetadata\":{\"promptTokenCount\":8,\"candidatesTokenCount\":5}}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n",
    );
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse);
    });

    let resp = vertex_client(&server)
        .stream_completion(
            user("weather?"),
            &[],
            ReasoningConfig::default(),
            noop_chunk(),
        )
        .await
        .expect("stream ok");

    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&resp.tool_calls[0].arguments).unwrap();
    assert_eq!(args, serde_json::json!({"city": "NYC"}));
    assert!(resp.usage.is_some());
}

#[tokio::test]
async fn model_override_routes_the_url_path_segment() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST).path(
            "/v1/projects/test-proj/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent",
        );
        then.status(200)
            .header("content-type", "text/event-stream")
            .body("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n");
    });

    let client = vertex_client(&server);
    with_model_override("gemini-2.5-flash".into(), async {
        client
            .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
            .await
            .expect("stream ok");
    })
    .await;
    m.assert_calls(1);
}

#[tokio::test]
async fn vertex_sends_bearer_token_in_header() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST)
            .path(STREAM_PATH)
            .header("authorization", "Bearer test-token");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n");
    });

    vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect("stream ok");
    m.assert_calls(1);
}

#[tokio::test]
async fn api_key_mode_uses_header_and_v1beta_path() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST)
            .path("/v1beta/models/gemini-2.5-pro:streamGenerateContent")
            .header("x-goog-api-key", "my-api-key");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n");
    });

    let client = GoogleClient::new("my-api-key".into())
        .with_auth_mode(AuthMode::ApiKey)
        .with_base_url(server.url(""));
    client
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect("stream ok");
    m.assert_calls(1);
}

#[tokio::test]
async fn reasoning_thinking_budget_reaches_the_wire() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST)
            .path(STREAM_PATH)
            .body_includes("\"thinkingConfig\"")
            .body_includes("\"thinkingBudget\":2048");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n");
    });

    vertex_client(&server)
        .stream_completion(
            user("think"),
            &[],
            ReasoningConfig::with_thinking_budget(2048),
            noop_chunk(),
        )
        .await
        .expect("stream ok");
    m.assert_calls(1);
}

#[tokio::test]
async fn tool_schema_reaches_the_wire_as_function_declarations() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(POST)
            .path(STREAM_PATH)
            .body_includes("\"functionDeclarations\"")
            .body_includes("\"get_weather\"");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n");
    });

    let tools = vec![ToolDefinition::new(
        "get_weather",
        "Look up the weather",
        serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    )];
    vertex_client(&server)
        .stream_completion(user("hi"), &tools, ReasoningConfig::default(), noop_chunk())
        .await
        .expect("stream ok");
    m.assert_calls(1);
}

#[tokio::test]
async fn safety_block_is_an_informative_non_error_decline() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(200)
            .header("content-type", "text/event-stream")
            .body("data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\",\"safetyRatings\":[{\"category\":\"HARM_CATEGORY_DANGEROUS_CONTENT\",\"blocked\":true}]}}\n\n");
    });

    let err = vertex_client(&server)
        .stream_completion(user("bad"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect_err("safety block must surface a decline");

    let CoreError::Llm(detail) = err else {
        panic!("safety decline must be a plain Llm error, got {err:?}");
    };
    assert!(
        detail.contains("HARM_CATEGORY_DANGEROUS_CONTENT"),
        "names category: {detail}"
    );
    assert!(
        detail.to_ascii_lowercase().contains("safety"),
        "explains the refusal: {detail}"
    );
    // The flagged user content ("bad") must never be echoed back.
    assert!(
        !detail.contains("bad"),
        "decline must not echo the flagged prompt: {detail}"
    );
}

#[tokio::test]
async fn http_429_resource_exhausted_is_rate_limited() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(429).header("retry-after", "23").body(
            r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded for requests per minute"}}"#,
        );
    });

    let err = vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect_err("429 must fail");
    match err {
        CoreError::RateLimited { retry_after, .. } => {
            assert_eq!(retry_after, Some(Duration::from_secs(23)));
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[tokio::test]
async fn http_400_context_overflow_is_context_overflow() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(400).body(
            r#"{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"The input token count (1290000) exceeds the maximum number of tokens allowed (1048576)."}}"#,
        );
    });

    let err = vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect_err("overflow must fail");
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

#[tokio::test]
async fn http_401_is_generic_llm_without_leaking_the_token() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(401).body(
            r#"{"error":{"code":401,"status":"UNAUTHENTICATED","message":"Request had invalid authentication credentials."}}"#,
        );
    });

    let err = vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect_err("401 must fail");
    let CoreError::Llm(detail) = err else {
        panic!("expected Llm, got {err:?}");
    };
    assert!(
        !detail.contains("test-token"),
        "bearer token leaked: {detail}"
    );
    assert!(detail.to_ascii_lowercase().contains("credential"));
}

#[tokio::test]
async fn http_500_is_rate_limited() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(500)
            .body(r#"{"error":{"code":500,"status":"INTERNAL","message":"Internal error."}}"#);
    });

    let err = vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect_err("500 must fail");
    assert!(matches!(err, CoreError::RateLimited { .. }), "got {err:?}");
}

#[tokio::test]
async fn tolerates_a_malformed_middle_frame() {
    let server = MockServer::start();
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"foo\"}]}}]}\n\n",
        "data: {not valid json at all]\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"bar\"}]},\"finishReason\":\"STOP\"}]}\n\n",
    );
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse);
    });

    let resp = vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect("malformed middle frame must not fail the turn");
    assert_eq!(resp.text, "foobar");
}

#[tokio::test]
async fn callback_abort_preserves_accumulated_tool_call_and_usage() {
    let server = MockServer::start();
    // A whole functionCall + usage lands first, then a text chunk the callback
    // rejects. The tool call and usage seen before the abort must survive.
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"NYC\"}}}]}}],\"usageMetadata\":{\"promptTokenCount\":11,\"candidatesTokenCount\":0}}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\" more\"}]},\"finishReason\":\"STOP\"}]}\n\n",
    );
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse);
    });

    // Abort on the first text chunk.
    let on_chunk: ChunkCallback = Box::new(|_| false);
    let resp = vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), on_chunk)
        .await
        .expect("aborted stream still returns Ok");

    assert_eq!(resp.tool_calls.len(), 1, "tool call before abort preserved");
    assert_eq!(resp.tool_calls[0].name, "get_weather");
    assert!(resp.usage.is_some(), "usage before abort preserved");
    assert_eq!(
        resp.text, "partial",
        "only the first chunk was appended before abort"
    );
}

#[tokio::test]
async fn cancellation_mid_stream_aborts_promptly() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(200)
            .header("content-type", "text/event-stream")
            .delay(Duration::from_secs(5))
            .body("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]}}]}\n\n");
    });

    let client = vertex_client(&server);
    let token = CancellationToken::new();
    let handle = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        handle.cancel();
    });

    let start = std::time::Instant::now();
    let result = with_cancellation_token(token, async {
        client
            .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
            .await
    })
    .await;
    let elapsed = start.elapsed();

    assert!(
        matches!(result, Err(CoreError::Cancelled)),
        "got {result:?}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "should abort promptly; took {elapsed:?}"
    );
}

#[tokio::test]
async fn missing_usage_metadata_yields_no_usage() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path(STREAM_PATH);
        then.status(200)
            .header("content-type", "text/event-stream")
            .body("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}]}\n\n");
    });

    let resp = vertex_client(&server)
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect("ok");
    assert_eq!(resp.text, "hi");
    assert!(resp.usage.is_none(), "no usageMetadata -> no usage");
}

#[tokio::test]
async fn vertex_without_project_fails_with_actionable_message() {
    // No project set: the URL cannot be composed, so the turn fails before any
    // request with a message that names the missing field.
    let client = GoogleClient::new(String::new())
        .with_token_provider(Arc::new(StaticTokenProvider::new("t")));
    let err = client
        .stream_completion(user("hi"), &[], ReasoningConfig::default(), noop_chunk())
        .await
        .expect_err("missing project must fail");
    let CoreError::Llm(detail) = err else {
        panic!("expected Llm, got {err:?}");
    };
    assert!(
        detail.contains("project"),
        "names the missing field: {detail}"
    );
}
