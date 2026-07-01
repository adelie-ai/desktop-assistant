//! Integration tests for the remote streamable-HTTP MCP client transport
//! (issue #455): drive `McpClient::connect_http` against an httpmock server,
//! covering the initialize handshake, a single-JSON `tools/list` reply, an SSE
//! `tools/call` reply, and bearer-token auth.

use std::time::Duration;

use desktop_assistant_mcp_client::McpClient;
use httpmock::prelude::*;
use serde_json::json;

/// Register the initialize handshake mocks (request + the `initialized`
/// notification) that every connection performs.
async fn mock_handshake(server: &MockServer) {
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"initialize""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}"#);
        })
        .await;
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"notifications/initialized""#);
            then.status(202);
        })
        .await;
}

#[tokio::test]
async fn http_transport_initialize_list_and_call() {
    let server = MockServer::start_async().await;
    mock_handshake(&server).await;

    // `tools/list` answered with a single JSON body.
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"tools/list""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"list_events","description":"List calendar events","inputSchema":{"type":"object"}}]}}"#);
        })
        .await;

    // `tools/call` answered with a `text/event-stream` (SSE) body — exercises
    // the SSE parser rather than the single-JSON path.
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"tools/call""#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"event created\"}]}}\n\n");
        })
        .await;

    let mut client = McpClient::connect_http(&server.url("/mcp"), None)
        .await
        .expect("connect_http should complete the initialize handshake");

    let tools = client.list_tools().await.expect("tools/list");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "list_events");

    let result = client
        .call_tool("list_events", json!({"calendarId": "primary"}))
        .await
        .expect("tools/call over SSE");
    assert_eq!(result, "event created");
}

#[tokio::test]
async fn http_transport_sends_bearer_token() {
    let server = MockServer::start_async().await;

    // Both handshake requests must carry the bearer token, or they won't match
    // and the connection fails — so a successful connect proves the header is
    // sent on every request.
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .header("authorization", "Bearer test-token-123")
                .body_includes(r#""method":"initialize""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{}}"#);
        })
        .await;
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .header("authorization", "Bearer test-token-123")
                .body_includes(r#""method":"notifications/initialized""#);
            then.status(202);
        })
        .await;

    let client = McpClient::connect_http_with_request_timeout(
        &server.url("/mcp"),
        Some("test-token-123".to_string()),
        Duration::from_secs(5),
    )
    .await;
    assert!(
        client.is_ok(),
        "connect_http must send the bearer token on every request; err: {:?}",
        client.err()
    );
}
