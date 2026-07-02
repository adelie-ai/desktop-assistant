//! Integration tests for the remote streamable-HTTP MCP client transport
//! (issue #455): drive `McpClient::connect_http` against an httpmock server,
//! covering the initialize handshake, a single-JSON `tools/list` reply, an SSE
//! `tools/call` reply, and bearer-token auth.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use desktop_assistant_mcp_client::McpClient;
use desktop_assistant_mcp_client::oauth::{
    InMemoryTokenStore, OAuthClient, TokenProvider, TokenSet,
};
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

/// Register handshake mocks that require a specific bearer token (so the OAuth
/// access token is proven to be attached to the initialize handshake too).
async fn mock_handshake_with_bearer(server: &MockServer, bearer: &str) {
    let auth = format!("Bearer {bearer}");
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .header("authorization", &auth)
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
                .header("authorization", &auth)
                .body_includes(r#""method":"notifications/initialized""#);
            then.status(202);
        })
        .await;
}

/// A valid, non-expired token so the provider serves it from cache without an
/// eager refresh. Uses the real clock because the transport calls the
/// zero-arg `current_token()`.
fn valid_token(access: &str) -> TokenSet {
    TokenSet {
        access_token: access.to_string(),
        refresh_token: Some("rt".to_string()),
        expires_at: Some(Utc::now() + chrono::Duration::hours(1)),
        token_type: "Bearer".to_string(),
        scope: None,
    }
}

#[tokio::test]
async fn http_transport_oauth_attaches_cached_token_without_refreshing() {
    let server = MockServer::start_async().await;
    mock_handshake_with_bearer(&server, "tok-A").await;
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .header("authorization", "Bearer tok-A")
                .body_includes(r#""method":"tools/list""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"list_events","inputSchema":{"type":"object"}}]}}"#);
        })
        .await;
    // The token endpoint must NOT be hit — the cached token is still valid.
    let token_mock = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .body(r#"{"access_token":"unexpected","expires_in":3600}"#);
        })
        .await;

    let oauth = OAuthClient::new("client-id", None, server.url("/token")).unwrap();
    let provider = TokenProvider::new(
        oauth,
        "acct@example.com",
        Arc::new(InMemoryTokenStore::default()),
        chrono::Duration::seconds(60),
        Some(valid_token("tok-A")),
    );

    let mut client = McpClient::connect_http_oauth(&server.url("/mcp"), Arc::new(provider))
        .await
        .expect("connect with a valid cached OAuth token");
    let tools = client.list_tools().await.expect("tools/list");
    assert_eq!(tools.len(), 1);
    token_mock.assert_calls_async(0).await;
}

#[tokio::test]
async fn http_transport_refreshes_and_retries_on_401() {
    let server = MockServer::start_async().await;
    // Handshake succeeds with the stale token, so the 401 is exercised on the
    // subsequent tools/list call rather than at connect time.
    mock_handshake_with_bearer(&server, "stale").await;

    // tools/list with the stale token is rejected.
    let stale_call = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .header("authorization", "Bearer stale")
                .body_includes(r#""method":"tools/list""#);
            then.status(401).body("token expired");
        })
        .await;
    // ...and accepted once refreshed.
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .header("authorization", "Bearer fresh")
                .body_includes(r#""method":"tools/list""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"send_email","inputSchema":{"type":"object"}}]}}"#);
        })
        .await;
    // The refresh mints the fresh token.
    let token_mock = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_includes("grant_type=refresh_token");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"fresh","expires_in":3600,"token_type":"Bearer"}"#);
        })
        .await;

    let oauth = OAuthClient::new("client-id", None, server.url("/token")).unwrap();
    let provider = TokenProvider::new(
        oauth,
        "acct@example.com",
        Arc::new(InMemoryTokenStore::default()),
        chrono::Duration::seconds(60),
        Some(valid_token("stale")),
    );

    let mut client = McpClient::connect_http_oauth(&server.url("/mcp"), Arc::new(provider))
        .await
        .expect("connect with the stale token (handshake accepts it)");
    let tools = client
        .list_tools()
        .await
        .expect("tools/list should succeed after a 401-triggered refresh");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "send_email");

    stale_call.assert_calls_async(1).await;
    token_mock.assert_calls_async(1).await;
}
