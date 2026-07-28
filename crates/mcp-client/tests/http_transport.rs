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
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}"#);
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

// ----- MCP-Protocol-Version header (issue #932) -----
//
// Required on every HTTP request after initialization since spec revision
// 2025-06-18. A compliant server that receives no header is told to assume
// 2025-03-26 and to answer 400 for a version it does not support — so against a
// server that has retired 2025-03-26 (which is exactly what mcp-core just did)
// every request we make is entitled to fail.

/// Register a handshake whose `initialize` reply negotiates `version`, and
/// return a mock matching any request that carries `MCP-Protocol-Version`.
async fn mock_handshake_negotiating(server: &MockServer, version: &str) {
    let body = format!(
        r#"{{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":"{version}","capabilities":{{}},"serverInfo":{{"name":"mock","version":"0"}}}}}}"#
    );
    server
        .mock_async(move |when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"initialize""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(body);
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

/// Nothing is negotiated until the server replies, so the initialize request
/// itself must not carry the header.
#[tokio::test]
async fn http_initialize_omits_protocol_version_header() {
    let server = MockServer::start_async().await;
    let without_header = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"initialize""#)
                .header_missing("MCP-Protocol-Version");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}"#);
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

    McpClient::connect_http(&server.url("/mcp"), None)
        .await
        .expect("handshake must succeed");
    without_header.assert_calls_async(1).await;
}

/// The `notifications/initialized` POST is the first post-initialize message,
/// so it is already subject to the rule.
#[tokio::test]
async fn http_initialized_notification_carries_protocol_version_header() {
    let server = MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"initialize""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}"#);
        })
        .await;
    let notified = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"notifications/initialized""#)
                .header("MCP-Protocol-Version", "2025-11-25");
            then.status(202);
        })
        .await;

    McpClient::connect_http(&server.url("/mcp"), None)
        .await
        .expect("handshake must succeed");
    notified.assert_calls_async(1).await;
}

#[tokio::test]
async fn http_requests_carry_negotiated_protocol_version_header() {
    let server = MockServer::start_async().await;
    mock_handshake_negotiating(&server, "2025-11-25").await;
    let listed = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"tools/list""#)
                .header("MCP-Protocol-Version", "2025-11-25");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}"#);
        })
        .await;

    let mut client = McpClient::connect_http(&server.url("/mcp"), None)
        .await
        .expect("handshake must succeed");
    client.list_tools().await.expect("tools/list");
    listed.assert_calls_async(1).await;
}

/// The header must carry what the server agreed to, not what we asked for.
#[tokio::test]
async fn http_header_reflects_downgraded_version() {
    let server = MockServer::start_async().await;
    mock_handshake_negotiating(&server, "2025-06-18").await;
    let listed = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"tools/list""#)
                .header("MCP-Protocol-Version", "2025-06-18");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}"#);
        })
        .await;

    let mut client = McpClient::connect_http(&server.url("/mcp"), None)
        .await
        .expect("a 2025-06-18 server must still connect");
    client.list_tools().await.expect("tools/list");
    listed.assert_calls_async(1).await;
}

/// Adding the protocol header must not displace the session header.
#[tokio::test]
async fn http_session_and_protocol_headers_coexist() {
    let server = MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"initialize""#);
            then.status(200)
                .header("content-type", "application/json")
                .header("Mcp-Session-Id", "sess-42")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}"#);
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
    let listed = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"tools/list""#)
                .header("Mcp-Session-Id", "sess-42")
                .header("MCP-Protocol-Version", "2025-11-25");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}"#);
        })
        .await;

    let mut client = McpClient::connect_http(&server.url("/mcp"), None)
        .await
        .expect("handshake must succeed");
    client.list_tools().await.expect("tools/list");
    listed.assert_calls_async(1).await;
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
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}"#);
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
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}"#);
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
