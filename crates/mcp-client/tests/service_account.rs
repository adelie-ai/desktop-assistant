//! Integration tests for resolving an MCP server's OAuth via a reusable
//! **service account** (issue #479). These drive the executor's real connect
//! path — resolver → `build_token_provider` → `connect_http_oauth` — against an
//! `httpmock` token endpoint + MCP endpoint, proving that a server referencing
//! an account authenticates with that account's token, and that two servers
//! sharing an account share one minted token.

use std::collections::HashMap;

use desktop_assistant_mcp_client::executor::{
    HttpTransportConfig, McpServerConfig, McpToolExecutor, ServiceAccount,
};
use httpmock::prelude::*;

/// Handshake mocks (`initialize` + `initialized`) requiring a specific bearer,
/// so a successful connect proves the account's access token was attached.
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

/// A `tools/list` reply carrying the bearer, so the connected server exposes a
/// tool the executor can register.
async fn mock_tools_list_with_bearer(server: &MockServer, bearer: &str) {
    let auth = format!("Bearer {bearer}");
    server
        .mock_async(move |when, then| {
            when.method(POST)
                .path("/mcp")
                .header("authorization", &auth)
                .body_includes(r#""method":"tools/list""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"do_thing","inputSchema":{"type":"object"}}]}}"#);
        })
        .await;
}

/// The account's token endpoint: exchanges the bootstrap refresh token for an
/// access token. Returns the mock so tests can assert how many times it's hit
/// (a shared account should refresh only once across its servers).
async fn mock_token_endpoint<'a>(server: &'a MockServer, access: &str) -> httpmock::Mock<'a> {
    server
        .mock_async(move |when, then| {
            when.method(POST)
                .path("/token")
                .body_includes("grant_type=refresh_token")
                .body_includes("refresh_token=rt-original");
            then.status(200)
                .header("content-type", "application/json")
                .body(format!(
                    r#"{{"access_token":"{access}","expires_in":3600,"token_type":"Bearer"}}"#
                ));
        })
        .await
}

/// A server that reaches `url` and authenticates via service account `work`.
fn account_server(name: &str, url: String, scopes: &[&str]) -> McpServerConfig {
    McpServerConfig {
        name: name.into(),
        command: String::new(),
        args: vec![],
        namespace: Some(name.into()),
        enabled: true,
        env: HashMap::new(),
        env_secrets: HashMap::new(),
        http: Some(HttpTransportConfig {
            url,
            auth_bearer_secret: None,
            oauth: None,
            oauth_account: Some("work".into()),
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
        }),
    }
}

/// The `work` service account, pointing its token endpoint at the mock server.
fn work_account(token_url: String, granted: &[&str]) -> ServiceAccount {
    ServiceAccount {
        id: "work".into(),
        display_name: "Work Google".into(),
        client_id: "client-abc".into(),
        client_secret_ref: None, // public PKCE client
        authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
        token_url,
        account: Some("acct@example.com".into()),
        refresh_token_ref: "acct_refresh".into(),
        granted_scopes: granted.iter().map(|s| s.to_string()).collect(),
    }
}

fn secrets_with_refresh() -> HashMap<String, String> {
    let mut s = HashMap::new();
    s.insert("acct_refresh".to_string(), "rt-original".to_string());
    s
}

#[tokio::test]
async fn server_referencing_account_connects_with_account_token() {
    let server = MockServer::start_async().await;
    let token_mock = mock_token_endpoint(&server, "acct-access").await;
    mock_handshake_with_bearer(&server, "acct-access").await;
    mock_tools_list_with_bearer(&server, "acct-access").await;

    let executor = McpToolExecutor::new(vec![account_server(
        "gmail",
        server.url("/mcp"),
        &["scope.read"],
    )]);
    let handle = executor.control_handle();
    handle.replace_secrets(secrets_with_refresh()).await;
    handle
        .replace_service_accounts(vec![work_account(server.url("/token"), &["scope.read"])])
        .await;

    executor.start().await.expect("start connects the server");

    // Connected via the account's token, and its tool is registered.
    let status = handle.status(Some("gmail")).await;
    assert_eq!(
        status[0].status, "running",
        "detail: {:?}",
        status[0].detail
    );
    let tools = executor.tools_by_service().await;
    assert!(
        tools
            .iter()
            .any(|(svc, tool)| svc == "gmail" && tool.contains("do_thing")),
        "gmail's tool should be registered: {tools:?}"
    );
    // The account's refresh token was exchanged for an access token.
    token_mock.assert_calls_async(1).await;
}

#[tokio::test]
async fn two_servers_share_one_account_token() {
    let server = MockServer::start_async().await;
    // The token endpoint must be hit only ONCE: the first server mints the
    // token, the second adopts it from the shared store (keyed by the account).
    let token_mock = mock_token_endpoint(&server, "shared-access").await;
    mock_handshake_with_bearer(&server, "shared-access").await;
    mock_tools_list_with_bearer(&server, "shared-access").await;

    let executor = McpToolExecutor::new(vec![
        account_server("gmail", server.url("/mcp"), &["scope.read"]),
        account_server("calendar", server.url("/mcp"), &["scope.read"]),
    ]);
    let handle = executor.control_handle();
    handle.replace_secrets(secrets_with_refresh()).await;
    handle
        .replace_service_accounts(vec![work_account(server.url("/token"), &["scope.read"])])
        .await;

    executor.start().await.expect("start connects both servers");

    let status = handle.status(None).await;
    for s in &status {
        assert_eq!(s.status, "running", "{} detail: {:?}", s.name, s.detail);
    }
    // One account → one refresh, shared across both servers.
    token_mock.assert_calls_async(1).await;
}

#[tokio::test]
async fn inline_oauth_server_still_connects() {
    // Regression: a server with an inline [http.oauth] block (no account
    // reference) resolves and connects exactly as before #479.
    use desktop_assistant_mcp_client::executor::OAuthServerConfig;

    let server = MockServer::start_async().await;
    let token_mock = mock_token_endpoint(&server, "inline-access").await;
    mock_handshake_with_bearer(&server, "inline-access").await;
    mock_tools_list_with_bearer(&server, "inline-access").await;

    let config = McpServerConfig {
        name: "legacy".into(),
        command: String::new(),
        args: vec![],
        namespace: Some("legacy".into()),
        enabled: true,
        env: HashMap::new(),
        env_secrets: HashMap::new(),
        http: Some(HttpTransportConfig {
            url: server.url("/mcp"),
            auth_bearer_secret: None,
            oauth: Some(OAuthServerConfig {
                client_id: "client-abc".into(),
                token_url: server.url("/token"),
                refresh_token_ref: "acct_refresh".into(),
                client_secret_ref: None,
                authorize_url: Some("https://accounts.google.com/o/oauth2/v2/auth".into()),
                scopes: vec!["scope.read".into()],
                account: Some("acct@example.com".into()),
                refresh_skew_seconds: None,
            }),
            oauth_account: None,
            scopes: vec![],
        }),
    };
    let executor = McpToolExecutor::new(vec![config]);
    let handle = executor.control_handle();
    handle.replace_secrets(secrets_with_refresh()).await;

    executor
        .start()
        .await
        .expect("inline oauth server connects");

    let status = handle.status(Some("legacy")).await;
    assert_eq!(
        status[0].status, "running",
        "detail: {:?}",
        status[0].detail
    );
    token_mock.assert_calls_async(1).await;
}
