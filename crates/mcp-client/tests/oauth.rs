//! Integration tests for the OAuth 2.0 token acquisition/refresh support that
//! backs remote (HTTP) MCP servers (issue #455 follow-up).
//!
//! These drive the real HTTP paths (`OAuthClient::refresh`/`exchange_code`, the
//! `TokenProvider` cache/refresh cycle, and the interactive loopback flow)
//! against an `httpmock` token endpoint. The pure/offline pieces (PKCE vectors,
//! authorize-URL construction, `TokenSet` expiry and redaction) are unit-tested
//! inline in `src/oauth.rs`.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use desktop_assistant_mcp_client::oauth::{
    InMemoryTokenStore, OAuthClient, OAuthError, TokenProvider, TokenSet, TokenStore,
    run_loopback_login,
};
use httpmock::prelude::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A fixed instant so token expiry math is deterministic (2026-07-02T00:00:00Z).
fn fixed_now() -> DateTime<Utc> {
    DateTime::from_timestamp(1_782_000_000, 0).expect("valid timestamp")
}

fn client_for(server: &MockServer) -> OAuthClient {
    OAuthClient::new("client-abc", None, server.url("/token")).expect("build oauth client")
}

// ---------------------------------------------------------------------------
// OAuthClient::refresh
// ---------------------------------------------------------------------------

#[tokio::test]
async fn refresh_success_populates_tokens_and_expiry() {
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_includes("grant_type=refresh_token")
                .body_includes("refresh_token=rt-original")
                .body_includes("client_id=client-abc");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"at-new","expires_in":3600,"token_type":"Bearer","scope":"calendar"}"#);
        })
        .await;

    let client = client_for(&server);
    let now = fixed_now();
    let tokens = client
        .refresh_at("rt-original", now)
        .await
        .expect("refresh should succeed");

    mock.assert_async().await;
    assert_eq!(tokens.access_token, "at-new");
    assert_eq!(tokens.token_type, "Bearer");
    assert_eq!(tokens.scope.as_deref(), Some("calendar"));
    // expires_at = now + expires_in.
    assert_eq!(
        tokens.expires_at,
        Some(now + chrono::Duration::seconds(3600))
    );
    // Google omits refresh_token on refresh; the client reports None and the
    // provider is responsible for carrying the old one forward.
    assert_eq!(tokens.refresh_token, None);
}

#[tokio::test]
async fn refresh_invalid_grant_maps_to_reauth_error() {
    let server = MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(400)
                .header("content-type", "application/json")
                .body(r#"{"error":"invalid_grant","error_description":"Token has been expired or revoked."}"#);
        })
        .await;

    let err = client_for(&server)
        .refresh_at("rt-dead", fixed_now())
        .await
        .expect_err("invalid_grant must be an error");
    assert!(
        matches!(err, OAuthError::InvalidGrant(_)),
        "expected InvalidGrant, got {err:?}"
    );
}

#[tokio::test]
async fn refresh_server_error_is_endpoint_error() {
    let server = MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(503).body("upstream down");
        })
        .await;

    let err = client_for(&server)
        .refresh_at("rt", fixed_now())
        .await
        .expect_err("5xx must be an error");
    match err {
        OAuthError::Endpoint { status, .. } => assert_eq!(status, 503),
        other => panic!("expected Endpoint error, got {other:?}"),
    }
}

#[tokio::test]
async fn refresh_malformed_success_body_is_error() {
    let server = MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            // 200 OK but no access_token field.
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"token_type":"Bearer"}"#);
        })
        .await;

    let err = client_for(&server)
        .refresh_at("rt", fixed_now())
        .await
        .expect_err("missing access_token must be an error");
    assert!(
        matches!(err, OAuthError::Malformed(_)),
        "expected Malformed, got {err:?}"
    );
}

#[tokio::test]
async fn exchange_code_returns_refresh_token() {
    let server = MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_includes("grant_type=authorization_code")
                .body_includes("code=auth-code-xyz")
                .body_includes("code_verifier=the-verifier")
                .body_includes("redirect_uri=");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"at-1","refresh_token":"rt-1","expires_in":3599,"token_type":"Bearer"}"#);
        })
        .await;

    let tokens = client_for(&server)
        .exchange_code_at(
            "auth-code-xyz",
            "the-verifier",
            "http://127.0.0.1:9999",
            fixed_now(),
        )
        .await
        .expect("code exchange should succeed");
    assert_eq!(tokens.access_token, "at-1");
    assert_eq!(tokens.refresh_token.as_deref(), Some("rt-1"));
}

// ---------------------------------------------------------------------------
// TokenProvider caching / refresh cycle
// ---------------------------------------------------------------------------

/// Seed a provider whose only credential is a refresh token (the daemon's
/// startup state: refresh token from secrets.toml, no access token yet).
fn bootstrap_provider(server: &MockServer, store: Arc<dyn TokenStore>) -> TokenProvider {
    TokenProvider::bootstrap_from_refresh_token(
        client_for(server),
        "acct@example.com",
        store,
        chrono::Duration::seconds(60),
        "rt-original".to_string(),
    )
}

#[tokio::test]
async fn provider_refreshes_once_then_serves_from_cache() {
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"cached-at","expires_in":3600,"token_type":"Bearer"}"#);
        })
        .await;

    let provider = bootstrap_provider(&server, Arc::new(InMemoryTokenStore::default()));
    let now = fixed_now();

    // First call has no access token → must refresh.
    assert_eq!(provider.current_token_at(now).await.unwrap(), "cached-at");
    // Second call, still well within validity → served from cache, no new hit.
    assert_eq!(
        provider
            .current_token_at(now + chrono::Duration::seconds(60))
            .await
            .unwrap(),
        "cached-at"
    );

    mock.assert_calls_async(1).await;
}

#[tokio::test]
async fn provider_refreshes_again_after_expiry_within_skew() {
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"at","expires_in":3600,"token_type":"Bearer"}"#);
        })
        .await;

    let provider = bootstrap_provider(&server, Arc::new(InMemoryTokenStore::default()));
    let now = fixed_now();
    provider.current_token_at(now).await.unwrap();

    // Jump to 30s before expiry: inside the 60s skew ⇒ treated as expired ⇒
    // a second refresh happens.
    let near_expiry = now + chrono::Duration::seconds(3600 - 30);
    provider.current_token_at(near_expiry).await.unwrap();

    mock.assert_calls_async(2).await;
}

#[tokio::test]
async fn provider_without_refresh_token_reports_needs_login() {
    let server = MockServer::start_async().await;
    // No mock needed: we must fail before any HTTP call.
    let provider = TokenProvider::new(
        client_for(&server),
        "acct@example.com",
        Arc::new(InMemoryTokenStore::default()),
        chrono::Duration::seconds(60),
        None,
    );
    let err = provider
        .current_token_at(fixed_now())
        .await
        .expect_err("no refresh token ⇒ error");
    assert!(
        matches!(err, OAuthError::NoRefreshToken),
        "expected NoRefreshToken, got {err:?}"
    );
}

#[tokio::test]
async fn provider_preserves_refresh_token_across_refreshes() {
    let server = MockServer::start_async().await;
    // The token endpoint requires the *original* refresh token every time — so
    // if the provider lost it after the first refresh (Google omits it in the
    // response), the second refresh would 400 and this test would fail.
    let mock = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_includes("refresh_token=rt-original");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"at","expires_in":10,"token_type":"Bearer"}"#);
        })
        .await;

    let store = Arc::new(InMemoryTokenStore::default());
    let provider = bootstrap_provider(&server, store.clone());
    let now = fixed_now();
    provider.current_token_at(now).await.unwrap();
    // 10s token, jump well past expiry → second refresh, still using rt-original.
    provider
        .current_token_at(now + chrono::Duration::seconds(120))
        .await
        .unwrap();

    mock.assert_calls_async(2).await;
    // The store also carries the refresh token forward for the next process.
    let persisted = store.load("acct@example.com").unwrap().unwrap();
    assert_eq!(persisted.refresh_token.as_deref(), Some("rt-original"));
}

#[tokio::test]
async fn provider_loads_valid_token_from_store_without_refreshing() {
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .body(r#"{"access_token":"should-not-be-used"}"#);
        })
        .await;

    // Store already has a valid access token (e.g. persisted by a prior run).
    let now = fixed_now();
    let store = Arc::new(InMemoryTokenStore::default());
    store
        .save(
            "acct@example.com",
            &TokenSet {
                access_token: "still-good".into(),
                refresh_token: Some("rt".into()),
                expires_at: Some(now + chrono::Duration::seconds(3600)),
                token_type: "Bearer".into(),
                scope: None,
            },
        )
        .unwrap();

    let provider = TokenProvider::new(
        client_for(&server),
        "acct@example.com",
        store,
        chrono::Duration::seconds(60),
        None,
    );
    assert_eq!(provider.current_token_at(now).await.unwrap(), "still-good");
    mock.assert_calls_async(0).await;
}

#[tokio::test]
async fn provider_adopts_cached_token_matching_bootstrap_refresh_token() {
    // Daemon-restart case: bootstrapped from secrets.toml (rt-original) AND a
    // store (keyring) that cached a still-valid access token minted from the
    // same refresh token ⇒ adopt the cache, skip the startup refresh.
    let server = MockServer::start_async().await;
    let token_mock = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .body(r#"{"access_token":"should-not-be-used"}"#);
        })
        .await;

    let now = fixed_now();
    let store = Arc::new(InMemoryTokenStore::default());
    store
        .save(
            "acct@example.com",
            &TokenSet {
                access_token: "cached-and-valid".into(),
                refresh_token: Some("rt-original".into()),
                expires_at: Some(now + chrono::Duration::seconds(3600)),
                token_type: "Bearer".into(),
                scope: None,
            },
        )
        .unwrap();

    let provider = bootstrap_provider(&server, store);
    assert_eq!(
        provider.current_token_at(now).await.unwrap(),
        "cached-and-valid"
    );
    token_mock.assert_calls_async(0).await;
}

#[tokio::test]
async fn provider_ignores_stale_store_after_relogin() {
    // Re-login case: secrets.toml now has rt-new, but the store still caches a
    // (valid-looking) token for the OLD refresh token ⇒ ignore the stale cache
    // and refresh with rt-new. The token endpoint asserts rt-new is used.
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_includes("refresh_token=rt-new");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"fresh-from-rt-new","expires_in":3600,"token_type":"Bearer"}"#);
        })
        .await;

    let now = fixed_now();
    let store = Arc::new(InMemoryTokenStore::default());
    store
        .save(
            "acct@example.com",
            &TokenSet {
                access_token: "stale-access".into(),
                refresh_token: Some("rt-old".into()),
                expires_at: Some(now + chrono::Duration::seconds(3600)),
                token_type: "Bearer".into(),
                scope: None,
            },
        )
        .unwrap();

    let provider = TokenProvider::bootstrap_from_refresh_token(
        client_for(&server),
        "acct@example.com",
        store.clone(),
        chrono::Duration::seconds(60),
        "rt-new".to_string(),
    );
    assert_eq!(
        provider.current_token_at(now).await.unwrap(),
        "fresh-from-rt-new"
    );
    mock.assert_calls_async(1).await;
    // The store is overwritten with the token for the current refresh token.
    let persisted = store.load("acct@example.com").unwrap().unwrap();
    assert_eq!(persisted.access_token, "fresh-from-rt-new");
    assert_eq!(persisted.refresh_token.as_deref(), Some("rt-new"));
}

#[tokio::test]
async fn force_refresh_bypasses_cache() {
    let server = MockServer::start_async().await;
    let mock = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"at","expires_in":3600,"token_type":"Bearer"}"#);
        })
        .await;

    let provider = bootstrap_provider(&server, Arc::new(InMemoryTokenStore::default()));
    let now = fixed_now();
    provider.current_token_at(now).await.unwrap(); // hit 1
    provider.force_refresh().await.unwrap(); // hit 2 despite valid cache
    mock.assert_calls_async(2).await;
}

// ---------------------------------------------------------------------------
// Interactive loopback flow
// ---------------------------------------------------------------------------

/// Drive the callback like a browser would: parse the authorize URL the flow
/// hands to `open_browser`, then connect to its `redirect_uri` with the given
/// `code`/`state` query. Returns nothing; the flow itself does the exchange.
fn browser_that_calls_back(
    code: &'static str,
    state_override: Option<&'static str>,
) -> impl FnOnce(&str) -> Result<(), OAuthError> {
    move |authorize_url: &str| {
        let parsed = url::Url::parse(authorize_url)
            .map_err(|e| OAuthError::Flow(format!("test browser could not parse url: {e}")))?;
        let mut redirect = None;
        let mut state = None;
        for (k, v) in parsed.query_pairs() {
            match k.as_ref() {
                "redirect_uri" => redirect = Some(v.into_owned()),
                "state" => state = Some(v.into_owned()),
                _ => {}
            }
        }
        let redirect = redirect.expect("authorize url must carry redirect_uri");
        let state = state_override
            .map(str::to_string)
            .unwrap_or_else(|| state.expect("authorize url must carry state"));
        tokio::spawn(async move {
            let target = url::Url::parse(&redirect).unwrap();
            let host = target.host_str().unwrap().to_string();
            let port = target.port().unwrap();
            // Small retry loop: the flow binds before calling open_browser, so a
            // connect should succeed almost immediately.
            let mut stream = None;
            for _ in 0..50 {
                match tokio::net::TcpStream::connect((host.as_str(), port)).await {
                    Ok(s) => {
                        stream = Some(s);
                        break;
                    }
                    Err(_) => tokio::task::yield_now().await,
                }
            }
            let mut stream = stream.expect("connect to loopback listener");
            let req =
                format!("GET /?code={code}&state={state} HTTP/1.1\r\nHost: localhost\r\n\r\n");
            stream.write_all(req.as_bytes()).await.unwrap();
            // Read (and discard) the response so the server-side write completes.
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf).await;
        });
        Ok(())
    }
}

#[tokio::test]
async fn loopback_flow_happy_path_returns_tokens() {
    let server = MockServer::start_async().await;
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_includes("grant_type=authorization_code")
                .body_includes("code=good-code");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"at-loop","refresh_token":"rt-loop","expires_in":3600,"token_type":"Bearer"}"#);
        })
        .await;

    let client = client_for(&server);
    let scopes = vec!["https://www.googleapis.com/auth/calendar".to_string()];
    let tokens = run_loopback_login(
        &client,
        "https://accounts.google.com/o/oauth2/v2/auth",
        &scopes,
        "127.0.0.1",
        Duration::from_secs(10),
        browser_that_calls_back("good-code", None),
    )
    .await
    .expect("loopback flow should complete");

    assert_eq!(tokens.access_token, "at-loop");
    assert_eq!(tokens.refresh_token.as_deref(), Some("rt-loop"));
}

#[tokio::test]
async fn loopback_flow_rejects_state_mismatch() {
    let server = MockServer::start_async().await;
    // Token endpoint must never be reached on a CSRF/state mismatch.
    let mock = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200).body("{}");
        })
        .await;

    let client = client_for(&server);
    let err = run_loopback_login(
        &client,
        "https://accounts.google.com/o/oauth2/v2/auth",
        &["scope".to_string()],
        "127.0.0.1",
        Duration::from_secs(10),
        browser_that_calls_back("good-code", Some("attacker-state")),
    )
    .await
    .expect_err("state mismatch must abort");
    assert!(
        matches!(err, OAuthError::StateMismatch),
        "expected StateMismatch, got {err:?}"
    );
    mock.assert_calls_async(0).await;
}

#[tokio::test]
async fn loopback_flow_surfaces_provider_denial() {
    let server = MockServer::start_async().await;
    let client = client_for(&server);
    // A browser that returns `error=access_denied` instead of a code.
    let deny_browser = move |authorize_url: &str| -> Result<(), OAuthError> {
        let parsed = url::Url::parse(authorize_url).unwrap();
        let redirect = parsed
            .query_pairs()
            .find(|(k, _)| k == "redirect_uri")
            .map(|(_, v)| v.into_owned())
            .unwrap();
        tokio::spawn(async move {
            let target = url::Url::parse(&redirect).unwrap();
            let host = target.host_str().unwrap().to_string();
            let port = target.port().unwrap();
            let mut stream = None;
            for _ in 0..50 {
                if let Ok(s) = tokio::net::TcpStream::connect((host.as_str(), port)).await {
                    stream = Some(s);
                    break;
                }
                tokio::task::yield_now().await;
            }
            let mut stream = stream.unwrap();
            let req = "GET /?error=access_denied HTTP/1.1\r\nHost: localhost\r\n\r\n";
            stream.write_all(req.as_bytes()).await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf).await;
        });
        Ok(())
    };

    let err = run_loopback_login(
        &client,
        "https://accounts.google.com/o/oauth2/v2/auth",
        &["scope".to_string()],
        "127.0.0.1",
        Duration::from_secs(10),
        deny_browser,
    )
    .await
    .expect_err("provider denial must surface");
    assert!(
        matches!(err, OAuthError::Authorization(_)),
        "expected Authorization error, got {err:?}"
    );
}
