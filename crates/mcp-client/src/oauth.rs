//! OAuth 2.0 token acquisition and refresh for remote (HTTP) MCP servers
//! (issue #455 follow-up).
//!
//! The remote-HTTP transport (see `lib.rs`) can authenticate either with a
//! static bearer token from `secrets.toml` or with an OAuth 2.0 [`TokenProvider`]
//! that keeps a short-lived access token fresh from a long-lived refresh token.
//! That split matches how the daemon runs:
//!
//! * **Acquisition** (interactive, one-time per account) — [`run_loopback_login`]
//!   drives the installed-app loopback + PKCE flow: open the browser, capture
//!   the authorization code on a localhost listener, exchange it for tokens.
//!   The daemon is headless, so this runs from a CLI subcommand at a terminal.
//! * **Refresh** (non-interactive, ongoing) — [`TokenProvider`] exchanges the
//!   stored refresh token for access tokens on demand, caching each until it is
//!   near expiry. This is what the transport calls on every request.
//!
//! Everything here is transport- and vendor-agnostic; Google is simply the
//! first target (`https://oauth2.googleapis.com/token`).

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use chrono::{DateTime, Utc};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt as _, AsyncWriteExt};
use tokio::sync::Mutex;
use url::Url;

/// Errors from the OAuth token/authorization machinery.
#[derive(Debug, thiserror::Error)]
pub enum OAuthError {
    #[error("OAuth configuration error: {0}")]
    Config(String),

    #[error("OAuth HTTP request failed: {0}")]
    Http(String),

    #[error("OAuth token endpoint returned HTTP {status}: {body}")]
    Endpoint { status: u16, body: String },

    /// The refresh token was rejected (`invalid_grant`) — it was revoked or has
    /// expired (e.g. the 7-day cap on a personal/"Testing" consent screen). The
    /// only fix is to re-run the interactive login.
    #[error("refresh token is no longer valid ({0}); re-run the OAuth login for this account")]
    InvalidGrant(String),

    #[error("malformed OAuth token response: {0}")]
    Malformed(String),

    #[error("no refresh token available; run the OAuth login for this account first")]
    NoRefreshToken,

    #[error("authorization was denied or failed: {0}")]
    Authorization(String),

    #[error("OAuth state mismatch on redirect: possible CSRF; login aborted")]
    StateMismatch,

    #[error("token store error: {0}")]
    Store(String),

    #[error("OAuth login flow error: {0}")]
    Flow(String),
}

/// A set of OAuth tokens plus a derived absolute expiry.
///
/// `Debug` is implemented by hand to redact the token material — a `TokenSet`
/// must never print secrets into logs.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSet {
    /// The short-lived bearer token sent to the resource server.
    pub access_token: String,
    /// The long-lived token used to mint new access tokens. Google omits this
    /// on a *refresh* response, so callers must carry the previous one forward
    /// (see [`TokenProvider`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Absolute expiry (`issued_at + expires_in`). `None` means the endpoint
    /// gave no `expires_in`; such a token is never *proactively* refreshed —
    /// only a `401` forces it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    /// Token type; almost always `Bearer`.
    #[serde(default)]
    pub token_type: String,
    /// Space-delimited granted scopes, if the endpoint reported them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

impl TokenSet {
    /// True if this token is at or past expiry once `skew` is subtracted — i.e.
    /// refresh a little *before* the hard deadline to absorb clock skew and
    /// in-flight latency. A token with no known expiry is never proactively
    /// expired.
    pub fn is_expired_at(&self, now: DateTime<Utc>, skew: chrono::Duration) -> bool {
        match self.expires_at {
            None => false,
            Some(exp) => now + skew >= exp,
        }
    }
}

impl fmt::Debug for TokenSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenSet")
            .field("access_token", &Redacted(self.access_token.len()))
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|t| Redacted(t.len())),
            )
            .field("expires_at", &self.expires_at)
            .field("token_type", &self.token_type)
            .field("scope", &self.scope)
            .finish()
    }
}

/// A `Debug` stand-in that reveals only a length, never the bytes.
struct Redacted(usize);

impl fmt::Debug for Redacted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<redacted {} bytes>", self.0)
    }
}

/// The raw JSON shape of an OAuth token-endpoint success response.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

impl TokenResponse {
    fn into_token_set(self, now: DateTime<Utc>) -> TokenSet {
        TokenSet {
            expires_at: self
                .expires_in
                .map(|secs| now + chrono::Duration::seconds(secs)),
            access_token: self.access_token,
            refresh_token: self.refresh_token,
            token_type: self.token_type.unwrap_or_else(|| "Bearer".to_string()),
            scope: self.scope,
        }
    }
}

/// PKCE (RFC 7636) verifier/challenge pair. Public clients must use PKCE so an
/// intercepted authorization code cannot be redeemed without the verifier.
pub struct Pkce {
    /// High-entropy secret kept by the client and sent on the code exchange.
    pub verifier: String,
    /// `BASE64URL(SHA256(verifier))`, sent on the authorization request.
    pub challenge: String,
    /// Challenge method identifier; always `S256`.
    pub method: &'static str,
}

impl Pkce {
    /// Generate a fresh verifier from 32 bytes of CSPRNG entropy and derive its
    /// S256 challenge.
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        Self::from_verifier(base64url(&bytes))
    }

    /// Derive the challenge for a specific verifier (used to test against the
    /// RFC 7636 vector).
    pub fn from_verifier(verifier: impl Into<String>) -> Self {
        let verifier = verifier.into();
        let digest = Sha256::digest(verifier.as_bytes());
        Self {
            verifier,
            challenge: base64url(digest.as_ref()),
            method: "S256",
        }
    }
}

/// URL-safe base64 without padding — the encoding RFC 7636 mandates for PKCE
/// and a safe default for opaque random tokens (`state`).
fn base64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// An opaque, high-entropy value for the `state` CSRF parameter.
fn random_state() -> String {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    base64url(&bytes)
}

/// Reject endpoints that would send credentials in the clear. HTTPS is required
/// except for loopback (`localhost`/`127.0.0.1`/`::1`), which is safe and lets
/// tests and local IdPs use plain HTTP.
fn validate_endpoint_url(raw: &str) -> Result<(), OAuthError> {
    let url =
        Url::parse(raw).map_err(|e| OAuthError::Config(format!("invalid URL '{raw}': {e}")))?;
    match url.scheme() {
        "https" => Ok(()),
        "http" if matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1")) => Ok(()),
        "http" => Err(OAuthError::Config(format!(
            "refusing insecure OAuth endpoint (http:// is only allowed for loopback): {raw}"
        ))),
        other => Err(OAuthError::Config(format!(
            "unsupported URL scheme '{other}' in {raw}"
        ))),
    }
}

/// Build an OAuth 2.0 authorization-request URL (authorization-code grant with
/// PKCE). `access_type=offline` + `prompt=consent` ask Google to return a
/// refresh token even on re-authorization; both are harmless to other IdPs.
pub fn build_authorize_url(
    authorize_url: &str,
    client_id: &str,
    redirect_uri: &str,
    scopes: &[String],
    state: &str,
    pkce: &Pkce,
) -> Result<String, OAuthError> {
    validate_endpoint_url(authorize_url)?;
    let mut url = Url::parse(authorize_url)
        .map_err(|e| OAuthError::Config(format!("invalid authorize_url '{authorize_url}': {e}")))?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", &scopes.join(" "))
        .append_pair("state", state)
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", pkce.method)
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");
    Ok(url.to_string())
}

/// A configured OAuth 2.0 client: it knows the token endpoint and how to
/// identify itself, and performs the `refresh_token` and `authorization_code`
/// grant exchanges.
pub struct OAuthClient {
    http: reqwest::Client,
    client_id: String,
    client_secret: Option<String>,
    token_url: String,
}

impl OAuthClient {
    /// Build a client. `client_secret` is optional (public/PKCE clients omit
    /// it). The token endpoint must be HTTPS (or loopback).
    pub fn new(
        client_id: impl Into<String>,
        client_secret: Option<String>,
        token_url: impl Into<String>,
    ) -> Result<Self, OAuthError> {
        let token_url = token_url.into();
        validate_endpoint_url(&token_url)?;
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| OAuthError::Config(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            http,
            client_id: client_id.into(),
            client_secret,
            token_url,
        })
    }

    /// The configured client identifier (used to build authorization URLs).
    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    /// Exchange a refresh token for a fresh access token, stamping expiry
    /// against the current clock.
    pub async fn refresh(&self, refresh_token: &str) -> Result<TokenSet, OAuthError> {
        self.refresh_at(refresh_token, Utc::now()).await
    }

    /// [`Self::refresh`] with an explicit `now` for deterministic expiry math.
    pub async fn refresh_at(
        &self,
        refresh_token: &str,
        now: DateTime<Utc>,
    ) -> Result<TokenSet, OAuthError> {
        let mut form: Vec<(&str, String)> = vec![
            ("grant_type", "refresh_token".to_string()),
            ("refresh_token", refresh_token.to_string()),
            ("client_id", self.client_id.clone()),
        ];
        if let Some(secret) = &self.client_secret {
            form.push(("client_secret", secret.clone()));
        }
        self.post_token(&form, now).await
    }

    /// Exchange an authorization code (with its PKCE verifier) for tokens.
    pub async fn exchange_code(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
    ) -> Result<TokenSet, OAuthError> {
        self.exchange_code_at(code, verifier, redirect_uri, Utc::now())
            .await
    }

    /// [`Self::exchange_code`] with an explicit `now`.
    pub async fn exchange_code_at(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
        now: DateTime<Utc>,
    ) -> Result<TokenSet, OAuthError> {
        let mut form: Vec<(&str, String)> = vec![
            ("grant_type", "authorization_code".to_string()),
            ("code", code.to_string()),
            ("redirect_uri", redirect_uri.to_string()),
            ("client_id", self.client_id.clone()),
            ("code_verifier", verifier.to_string()),
        ];
        if let Some(secret) = &self.client_secret {
            form.push(("client_secret", secret.clone()));
        }
        self.post_token(&form, now).await
    }

    /// POST a form-encoded grant to the token endpoint and parse the result,
    /// classifying failures (`invalid_grant` vs. other endpoint errors vs.
    /// malformed success bodies).
    async fn post_token(
        &self,
        form: &[(&str, String)],
        now: DateTime<Utc>,
    ) -> Result<TokenSet, OAuthError> {
        // Build the form body in a block so the (non-`Send`) serializer is
        // dropped before the `await` below — otherwise the whole future would
        // be `!Send` and couldn't run on the multithreaded runtime.
        let body = {
            let mut serializer = url::form_urlencoded::Serializer::new(String::new());
            for (key, value) in form {
                serializer.append_pair(key, value);
            }
            serializer.finish()
        };
        let response = self
            .http
            .post(&self.token_url)
            .header("content-type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| OAuthError::Http(e.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| OAuthError::Http(e.to_string()))?;

        if !status.is_success() {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body)
                && value.get("error").and_then(serde_json::Value::as_str) == Some("invalid_grant")
            {
                return Err(OAuthError::InvalidGrant(snippet(&body)));
            }
            return Err(OAuthError::Endpoint {
                status: status.as_u16(),
                body: snippet(&body),
            });
        }

        let parsed: TokenResponse =
            serde_json::from_str(&body).map_err(|e| OAuthError::Malformed(e.to_string()))?;
        Ok(parsed.into_token_set(now))
    }
}

/// Truncate an endpoint body for inclusion in an error, so a huge or hostile
/// response can't blow up a log line.
fn snippet(body: &str) -> String {
    body.chars().take(500).collect()
}

/// Persistence for a [`TokenSet`], keyed by an account identifier (e.g. email).
/// The daemon can back this with the keyring; tests use [`InMemoryTokenStore`].
pub trait TokenStore: Send + Sync {
    fn load(&self, key: &str) -> Result<Option<TokenSet>, OAuthError>;
    fn save(&self, key: &str, token: &TokenSet) -> Result<(), OAuthError>;
}

/// A process-local [`TokenStore`]. Refresh tokens survive only for the life of
/// the process; the durable refresh token comes from config/secrets on start.
#[derive(Default)]
pub struct InMemoryTokenStore {
    inner: std::sync::Mutex<HashMap<String, TokenSet>>,
}

impl TokenStore for InMemoryTokenStore {
    fn load(&self, key: &str) -> Result<Option<TokenSet>, OAuthError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| OAuthError::Store("in-memory token store lock poisoned".into()))?;
        Ok(guard.get(key).cloned())
    }

    fn save(&self, key: &str, token: &TokenSet) -> Result<(), OAuthError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| OAuthError::Store("in-memory token store lock poisoned".into()))?;
        guard.insert(key.to_string(), token.clone());
        Ok(())
    }
}

struct ProviderState {
    /// Whether we've consulted the store yet (to distinguish "not loaded" from
    /// "loaded, nothing there").
    loaded_from_store: bool,
    tokens: Option<TokenSet>,
}

/// Keeps a live OAuth access token for one account: caches the current token,
/// refreshes it when it's near expiry (or on demand), and persists the result.
///
/// All access goes through an async mutex, so concurrent requests on the same
/// account can never stampede the token endpoint with parallel refreshes.
pub struct TokenProvider {
    client: OAuthClient,
    account_key: String,
    store: Arc<dyn TokenStore>,
    skew: chrono::Duration,
    state: Mutex<ProviderState>,
}

impl TokenProvider {
    /// Construct a provider. `seed` optionally supplies an initial [`TokenSet`]
    /// (e.g. a bootstrap refresh token); when `None`, the store is consulted on
    /// first use.
    pub fn new(
        client: OAuthClient,
        account_key: impl Into<String>,
        store: Arc<dyn TokenStore>,
        skew: chrono::Duration,
        seed: Option<TokenSet>,
    ) -> Self {
        Self {
            client,
            account_key: account_key.into(),
            store,
            skew,
            state: Mutex::new(ProviderState {
                loaded_from_store: false,
                tokens: seed,
            }),
        }
    }

    /// Construct a provider seeded with just a refresh token — the daemon's
    /// startup state (refresh token from `secrets.toml`, no access token yet).
    pub fn bootstrap_from_refresh_token(
        client: OAuthClient,
        account_key: impl Into<String>,
        store: Arc<dyn TokenStore>,
        skew: chrono::Duration,
        refresh_token: String,
    ) -> Self {
        let seed = TokenSet {
            access_token: String::new(),
            refresh_token: Some(refresh_token),
            expires_at: None,
            token_type: "Bearer".to_string(),
            scope: None,
        };
        Self::new(client, account_key, store, skew, Some(seed))
    }

    /// Return a valid access token, refreshing first if the cached one is
    /// missing or near expiry.
    pub async fn current_token(&self) -> Result<String, OAuthError> {
        self.get(Utc::now(), false).await
    }

    /// [`Self::current_token`] with an explicit `now`.
    pub async fn current_token_at(&self, now: DateTime<Utc>) -> Result<String, OAuthError> {
        self.get(now, false).await
    }

    /// Force a refresh regardless of cache validity (used when the resource
    /// server rejects the current token with `401`).
    pub async fn force_refresh(&self) -> Result<String, OAuthError> {
        self.get(Utc::now(), true).await
    }

    async fn get(&self, now: DateTime<Utc>, force: bool) -> Result<String, OAuthError> {
        let mut state = self.state.lock().await;

        // Lazily consult the store once. The store is a best-effort cache: a
        // load failure is logged and ignored (we fall back to the seed), never
        // fatal to the request. `reconcile_stored` decides whether the cached
        // token is still authoritative for our bootstrap refresh token.
        if !state.loaded_from_store {
            state.loaded_from_store = true;
            match self.store.load(&self.account_key) {
                Ok(Some(stored)) => {
                    let seed = state.tokens.take();
                    state.tokens = Some(reconcile_stored(seed, stored));
                }
                Ok(None) => {}
                Err(error) => tracing::warn!(
                    "token store load failed for account '{}': {error}; continuing without cache",
                    self.account_key
                ),
            }
        }

        let needs_refresh = force
            || match &state.tokens {
                None => true,
                Some(t) => t.access_token.is_empty() || t.is_expired_at(now, self.skew),
            };

        if needs_refresh {
            let refresh_token = state
                .tokens
                .as_ref()
                .and_then(|t| t.refresh_token.clone())
                .ok_or(OAuthError::NoRefreshToken)?;
            let refreshed = self.client.refresh_at(&refresh_token, now).await?;
            let merged = merge_tokens(state.tokens.take(), refreshed);
            // Persist is best-effort too: a store write failure keeps the token
            // in memory (it just won't survive a restart), never fails the call.
            if let Err(error) = self.store.save(&self.account_key, &merged) {
                tracing::warn!(
                    "token store save failed for account '{}': {error}; token kept in memory only",
                    self.account_key
                );
            }
            state.tokens = Some(merged);
        }

        state
            .tokens
            .as_ref()
            .map(|t| t.access_token.clone())
            .ok_or(OAuthError::NoRefreshToken)
    }
}

/// Carry a refresh token forward across a refresh: OAuth servers (Google
/// included) usually omit `refresh_token` in the refresh response, so keep the
/// previous one unless a new one was explicitly issued (rotation).
fn merge_tokens(old: Option<TokenSet>, mut new: TokenSet) -> TokenSet {
    if new.refresh_token.is_none() {
        new.refresh_token = old.and_then(|t| t.refresh_token);
    }
    new
}

/// Reconcile a token loaded from the store with the bootstrap `seed`.
///
/// The store is authoritative when its cached token was minted from the *same*
/// refresh token we were bootstrapped with — then its (possibly still-valid)
/// access token lets the daemon skip a refresh across restarts, and any rotated
/// refresh token it holds is preserved. If the refresh tokens differ, the cache
/// is stale (e.g. the user re-ran the login, writing a new refresh token to
/// secrets.toml) and we fall back to the seed. With no seed (a provider built
/// purely from the store), the store is trusted.
fn reconcile_stored(seed: Option<TokenSet>, stored: TokenSet) -> TokenSet {
    let seed_refresh = seed.as_ref().and_then(|s| s.refresh_token.clone());
    match (&seed_refresh, &stored.refresh_token) {
        (Some(seed_rt), Some(stored_rt)) if seed_rt != stored_rt => {
            seed.expect("seed is Some when its refresh token was read")
        }
        _ => merge_tokens(seed, stored),
    }
}

/// Run the installed-app loopback + PKCE authorization flow and return the
/// acquired tokens (including a refresh token).
///
/// Binds an ephemeral `127.0.0.1` port as the redirect target, hands the
/// authorization URL to `open_browser`, waits (up to `accept_timeout`) for the
/// provider to redirect back with an authorization code, validates the `state`,
/// and exchanges the code. `open_browser` is injected so callers (and tests)
/// control how the URL is surfaced.
pub async fn run_loopback_login<F>(
    oauth: &OAuthClient,
    authorize_url: &str,
    scopes: &[String],
    bind_host: &str,
    accept_timeout: Duration,
    open_browser: F,
) -> Result<TokenSet, OAuthError>
where
    F: FnOnce(&str) -> Result<(), OAuthError>,
{
    let listener = tokio::net::TcpListener::bind((bind_host, 0))
        .await
        .map_err(|e| OAuthError::Flow(format!("failed to bind loopback listener: {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| OAuthError::Flow(format!("failed to read loopback address: {e}")))?
        .port();
    let redirect_uri = format!("http://{bind_host}:{port}");

    let pkce = Pkce::generate();
    let state = random_state();
    let url = build_authorize_url(
        authorize_url,
        oauth.client_id(),
        &redirect_uri,
        scopes,
        &state,
        &pkce,
    )?;

    open_browser(&url)?;

    let (mut socket, _peer) = tokio::time::timeout(accept_timeout, listener.accept())
        .await
        .map_err(|_| {
            OAuthError::Flow(format!(
                "timed out after {accept_timeout:?} waiting for the OAuth redirect"
            ))
        })?
        .map_err(|e| OAuthError::Flow(format!("failed to accept redirect connection: {e}")))?;

    // Read just the request line: `GET /path?query HTTP/1.1`. Bound the read so
    // a local process racing the browser to the loopback port can't make us
    // buffer an unbounded line.
    let request_line = {
        const MAX_REQUEST_LINE: u64 = 64 * 1024;
        let mut reader = tokio::io::BufReader::new((&mut socket).take(MAX_REQUEST_LINE));
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .await
            .map_err(|e| OAuthError::Flow(format!("failed to read redirect request: {e}")))?;
        line
    };

    let target = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| OAuthError::Flow("malformed redirect request line".to_string()))?;
    let parsed = Url::parse(&format!("http://localhost{target}"))
        .map_err(|e| OAuthError::Flow(format!("failed to parse redirect target: {e}")))?;

    let mut code = None;
    let mut got_state = None;
    let mut denied = None;
    for (key, value) in parsed.query_pairs() {
        match key.as_ref() {
            "code" => code = Some(value.into_owned()),
            "state" => got_state = Some(value.into_owned()),
            "error" => denied = Some(value.into_owned()),
            _ => {}
        }
    }

    let outcome: Result<(), OAuthError> = if let Some(error) = denied {
        Err(OAuthError::Authorization(error))
    } else if got_state.as_deref() != Some(state.as_str()) {
        Err(OAuthError::StateMismatch)
    } else if code.is_none() {
        Err(OAuthError::Authorization(
            "redirect carried no authorization code".to_string(),
        ))
    } else {
        Ok(())
    };

    // Always answer the browser so the user sees a clean page, then act.
    let page = if outcome.is_ok() {
        "<html><body><h2>Authorization complete.</h2>\
         <p>You may close this window and return to the terminal.</p></body></html>"
    } else {
        "<html><body><h2>Authorization failed.</h2>\
         <p>Return to the terminal for details.</p></body></html>"
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        page.len(),
        page
    );
    let _ = socket.write_all(response.as_bytes()).await;
    let _ = socket.shutdown().await;

    outcome?;
    let code = code.expect("code is Some when outcome is Ok");
    oauth
        .exchange_code(&code, &pkce.verifier, &redirect_uri)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(secs, 0).expect("valid timestamp")
    }

    #[test]
    fn pkce_matches_rfc7636_vector() {
        // Appendix B of RFC 7636.
        let pkce = Pkce::from_verifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk");
        assert_eq!(
            pkce.challenge,
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
        assert_eq!(pkce.method, "S256");
    }

    #[test]
    fn pkce_generate_is_url_safe_and_high_entropy() {
        let a = Pkce::generate();
        let b = Pkce::generate();
        assert_ne!(a.verifier, b.verifier, "verifiers must be random");
        // base64url(32 bytes) has no padding and only url-safe chars.
        assert!(
            !a.verifier.contains('=') && !a.verifier.contains('+') && !a.verifier.contains('/')
        );
        assert!(
            !a.challenge.contains('=') && !a.challenge.contains('+') && !a.challenge.contains('/')
        );
        // A verifier derived from its own challenge is deterministic.
        assert_eq!(
            Pkce::from_verifier(a.verifier.clone()).challenge,
            a.challenge
        );
    }

    #[test]
    fn token_set_expiry_semantics() {
        let base = at(1_000_000);
        let token = TokenSet {
            access_token: "x".into(),
            refresh_token: None,
            expires_at: Some(base + chrono::Duration::seconds(3600)),
            token_type: "Bearer".into(),
            scope: None,
        };
        let skew = chrono::Duration::seconds(60);
        // Well before expiry.
        assert!(!token.is_expired_at(base, skew));
        // 30s before the deadline but inside the 60s skew ⇒ treated as expired.
        assert!(token.is_expired_at(base + chrono::Duration::seconds(3600 - 30), skew));
        // Past the deadline.
        assert!(token.is_expired_at(base + chrono::Duration::seconds(4000), skew));

        // A token with no expiry is never proactively expired.
        let no_expiry = TokenSet {
            expires_at: None,
            ..token
        };
        assert!(!no_expiry.is_expired_at(base + chrono::Duration::seconds(100_000), skew));
    }

    #[test]
    fn token_set_debug_redacts_secrets() {
        let token = TokenSet {
            access_token: "SUPER-SECRET-ACCESS".into(),
            refresh_token: Some("SUPER-SECRET-REFRESH".into()),
            expires_at: None,
            token_type: "Bearer".into(),
            scope: Some("calendar".into()),
        };
        let debug = format!("{token:?}");
        assert!(
            !debug.contains("SUPER-SECRET-ACCESS"),
            "access token leaked: {debug}"
        );
        assert!(
            !debug.contains("SUPER-SECRET-REFRESH"),
            "refresh token leaked: {debug}"
        );
        assert!(debug.contains("redacted"));
        // Non-secret metadata is still visible for diagnostics.
        assert!(debug.contains("Bearer"));
        assert!(debug.contains("calendar"));
    }

    #[test]
    fn authorize_url_carries_required_params() {
        let pkce = Pkce::from_verifier("verifier-value");
        let scopes = vec![
            "https://www.googleapis.com/auth/gmail.readonly".to_string(),
            "https://www.googleapis.com/auth/calendar".to_string(),
        ];
        let url = build_authorize_url(
            "https://accounts.google.com/o/oauth2/v2/auth",
            "client-123",
            "http://127.0.0.1:8080",
            &scopes,
            "state-xyz",
            &pkce,
        )
        .unwrap();

        let parsed = Url::parse(&url).unwrap();
        let params: HashMap<_, _> = parsed.query_pairs().into_owned().collect();
        assert_eq!(params["response_type"], "code");
        assert_eq!(params["client_id"], "client-123");
        assert_eq!(params["redirect_uri"], "http://127.0.0.1:8080");
        assert_eq!(params["code_challenge"], pkce.challenge);
        assert_eq!(params["code_challenge_method"], "S256");
        assert_eq!(params["state"], "state-xyz");
        assert_eq!(params["access_type"], "offline");
        // Scopes are space-delimited within the single `scope` param.
        assert_eq!(
            params["scope"],
            "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/calendar"
        );
    }

    #[test]
    fn endpoint_url_validation() {
        assert!(validate_endpoint_url("https://oauth2.googleapis.com/token").is_ok());
        assert!(validate_endpoint_url("http://127.0.0.1:9000/token").is_ok());
        assert!(validate_endpoint_url("http://localhost/token").is_ok());
        // Plain HTTP to a remote host would send the client secret in the clear.
        assert!(validate_endpoint_url("http://evil.example.com/token").is_err());
        assert!(validate_endpoint_url("ftp://example.com/token").is_err());
        assert!(validate_endpoint_url("not a url").is_err());
    }

    #[test]
    fn oauth_client_rejects_insecure_token_url() {
        // `matches!` (not `unwrap_err`) so the test doesn't force `Debug` on
        // `OAuthClient`, which holds the client secret and must not be printable.
        assert!(matches!(
            OAuthClient::new("id", None, "http://evil.example.com/token"),
            Err(OAuthError::Config(_))
        ));
    }

    #[test]
    fn merge_tokens_carries_refresh_forward_but_honours_rotation() {
        let old = Some(TokenSet {
            access_token: "old".into(),
            refresh_token: Some("rt-old".into()),
            expires_at: None,
            token_type: "Bearer".into(),
            scope: None,
        });
        // Refresh response without a refresh_token ⇒ keep the old one.
        let refreshed = TokenSet {
            access_token: "new".into(),
            refresh_token: None,
            expires_at: None,
            token_type: "Bearer".into(),
            scope: None,
        };
        let merged = merge_tokens(old.clone(), refreshed);
        assert_eq!(merged.access_token, "new");
        assert_eq!(merged.refresh_token.as_deref(), Some("rt-old"));

        // Refresh response WITH a new refresh_token ⇒ rotation wins.
        let rotated = TokenSet {
            access_token: "new".into(),
            refresh_token: Some("rt-new".into()),
            expires_at: None,
            token_type: "Bearer".into(),
            scope: None,
        };
        assert_eq!(
            merge_tokens(old, rotated).refresh_token.as_deref(),
            Some("rt-new")
        );
    }

    fn token(access: &str, refresh: &str) -> TokenSet {
        TokenSet {
            access_token: access.into(),
            refresh_token: Some(refresh.into()),
            expires_at: None,
            token_type: "Bearer".into(),
            scope: None,
        }
    }

    #[test]
    fn reconcile_adopts_store_when_refresh_token_matches() {
        // Bootstrap seed (empty access) + a cached token minted from the same
        // refresh token ⇒ adopt the cache (its access token skips a refresh).
        let seed = TokenSet {
            access_token: String::new(),
            ..token("", "rt-1")
        };
        let stored = token("cached-access", "rt-1");
        let result = reconcile_stored(Some(seed), stored);
        assert_eq!(result.access_token, "cached-access");
        assert_eq!(result.refresh_token.as_deref(), Some("rt-1"));
    }

    #[test]
    fn reconcile_prefers_seed_when_refresh_token_differs() {
        // A re-login wrote rt-2 to secrets.toml; the store still caches a token
        // for rt-1 ⇒ ignore the stale cache and use the seed.
        let seed = TokenSet {
            access_token: String::new(),
            ..token("", "rt-2")
        };
        let stored = token("stale-access", "rt-1");
        let result = reconcile_stored(Some(seed), stored);
        assert!(
            result.access_token.is_empty(),
            "must not adopt stale access token"
        );
        assert_eq!(result.refresh_token.as_deref(), Some("rt-2"));
    }

    #[test]
    fn reconcile_trusts_store_when_no_seed() {
        let stored = token("cached-access", "rt-1");
        let result = reconcile_stored(None, stored);
        assert_eq!(result.access_token, "cached-access");
        assert_eq!(result.refresh_token.as_deref(), Some("rt-1"));
    }
}
