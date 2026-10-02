//! OAuth2 bearer acquisition for the Vertex surface, behind a `TokenProvider`
//! seam.
//!
//! Vertex authenticates with a short-lived OAuth2 access token minted from a
//! GCP service account (the cloud-credential analogue of Bedrock's AWS chain).
//! The [`TokenProvider`] trait is the seam: production uses
//! [`ServiceAccountTokenProvider`] (a service-account JWT -> token exchange via
//! the workspace `jsonwebtoken` crate, no vendor SDK); tests inject
//! [`StaticTokenProvider`]. The Gemini API (AI Studio) surface uses an API-key
//! header instead and needs no `TokenProvider`.
//!
//! Credential safety is load-bearing: the bearer token and the service-account
//! private key never appear in a URL, a log line, or an error string. Every
//! type that holds credential material implements a redacting `Debug`.

use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use desktop_assistant_core::CoreError;
use serde::{Deserialize, Serialize};

/// Token scope requested for Vertex access.
pub const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Refresh a cached token this many seconds before its stated expiry, so a
/// request never rides a token that expires mid-flight.
const EXPIRY_SKEW: u64 = 60;

/// Percent-encode a value for an `application/x-www-form-urlencoded` body,
/// leaving the RFC 3986 unreserved set untouched. Avoids a `url`/`.form()`
/// dependency for the two-field token-exchange body.
fn form_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Seam for acquiring a Vertex OAuth2 bearer token. Implementations must never
/// log or otherwise leak the returned token.
#[async_trait::async_trait]
pub trait TokenProvider: Send + Sync {
    /// Return a currently-valid bearer token (without the `Bearer ` prefix).
    async fn token(&self) -> Result<String, CoreError>;
}

/// A fixed, pre-minted token. Used by tests as the mock provider, and usable
/// in production when a caller already holds a bearer (e.g. an ADC-resolved
/// token from an outer layer).
#[derive(Clone)]
pub struct StaticTokenProvider {
    token: String,
}

impl StaticTokenProvider {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

/// Redacting `Debug`: the token renders as its length only.
impl std::fmt::Debug for StaticTokenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticTokenProvider")
            .field(
                "token",
                &format_args!("<redacted; len={}>", self.token.len()),
            )
            .finish()
    }
}

#[async_trait::async_trait]
impl TokenProvider for StaticTokenProvider {
    async fn token(&self) -> Result<String, CoreError> {
        Ok(self.token.clone())
    }
}

/// The default provider before any credential is configured. Every `token()`
/// call fails with an actionable message rather than a raw 401 at request
/// time, so a misconfigured Vertex connection says exactly what is missing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopTokenProvider;

#[async_trait::async_trait]
impl TokenProvider for NoopTokenProvider {
    async fn token(&self) -> Result<String, CoreError> {
        Err(CoreError::Llm(
            "Vertex has no resolvable GCP credential; set a credentials_path \
             (service-account JSON) or GOOGLE_APPLICATION_CREDENTIALS, or use \
             auth_mode=api_key with GOOGLE_API_KEY"
                .into(),
        ))
    }
}

/// A parsed GCP service-account key (the JSON downloaded from the console).
/// Only the fields this connector needs are modeled.
#[derive(Deserialize, Clone)]
pub struct ServiceAccountKey {
    /// The service account's email; the JWT `iss`/`sub`.
    pub client_email: String,
    /// PEM-encoded RSA private key used to sign the assertion.
    pub private_key: String,
    /// Token-exchange endpoint (`https://oauth2.googleapis.com/token`).
    pub token_uri: String,
    /// Key id, set as the JWT header `kid` when present.
    #[serde(default)]
    pub private_key_id: Option<String>,
    /// The owning project; not required by the grant but handy for logging.
    #[serde(default)]
    pub project_id: Option<String>,
}

/// Redacting `Debug`: the private key never renders.
impl std::fmt::Debug for ServiceAccountKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceAccountKey")
            .field("client_email", &self.client_email)
            .field("token_uri", &self.token_uri)
            .field("private_key_id", &self.private_key_id)
            .field("project_id", &self.project_id)
            .field(
                "private_key",
                &format_args!("<redacted; len={}>", self.private_key.len()),
            )
            .finish()
    }
}

impl ServiceAccountKey {
    /// Parse a service-account key from its JSON text.
    pub fn from_json(json: &str) -> Result<Self, CoreError> {
        serde_json::from_str(json)
            .map_err(|e| CoreError::Llm(format!("failed to parse service-account key JSON: {e}")))
    }

    /// Read and parse a service-account key from a file path.
    pub fn from_file(path: &str) -> Result<Self, CoreError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            CoreError::Llm(format!(
                "failed to read service-account key file {path}: {e}"
            ))
        })?;
        Self::from_json(&text)
    }
}

/// JWT claims for the service-account assertion (RFC 7523 `jwt-bearer` grant).
#[derive(Serialize)]
struct JwtClaims {
    iss: String,
    sub: String,
    aud: String,
    scope: String,
    iat: u64,
    exp: u64,
}

/// Build the assertion claim set for `key` at `now_unix` (seconds). The
/// assertion is valid for one hour.
fn assertion_claims(key: &ServiceAccountKey, scope: &str, now_unix: u64) -> JwtClaims {
    JwtClaims {
        iss: key.client_email.clone(),
        sub: key.client_email.clone(),
        aud: key.token_uri.clone(),
        scope: scope.to_string(),
        iat: now_unix,
        exp: now_unix + 3600,
    }
}

/// Sign the service-account assertion (RS256) for `key` at `now_unix`.
pub fn build_assertion(
    key: &ServiceAccountKey,
    scope: &str,
    now_unix: u64,
) -> Result<String, CoreError> {
    let claims = assertion_claims(key, scope, now_unix);
    let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
    header.kid = key.private_key_id.clone();
    let encoding =
        jsonwebtoken::EncodingKey::from_rsa_pem(key.private_key.as_bytes()).map_err(|e| {
            CoreError::Llm(format!(
                "service-account private key is not valid RSA PEM: {e}"
            ))
        })?;
    jsonwebtoken::encode(&header, &claims, &encoding)
        .map_err(|e| CoreError::Llm(format!("failed to sign service-account assertion: {e}")))
}

/// Where a [`ServiceAccountTokenProvider`] gets its key material.
enum KeySource {
    /// A filesystem path read lazily on first mint (and each refresh).
    Path(String),
    /// A pre-parsed key (used by tests and by callers that already hold one).
    Key(ServiceAccountKey),
}

/// Mints and caches Vertex bearer tokens from a GCP service account via the
/// RFC 7523 JWT-bearer grant. Thread-safe; the cached token is shared and
/// refreshed just before expiry.
pub struct ServiceAccountTokenProvider {
    source: KeySource,
    scope: String,
    http: reqwest::Client,
    /// Cached `(token, expiry_instant)`; refreshed when within [`EXPIRY_SKEW`].
    cache: Mutex<Option<(String, Instant)>>,
}

impl ServiceAccountTokenProvider {
    /// Build a provider that reads its key from `path` on demand, requesting
    /// the cloud-platform scope.
    pub fn from_credentials_path(path: impl Into<String>) -> Self {
        Self {
            source: KeySource::Path(path.into()),
            scope: CLOUD_PLATFORM_SCOPE.to_string(),
            http: reqwest::Client::new(),
            cache: Mutex::new(None),
        }
    }

    /// Build a provider from an already-parsed key.
    pub fn from_key(key: ServiceAccountKey) -> Self {
        Self {
            source: KeySource::Key(key),
            scope: CLOUD_PLATFORM_SCOPE.to_string(),
            http: reqwest::Client::new(),
            cache: Mutex::new(None),
        }
    }

    /// Override the requested scope (defaults to [`CLOUD_PLATFORM_SCOPE`]).
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = scope.into();
        self
    }
}

/// Redacting `Debug`: neither the key material nor the cached token renders.
impl std::fmt::Debug for ServiceAccountTokenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let source = match &self.source {
            KeySource::Path(p) => format!("Path({p})"),
            KeySource::Key(_) => "Key(<redacted>)".to_string(),
        };
        f.debug_struct("ServiceAccountTokenProvider")
            .field("source", &format_args!("{source}"))
            .field("scope", &self.scope)
            .field("cache", &format_args!("<redacted>"))
            .finish()
    }
}

#[async_trait::async_trait]
impl TokenProvider for ServiceAccountTokenProvider {
    async fn token(&self) -> Result<String, CoreError> {
        // Fast path: a cached token that still has headroom. The guard is
        // dropped before any `.await` (no lock held across await).
        {
            let guard = self.cache.lock().expect("token cache mutex poisoned");
            if let Some((tok, expiry)) = guard.as_ref()
                && *expiry > Instant::now()
            {
                return Ok(tok.clone());
            }
        }

        let key = match &self.source {
            KeySource::Path(path) => ServiceAccountKey::from_file(path)?,
            KeySource::Key(k) => k.clone(),
        };

        let now_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| CoreError::Llm(format!("system clock before UNIX epoch: {e}")))?
            .as_secs();
        let assertion = build_assertion(&key, &self.scope, now_unix)?;

        let body = format!(
            "grant_type={}&assertion={}",
            form_encode("urn:ietf:params:oauth:grant-type:jwt-bearer"),
            form_encode(&assertion),
        );
        let response = self
            .http
            .post(&key.token_uri)
            .header("content-type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| {
                CoreError::Llm(format!(
                    "service-account token exchange request failed: {e}"
                ))
            })?;
        let response =
            desktop_assistant_llm_http::bail_for_status(response, "Google token exchange").await?;
        let token_resp: AccessTokenResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse token-exchange response: {e}")))?;

        let ttl = token_resp
            .expires_in
            .unwrap_or(3600)
            .saturating_sub(EXPIRY_SKEW);
        let expiry = Instant::now() + Duration::from_secs(ttl);
        {
            let mut guard = self.cache.lock().expect("token cache mutex poisoned");
            *guard = Some((token_resp.access_token.clone(), expiry));
        }
        Ok(token_resp.access_token)
    }
}

/// Access-token response from the token endpoint.
#[derive(Deserialize)]
struct AccessTokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SA_KEY_PEM: &str = include_str!("testdata/sa_test_key.pem");

    fn test_key() -> ServiceAccountKey {
        ServiceAccountKey {
            client_email: "svc@proj.iam.gserviceaccount.com".into(),
            private_key: TEST_SA_KEY_PEM.into(),
            token_uri: "https://oauth2.googleapis.com/token".into(),
            private_key_id: Some("kid-123".into()),
            project_id: Some("proj".into()),
        }
    }

    #[tokio::test]
    async fn static_provider_returns_token() {
        let p = StaticTokenProvider::new("tok-abc");
        assert_eq!(p.token().await.unwrap(), "tok-abc");
    }

    #[test]
    fn static_provider_debug_redacts_token() {
        let rendered = format!("{:?}", StaticTokenProvider::new("super-secret-token"));
        assert!(
            !rendered.contains("super-secret-token"),
            "leaked: {rendered}"
        );
        assert!(rendered.contains("redacted"));
    }

    #[tokio::test]
    async fn noop_provider_errors_with_actionable_message() {
        let err = NoopTokenProvider.token().await.expect_err("must error");
        let CoreError::Llm(detail) = err else {
            panic!("expected Llm, got {err:?}");
        };
        assert!(
            detail.contains("credentials_path")
                || detail.contains("GOOGLE_APPLICATION_CREDENTIALS")
        );
        assert!(
            detail.contains("api_key"),
            "should name the alternative mode"
        );
    }

    #[test]
    fn service_account_key_from_json_parses() {
        let json = r#"{
            "type": "service_account",
            "client_email": "svc@proj.iam.gserviceaccount.com",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMII...\n-----END PRIVATE KEY-----\n",
            "token_uri": "https://oauth2.googleapis.com/token",
            "private_key_id": "kid-123",
            "project_id": "proj"
        }"#;
        let key = ServiceAccountKey::from_json(json).expect("parse");
        assert_eq!(key.client_email, "svc@proj.iam.gserviceaccount.com");
        assert_eq!(key.token_uri, "https://oauth2.googleapis.com/token");
        assert_eq!(key.private_key_id.as_deref(), Some("kid-123"));
    }

    #[test]
    fn service_account_key_debug_redacts_private_key() {
        let key = ServiceAccountKey {
            client_email: "svc@proj.iam.gserviceaccount.com".into(),
            private_key: "-----BEGIN PRIVATE KEY-----SECRETMATERIAL-----END-----".into(),
            token_uri: "https://oauth2.googleapis.com/token".into(),
            private_key_id: Some("kid".into()),
            project_id: None,
        };
        let rendered = format!("{key:?}");
        assert!(!rendered.contains("SECRETMATERIAL"), "leaked: {rendered}");
        assert!(rendered.contains("redacted"));
        // Non-secret fields remain visible.
        assert!(rendered.contains("svc@proj.iam.gserviceaccount.com"));
    }

    #[test]
    fn assertion_claims_carry_iss_aud_scope_and_expiry() {
        let key = test_key();
        let claims = assertion_claims(&key, CLOUD_PLATFORM_SCOPE, 1_000);
        assert_eq!(claims.iss, "svc@proj.iam.gserviceaccount.com");
        assert_eq!(claims.sub, "svc@proj.iam.gserviceaccount.com");
        assert_eq!(claims.aud, "https://oauth2.googleapis.com/token");
        assert_eq!(claims.scope, CLOUD_PLATFORM_SCOPE);
        assert_eq!(claims.iat, 1_000);
        assert_eq!(claims.exp, 4_600, "assertion is valid for one hour");
    }

    #[test]
    fn build_assertion_signs_rs256_with_kid() {
        let key = test_key();
        let jwt = build_assertion(&key, CLOUD_PLATFORM_SCOPE, 1_000).expect("sign");
        // A JWT is three dot-separated base64url segments.
        assert_eq!(jwt.split('.').count(), 3, "not a JWT: {jwt}");
        // The header is decodable without a key and carries alg + kid.
        let header = jsonwebtoken::decode_header(&jwt).expect("decode header");
        assert_eq!(header.alg, jsonwebtoken::Algorithm::RS256);
        assert_eq!(header.kid.as_deref(), Some("kid-123"));
    }

    #[test]
    fn service_account_provider_debug_redacts() {
        let p = ServiceAccountTokenProvider::from_key(test_key());
        let rendered = format!("{p:?}");
        assert!(
            !rendered.contains(TEST_SA_KEY_PEM),
            "private key leaked in Debug"
        );
        assert!(rendered.contains("redacted"));
    }
}
