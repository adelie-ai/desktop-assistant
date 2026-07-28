use std::path::Path;
use std::time::Duration;

use anyhow::Result;

use crate::config::ConnectionConfig;

/// How long to wait before retrying `/login` when the daemon refused the
/// attempt but named no wait. Long enough not to re-arm a short lockout, short
/// enough that a client recovers promptly once the door reopens.
const DEFAULT_LOGIN_RETRY_AFTER: Duration = Duration::from_secs(60);

/// Longest wait this client will honour from a `Retry-After` header.
///
/// The header arrives over the wire, and `/login` runs over plain HTTP when the
/// daemon serves `ws://`, so an on-path attacker or a broken proxy writes it.
/// Unclamped, a huge value parks the client for years, and one near
/// `u64::MAX` panics the reconnect task outright, because `Duration`'s `Add`
/// aborts on overflow. Five minutes is far longer than any wait this daemon
/// asks for and short enough to recover from.
const MAX_LOGIN_RETRY_AFTER: Duration = Duration::from_secs(5 * 60);

/// The daemon refused a `/login` attempt because too many have failed recently
/// (#808).
///
/// It is not a credential verdict: the daemon did not check the password. A
/// caller that retries must wait [`Self::retry_after`] first, because an early
/// retry spends from the same budget and pushes the wait out again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginThrottled {
    /// How long the daemon asked the caller to wait.
    pub retry_after: Duration,
}

impl std::fmt::Display for LoginThrottled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "too many recent login attempts; retry in {} s",
            self.retry_after.as_secs()
        )
    }
}

impl std::error::Error for LoginThrottled {}

/// The wait a login failure asked for, if it was a throttle refusal rather than
/// a credential failure or a transport error.
///
/// Retry loops call this instead of matching the error text, so a refusal is
/// told apart from a wrong password without parsing prose.
pub fn login_retry_after(error: &anyhow::Error) -> Option<Duration> {
    error
        .downcast_ref::<LoginThrottled>()
        .map(|throttled| throttled.retry_after)
}

/// Read a `Retry-After` header expressed in whole seconds.
///
/// The HTTP-date form is not read: the daemon always sends seconds, and a date
/// would need a clock both ends agree on. An unreadable value yields `None`, and
/// the caller falls back to [`DEFAULT_LOGIN_RETRY_AFTER`]. The caller also
/// clamps whatever comes back to [`MAX_LOGIN_RETRY_AFTER`].
fn retry_after_from(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

pub fn derive_login_url_from_ws_url(ws_url: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(ws_url)
        .map_err(|error| anyhow::anyhow!("invalid websocket URL '{ws_url}': {error}"))?;

    let next_scheme = match url.scheme() {
        "ws" => "http",
        "wss" => "https",
        other => {
            return Err(anyhow::anyhow!(
                "websocket URL must use ws:// or wss:// (got {other}://)"
            ));
        }
    };

    url.set_scheme(next_scheme).map_err(|_| {
        anyhow::anyhow!("failed to rewrite websocket URL scheme for login endpoint")
    })?;
    url.set_path("/login");
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string())
}

/// Loads the extra trust anchors for the `/login` request, layered on top of
/// reqwest's built-in roots.
///
/// Mirrors `ws_client::build_root_store` deliberately: both halves of the
/// connect flow must trust the same anchors, or login succeeds and the socket
/// that follows it fails (#521). `from_pem_bundle` rather than `from_pem`
/// because the latter stops after the first certificate in a concatenated file.
fn load_login_root_certs(tls_ca_cert: Option<&Path>) -> Result<Vec<reqwest::tls::Certificate>> {
    let Some(ca_path) = tls_ca_cert else {
        return Ok(Vec::new());
    };
    let Some(pem_bytes) = crate::config::read_optional_ca_pem(Some(ca_path))? else {
        return Ok(Vec::new());
    };
    let certs = reqwest::tls::Certificate::from_pem_bundle(&pem_bytes)
        .map_err(|e| anyhow::anyhow!("parsing CA cert {}: {e}", ca_path.display()))?;
    if certs.is_empty() {
        return Err(anyhow::anyhow!(
            "CA cert {} contains no certificates",
            ca_path.display()
        ));
    }
    Ok(certs)
}

pub async fn request_ws_login_token(
    ws_url: &str,
    username: &str,
    password: &str,
    tls_ca_cert: Option<&Path>,
) -> Result<String> {
    let login_url = derive_login_url_from_ws_url(ws_url)?;
    let mut builder = reqwest::Client::builder().timeout(Duration::from_secs(10));
    for cert in load_login_root_certs(tls_ca_cert)? {
        builder = builder.add_root_certificate(cert);
    }
    let client = builder.build()?;

    let response = client
        .post(login_url)
        .basic_auth(username, Some(password))
        .send()
        .await?;
    let status = response.status();
    let retry_after = retry_after_from(response.headers());
    let body = response.text().await?;

    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        // The daemon is throttling failed logins (#808). This is NOT a verdict
        // on the credential: it did not read it. Carry the wait as a typed
        // error so a retry loop can honour it instead of coming back on its own
        // schedule - retrying early re-arms the lockout, and a client left
        // running with a stale password would otherwise hold the door shut for
        // everyone, including whoever has the right password.
        let throttled = LoginThrottled {
            retry_after: retry_after
                .unwrap_or(DEFAULT_LOGIN_RETRY_AFTER)
                .min(MAX_LOGIN_RETRY_AFTER),
        };
        return Err(anyhow::Error::new(throttled)
            .context("remote /login is refusing attempts for now".to_string()));
    }

    if !status.is_success() {
        let trimmed = body.trim();
        if trimmed.is_empty() {
            return Err(anyhow::anyhow!("remote /login failed with HTTP {}", status));
        }
        return Err(anyhow::anyhow!(
            "remote /login failed with HTTP {}: {}",
            status,
            trimmed
        ));
    }

    let payload: serde_json::Value = serde_json::from_str(&body)
        .map_err(|error| anyhow::anyhow!("invalid /login JSON: {error}"))?;
    let token = payload
        .get("token")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .unwrap_or("");
    if token.is_empty() {
        return Err(anyhow::anyhow!("/login response did not include token"));
    }
    Ok(token.to_string())
}

/// Resolve a bearer token for the **network** door (WebSocket). Local UDS no
/// longer calls this — it authenticates by kernel peer-cred (#407) — so this is
/// the remote-client path only: an explicit `ws_jwt`, else a D-Bus `GenerateWsJwt`
/// (built-in HS256 issuer), else a `/login` password exchange.
pub async fn resolve_ws_bearer_token(config: &ConnectionConfig) -> Result<String> {
    if let Some(token) = config.ws_jwt.clone() {
        return Ok(token);
    }

    #[cfg(feature = "dbus")]
    {
        match crate::dbus_client::generate_ws_jwt(&config.ws_subject).await {
            Ok(token) => Ok(token),
            Err(dbus_error) => {
                if let (Some(username), Some(password)) = (
                    config.ws_login_username.as_deref(),
                    config.ws_login_password.as_deref(),
                ) {
                    request_ws_login_token(
                        &config.ws_url,
                        username,
                        password,
                        config.tls_ca_cert.as_deref(),
                    )
                    .await
                    // `context`, not a new error: a throttle refusal must stay
                    // downcastable so `login_retry_after` still sees it.
                    .map_err(|login_error| {
                        login_error.context(format!(
                            "failed to obtain websocket token via D-Bus ({dbus_error}); \
                             fallback /login on websocket host also failed"
                        ))
                    })
                } else {
                    Err(anyhow::anyhow!(
                        "failed to obtain websocket token via D-Bus ({dbus_error}); \
                         provide --ws-jwt or --ws-login-username/--ws-login-password for /login fallback"
                    ))
                }
            }
        }
    }

    #[cfg(not(feature = "dbus"))]
    {
        if let (Some(username), Some(password)) = (
            config.ws_login_username.as_deref(),
            config.ws_login_password.as_deref(),
        ) {
            return request_ws_login_token(
                &config.ws_url,
                username,
                password,
                config.tls_ca_cert.as_deref(),
            )
            .await;
        }
        Err(anyhow::anyhow!(
            "no JWT provided and D-Bus not available; \
             provide --ws-jwt or --ws-login-username/--ws-login-password for /login fallback"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Answer exactly one HTTP request with `status`, `headers` and an empty
    /// body, then close. Enough to drive `/login`'s response handling without a
    /// mock-HTTP dependency.
    async fn one_shot_http(status: &'static str, headers: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("read the bound address");
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.expect("accept one connection");
            let mut scratch = [0u8; 2048];
            let _ = socket.read(&mut scratch).await;
            let response = format!(
                "HTTP/1.1 {status}\r\n{headers}Content-Length: 0\r\nConnection: close\r\n\r\n"
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        });
        format!("ws://{addr}/ws")
    }

    /// A refusal is not a credential verdict, and it names the wait, so a retry
    /// loop can honour it instead of coming back on its own schedule (#808).
    #[tokio::test]
    async fn a_throttled_login_carries_the_wait_the_daemon_asked_for() {
        let ws_url = one_shot_http("429 Too Many Requests", "Retry-After: 42\r\n").await;
        let error = request_ws_login_token(&ws_url, "alice", "wrong", None)
            .await
            .expect_err("a 429 must not be read as a token");
        assert_eq!(
            login_retry_after(&error),
            Some(Duration::from_secs(42)),
            "the caller must be able to read the wait without parsing the message"
        );
    }

    /// A refusal with no usable `Retry-After` still has to be told apart from a
    /// wrong password, so it falls back to a wait rather than to nothing.
    #[tokio::test]
    async fn a_throttled_login_without_a_header_still_names_a_wait() {
        let ws_url = one_shot_http("429 Too Many Requests", "").await;
        let error = request_ws_login_token(&ws_url, "alice", "wrong", None)
            .await
            .expect_err("a 429 must not be read as a token");
        assert_eq!(login_retry_after(&error), Some(DEFAULT_LOGIN_RETRY_AFTER));
    }

    /// A wrong password is a credential verdict and must not be mistaken for a
    /// refusal - retrying it after a wait would never help.
    #[tokio::test]
    async fn a_rejected_credential_is_not_a_throttle_refusal() {
        let ws_url = one_shot_http("401 Unauthorized", "").await;
        let error = request_ws_login_token(&ws_url, "alice", "wrong", None)
            .await
            .expect_err("a 401 must not be read as a token");
        assert_eq!(login_retry_after(&error), None);
    }

    /// The wait must survive the context a caller wraps the failure in, or the
    /// retry loop stops seeing it the moment someone adds a message.
    #[test]
    fn the_wait_survives_added_context() {
        let error = anyhow::Error::new(LoginThrottled {
            retry_after: Duration::from_secs(7),
        })
        .context("outer")
        .context("outer again");
        assert_eq!(login_retry_after(&error), Some(Duration::from_secs(7)));
        assert_eq!(login_retry_after(&anyhow::anyhow!("unrelated")), None);
    }

    /// The header comes off the wire, so a hostile or broken value must not be
    /// able to park the client for years or overflow the reconnect delay.
    #[tokio::test]
    async fn an_absurd_retry_after_is_clamped() {
        let ws_url = one_shot_http(
            "429 Too Many Requests",
            "Retry-After: 18446744073709551615\r\n",
        )
        .await;
        let error = request_ws_login_token(&ws_url, "alice", "wrong", None)
            .await
            .expect_err("a 429 must not be read as a token");
        let wait = login_retry_after(&error).expect("a refusal names a wait");
        assert!(
            wait <= MAX_LOGIN_RETRY_AFTER,
            "a wire-supplied wait of {wait:?} must be clamped"
        );
        // The clamped value must still be safe to add to, which is what the
        // reconnect loop does before it sleeps.
        let _ = wait + Duration::from_secs(1);
    }

    #[test]
    fn retry_after_reads_whole_seconds_only() {
        let header = |value: &str| {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::RETRY_AFTER,
                reqwest::header::HeaderValue::from_str(value).expect("valid header"),
            );
            headers
        };
        assert_eq!(
            retry_after_from(&header(" 30 ")),
            Some(Duration::from_secs(30))
        );
        assert_eq!(retry_after_from(&header("0")), Some(Duration::ZERO));
        // The HTTP-date form is deliberately not read; it needs a shared clock.
        assert_eq!(
            retry_after_from(&header("Wed, 21 Oct 2026 07:28:00 GMT")),
            None
        );
        assert_eq!(retry_after_from(&header("soon")), None);
        assert_eq!(retry_after_from(&reqwest::header::HeaderMap::new()), None);
    }

    #[test]
    fn derive_login_url_rewrites_ws_scheme_and_path() {
        let url = derive_login_url_from_ws_url("ws://127.0.0.1:11339/ws?x=1#frag").unwrap();
        assert_eq!(url, "http://127.0.0.1:11339/login");

        let secure = derive_login_url_from_ws_url("wss://daemon.example.com/ws").unwrap();
        assert_eq!(secure, "https://daemon.example.com/login");
    }

    #[test]
    fn derive_login_url_rejects_non_ws_scheme() {
        let error = derive_login_url_from_ws_url("http://example.com/ws")
            .expect_err("non-ws scheme should fail");
        assert!(error.to_string().contains("ws:// or wss://"));
    }

    fn ca_file(pem: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().expect("create temp CA file");
        f.write_all(pem.as_bytes()).expect("write temp CA file");
        f.flush().expect("flush temp CA file");
        f
    }

    fn self_signed_pem() -> String {
        rcgen::generate_simple_self_signed(vec!["ca.test".to_string()])
            .expect("generate self-signed cert")
            .cert
            .pem()
    }

    /// The login half of the connect flow must tolerate an absent local CA for
    /// the same reason the socket half does — otherwise a fresh machine cannot
    /// authenticate against a publicly-signed endpoint.
    #[test]
    fn missing_ca_file_yields_no_extra_login_roots() {
        let missing = std::path::Path::new("/nonexistent/desktop-assistant/tls/ca.pem");

        let certs =
            load_login_root_certs(Some(missing)).expect("missing CA file must not be fatal");

        assert!(
            certs.is_empty(),
            "expected no extra roots, got {}",
            certs.len()
        );
    }

    #[test]
    fn single_ca_file_yields_one_login_root() {
        let ca = ca_file(&self_signed_pem());

        let certs = load_login_root_certs(Some(ca.path())).expect("load single CA");

        assert_eq!(certs.len(), 1);
    }

    /// A concatenated bundle must contribute every certificate, matching the
    /// WebSocket trust store. `Certificate::from_pem` silently reads only the
    /// first, which would leave the two halves of the flow trusting different
    /// sets of anchors.
    #[test]
    fn ca_bundle_yields_every_login_root() {
        let bundle = ca_file(&format!("{}{}", self_signed_pem(), self_signed_pem()));

        let certs = load_login_root_certs(Some(bundle.path())).expect("load CA bundle");

        assert_eq!(
            certs.len(),
            2,
            "both certificates in the bundle should load"
        );
    }

    #[test]
    fn ca_file_without_certificates_fails_login_root_load() {
        let junk = ca_file("this is not a certificate\n");

        let err = load_login_root_certs(Some(junk.path()))
            .expect_err("a CA file with no certificates must be rejected");

        assert!(
            err.to_string().contains("no certificates"),
            "error should name the empty-bundle cause, got: {err}"
        );
    }
}
