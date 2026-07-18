use std::path::Path;
use std::time::Duration;

use anyhow::Result;

use crate::config::ConnectionConfig;

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
    let Some(pem_bytes) = crate::config::read_optional_ca_pem(tls_ca_cert)? else {
        return Ok(Vec::new());
    };
    let certs = reqwest::tls::Certificate::from_pem_bundle(&pem_bytes)?;
    if certs.is_empty() {
        let path = tls_ca_cert.map(Path::display);
        return Err(anyhow::anyhow!(
            "CA cert {} contains no certificates",
            path.expect("a bundle was read, so a path was configured")
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
    let body = response.text().await?;

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
                    .map_err(|login_error| {
                        anyhow::anyhow!(
                            "failed to obtain websocket token via D-Bus ({dbus_error}); \
                         fallback /login on websocket host also failed ({login_error})"
                        )
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
