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

pub async fn request_ws_login_token(
    ws_url: &str,
    username: &str,
    password: &str,
) -> Result<String> {
    let login_url = derive_login_url_from_ws_url(ws_url)?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;

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
                    request_ws_login_token(&config.ws_url, username, password).await
                        .map_err(|login_error| anyhow::anyhow!(
                            "failed to obtain websocket token via D-Bus ({dbus_error}); \
                             fallback /login on websocket host also failed ({login_error})"
                        ))
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
            return request_ws_login_token(&config.ws_url, username, password).await;
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
            .err()
            .expect("non-ws scheme should fail");
        assert!(error.to_string().contains("ws:// or wss://"));
    }
}
