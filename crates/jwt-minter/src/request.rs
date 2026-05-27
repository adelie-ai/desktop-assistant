//! Wire types and the pure `handle_request` function.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context};
use desktop_assistant_auth_jwt::{self as auth_jwt, Claims};
use serde::{Deserialize, Serialize};

use crate::config::MintConfig;
use crate::peer::PeerIdentity;

/// Caller-supplied request payload. All fields optional — callers commonly
/// send `{}`.
#[derive(Debug, Default, Deserialize)]
pub struct MintRequest {
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
    #[serde(default)]
    pub audience: Option<String>,
}

/// Response payload. `error` is `Some` iff minting failed; otherwise
/// `token`/`exp` carry the result.
#[derive(Debug, Serialize, Deserialize)]
pub struct MintResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl MintResponse {
    fn ok(token: String, exp: u64) -> Self {
        Self {
            token: Some(token),
            exp: Some(exp),
            error: None,
        }
    }

    fn err(message: impl Into<String>) -> Self {
        Self {
            token: None,
            exp: None,
            error: Some(message.into()),
        }
    }
}

/// Parse `raw_request`, mint a token for `peer`, return the serialized
/// JSON response.
///
/// This is the unit-testable core. The server loop only adds the I/O
/// wrapper around it.
pub fn handle_request(
    raw_request: &str,
    peer: &PeerIdentity,
    config: &MintConfig,
    signing_key: &str,
) -> String {
    let response = handle_request_inner(raw_request, peer, config, signing_key);
    serde_json::to_string(&response)
        .unwrap_or_else(|_| r#"{"error":"failed to serialize response"}"#.to_string())
}

fn handle_request_inner(
    raw_request: &str,
    peer: &PeerIdentity,
    config: &MintConfig,
    signing_key: &str,
) -> MintResponse {
    let request: MintRequest = match parse_request(raw_request) {
        Ok(r) => r,
        Err(err) => return MintResponse::err(format!("invalid request: {err}")),
    };

    let ttl = config.clamp_ttl(request.ttl_seconds);
    let audience = request
        .audience
        .filter(|a| !a.trim().is_empty())
        .unwrap_or_else(|| config.default_audience.clone());

    let now = match unix_now() {
        Ok(t) => t,
        Err(err) => return MintResponse::err(format!("clock error: {err}")),
    };
    let exp = now.saturating_add(ttl.as_secs());

    let claims = Claims {
        iss: config.issuer.clone(),
        sub: peer.username.clone(),
        aud: audience,
        exp,
        iat: now,
        nbf: now.saturating_sub(1),
        jti: uuid::Uuid::new_v4().to_string(),
    };

    match auth_jwt::encode(&claims, signing_key) {
        Ok(token) => MintResponse::ok(token, exp),
        Err(err) => MintResponse::err(format!("encode error: {err}")),
    }
}

/// Treat an entirely empty body as `{}` to give clients a friendly
/// "default everything" shortcut, but fail clean on any non-empty
/// non-JSON.
fn parse_request(raw: &str) -> anyhow::Result<MintRequest> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(MintRequest::default());
    }
    serde_json::from_str(trimmed).context("malformed JSON")
}

fn unix_now() -> anyhow::Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|e| anyhow!("system clock before unix epoch: {e}"))
}
