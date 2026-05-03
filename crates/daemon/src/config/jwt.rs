//! WebSocket JWT signing — generates and validates the bearer tokens
//! the WS adapter uses for local authentication.
//!
//! Extracted from `config.rs` (#41). The signing key is HS256 with a
//! key persisted via the secret-store backends in
//! [`super::secrets::read_common_file_secret`] / [`super::write_common_file_secret`].
//! Issuer and audience are fixed local strings so a token from a
//! different daemon instance can't pass validation.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::anyhow;
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

/// JWT claim payload. `pub(super)` so the JWT round-trip and forged-
/// token tests in the parent test module can mutate fields and
/// re-encode without going through a separate test-only API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct WsJwtClaims {
    pub(super) iss: String,
    pub(super) sub: String,
    pub(super) aud: String,
    pub(super) exp: u64,
    pub(super) iat: u64,
    pub(super) nbf: u64,
    pub(super) jti: String,
}

fn ws_jwt_signing_key_account() -> &'static str {
    "ws_jwt_hs256_signing_key"
}

pub(super) fn default_ws_jwt_issuer() -> &'static str {
    "org.desktopAssistant.local"
}

pub(super) fn default_ws_jwt_audience() -> &'static str {
    "desktop-assistant-ws"
}

fn default_ws_jwt_ttl_seconds() -> u64 {
    60 * 60 * 24 * 30
}

pub fn current_username() -> String {
    std::env::var("USER")
        .ok()
        .or_else(|| std::env::var("LOGNAME").ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "desktop-user".to_string())
}

fn normalize_ws_jwt_subject(subject: Option<String>) -> String {
    subject
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(current_username)
}

fn unix_timestamp_seconds() -> anyhow::Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| anyhow!("failed to read system clock: {error}"))
}

fn ensure_ws_jwt_signing_key() -> anyhow::Result<String> {
    if let Some(existing) = super::secrets::read_common_file_secret(ws_jwt_signing_key_account()) {
        return Ok(existing);
    }

    // 64 hex chars from two UUIDv4 values gives a sufficiently strong local HMAC secret.
    let generated = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    super::secrets::write_common_file_secret(ws_jwt_signing_key_account(), &generated)?;
    Ok(generated)
}

fn read_ws_jwt_signing_key() -> anyhow::Result<String> {
    super::secrets::read_common_file_secret(ws_jwt_signing_key_account())
        .ok_or_else(|| anyhow!("ws jwt signing key is not initialized"))
}

fn ws_jwt_validation() -> Validation {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.validate_exp = true;
    validation.set_issuer(&[default_ws_jwt_issuer()]);
    validation.set_audience(&[default_ws_jwt_audience()]);
    validation
}

pub(super) fn encode_ws_jwt(claims: &WsJwtClaims) -> anyhow::Result<String> {
    let signing_key = ensure_ws_jwt_signing_key()?;
    jsonwebtoken::encode(
        &Header::new(Algorithm::HS256),
        claims,
        &EncodingKey::from_secret(signing_key.as_bytes()),
    )
    .map_err(|error| anyhow!("failed to encode ws jwt: {error}"))
}

pub(super) fn decode_ws_jwt_claims(token: &str) -> anyhow::Result<WsJwtClaims> {
    let signing_key = read_ws_jwt_signing_key()?;
    let decoded = jsonwebtoken::decode::<WsJwtClaims>(
        token,
        &DecodingKey::from_secret(signing_key.as_bytes()),
        &ws_jwt_validation(),
    )
    .map_err(|error| anyhow!("failed to decode ws jwt: {error}"))?;
    Ok(decoded.claims)
}

pub fn generate_ws_jwt(subject: Option<String>) -> anyhow::Result<String> {
    let now = unix_timestamp_seconds()?;
    let claims = WsJwtClaims {
        iss: default_ws_jwt_issuer().to_string(),
        sub: normalize_ws_jwt_subject(subject),
        aud: default_ws_jwt_audience().to_string(),
        exp: now.saturating_add(default_ws_jwt_ttl_seconds()),
        iat: now,
        nbf: now.saturating_sub(1),
        jti: uuid::Uuid::new_v4().to_string(),
    };

    encode_ws_jwt(&claims)
}

pub fn validate_ws_jwt(token: &str) -> anyhow::Result<bool> {
    let token = token.trim();
    if token.is_empty() {
        return Ok(false);
    }

    match decode_ws_jwt_claims(token) {
        Ok(_) => Ok(true),
        Err(error) => {
            tracing::debug!("invalid ws jwt: {error}");
            Ok(false)
        }
    }
}
