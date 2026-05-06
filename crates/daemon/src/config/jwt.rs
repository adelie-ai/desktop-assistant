//! WebSocket JWT signing — delegates to the shared `auth-jwt` crate
//! (extracted in #?). The daemon owns the issuer/audience/TTL policy and
//! the key-file path; the codec and atomic file IO live in `auth-jwt` so
//! the JWT minter can produce tokens this validator accepts.
//!
//! Public API (`current_username`, `generate_ws_jwt`, `validate_ws_jwt`)
//! is preserved exactly. The `pub(super)` test API
//! (`WsJwtClaims`, `encode_ws_jwt`, `decode_ws_jwt_claims`) is preserved
//! so the forge-token tests in `super::tests` keep working without
//! changes.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::anyhow;
use desktop_assistant_auth_jwt as auth_jwt;

/// Re-export of the shared claim type so the existing `pub(super)` test
/// helpers keep their type names. The fields are public (the whole point
/// of moving to the shared crate is letting the minter construct claims
/// directly), but `pub(super) use` keeps the alias scoped to this module's
/// supermodule, matching the pre-refactor visibility.
pub(super) use auth_jwt::Claims as WsJwtClaims;

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

fn signing_key_path() -> PathBuf {
    super::default_secret_store_dir().join(ws_jwt_signing_key_account())
}

pub(super) fn encode_ws_jwt(claims: &WsJwtClaims) -> anyhow::Result<String> {
    let signing_key = auth_jwt::ensure_signing_key_at(&signing_key_path())?;
    auth_jwt::encode(claims, &signing_key)
}

pub(super) fn decode_ws_jwt_claims(token: &str) -> anyhow::Result<WsJwtClaims> {
    let signing_key = auth_jwt::read_signing_key_at(&signing_key_path())
        .ok_or_else(|| anyhow!("ws jwt signing key is not initialized"))?;
    auth_jwt::decode(
        token,
        &signing_key,
        default_ws_jwt_issuer(),
        default_ws_jwt_audience(),
    )
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
