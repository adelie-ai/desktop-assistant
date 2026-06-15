//! WebSocket JWT signing — delegates to the shared `auth-jwt` crate
//! (extracted in #?). The daemon owns the issuer/audience/TTL policy and
//! the key-file path; the codec and atomic file IO live in `auth-jwt`. JWT is a
//! network-door (WebSocket) concern only — local transports authenticate by
//! kernel peer-cred (#407), and the standalone `adelie-mint` minter is retired.
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

/// Default lifetime of a daemon-minted WS JWT.
///
/// DT-3 (#269): dropped from 30 days to 1 hour to match the jwt-minter's own
/// default (`dbus-bridge`'s `token_ttl_seconds`). A leaked token is now a
/// one-hour exposure, not a month, and clients re-mint transparently on
/// expiry (the `Connector` already reconnects and replays). Existing configs
/// that explicitly request a longer TTL via the minter keep working — this is
/// only the default applied when no TTL is supplied.
pub(super) fn default_ws_jwt_ttl_seconds() -> u64 {
    60 * 60
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

/// Decode and verify `token`, then reject it if its `jti` is on the
/// revocation deny-list (DT-3 #269). This is the single chokepoint that both
/// [`validate_ws_jwt`] and [`ws_jwt_sub`] route through, so a revoked token is
/// rejected on every auth path.
pub(super) fn decode_ws_jwt_claims(token: &str) -> anyhow::Result<WsJwtClaims> {
    let claims = decode_ws_jwt_claims_ignoring_revocation(token)?;
    if is_jti_revoked(&claims.jti) {
        return Err(anyhow!("ws jwt has been revoked"));
    }
    Ok(claims)
}

/// Decode and verify `token` (signature, issuer, audience, exp/nbf) WITHOUT
/// consulting the revocation list. Used internally so revocation can be added
/// on top, and by [`revoke_ws_jwt`] to read a token's `jti`/`exp` before
/// denying it.
pub(super) fn decode_ws_jwt_claims_ignoring_revocation(token: &str) -> anyhow::Result<WsJwtClaims> {
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

/// Decode and validate `token`, then return the `sub` claim.
///
/// Used by the WS auth path (#105) to map a validated bearer token to
/// the `user_id` that scopes every subsequent storage query. Returns
/// `None` when the token is missing, malformed, expired, or otherwise
/// rejected by the same checks [`validate_ws_jwt`] applies — a caller
/// that wants to distinguish "valid but no sub" from "invalid" should
/// call this AFTER `validate_ws_jwt` and treat `None` as "validator
/// opted out of identity extraction".
pub fn ws_jwt_sub(token: &str) -> Option<String> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    decode_ws_jwt_claims(token).ok().map(|claims| claims.sub)
}

// ── DT-3 (#269): jti revocation deny-list ────────────────────────────────────
//
// Tokens are short-lived (1h default), so a revoked `jti` only needs to be
// remembered until its own `exp` passes — after that the token is rejected by
// the `exp` check anyway and the entry is pruned. The list is a JSON map of
// `jti -> exp` persisted next to the signing key. All access is serialised
// through a process-wide mutex so concurrent revokes/validations can't race on
// the file; the lock is cheap because the file is tiny (it only holds
// not-yet-expired revoked jtis).

use std::collections::HashMap;
use std::sync::Mutex;

fn revocation_list_path() -> PathBuf {
    super::default_secret_store_dir().join("ws_jwt_revocations.json")
}

fn revocation_lock() -> &'static Mutex<()> {
    static LOCK: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Read the persisted deny-list (`jti -> exp`). Missing/corrupt file → empty.
fn read_revocations() -> HashMap<String, u64> {
    let path = revocation_list_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return HashMap::new();
    };
    serde_json::from_str(&raw).unwrap_or_else(|error| {
        tracing::warn!(
            path = %path.display(),
            %error,
            "ws jwt revocation list is corrupt; treating as empty"
        );
        HashMap::new()
    })
}

/// Atomically persist the deny-list with the same 0600/0700 tightening the
/// signing key gets (reuse `auth_jwt`'s atomic writer).
fn write_revocations(map: &HashMap<String, u64>) -> anyhow::Result<()> {
    let json = serde_json::to_string(map).map_err(|e| anyhow!("serialize revocations: {e}"))?;
    auth_jwt::write_signing_key_at(&revocation_list_path(), &json)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// True if `jti` is on the (non-expired) deny-list.
pub(super) fn is_jti_revoked(jti: &str) -> bool {
    let _guard = revocation_lock().lock().unwrap_or_else(|e| e.into_inner());
    match read_revocations().get(jti) {
        Some(&exp) => exp > now_secs(),
        None => false,
    }
}

/// Drop deny-list entries whose `exp` has already passed (they're rejected by
/// the `exp` check regardless), keeping the file from growing without bound.
pub fn prune_revocations() {
    let _guard = revocation_lock().lock().unwrap_or_else(|e| e.into_inner());
    let now = now_secs();
    let mut map = read_revocations();
    let before = map.len();
    map.retain(|_, &mut exp| exp > now);
    if map.len() != before
        && let Err(error) = write_revocations(&map)
    {
        tracing::warn!(%error, "failed to persist pruned ws jwt revocation list");
    }
}

/// Revoke a WS JWT by its `jti`.
///
/// The token must currently decode (valid signature/issuer/audience) so an
/// operator can't accidentally poison the list with garbage; `exp` is read
/// from the token so the entry self-expires. Idempotent: re-revoking is fine.
/// Returns an error for a missing/malformed/expired token rather than silently
/// succeeding, so callers know nothing was added.
pub fn revoke_ws_jwt(token: &str) -> anyhow::Result<()> {
    let token = token.trim();
    if token.is_empty() {
        return Err(anyhow!("cannot revoke an empty token"));
    }
    // Decode ignoring revocation (a token already revoked must still be
    // re-revocable / inspectable) but still requiring a valid signature+exp.
    let claims = decode_ws_jwt_claims_ignoring_revocation(token)?;

    let _guard = revocation_lock().lock().unwrap_or_else(|e| e.into_inner());
    let now = now_secs();
    let mut map = read_revocations();
    map.retain(|_, &mut exp| exp > now); // opportunistic prune
    map.insert(claims.jti, claims.exp);
    write_revocations(&map)
}

#[cfg(test)]
pub(super) fn record_revocation_for_test(jti: &str, exp: u64) {
    let _guard = revocation_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut map = read_revocations();
    map.insert(jti.to_string(), exp);
    write_revocations(&map).expect("write test revocation");
}
