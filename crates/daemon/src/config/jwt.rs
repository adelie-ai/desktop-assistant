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
use std::sync::OnceLock;
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

/// The resolved built-in HS256 issuer identity (`iss`/`aud`), shared by the
/// issue and validate paths so they can never drift. Seeded once at daemon
/// startup from `[ws_auth.hs256]` via [`init_hs256_identity`]; any access before
/// that (e.g. a unit test) lazily falls back to the per-host default.
#[derive(Debug, Clone)]
struct Hs256Identity {
    issuer: String,
    audience: String,
}

static HS256_IDENTITY: OnceLock<Hs256Identity> = OnceLock::new();

/// Best-effort local hostname for the default `iss`. Dependency-free, mirroring
/// `client-common`'s resolver: kernel hostname, then `/etc/hostname`, then
/// `$HOSTNAME`. Falls back to a fixed label so a token always has an issuer.
fn local_hostname() -> String {
    let from_file = |path: &str| {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    from_file("/proc/sys/kernel/hostname")
        .or_else(|| from_file("/etc/hostname"))
        .or_else(|| {
            std::env::var("HOSTNAME")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "desktop-assistant.local".to_string())
}

/// The per-host default identity used when config leaves a field unset:
/// `iss` = local hostname, `aud` = `"<user>.adelie-ai"`.
fn default_hs256_identity() -> Hs256Identity {
    Hs256Identity {
        issuer: local_hostname(),
        audience: format!("{}.adelie-ai", current_username()),
    }
}

/// Resolve the effective identity from config fields: a non-empty `issuer` /
/// `audience` wins, otherwise the per-host default. Pure (no global state) so it
/// can be unit-tested directly.
fn resolve_hs256_identity(issuer: Option<String>, audience: Option<String>) -> Hs256Identity {
    let non_empty = |s: Option<String>| s.map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let default = default_hs256_identity();
    Hs256Identity {
        issuer: non_empty(issuer).unwrap_or(default.issuer),
        audience: non_empty(audience).unwrap_or(default.audience),
    }
}

/// Seed the process-wide HS256 issuer identity from `[ws_auth.hs256]`. Call once
/// at daemon startup, before serving. Empty/`None` fields fall back to the
/// per-host default. First call wins; later calls are ignored (the identity is
/// immutable for the process lifetime, which is what keeps issue == validate).
pub(crate) fn init_hs256_identity(issuer: Option<String>, audience: Option<String>) {
    let _ = HS256_IDENTITY.set(resolve_hs256_identity(issuer, audience));
}

fn hs256_identity() -> &'static Hs256Identity {
    HS256_IDENTITY.get_or_init(default_hs256_identity)
}

pub(super) fn ws_jwt_issuer() -> &'static str {
    &hs256_identity().issuer
}

pub(super) fn ws_jwt_audience() -> &'static str {
    &hs256_identity().audience
}

/// Default lifetime of a daemon-minted WS JWT.
///
/// DT-3 (#269): dropped from 30 days to 1 hour. A leaked token is now a
/// one-hour exposure, not a month, and clients re-issue transparently on expiry
/// (the `Connector` already reconnects and replays).
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
    auth_jwt::decode(token, &signing_key, ws_jwt_issuer(), ws_jwt_audience())
}

pub fn generate_ws_jwt(subject: Option<String>) -> anyhow::Result<String> {
    let now = unix_timestamp_seconds()?;
    let claims = WsJwtClaims {
        iss: ws_jwt_issuer().to_string(),
        sub: normalize_ws_jwt_subject(subject),
        aud: ws_jwt_audience().to_string(),
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

#[cfg(test)]
mod hs256_identity_tests {
    //! Unit tests for the HS256 issuer-identity resolution (#407 step 5). These
    //! exercise the pure `resolve_hs256_identity` so they don't touch the
    //! process-global `OnceLock` (which would race other tests in this binary).

    use super::{current_username, local_hostname, resolve_hs256_identity};

    #[test]
    fn config_values_win_when_present() {
        let id = resolve_hs256_identity(
            Some("issuer.example.com".to_string()),
            Some("team.adelie-ai".to_string()),
        );
        assert_eq!(id.issuer, "issuer.example.com");
        assert_eq!(id.audience, "team.adelie-ai");
    }

    #[test]
    fn unset_fields_fall_back_to_per_host_defaults() {
        let id = resolve_hs256_identity(None, None);
        assert_eq!(id.issuer, local_hostname(), "default iss is the hostname");
        assert_eq!(
            id.audience,
            format!("{}.adelie-ai", current_username()),
            "default aud is <user>.adelie-ai"
        );
    }

    #[test]
    fn blank_fields_are_treated_as_unset() {
        // Whitespace-only config values must not become the literal issuer/aud —
        // they fall back to the defaults, so a stray empty string can't lock you
        // out by minting tokens with an `aud` the validator won't expect.
        let id = resolve_hs256_identity(Some("   ".to_string()), Some(String::new()));
        assert_eq!(id.issuer, local_hostname());
        assert_eq!(id.audience, format!("{}.adelie-ai", current_username()));
    }

    #[test]
    fn one_field_overridden_other_defaulted() {
        let id = resolve_hs256_identity(Some("pinned-issuer".to_string()), None);
        assert_eq!(id.issuer, "pinned-issuer");
        assert_eq!(id.audience, format!("{}.adelie-ai", current_username()));
    }
}
