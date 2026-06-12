//! Transport setup: auth validators, the WS login service, transport
//! enable/disable defaults, and the env/host resolution helpers shared by the
//! daemon's per-transport wiring (#279 item 4).
//!
//! Extracted verbatim from `main.rs` to slim the wiring god-function. The
//! types are `pub(crate)` so `main.rs` can name them while wiring each
//! transport; behavior is unchanged.

use std::sync::Arc;

use async_trait::async_trait;
use desktop_assistant_core::ports::inbound::SettingsService;

use crate::config;
use desktop_assistant_uds as uds;
use desktop_assistant_ws as ws;

pub(crate) struct WsSettingsAuth<S: SettingsService + 'static> {
    settings: Arc<S>,
}

impl<S: SettingsService + 'static> WsSettingsAuth<S> {
    pub(crate) fn new(settings: Arc<S>) -> Self {
        Self { settings }
    }
}

#[async_trait]
impl<S: SettingsService + 'static> ws::WsAuthValidator for WsSettingsAuth<S> {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        self.settings
            .validate_ws_jwt(token.to_string())
            .await
            .unwrap_or(false)
    }

    async fn extract_user_id(&self, token: &str) -> Option<desktop_assistant_application::UserId> {
        // #105 mapping rule: JWT `sub` → `UserId`. Returns `None` for
        // tokens this validator would reject; the ws-interface
        // handler then falls back to `UserId::default` (the schema
        // sentinel) so a single-tenant deploy without identity
        // information still resolves correctly.
        config::ws_jwt_sub(token).map(desktop_assistant_application::UserId::from)
    }
}

/// Auth validator that tries the local HS256 JWT first, then falls back to OIDC RS256.
pub(crate) struct OidcAwareAuth<S: SettingsService + 'static> {
    pub(crate) local: WsSettingsAuth<S>,
    pub(crate) oidc_validator: config::OidcValidator,
}

#[async_trait]
impl<S: SettingsService + 'static> ws::WsAuthValidator for OidcAwareAuth<S> {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        // Try local HS256 JWT first
        if self.local.validate_bearer_token(token).await {
            return true;
        }
        // Fall back to OIDC RS256 validation
        self.oidc_validator.validate_token(token).await
    }

    async fn extract_user_id(&self, token: &str) -> Option<desktop_assistant_application::UserId> {
        // Identity extraction must follow *acceptance*, not run independently
        // of it (#279 item 6). Each validator's `sub` is trusted only when
        // that same validator accepted the token: gate the local HS256
        // extraction on the local validator accepting, and likewise for OIDC.
        // Order mirrors `validate_bearer_token` — local HS256 mint first
        // (single-tenant desktop primary path), then OIDC RS256 (multi-tenant
        // deploys) — so the validator that would accept the token is the one
        // that yields its `sub`.
        if self.local.validate_bearer_token(token).await {
            return self.local.extract_user_id(token).await;
        }
        if self.oidc_validator.validate_token(token).await {
            return self
                .oidc_validator
                .extract_sub(token)
                .await
                .map(desktop_assistant_application::UserId::from);
        }
        None
    }
}

/// Provides auth discovery info from the daemon config.
pub(crate) struct WsAuthDiscoveryProvider {
    pub(crate) discovery: config::WsAuthDiscoveryInfo,
}

#[async_trait]
impl ws::WsAuthDiscovery for WsAuthDiscoveryProvider {
    async fn auth_config(&self) -> serde_json::Value {
        serde_json::to_value(&self.discovery)
            .unwrap_or_else(|_| serde_json::json!({ "methods": ["password"] }))
    }
}

/// Adapter: reuses the WS bearer-token validator for UDS connections so
/// both transports honor the same JWT policy (local HS256 + OIDC RS256
/// fallback) per `architecture-evolution.md` rule #2 (uniform JWT auth).
pub(crate) struct WsAsUdsAuth {
    validator: Arc<dyn ws::WsAuthValidator>,
}

impl WsAsUdsAuth {
    pub(crate) fn new(validator: Arc<dyn ws::WsAuthValidator>) -> Self {
        Self { validator }
    }
}

#[async_trait]
impl uds::UdsAuthValidator for WsAsUdsAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        self.validator.validate_bearer_token(token).await
    }

    async fn extract_user_id(&self, token: &str) -> Option<desktop_assistant_application::UserId> {
        // Delegate to the same WS validator so UDS and WS share the
        // JWT-to-user_id mapping. The bridge above already enforces
        // uniform validation; #105's identity extraction follows the
        // same path.
        self.validator.extract_user_id(token).await
    }
}

pub(crate) fn resolve_uds_socket_path() -> Option<std::path::PathBuf> {
    if let Some(explicit) = std::env::var_os("DESKTOP_ASSISTANT_UDS_SOCKET") {
        let s = explicit.to_string_lossy().trim().to_string();
        if s.is_empty() {
            return None;
        }
        return Some(std::path::PathBuf::from(s));
    }
    uds::default_desktop_socket_path()
}

pub(crate) struct WsBasicLogin<S: SettingsService + 'static> {
    settings: Arc<S>,
    username: String,
    mode: WsLoginMode,
}

pub(crate) enum WsLoginMode {
    StaticPassword(String),
    SystemPassword,
}

impl<S: SettingsService + 'static> WsBasicLogin<S> {
    pub(crate) fn new(settings: Arc<S>, username: String, mode: WsLoginMode) -> Self {
        Self {
            settings,
            username,
            mode,
        }
    }
}

#[async_trait]
impl<S: SettingsService + 'static> ws::WsLoginService for WsBasicLogin<S> {
    async fn authenticate_basic(&self, username: &str, password: &str) -> bool {
        if username != self.username {
            return false;
        }

        match &self.mode {
            // Constant-time compare so a byte-by-byte timing attacker
            // can't peel the password one prefix at a time. Mostly
            // theoretical for local-loopback HTTPS, but trivial to
            // get right (#37). `ct_eq` returns false on length
            // mismatch without short-circuiting per byte.
            WsLoginMode::StaticPassword(expected) => {
                use subtle::ConstantTimeEq;
                password.as_bytes().ct_eq(expected.as_bytes()).into()
            }
            WsLoginMode::SystemPassword => {
                match config::authenticate_os_user_password(username, password) {
                    Ok(valid) => valid,
                    Err(error) => {
                        tracing::warn!("system-password auth check failed: {error}");
                        false
                    }
                }
            }
        }
    }

    async fn issue_token_for_subject(&self, subject: &str) -> std::result::Result<String, String> {
        self.settings
            .generate_ws_jwt(Some(subject.to_string()))
            .await
            .map_err(|error| error.to_string())
    }
}

pub(crate) fn env_bool(name: &str, default: bool) -> bool {
    parse_env_bool(std::env::var(name).ok().as_deref(), default)
}

/// The daemon's self-identity **display label** for server-side tool localities
/// (#243) — the human-readable `host` shown in the tool note (e.g.
/// `terminal — server 'daemon-host'`). Co-location is decided separately by the
/// per-machine system-id handshake (#248), not by this label. Resolution is
/// dependency-free and best-effort: the Linux kernel hostname
/// (`/proc/sys/kernel/hostname`), then `/etc/hostname`, then the `HOSTNAME`
/// env var, falling back to `"this machine"` so the tool note is always
/// coherent.
pub(crate) fn daemon_host_label() -> String {
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
        .unwrap_or_else(|| "this machine".to_string())
}

/// Pure parser behind [`env_bool`], split out so the flag semantics are
/// unit-testable without touching the process environment. `None` (unset) and
/// unrecognized values fall back to `default`.
pub(crate) fn parse_env_bool(value: Option<&str>, default: bool) -> bool {
    match value {
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        None => default,
    }
}

/// Local-first transport defaults. Out of the box the daemon serves the local
/// transports (the D-Bus minter + UDS) and leaves the remote WebSocket
/// endpoint off until explicitly enabled. Each is overridable via the matching
/// `DESKTOP_ASSISTANT_*` env var; centralized here so the policy is documented
/// and pinned by tests.
pub(crate) mod transport_defaults {
    /// WebSocket listener is OFF by default (`DESKTOP_ASSISTANT_WS_ENABLED`).
    pub const WS_ENABLED: bool = false;
    /// D-Bus is best-effort by default — a missing/unavailable bus logs and
    /// the daemon continues (`DESKTOP_ASSISTANT_DBUS_REQUIRED`).
    pub const DBUS_REQUIRED: bool = false;
    /// UDS is ON by default on Unix targets (`DESKTOP_ASSISTANT_UDS_ENABLED`).
    pub fn uds_enabled() -> bool {
        cfg!(unix)
    }
}

pub(crate) fn is_container_environment() -> bool {
    std::env::var("container")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .is_some()
        || std::path::Path::new("/.dockerenv").exists()
        || std::path::Path::new("/run/.containerenv").exists()
}

pub(crate) fn resolve_ws_login_mode_decision(
    current_username: String,
    configured_username: Option<String>,
    configured_password: Option<String>,
    local_system_auth_enabled: bool,
    is_container: bool,
) -> Option<(String, WsLoginMode)> {
    if let Some(password) = configured_password {
        let username = configured_username.unwrap_or(current_username);
        return Some((username, WsLoginMode::StaticPassword(password)));
    }

    if local_system_auth_enabled && !is_container {
        return Some((current_username, WsLoginMode::SystemPassword));
    }

    None
}

pub(crate) fn resolve_ws_login_mode() -> Option<(String, WsLoginMode)> {
    let current_username = config::current_username();
    let configured_username = std::env::var("DESKTOP_ASSISTANT_WS_LOGIN_USERNAME")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let configured_password = std::env::var("DESKTOP_ASSISTANT_WS_LOGIN_PASSWORD")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let local_system_auth_enabled = env_bool("DESKTOP_ASSISTANT_WS_LOGIN_LOCAL_SYSTEM_AUTH", true);
    resolve_ws_login_mode_decision(
        current_username,
        configured_username,
        configured_password,
        local_system_auth_enabled,
        is_container_environment(),
    )
}

#[cfg(test)]
mod tests {
    use super::{WsLoginMode, parse_env_bool, resolve_ws_login_mode_decision, transport_defaults};

    #[test]
    fn parse_env_bool_recognizes_truthy_and_falsy() {
        for v in ["1", "true", "TRUE", "Yes", " on "] {
            assert!(parse_env_bool(Some(v), false), "{v:?} should parse true");
        }
        for v in ["0", "false", "No", "off"] {
            assert!(!parse_env_bool(Some(v), true), "{v:?} should parse false");
        }
    }

    #[test]
    fn parse_env_bool_falls_back_to_default() {
        assert!(parse_env_bool(None, true));
        assert!(!parse_env_bool(None, false));
        // Unrecognized values fall through to the supplied default.
        assert!(parse_env_bool(Some("maybe"), true));
        assert!(!parse_env_bool(Some("maybe"), false));
    }

    #[test]
    fn transport_defaults_are_local_first() {
        // Local-first policy: WebSocket off, D-Bus best-effort (not required),
        // UDS on (Unix). Bind to locals so the asserts are runtime checks of
        // the policy constants rather than constant-folded tautologies.
        let ws_enabled = transport_defaults::WS_ENABLED;
        let dbus_required = transport_defaults::DBUS_REQUIRED;
        assert!(!ws_enabled, "WS must default off");
        assert!(!dbus_required, "D-Bus must be optional by default");
        assert_eq!(transport_defaults::uds_enabled(), cfg!(unix));

        // The env knobs still flip each policy.
        assert!(parse_env_bool(Some("true"), transport_defaults::WS_ENABLED));
        assert!(parse_env_bool(
            Some("true"),
            transport_defaults::DBUS_REQUIRED
        ));
        assert!(!parse_env_bool(
            Some("false"),
            transport_defaults::uds_enabled()
        ));
    }

    #[test]
    fn static_password_mode_uses_configured_username() {
        let result = resolve_ws_login_mode_decision(
            "local-user".to_string(),
            Some("api-user".to_string()),
            Some("secret".to_string()),
            true,
            false,
        );

        match result {
            Some((username, WsLoginMode::StaticPassword(password))) => {
                assert_eq!(username, "api-user");
                assert_eq!(password, "secret");
            }
            _ => panic!("expected static password mode"),
        }
    }

    #[test]
    fn static_password_mode_defaults_to_current_username() {
        let result = resolve_ws_login_mode_decision(
            "local-user".to_string(),
            None,
            Some("secret".to_string()),
            true,
            false,
        );

        match result {
            Some((username, WsLoginMode::StaticPassword(password))) => {
                assert_eq!(username, "local-user");
                assert_eq!(password, "secret");
            }
            _ => panic!("expected static password mode"),
        }
    }

    #[test]
    fn system_password_mode_ignores_configured_username() {
        let result = resolve_ws_login_mode_decision(
            "local-user".to_string(),
            Some("other-user".to_string()),
            None,
            true,
            false,
        );

        match result {
            Some((username, WsLoginMode::SystemPassword)) => {
                assert_eq!(username, "local-user");
            }
            _ => panic!("expected system password mode"),
        }
    }

    #[test]
    fn login_mode_disabled_in_container_without_static_password() {
        let result =
            resolve_ws_login_mode_decision("local-user".to_string(), None, None, true, true);
        assert!(result.is_none());
    }

    #[test]
    fn login_mode_disabled_when_local_system_auth_is_off() {
        let result =
            resolve_ws_login_mode_decision("local-user".to_string(), None, None, false, false);
        assert!(result.is_none());
    }
}
