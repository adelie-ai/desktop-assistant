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

/// UDS auth for the **local trust model** (#407): authenticate by kernel
/// peer-credentials, deriving the `UserId` from the connecting peer's username.
/// No bearer token is required on a local Unix socket — the kernel-attested peer
/// UID (`SO_PEERCRED`, unforgeable) *is* the authentication. This is the
/// username the retired `adelie-mint` minter used to stamp as the JWT `sub`, so
/// per-user identity is preserved.
///
/// This **reverses** `architecture-evolution.md` rule #2 (uniform JWT on every
/// transport): JWT auth now belongs to the *remote* WS door only.
///
/// During the migration off the minter this stays **tolerant**: if the OS can't
/// supply peer credentials but the client still presents a valid bearer token
/// (the old uniform-JWT path), the token is accepted as a fallback. Once every
/// local client has stopped minting tokens (#407 step 3) the fallback can go.
pub(crate) struct PeerCredUdsAuth {
    /// JWT fallback for the (rare) peer-cred-unavailable case during migration.
    jwt_fallback: Arc<dyn ws::WsAuthValidator>,
}

impl PeerCredUdsAuth {
    pub(crate) fn new(jwt_fallback: Arc<dyn ws::WsAuthValidator>) -> Self {
        Self { jwt_fallback }
    }
}

#[async_trait]
impl uds::UdsAuthValidator for PeerCredUdsAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        self.jwt_fallback.validate_bearer_token(token).await
    }

    async fn extract_user_id(&self, token: &str) -> Option<desktop_assistant_application::UserId> {
        self.jwt_fallback.extract_user_id(token).await
    }

    async fn authenticate(
        &self,
        token: Option<&str>,
        peer: Option<&uds::PeerIdentity>,
    ) -> uds::UdsAuth {
        // Local trust: the kernel-attested peer is the authentication. Derive
        // the per-user identity from the peer username.
        if let Some(peer) = peer {
            return uds::UdsAuth::Allow(desktop_assistant_application::UserId::from(
                peer.username.clone(),
            ));
        }
        // Peer-cred unavailable — fall back to a valid bearer token (migration
        // tolerance; see the struct docs).
        match token {
            Some(t) if self.jwt_fallback.validate_bearer_token(t).await => uds::UdsAuth::Allow(
                self.jwt_fallback
                    .extract_user_id(t)
                    .await
                    .unwrap_or_default(),
            ),
            _ => uds::UdsAuth::Reject(
                "auth: no peer credentials and no valid bearer token".to_string(),
            ),
        }
    }
}

/// Resolve the UDS socket path. Precedence: the
/// `DESKTOP_ASSISTANT_UDS_SOCKET` env var (an empty value disables the socket),
/// then the `[transports].uds_socket` config override (`config_socket`; empty
/// disables), then the default desktop socket path.
pub(crate) fn resolve_uds_socket_path(config_socket: Option<&str>) -> Option<std::path::PathBuf> {
    if let Some(explicit) = std::env::var_os("DESKTOP_ASSISTANT_UDS_SOCKET") {
        let s = explicit.to_string_lossy().trim().to_string();
        if s.is_empty() {
            return None;
        }
        return Some(std::path::PathBuf::from(s));
    }
    if let Some(configured) = config_socket {
        let s = configured.trim();
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
    use super::{WsLoginMode, parse_env_bool, resolve_ws_login_mode_decision};
    use crate::config::TransportsConfig;

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
        // Local-first policy lives in `[transports]` (#279 item 3): WebSocket
        // off, UDS on (Unix). Bind to locals so the asserts are runtime checks
        // rather than constant-folded tautologies.
        let defaults = TransportsConfig::default();
        assert!(!defaults.ws_enabled, "WS must default off");
        assert_eq!(defaults.uds_enabled, cfg!(unix));
        assert_eq!(defaults.ws_bind, "127.0.0.1:11339");

        // The env knobs (via `parse_env_bool`) still flip each policy.
        assert!(parse_env_bool(Some("true"), defaults.ws_enabled));
        assert!(!parse_env_bool(Some("false"), defaults.uds_enabled));
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

    mod peer_cred_uds_auth {
        use std::sync::Arc;

        use async_trait::async_trait;
        use desktop_assistant_application::UserId;
        use desktop_assistant_uds::{PeerIdentity, UdsAuth, UdsAuthValidator};
        use desktop_assistant_ws as ws;

        use crate::transports::PeerCredUdsAuth;

        /// JWT fallback stub: accepts only the literal token `"good"`, whose
        /// `sub` is `"jwtuser"`.
        struct StubJwt;

        #[async_trait]
        impl ws::WsAuthValidator for StubJwt {
            async fn validate_bearer_token(&self, token: &str) -> bool {
                token == "good"
            }
            async fn extract_user_id(&self, token: &str) -> Option<UserId> {
                (token == "good").then(|| UserId::from("jwtuser"))
            }
        }

        fn auth() -> PeerCredUdsAuth {
            PeerCredUdsAuth::new(Arc::new(StubJwt))
        }

        fn peer(username: &str) -> PeerIdentity {
            PeerIdentity {
                uid: 1000,
                username: username.to_string(),
            }
        }

        fn allowed(outcome: UdsAuth) -> UserId {
            match outcome {
                UdsAuth::Allow(user) => user,
                UdsAuth::Reject(reason) => panic!("expected Allow, got Reject({reason})"),
            }
        }

        /// Peer-cred alone (no token) authenticates, and the `UserId` is the
        /// peer's username — the local trust model (#407).
        #[tokio::test]
        async fn peer_cred_without_token_authenticates_as_peer_user() {
            let outcome = auth().authenticate(None, Some(&peer("dave"))).await;
            assert_eq!(allowed(outcome), UserId::from("dave"));
        }

        /// Peer-cred wins even when a (valid) token is also presented — the
        /// kernel identity is ground truth on a local socket.
        #[tokio::test]
        async fn peer_cred_takes_precedence_over_a_token() {
            let outcome = auth().authenticate(Some("good"), Some(&peer("dave"))).await;
            assert_eq!(allowed(outcome), UserId::from("dave"));
        }

        /// Migration tolerance: with no peer-cred but a valid token, the token
        /// is accepted and its `sub` is the identity.
        #[tokio::test]
        async fn valid_token_is_accepted_when_peer_cred_is_unavailable() {
            let outcome = auth().authenticate(Some("good"), None).await;
            assert_eq!(allowed(outcome), UserId::from("jwtuser"));
        }

        /// Neither peer-cred nor a valid token → rejected.
        #[tokio::test]
        async fn no_peer_cred_and_no_valid_token_is_rejected() {
            assert!(matches!(
                auth().authenticate(None, None).await,
                UdsAuth::Reject(_)
            ));
            assert!(matches!(
                auth().authenticate(Some("bogus"), None).await,
                UdsAuth::Reject(_)
            ));
        }
    }
}
