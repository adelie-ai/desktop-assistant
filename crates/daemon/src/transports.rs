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
use desktop_assistant_transport_dispatch::{AdminSubjects, Capability, capability_for_local_peer};
use desktop_assistant_uds as uds;
use desktop_assistant_ws as ws;

pub(crate) struct WsSettingsAuth<S: SettingsService + 'static> {
    settings: Arc<S>,
    /// The operator's remote-administrator allowlist (#728), read once from
    /// `[authz] admin_subjects`. Empty by default, so an unconfigured daemon
    /// admits nobody to the admin surface over the network.
    admin_subjects: Arc<AdminSubjects>,
}

impl<S: SettingsService + 'static> WsSettingsAuth<S> {
    pub(crate) fn new(settings: Arc<S>, admin_subjects: Arc<AdminSubjects>) -> Self {
        Self {
            settings,
            admin_subjects,
        }
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

    fn capability_for_subject(&self, subject: &str) -> Capability {
        // Remote is admin only by explicit allowlist (#728): a token proves who
        // the caller is, never that they own the service.
        self.admin_subjects.capability_for(subject)
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

    fn capability_for_subject(&self, subject: &str) -> Capability {
        // One allowlist for the door, whichever issuer authenticated the token:
        // OIDC validates issuer and audience, never who administers this daemon.
        self.local.capability_for_subject(subject)
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
    /// The uid this daemon process runs as (#728). A peer that matches it is
    /// the person who runs the daemon, so it administers the daemon - which is
    /// what makes the single-user desktop need no configuration at all.
    daemon_uid: u32,
    /// The operator's allowlist, so a multi-user host can name a second
    /// administrator without a code change. Also the only promotion available
    /// on the token-fallback path, which has no unforgeable uid to compare.
    admin_subjects: Arc<AdminSubjects>,
}

impl PeerCredUdsAuth {
    pub(crate) fn new(
        jwt_fallback: Arc<dyn ws::WsAuthValidator>,
        daemon_uid: u32,
        admin_subjects: Arc<AdminSubjects>,
    ) -> Self {
        Self {
            jwt_fallback,
            daemon_uid,
            admin_subjects,
        }
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
            // Two independent grants, and the higher one wins (#728): the peer
            // uid says whether this is the daemon's own account, and the
            // allowlist can name another local account as an administrator.
            let capability = capability_for_local_peer(peer.uid, self.daemon_uid)
                .strongest(self.admin_subjects.capability_for(&peer.username));
            return uds::UdsAuth::Allow {
                user: desktop_assistant_application::UserId::from(peer.username.clone()),
                capability,
            };
        }
        // Peer-cred unavailable — fall back to a valid bearer token (migration
        // tolerance; see the struct docs).
        match token {
            Some(t) if self.jwt_fallback.validate_bearer_token(t).await => {
                let user = self
                    .jwt_fallback
                    .extract_user_id(t)
                    .await
                    .unwrap_or_default();
                // No peer credentials means no unforgeable uid to compare, so
                // the local grant does not apply: only the allowlist can
                // promote this connection.
                let capability = self.admin_subjects.capability_for(user.as_str());
                uds::UdsAuth::Allow { user, capability }
            }
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

/// The WebSocket `/login` door: HTTP Basic against a single configured
/// account, exchanged for an HS256 bearer token.
///
/// One account, so one identity. `username` is both the only credential this
/// door accepts and the `sub` it stamps on the token it returns — and that
/// `sub` is the `UserId` every conversation, knowledge entry and scratchpad
/// note the connection writes is scoped by. On a desktop that account is the
/// daemon's OS user; in a container it is whatever
/// `DESKTOP_ASSISTANT_WS_LOGIN_USERNAME` names, which is then the tenant.
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
        // Defense in depth. `authenticate_basic` already accepts only
        // `self.username`, so the handler cannot reach here with anything else;
        // re-checking keeps that true if a future caller separates the two
        // steps. The token's `sub` becomes the storage `user_id`, so minting one
        // for an unauthenticated identity would hand out another tenant's
        // partition — fail loudly instead.
        if subject != self.username {
            tracing::warn!(
                "ws login: refusing to issue a token for subject {subject:?}; \
                 only the configured login user is authenticated here"
            );
            return Err(
                "login cannot issue a token for a subject it did not authenticate".to_string(),
            );
        }

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
            None,
            false,
            true,
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
            None,
            false,
            true,
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
            None,
            false,
            true,
        );

        match result {
            Some((username, WsLoginMode::SystemPassword(_))) => {
                assert_eq!(username, "local-user");
            }
            _ => panic!("expected system password mode"),
        }
    }

    #[test]
    fn login_mode_disabled_in_container_without_static_password() {
        let result =
            resolve_ws_login_mode_decision("local-user".to_string(), None, None, None, true, true);
        assert!(result.is_none());
    }

    #[test]
    fn login_mode_disabled_when_local_system_auth_is_off() {
        let result = resolve_ws_login_mode_decision(
            "local-user".to_string(),
            None,
            None,
            Some(false),
            false,
            true,
        );
        assert!(result.is_none());
    }

    /// #806: the OS-password door is a *local* convenience, and the bind
    /// address is what says whether the door is local. A daemon bound past
    /// loopback with the flag untouched must not turn `/login` into a
    /// network-reachable PAM oracle for a real system account.
    mod system_password_needs_a_local_door {
        use super::super::{WsLoginMode, resolve_ws_login_mode_decision};

        fn decide(
            local_system_auth: Option<bool>,
            is_container: bool,
            bind_is_loopback: bool,
        ) -> Option<(String, WsLoginMode)> {
            resolve_ws_login_mode_decision(
                "local-user".to_string(),
                None,
                None,
                local_system_auth,
                is_container,
                bind_is_loopback,
            )
        }

        #[test]
        fn system_password_login_is_off_by_default_on_a_non_loopback_bind() {
            assert!(
                decide(None, false, false).is_none(),
                "a daemon bound past loopback must not accept the host account password \
                 unless the operator asked for it"
            );
        }

        #[test]
        fn system_password_login_on_a_non_loopback_bind_needs_an_explicit_opt_in() {
            assert!(
                matches!(decide(Some(true), false, false), Some((_, WsLoginMode::SystemPassword(_)))),
                "an operator who sets the flag deliberately still gets the mode"
            );
        }

        /// The single-user desktop case, which must not change: the daemon
        /// binds loopback, so the OS-password door stays on with no
        /// configuration at all.
        #[test]
        fn system_password_login_stays_on_for_a_loopback_bind() {
            assert!(
                matches!(decide(None, false, true), Some((_, WsLoginMode::SystemPassword(_)))),
                "the loopback door is the case this mode was designed for"
            );
        }

        #[test]
        fn an_explicit_no_disables_system_password_login_on_every_bind() {
            assert!(decide(Some(false), false, true).is_none());
            assert!(decide(Some(false), false, false).is_none());
        }

        #[test]
        fn a_container_never_gets_system_password_login() {
            assert!(decide(None, true, true).is_none());
            assert!(decide(Some(true), true, false).is_none());
        }

        /// The static-password door is the deployed shape and is unrelated to
        /// the host's OS accounts, so the bind address does not gate it.
        #[test]
        fn static_password_login_is_unaffected_by_the_bind_address() {
            let decided = resolve_ws_login_mode_decision(
                "local-user".to_string(),
                Some("api-user".to_string()),
                Some("secret".to_string()),
                None,
                false,
                false,
            );
            assert!(matches!(decided, Some((_, WsLoginMode::StaticPassword(_)))));
        }
    }

    /// The flag now has three states, not two: an operator who never set it is
    /// not the same as one who set it to `false`, because only the first can be
    /// overridden by the bind address.
    #[test]
    fn parse_env_opt_bool_distinguishes_unset_from_a_stated_value() {
        use super::parse_env_opt_bool;
        assert_eq!(parse_env_opt_bool(None), None);
        assert_eq!(parse_env_opt_bool(Some("")), None);
        assert_eq!(parse_env_opt_bool(Some("maybe")), None);
        assert_eq!(parse_env_opt_bool(Some("true")), Some(true));
        assert_eq!(parse_env_opt_bool(Some(" ON ")), Some(true));
        assert_eq!(parse_env_opt_bool(Some("0")), Some(false));
    }

    /// #806: the blocking PAM call must not run on the async runtime. Each
    /// failed guess used to park a tokio worker thread for the whole of
    /// libpam's fail delay, so an attacker throttled the daemon itself.
    mod os_password_check_is_blocking_work {
        use std::path::PathBuf;
        use std::sync::Arc;
        use std::time::Duration;

        use desktop_assistant_ws::WsLoginService;

        use crate::settings_service::DaemonSettingsService;
        use crate::transports::{WsBasicLogin, WsLoginMode};

        fn login(mode: WsLoginMode) -> WsBasicLogin<DaemonSettingsService> {
            WsBasicLogin::new(
                Arc::new(DaemonSettingsService::new(PathBuf::from(
                    "/nonexistent/desktop-assistant-ws-login-806.toml",
                ))),
                "local-user".to_string(),
                mode,
            )
        }

        /// Stands in for libpam's fail delay without needing a PAM stack.
        fn slow_check(_username: &str, _password: &str) -> anyhow::Result<bool> {
            std::thread::sleep(Duration::from_millis(300));
            Ok(false)
        }

        fn broken_check(_username: &str, _password: &str) -> anyhow::Result<bool> {
            Err(anyhow::anyhow!("PAM is not available on this host"))
        }

        #[tokio::test]
        async fn the_os_password_check_does_not_block_the_async_runtime() {
            let door = login(WsLoginMode::SystemPassword(slow_check));
            let check = tokio::spawn(async move { door.authenticate_basic("local-user", "guess").await });

            // This test runtime is single-threaded, so an inline blocking call
            // would hold its only worker for the whole 300 ms and this short
            // timer could not fire.
            tokio::time::timeout(
                Duration::from_millis(120),
                tokio::time::sleep(Duration::from_millis(10)),
            )
            .await
            .expect("the runtime must stay responsive while the OS password check runs");

            assert!(!check.await.expect("the check task must finish"));
        }

        /// PAM unavailable is not an authentication. The check reports the
        /// error and the door stays shut.
        #[tokio::test]
        async fn a_failing_os_password_check_is_not_an_authentication() {
            let door = login(WsLoginMode::SystemPassword(broken_check));
            assert!(!door.authenticate_basic("local-user", "whatever").await);
        }
    }

    /// The `/login` door's end of the tenant boundary (#726): the token it hands
    /// back must name the account it authenticated, because that `sub` is the
    /// `user_id` every later write is filed under.
    mod ws_login_subject {
        use std::path::PathBuf;
        use std::sync::Arc;

        use desktop_assistant_ws as ws;
        use ws::WsLoginService;

        use crate::config;
        use crate::settings_service::DaemonSettingsService;
        use crate::transports::{WsBasicLogin, WsLoginMode};

        const LOGIN_USER: &str = "api-user";
        const LOGIN_PASSWORD: &str = "correct-horse";

        /// A login door in the deployed container shape: static password, with
        /// an operator-configured username that is deliberately not the OS
        /// account the daemon runs as.
        fn login() -> WsBasicLogin<DaemonSettingsService> {
            WsBasicLogin::new(
                Arc::new(DaemonSettingsService::new(PathBuf::from(
                    "/nonexistent/desktop-assistant-ws-login-726.toml",
                ))),
                LOGIN_USER.to_string(),
                WsLoginMode::StaticPassword(LOGIN_PASSWORD.to_string()),
            )
        }

        /// Drive `future` to completion on a fresh current-thread runtime, so
        /// the `XDG_DATA_HOME` guard is never held across an `.await`.
        fn run_async<F: std::future::Future>(future: F) -> F::Output {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build current-thread runtime")
                .block_on(future)
        }

        #[test]
        fn login_token_subject_is_the_configured_login_user_not_the_daemon_os_user() {
            let sub = config::with_isolated_xdg_data_home("ws-login-subject", || {
                let token = run_async(login().issue_token_for_subject(LOGIN_USER))
                    .expect("issuing a token for the authenticated user should succeed");
                config::ws_jwt_sub(&token).expect("issued token must validate and carry a sub")
            });

            assert_eq!(
                sub, LOGIN_USER,
                "the token's sub is the storage user_id; it must be the account /login authenticated"
            );
        }

        /// Cross-tenant guard: the door only mints for the identity it can
        /// authenticate. Anything else fails loudly rather than silently
        /// meaning someone else.
        #[test]
        fn login_refuses_to_mint_a_token_for_a_subject_it_did_not_authenticate() {
            // Map to a token-free outcome before asserting: a failure message
            // must never print the bearer token itself.
            let issued = config::with_isolated_xdg_data_home("ws-login-subject-mismatch", || {
                run_async(login().issue_token_for_subject("someone-else")).is_ok()
            });

            assert!(
                !issued,
                "a subject other than the configured login user must not be issued a token"
            );
        }

        /// The gate that makes honouring the requested subject safe: basic auth
        /// accepts only the configured username, so the subject reaching
        /// `issue_token_for_subject` is always an authenticated one.
        #[test]
        fn basic_auth_rejects_a_username_other_than_the_configured_one() {
            assert!(
                run_async(login().authenticate_basic(LOGIN_USER, LOGIN_PASSWORD)),
                "the configured user with the right password must authenticate"
            );
            assert!(
                !run_async(login().authenticate_basic("someone-else", LOGIN_PASSWORD)),
                "a different username must not authenticate, even with the right password"
            );
            assert!(
                !run_async(login().authenticate_basic(LOGIN_USER, "wrong")),
                "the configured user with a wrong password must not authenticate"
            );
        }
    }

    mod peer_cred_uds_auth {
        use std::sync::Arc;

        use async_trait::async_trait;
        use desktop_assistant_application::UserId;
        use desktop_assistant_transport_dispatch::{AdminSubjects, Capability};
        use desktop_assistant_uds::{PeerIdentity, UdsAuth, UdsAuthValidator};
        use desktop_assistant_ws as ws;

        use crate::transports::PeerCredUdsAuth;

        fn capability(outcome: UdsAuth) -> Capability {
            match outcome {
                UdsAuth::Allow { capability, .. } => capability,
                UdsAuth::Reject(reason) => panic!("expected Allow, got Reject({reason})"),
            }
        }

        /// JWT fallback stub. `"good"` is the ordinary case: accepted, `sub` is
        /// `"jwtuser"`. The other two accepted tokens reproduce the identity
        /// fail-open this work closes - a token an issuer signs correctly but
        /// that names no usable subject.
        struct StubJwt;

        #[async_trait]
        impl ws::WsAuthValidator for StubJwt {
            async fn validate_bearer_token(&self, token: &str) -> bool {
                matches!(token, "good" | "no-subject" | "blank-subject")
            }
            async fn extract_user_id(&self, token: &str) -> Option<UserId> {
                match token {
                    "good" => Some(UserId::from("jwtuser")),
                    "blank-subject" => Some(UserId::from("   ")),
                    _ => None,
                }
            }
        }

        /// The uid this daemon pretends to run as in these tests.
        const DAEMON_UID: u32 = 1000;

        fn auth() -> PeerCredUdsAuth {
            PeerCredUdsAuth::new(
                Arc::new(StubJwt),
                DAEMON_UID,
                Arc::new(AdminSubjects::default()),
            )
        }

        fn auth_with_admins(subjects: &[&str]) -> PeerCredUdsAuth {
            PeerCredUdsAuth::new(
                Arc::new(StubJwt),
                DAEMON_UID,
                Arc::new(AdminSubjects::new(subjects.iter().copied())),
            )
        }

        fn peer(username: &str) -> PeerIdentity {
            peer_with_uid(username, DAEMON_UID)
        }

        fn peer_with_uid(username: &str, uid: u32) -> PeerIdentity {
            PeerIdentity {
                uid,
                username: username.to_string(),
                real_name: None,
                home_dir: None,
            }
        }

        fn allowed(outcome: UdsAuth) -> UserId {
            match outcome {
                UdsAuth::Allow { user, .. } => user,
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

        /// #807: identity resolution is part of acceptance. A token the
        /// validator accepts but whose subject it cannot name must be rejected,
        /// not collapsed to the schema sentinel `"default"` - which on a
        /// desktop-originated database is the operator's own partition, and
        /// which an allowlist can name.
        #[tokio::test]
        async fn token_fallback_is_rejected_when_the_token_names_no_subject() {
            assert!(matches!(
                auth().authenticate(Some("no-subject"), None).await,
                UdsAuth::Reject(_)
            ));
        }

        /// A blank subject is no subject.
        #[tokio::test]
        async fn token_fallback_is_rejected_when_the_subject_is_blank() {
            assert!(matches!(
                auth().authenticate(Some("blank-subject"), None).await,
                UdsAuth::Reject(_)
            ));
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

        // --- authorization tier (#728) -------------------------------------

        /// Local is admin by construction: a peer whose kernel-attested uid is
        /// the daemon's own owns the daemon. This is what lets the single-user
        /// desktop work with no `[authz]` configuration at all.
        #[tokio::test]
        async fn peer_cred_grants_admin_to_the_daemons_own_uid() {
            let outcome = auth().authenticate(None, Some(&peer("dave"))).await;
            assert_eq!(capability(outcome), Capability::Admin);
        }

        /// Another local account on the same host is a tenant. It still
        /// authenticates - it simply does not run the daemon.
        #[tokio::test]
        async fn peer_cred_grants_tenant_to_another_uid() {
            let outcome = auth()
                .authenticate(None, Some(&peer_with_uid("someone", 1001)))
                .await;
            assert_eq!(capability(outcome), Capability::Tenant);
        }

        /// The allowlist also promotes a named local account, so a multi-user
        /// host can name a second administrator without a code change.
        #[tokio::test]
        async fn allowlisted_local_subject_is_admin_despite_a_different_uid() {
            let outcome = auth_with_admins(&["someone"])
                .authenticate(None, Some(&peer_with_uid("someone", 1001)))
                .await;
            assert_eq!(capability(outcome), Capability::Admin);
        }

        /// The token fallback never inherits the local-peer grant: with no
        /// peer credentials there is no unforgeable uid to compare, so only
        /// the allowlist can promote the subject.
        #[tokio::test]
        async fn token_fallback_is_tenant_unless_the_subject_is_allowlisted() {
            let outcome = auth().authenticate(Some("good"), None).await;
            assert_eq!(capability(outcome), Capability::Tenant);

            let outcome = auth_with_admins(&["jwtuser"])
                .authenticate(Some("good"), None)
                .await;
            assert_eq!(capability(outcome), Capability::Admin);
        }
    }

    /// The remote door's end of the tier (#728): a WebSocket subject is an
    /// administrator only when `[authz] admin_subjects` names it.
    mod ws_admin_subjects {
        use std::path::PathBuf;
        use std::sync::Arc;

        use desktop_assistant_transport_dispatch::{AdminSubjects, Capability};
        use desktop_assistant_ws::WsAuthValidator;

        use crate::config::DaemonConfig;
        use crate::settings_service::DaemonSettingsService;
        use crate::transports::WsSettingsAuth;

        /// The settings service is irrelevant to the capability decision, so
        /// point it at a path that does not exist.
        fn validator(subjects: &[&str]) -> WsSettingsAuth<DaemonSettingsService> {
            WsSettingsAuth::new(
                Arc::new(DaemonSettingsService::new(PathBuf::from(
                    "/nonexistent/desktop-assistant-authz-728.toml",
                ))),
                Arc::new(AdminSubjects::new(subjects.iter().copied())),
            )
        }

        /// The default (empty) allowlist admits nobody remotely.
        #[test]
        fn absent_subject_is_a_tenant() {
            assert_eq!(
                validator(&[]).capability_for_subject("alice"),
                Capability::Tenant
            );
            assert_eq!(
                validator(&["operator"]).capability_for_subject("alice"),
                Capability::Tenant
            );
        }

        /// A listed subject is an administrator.
        #[test]
        fn listed_subject_is_an_admin() {
            assert_eq!(
                validator(&["operator", "alice"]).capability_for_subject("alice"),
                Capability::Admin
            );
        }

        /// The allowlist is built straight from `[authz] admin_subjects`, and a
        /// config with no `[authz]` section grants nobody.
        #[test]
        fn allowlist_comes_from_the_authz_config_section() {
            let empty = AdminSubjects::from(&DaemonConfig::default().authz);
            assert_eq!(empty.capability_for("alice"), Capability::Tenant);

            let cfg: DaemonConfig = toml::from_str("[authz]\nadmin_subjects = [\"alice\"]\n")
                .expect("parse authz config");
            let allowlist = AdminSubjects::from(&cfg.authz);
            assert_eq!(allowlist.capability_for("alice"), Capability::Admin);
            assert_eq!(allowlist.capability_for("bob"), Capability::Tenant);
        }
    }
}
