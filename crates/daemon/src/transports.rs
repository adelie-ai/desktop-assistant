//! Transport setup: auth validators, the WS login service, transport
//! enable/disable defaults, and the env/host resolution helpers shared by the
//! daemon's per-transport wiring (#279 item 4).
//!
//! Extracted verbatim from `main.rs` to slim the wiring god-function. The
//! types are `pub(crate)` so `main.rs` can name them while wiring each
//! transport; behavior is unchanged.

use std::path::Path;
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

/// What the remote WebSocket door will do at bind time (#805 review): stay
/// off entirely (the desktop default), or bind - optionally with TLS.
///
/// Why a combined type: `resolve_ws_tls` alone cannot demonstrate that the
/// fail-closed TLS check is unreachable when the door is off, because it
/// has no `ws_enabled` input - a caller has to trust that `main.rs` gates
/// the call correctly. `resolve_ws_door_plan` is that gate, made callable
/// and testable in its own right, so "TLS failure is irrelevant when the
/// door is off" is a fact this module proves by composing the real
/// production decision, not a fact asserted about an unrelated default.
pub(crate) enum WsDoorPlan {
    /// `ws_enabled` is `false`: the door does not bind at all.
    Disabled,
    /// The door binds. `tls` is `Some` when it serves TLS, `None` when TLS
    /// is deliberately off (`[tls] enabled = false`).
    Enabled {
        tls: Option<Arc<rustls::ServerConfig>>,
    },
}

/// Decide the WebSocket door's bind plan (#805 review): composes the
/// `ws_enabled` gate with [`crate::tls::resolve_ws_tls`]. When the door is
/// off, TLS is never resolved at all - a broken or missing cert/key
/// configuration cannot affect a desktop user who never turned the door on,
/// because this function returns `Disabled` before looking at it.
pub(crate) fn resolve_ws_door_plan(
    ws_enabled: bool,
    tls_enabled: bool,
    cert_file: Option<&Path>,
    key_file: Option<&Path>,
) -> anyhow::Result<WsDoorPlan> {
    if !ws_enabled {
        return Ok(WsDoorPlan::Disabled);
    }
    let tls = match crate::tls::resolve_ws_tls(tls_enabled, cert_file, key_file)? {
        crate::tls::WsTlsPosture::PlaintextByConfig => None,
        crate::tls::WsTlsPosture::Tls(server_config) => Some(server_config),
    };
    Ok(WsDoorPlan::Enabled { tls })
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

    /// The `resolve_ws_door_plan` acceptance suite (#805 review): the
    /// combined `ws_enabled` + TLS decision that `main.rs` wires directly,
    /// so these tests exercise the real production glue rather than a
    /// default value that happens to coincide with it.
    mod ws_door_plan {
        use std::path::PathBuf;

        use crate::transports::{WsDoorPlan, resolve_ws_door_plan};

        fn install_crypto_provider() {
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        }

        /// Writes a fresh self-signed cert + key PEM pair to `dir` and
        /// returns their paths. Mirrors `tls::tests::write_self_signed_cert`
        /// (kept local rather than shared across modules - two call sites,
        /// not three, so a shared test helper is not yet earned per 7.3).
        fn write_self_signed_cert(dir: &std::path::Path) -> (PathBuf, PathBuf) {
            let certified = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
                .expect("generate self-signed cert for test fixture");
            let cert_path = dir.join("cert.pem");
            let key_path = dir.join("key.pem");
            std::fs::write(&cert_path, certified.cert.pem()).expect("write test cert");
            std::fs::write(&key_path, certified.signing_key.serialize_pem())
                .expect("write test key");
            (cert_path, key_path)
        }

        /// Acceptance (#805): the single-user desktop case, where the remote
        /// door is off. It calls the real production decision function with a
        /// deliberately broken TLS configuration, and proves that a desktop
        /// user who never opted into the remote WebSocket door never reaches
        /// the TLS fail-closed startup check, whatever a stale or invalid
        /// `[tls]` section says. If a later edit hoisted TLS resolution above
        /// the `ws_enabled` gate, this test fails with an `Err` in place of
        /// `Disabled`.
        #[test]
        fn disabled_when_ws_is_off_even_with_a_broken_tls_config() {
            let dir = tempfile::tempdir().unwrap();
            let missing_cert = dir.path().join("does-not-exist-cert.pem");
            let missing_key = dir.path().join("does-not-exist-key.pem");

            let plan = resolve_ws_door_plan(false, true, Some(&missing_cert), Some(&missing_key))
                .expect("the door being off must short-circuit before TLS is ever resolved");

            assert!(
                matches!(plan, WsDoorPlan::Disabled),
                "ws_enabled = false must yield WsDoorPlan::Disabled regardless of the TLS \
                 configuration's validity"
            );
        }

        /// Acceptance: the door is on and TLS is configured and working.
        #[test]
        fn enabled_with_tls_when_configured_and_working() {
            install_crypto_provider();
            let dir = tempfile::tempdir().unwrap();
            let (cert_path, key_path) = write_self_signed_cert(dir.path());

            let plan = resolve_ws_door_plan(true, true, Some(&cert_path), Some(&key_path))
                .expect("valid cert/key must resolve, not error");

            assert!(
                matches!(plan, WsDoorPlan::Enabled { tls: Some(_) }),
                "TLS configured and working must yield an Enabled plan carrying a TLS acceptor"
            );
        }

        /// Acceptance: the door is on and TLS is deliberately off.
        #[test]
        fn enabled_without_tls_when_deliberately_off() {
            let plan = resolve_ws_door_plan(true, false, None, None)
                .expect("TLS disabled must resolve, not error");

            assert!(
                matches!(plan, WsDoorPlan::Enabled { tls: None }),
                "TLS disabled by configuration must yield an Enabled plan with no TLS acceptor"
            );
        }

        /// Acceptance: the door is on and TLS is configured and failing.
        /// This is the composed version of `tls::resolve_ws_tls`'s own
        /// failing-config test: it proves the `?` in `resolve_ws_door_plan`
        /// actually propagates the failure rather than swallowing it on the
        /// way through the extra layer of composition.
        #[test]
        fn fails_closed_when_ws_enabled_and_tls_configured_and_failing() {
            install_crypto_provider();
            let dir = tempfile::tempdir().unwrap();
            let missing_cert = dir.path().join("does-not-exist-cert.pem");
            let missing_key = dir.path().join("does-not-exist-key.pem");

            let result = resolve_ws_door_plan(true, true, Some(&missing_cert), Some(&missing_key));

            assert!(
                result.is_err(),
                "the door being on with a broken TLS config must fail closed (Err), \
                 never silently bind plaintext"
            );
        }
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
