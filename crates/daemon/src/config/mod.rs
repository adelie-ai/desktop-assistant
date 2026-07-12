// Submodules (#41 — split into focused modules).
mod jwt;
mod migration;
mod oidc;
#[cfg(target_os = "linux")]
mod pam_auth;
mod reload;
mod resolution;
mod secrets;
mod views;

// Hot-reload classification (#222): pure old/new config diff into a
// `ReloadPlan`. Re-exported at `config::` so `api_surface` and `main` address
// them without the submodule path.
pub use reload::{ReloadPlan, plan_reload};

// Re-export the settings views public API at `config::` so existing
// callers (the inbound API handler, the dbus settings adapter) keep
// working unchanged. `BackendTasksSettingsViewConfig` is a return
// type that flows through inference at every current call site, so
// the lint pass treats the named re-export as unused; allowing it
// keeps the type discoverable through `config::` for callers that
// want to spell it out.
#[allow(unused_imports)]
pub use views::BackendTasksSettingsViewConfig;
pub use views::{
    get_backend_tasks_settings_view, get_connector_defaults, get_database_settings_view,
    get_embeddings_settings_view, get_llm_settings_view, get_persistence_settings_view,
    get_ws_auth_discovery, get_ws_auth_settings, set_api_key, set_backend_tasks_settings,
    set_database_settings, set_embeddings_settings, set_llm_settings, set_persistence_settings,
    set_ws_auth_settings,
};

// Re-export the JWT + OIDC public API at the `config::` path so existing
// callers (`config::generate_ws_jwt`, `config::OidcValidator`, etc.)
// keep working unchanged.
pub use jwt::{
    current_username, generate_ws_jwt, prune_revocations, revoke_ws_jwt, validate_ws_jwt,
    ws_jwt_sub,
};
// Crate-internal: seeded once from main at startup (not part of the public API).
pub(crate) use jwt::init_hs256_identity;
pub use oidc::OidcValidator;

// Resolution helpers — public API stays at `config::` for callers in
// the registry, dispatch wrappers, dreaming/titling, etc. The lowercase
// helpers (`parse_connector_or_openai`, `default_*_model`, etc.) are
// re-exported so the sibling `views` module can keep addressing them
// through `super::`. `DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS` only has
// internal callers in the test module, so the re-export trips the
// unused-imports lint on bin builds; allowing it keeps the constant
// at `config::` for external visibility.
#[allow(unused_imports)]
pub use resolution::DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS;
pub use resolution::{
    apply_learned_cap, purpose_max_context_override, resolve_backend_tasks_llm_config,
    resolve_connection_llm_config, resolve_consolidation_llm_config, resolve_context_budget,
    resolve_database_config, resolve_embeddings_config, resolve_llm_config,
    resolve_persistence_config, resolve_purpose_llm_config,
};
pub(super) use resolution::{
    default_backend_llm_model, default_base_url, default_llm_model, normalize_optional_value,
    parse_connector_or_openai,
};
// Bring the secrets-backend helpers used by non-test code in
// `mod.rs` (settings setters, audit logging) into scope so call
// sites don't need `secrets::…` prefixes. The sibling `views`
// module addresses `write_secret_to_backend` through `super::` so
// the re-import keeps it visible. Test-only callers reference
// `secrets::sanitize_secret_value` directly to avoid a cfg-gated
// `use`.
use secrets::{
    bucket_secret_len, is_placeholder_secret_value, redacted_secret_audit, write_secret_to_backend,
};

use std::path::{Path, PathBuf};

use anyhow::Context;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::connections::{ConnectionConfig, ConnectionId, ConnectionsError, ConnectionsMap};
// `PurposeKind`, `BudgetSource` are only referenced by the test module
// inside this file. Bring them in unconditionally so test discovery
// works without an extra `#[cfg(test)] use` block; the warnings on
// non-test builds are silenced by `#[allow(unused_imports)]`.
#[allow(unused_imports)]
use crate::purposes::PurposeKind;
use crate::purposes::Purposes;
#[allow(unused_imports)]
use desktop_assistant_core::ports::llm::{BudgetSource, ContextBudget};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct DaemonConfig {
    #[serde(default)]
    pub llm: LlmConfig,
    /// Named connector instances. Each entry owns its own credentials and
    /// endpoint; the `type` tag selects which connector implementation is used.
    ///
    /// Populated by deserialize as `IndexMap<String, ConnectionConfig>` so TOML
    /// parse errors surface before id-slug validation. [`load_daemon_config`]
    /// re-wraps the map as a validated [`ConnectionsMap`], rejecting invalid
    /// or duplicate ids.
    ///
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub connections: IndexMap<String, ConnectionConfig>,
    #[serde(default)]
    pub embeddings: EmbeddingsConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
    #[serde(default)]
    pub database: DatabaseConfig,
    /// Backend-tasks (dreaming / titling) overrides. The legacy `llm` field
    /// is reshaped into `[purposes]` during migration; see
    /// [`maybe_migrate_legacy_purposes`]. Consumers that still read
    /// `backend_tasks.llm` see `None` after migration and fall back to
    /// the primary LLM.
    #[serde(default)]
    pub backend_tasks: BackendTasksConfig,
    /// Per-purpose LLM configs. Each purpose picks a connection + model
    /// (possibly inherited from `interactive`) and an optional effort
    /// level. Empty on fresh installs; synthesized by migration when a
    /// legacy `[llm]` / `[backend_tasks.llm]` pair is present.
    #[serde(default, skip_serializing_if = "Purposes::is_empty")]
    pub purposes: Purposes,
    #[serde(default)]
    pub profiling: ProfilingConfig,
    #[serde(default)]
    pub ws_auth: WsAuthConfig,
    #[serde(default)]
    pub tls: TlsConfig,
    /// Which transports the daemon serves and how they bind (#279 item 3).
    /// Source of truth for the WS/UDS/D-Bus enable + bind/socket/name knobs
    /// that previously lived only in `DESKTOP_ASSISTANT_*` env vars. The env
    /// vars still work and take precedence when set, so existing setups are
    /// unaffected; this table just gives the same knobs a home alongside the
    /// rest of daemon.toml (and hot reload / the planned health report).
    ///
    /// Skipped on serialize when it equals the default so migration output and
    /// freshly written configs stay minimal — an absent `[transports]` table
    /// already means "all defaults".
    #[serde(default, skip_serializing_if = "TransportsConfig::is_default")]
    pub transports: TransportsConfig,
    /// Configurable assistant disposition (issue #226, Phase 1: global). The
    /// resolved value is installed as a task-local on every send and rendered
    /// into a system-prompt blurb. Defaults to the Expressive-7 table when the
    /// `[personality]` section is absent. Reuses the core [`Personality`] type
    /// directly (re-exported below) so config, the api-model view, and the
    /// D-Bus surface share one schema.
    #[serde(default)]
    pub personality: Personality,
}

// Reuse the canonical core type rather than duplicating the trait set in the
// daemon. `Personality` derives `Serialize`/`Deserialize` with a lowercase
// representation, so it slots straight into the TOML `[personality]` section.
pub use desktop_assistant_core::prompts::Personality;
// `PersonalityLevel` is re-exported at `config::` for callers that spell the
// level out (and the in-file tests); only test code references it directly, so
// the non-test build sees it as unused — same pattern as the other `config::`
// re-exports above (e.g. `DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS`).
#[allow(unused_imports)]
pub use desktop_assistant_core::prompts::PersonalityLevel;

impl DaemonConfig {
    /// Validate the raw `connections` map and return a [`ConnectionsMap`].
    ///
    /// Rejects invalid id slugs and duplicates (deserialize preserves insertion
    /// order, so this is deterministic). An empty map is not itself an error
    /// here — that check is the caller's responsibility, because a freshly
    /// created config with no connections is valid during first-run migration.
    pub fn validated_connections(&self) -> Result<ConnectionsMap, ConnectionsError> {
        let pairs = self
            .connections
            .iter()
            .map(|(k, v)| {
                let id = ConnectionId::new(k.clone())?;
                Ok::<_, ConnectionsError>((id, v.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if pairs.is_empty() {
            return Err(ConnectionsError::Empty);
        }
        ConnectionsMap::from_pairs(pairs)
    }

    /// Whether the `[connections]` table is present (even if empty).
    pub fn has_connections(&self) -> bool {
        !self.connections.is_empty()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsAuthConfig {
    #[serde(default = "default_ws_auth_methods")]
    pub methods: Vec<String>,
    #[serde(default)]
    pub oidc: Option<OidcConfig>,
    /// Built-in HS256 issuer claims (`iss`/`aud`) for the daemon's own WS
    /// `/login` tokens. Both fields default per-host (see [`Hs256Config`]); set
    /// them to pin a stable identity (e.g. behind a load balancer). Omitted from
    /// serialized config when left at its all-default (empty) form.
    #[serde(default, skip_serializing_if = "Hs256Config::is_default")]
    pub hs256: Hs256Config,
    /// Allowed browser origins for WebSocket and login requests.
    /// Empty (default) means no browser clients are permitted.
    /// Native clients (which do not send an Origin header) are always allowed.
    #[serde(default)]
    pub allowed_origins: Vec<String>,
}

impl Default for WsAuthConfig {
    fn default() -> Self {
        Self {
            methods: default_ws_auth_methods(),
            oidc: None,
            hs256: Hs256Config::default(),
            allowed_origins: vec![],
        }
    }
}

/// Claims policy for the built-in HS256 issuer (the daemon's WS `/login`).
///
/// JWT is a network-door concern only — local transports authenticate by
/// peer-cred (#407) — so this governs just the WebSocket door's own tokens. Both
/// fields are issued **and** validated from this same config, so they can't
/// drift. Unset (`None`) ⇒ a per-host default resolved at startup:
/// `issuer` = the local hostname, `audience` = `"<user>.adelie-ai"` (a per-user
/// daemon is a distinct service instance, so the user-scoped audience is honest).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Hs256Config {
    /// JWT `iss`. `None`/empty ⇒ the local hostname.
    #[serde(default)]
    pub issuer: Option<String>,
    /// JWT `aud`. `None`/empty ⇒ `"<user>.adelie-ai"`.
    #[serde(default)]
    pub audience: Option<String>,
}

impl Hs256Config {
    /// `true` when both fields are unset — the all-default form, omitted from
    /// serialized config so an empty `[ws_auth.hs256]` table isn't emitted.
    fn is_default(&self) -> bool {
        self.issuer.is_none() && self.audience.is_none()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TlsConfig {
    /// Enable TLS for WebSocket connections. Default: true.
    /// Can be overridden by `DESKTOP_ASSISTANT_WS_TLS=false`.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Path to a PEM-encoded certificate chain (overrides auto-generated cert).
    #[serde(default)]
    pub cert_file: Option<std::path::PathBuf>,
    /// Path to a PEM-encoded private key (overrides auto-generated key).
    #[serde(default)]
    pub key_file: Option<std::path::PathBuf>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cert_file: None,
            key_file: None,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_ws_auth_methods() -> Vec<String> {
    vec!["password".to_string()]
}

/// Transport enable/bind configuration (#279 item 3).
///
/// Defaults are local-first: WebSocket off, UDS on (Unix). The matching
/// `DESKTOP_ASSISTANT_*` env var still overrides each field when set. The daemon
/// serves no in-process D-Bus surface — the standalone `adelie-dbus-bridge` owns
/// `org.desktopAssistant` since the cutover (#281/#319).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default)]
pub struct TransportsConfig {
    /// Serve the remote WebSocket endpoint. Env: `DESKTOP_ASSISTANT_WS_ENABLED`.
    pub ws_enabled: bool,
    /// WebSocket bind address. Env: `DESKTOP_ASSISTANT_WS_BIND`.
    pub ws_bind: String,
    /// Serve the local Unix-domain-socket endpoint (Unix only). Env:
    /// `DESKTOP_ASSISTANT_UDS_ENABLED`.
    pub uds_enabled: bool,
    /// Override the UDS socket path. Empty = use the default path. Env:
    /// `DESKTOP_ASSISTANT_UDS_SOCKET`.
    pub uds_socket: Option<String>,
}

/// Local-first WebSocket-off default, mirroring the historical env defaults.
pub const DEFAULT_WS_ENABLED: bool = false;
/// Historical default WebSocket bind address.
pub const DEFAULT_WS_BIND: &str = "127.0.0.1:11339";

impl Default for TransportsConfig {
    fn default() -> Self {
        Self {
            ws_enabled: DEFAULT_WS_ENABLED,
            ws_bind: DEFAULT_WS_BIND.to_string(),
            uds_enabled: cfg!(unix),
            uds_socket: None,
        }
    }
}

impl TransportsConfig {
    /// True when every field equals the default, so serialization can skip the
    /// whole `[transports]` table.
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OidcConfig {
    pub issuer_url: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub client_id: String,
    #[serde(default = "default_oidc_scopes")]
    pub scopes: String,
    #[serde(default)]
    pub jwks_uri: String,
    #[serde(default)]
    pub audience: String,
}

pub(super) fn default_oidc_scopes() -> String {
    "openid profile email".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BackendTasksConfig {
    /// Optional separate LLM config for backend tasks (title generation,
    /// context summary, dreaming extraction).
    /// Falls back to the top-level `[llm]` if omitted.
    #[serde(default)]
    pub llm: Option<LlmConfig>,
    /// Enable periodic fact extraction from conversations ("dreaming").
    #[serde(default)]
    pub dreaming_enabled: bool,
    #[serde(default = "default_dreaming_interval_secs")]
    pub dreaming_interval_secs: u64,
    /// Archive conversations older than this many days (0 = disabled).
    #[serde(default = "default_archive_after_days")]
    pub archive_after_days: u32,
    /// How often the background task regenerates missing/stale knowledge &
    /// tool embeddings. Embedding generation is decoupled from content writes,
    /// so this is the cadence at which new and edited entries gain semantic
    /// search coverage. A few minutes is a good default.
    #[serde(default = "default_embedding_backfill_interval_secs")]
    pub embedding_backfill_interval_secs: u64,
    /// How often the holistic knowledge-base consolidation runs (0 = disabled).
    /// Consolidation loads a user's whole KB and recomputes it, so it runs on a
    /// much slower cadence than extraction — daily by default.
    #[serde(default = "default_consolidation_interval_secs")]
    pub consolidation_interval_secs: u64,
    /// Optional dedicated LLM for holistic consolidation. Falls back to
    /// `[backend_tasks.llm]`, then the top-level `[llm]`. Lets extraction run on
    /// a cheap/local model while consolidation uses a stronger one.
    #[serde(default)]
    pub consolidation_llm: Option<LlmConfig>,
}

impl Default for BackendTasksConfig {
    fn default() -> Self {
        Self {
            llm: None,
            dreaming_enabled: false,
            dreaming_interval_secs: default_dreaming_interval_secs(),
            archive_after_days: default_archive_after_days(),
            embedding_backfill_interval_secs: default_embedding_backfill_interval_secs(),
            consolidation_interval_secs: default_consolidation_interval_secs(),
            consolidation_llm: None,
        }
    }
}

pub(super) fn default_archive_after_days() -> u32 {
    7
}

pub(super) fn default_dreaming_interval_secs() -> u64 {
    3600
}

pub(super) fn default_embedding_backfill_interval_secs() -> u64 {
    300
}

pub(super) fn default_consolidation_interval_secs() -> u64 {
    86400
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ProfilingConfig {
    /// Enable LLM call profiling. Default: false.
    #[serde(default)]
    pub enabled: bool,
    /// Path for the JSONL profile log.
    /// Defaults to `~/.local/share/desktop-assistant/llm-profile.jsonl`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_path: Option<String>,
    /// Log full message/response content instead of previews. Default: false.
    #[serde(default)]
    pub full_content: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL (e.g. "postgres://user:pass@localhost/desktop_assistant").
    /// Falls back to `DESKTOP_ASSISTANT_DATABASE_URL` env var.
    #[serde(default)]
    pub url: Option<String>,
    /// Maximum number of connections in the pool.
    #[serde(default = "default_database_max_connections")]
    pub max_connections: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: None,
            max_connections: default_database_max_connections(),
        }
    }
}

pub(super) fn default_database_max_connections() -> u32 {
    5
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PersistenceConfig {
    #[serde(default)]
    pub git: GitPersistenceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GitPersistenceConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_url: Option<String>,
    #[serde(default = "default_git_remote_name")]
    pub remote_name: String,
    #[serde(default = "default_push_on_update")]
    pub push_on_update: bool,
}

impl Default for GitPersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            remote_url: None,
            remote_name: default_git_remote_name(),
            push_on_update: default_push_on_update(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlmConfig {
    #[serde(default = "default_connector")]
    pub connector: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret: Option<SecretConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Enable provider-side hosted tool search (deferred loading / namespaces).
    /// When `None`, defaults to the connector's built-in capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hosted_tool_search: Option<bool>,
    /// AWS profile name for Bedrock connector (e.g. "my-work-profile").
    /// When `None`, uses the default AWS credential chain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aws_profile: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            connector: default_connector(),
            model: None,
            base_url: None,
            api_key_env: None,
            secret: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            hosted_tool_search: None,
            aws_profile: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SecretConfig {
    #[serde(default = "default_secret_backend")]
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
    #[serde(default = "default_wallet_name")]
    pub wallet: String,
    #[serde(default = "default_wallet_folder")]
    pub folder: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
}

impl Default for SecretConfig {
    fn default() -> Self {
        Self {
            backend: default_secret_backend(),
            service: Some(default_secret_service()),
            account: None,
            wallet: default_wallet_name(),
            folder: default_wallet_folder(),
            entry: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedLlmConfig {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    /// Explicit hosted-tool-search override from config, or `None` for connector default.
    pub hosted_tool_search: Option<bool>,
    /// AWS profile name for Bedrock connector.
    pub aws_profile: Option<String>,
    /// Per-connection override for the streaming first-response (connect) stall
    /// budget, in seconds. `None` falls back to the connector-shared default.
    pub connect_timeout_secs: Option<u64>,
    /// Per-connection override for the streaming per-chunk stall budget, in
    /// seconds. `None` falls back to the connector-shared default.
    pub stream_timeout_secs: Option<u64>,
    /// Ollama-only: keep this connection's interactive model resident in
    /// memory. `false`/`None` for every other connector. Consumed by the
    /// daemon's keep-warm loop, not by the connectors.
    pub keep_warm: bool,
    /// Per-connection hard ceiling on the effective context window, in tokens.
    /// `None` = "max available" (defer entirely to the model's reported max).
    /// `Some(n)` caps the window to `min(n, reported)`. See
    /// [`crate::connections::OllamaConnection::max_context_tokens`] and the
    /// connector's `max_context_tokens()` for how this folds together.
    pub max_context_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct LlmSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub hosted_tool_search: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct EmbeddingsConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingsSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    /// API key resolved from secret backend / env / shared LLM config. Used by
    /// `main.rs` to instantiate the OpenAI-compatible embedding client.
    /// Daemon-internal — not exposed across the settings API surface (see
    /// `settings_service::get_embeddings_settings`, which maps to a separate
    /// core view that omits this field).
    pub api_key: String,
    pub has_api_key: bool,
    pub available: bool,
    pub is_default: bool,
}

#[derive(Debug, Clone)]
pub struct ConnectorDefaultsView {
    pub llm_model: String,
    pub llm_base_url: String,
    pub backend_llm_model: String,
    pub embeddings_model: String,
    pub embeddings_base_url: String,
    pub embeddings_available: bool,
    pub hosted_tool_search_available: bool,
}

#[derive(Debug, Clone)]
pub struct ResolvedPersistenceConfig {
    pub enabled: bool,
    pub remote_url: Option<String>,
    pub remote_name: String,
    pub push_on_update: bool,
}

pub(super) fn default_connector() -> String {
    "openai".to_string()
}

fn default_secret_backend() -> String {
    "auto".to_string()
}

pub(super) fn default_git_remote_name() -> String {
    "origin".to_string()
}

pub(super) fn default_push_on_update() -> bool {
    true
}

fn default_secret_service() -> String {
    "org.desktopAssistant".to_string()
}

fn default_secret_account(connector: &str) -> String {
    format!("{}_api_key", normalized_connector_key_prefix(connector))
}

pub(super) fn default_api_key_env(connector: &str) -> String {
    format!(
        "{}_API_KEY",
        normalized_connector_key_prefix(connector).to_ascii_uppercase()
    )
}

pub(super) fn default_model_env(connector: &str) -> String {
    format!(
        "{}_MODEL",
        normalized_connector_key_prefix(connector).to_ascii_uppercase()
    )
}

pub(super) fn default_base_url_env(connector: &str) -> String {
    format!(
        "{}_BASE_URL",
        normalized_connector_key_prefix(connector).to_ascii_uppercase()
    )
}

fn normalized_connector_key_prefix(connector: &str) -> String {
    let mut normalized = String::new();
    let mut previous_was_separator = false;

    for ch in connector.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            previous_was_separator = false;
        } else if !normalized.is_empty() && !previous_was_separator {
            normalized.push('_');
            previous_was_separator = true;
        }
    }

    while normalized.ends_with('_') {
        normalized.pop();
    }

    if normalized.is_empty() {
        default_connector()
    } else {
        normalized
    }
}

fn default_wallet_name() -> String {
    "kdewallet".to_string()
}

fn default_wallet_folder() -> String {
    "desktop-assistant".to_string()
}

fn default_wallet_entry(connector: &str) -> String {
    default_secret_account(connector)
}

fn resolve_secret_account(secret: &SecretConfig, connector: &str) -> String {
    secret
        .account
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default_secret_account(connector))
}

fn resolve_wallet_entry(secret: &SecretConfig, connector: &str) -> String {
    secret
        .entry
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default_wallet_entry(connector))
}

pub fn default_daemon_config_path() -> PathBuf {
    let config_home = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string())).join(".config")
        });

    config_home.join("desktop-assistant").join("daemon.toml")
}

fn default_secret_store_dir() -> PathBuf {
    let data_home = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
                .join(".local")
                .join("share")
        });

    data_home.join("desktop-assistant").join("secrets")
}

fn common_secret_file_path(account: &str) -> PathBuf {
    default_secret_store_dir().join(account)
}

/// Secret-store account name for a named connection's credential.
///
/// Keyed by the connection **id** (not the connector type) so two connections
/// of the same connector never share — or clobber — each other's secret file.
/// The id is already validated (`ConnectionId`) to `[a-z0-9][a-z0-9_-]*`, so it
/// is safe as a bare filename fragment.
fn connection_secret_account(connection_id: &str) -> String {
    format!("connection_{connection_id}")
}

/// Store (or clear) the raw credential for a named connection and return the
/// [`SecretConfig`] coordinate to persist on the connection.
///
/// * Non-empty `value` ⇒ write it to the secret backend (the default `"auto"`
///   backend writes a 0600 file under [`default_secret_store_dir`]) keyed by the
///   connection id, and return `Some(secret)` for the caller to store on the
///   connection's `secret` field.
/// * Empty/whitespace `value` ⇒ best-effort remove the file-store entry and
///   return `None` (clear).
///
/// The raw credential is written **only** to the secret backend — never to
/// daemon.toml. `connector` is passed through to the backend for its
/// account-fallback logic but is not used to key the account (the id is), so
/// two connections of the same connector stay isolated.
pub fn store_connection_secret(
    connection_id: &str,
    connector: &str,
    value: &str,
) -> anyhow::Result<Option<SecretConfig>> {
    let account = connection_secret_account(connection_id);
    let secret = SecretConfig {
        account: Some(account.clone()),
        ..SecretConfig::default()
    };

    let trimmed = value.trim();
    if trimmed.is_empty() {
        // Clear: best-effort removal of the file-store entry. A missing file is
        // already "cleared", so treat NotFound as success.
        let path = common_secret_file_path(&account);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(
                    "failed to remove connection secret file {}: {error}",
                    path.display()
                );
            }
        }
        return Ok(None);
    }

    write_secret_to_backend(&secret, trimmed, connector)?;
    Ok(Some(secret))
}

pub fn load_daemon_config(path: &Path) -> anyhow::Result<Option<DaemonConfig>> {
    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(path)?;
    if content.trim().is_empty() {
        return Ok(None);
    }

    let parsed: DaemonConfig = toml::from_str(&content)?;
    let explicit_connections_table = migration::file_has_top_level_table(&content, "connections");
    let explicit_purposes_table = migration::file_has_top_level_table(&content, "purposes");
    // Purpose migration runs only when the file is in a legacy shape:
    // either `[llm]` or `[backend_tasks.llm]` is present. Pure
    // new-format configs (connections + no legacy markers) are left
    // alone so first-run users are not forced to accept synthesized
    // purposes.
    let legacy_shape_present = migration::file_has_top_level_table(&content, "llm")
        || migration::file_has_top_level_table(&content, "backend_tasks.llm");
    let parsed = migration::maybe_migrate_legacy_connections(path, parsed, &content)?;

    // Validate `[connections]` *after* migration so legacy-only configs still
    // succeed on first load. Two cases trigger validation:
    //
    // 1. The parsed map is non-empty (normal case).
    // 2. The user wrote an explicit `[connections]` header but left it empty.
    //    Catching this here surfaces the misconfiguration at startup rather
    //    than at the first request.
    if parsed.has_connections() || explicit_connections_table {
        parsed
            .validated_connections()
            .with_context(|| format!("invalid [connections] in {}", path.display()))?;
    }

    // Purpose migration runs after connection migration so it can reference
    // the synthesized `[connections.default]`. It also rewrites the config
    // file on first contact — only when a legacy shape was present and no
    // `[purposes]` table has been authored yet.
    let parsed = migration::maybe_migrate_legacy_purposes(
        path,
        parsed,
        explicit_purposes_table,
        legacy_shape_present,
    )?;

    // Validate purposes: structural checks (interactive required when set,
    // no `Primary` in interactive) happen here so misconfigurations surface
    // at startup rather than at the first dispatch.
    parsed
        .purposes
        .validate()
        .with_context(|| format!("invalid [purposes] in {}", path.display()))?;

    Ok(Some(parsed))
}

/// If the config has a legacy `[llm]` block and no `[connections]`, synthesize
/// a connection named `default`, write the new form back to disk, and back up
/// the original to `daemon.toml.bak` (or `.bak.N` if `.bak` already exists).
///
/// Also emits a one-time deprecation warning via `tracing::warn!`.
///
/// The `backend_tasks.llm` block is preserved as-is for #10 to reshape into a
/// purpose config. We deliberately do not synthesize a second connection for
/// it here, because that would force the user to manage two copies of the same
/// credentials when the common case is "backend tasks share the primary
/// connector".
/// Ensure a daemon config file exists at `path`, writing a default
/// [`DaemonConfig`] when it is absent (first run, or a fresh writable mount that
/// replaced a baked-in config). Returns `true` when a default was written.
///
/// Idempotent and non-clobbering: an existing file — even empty or unparsable —
/// is left untouched (`Ok(false)`). Best-effort at the call site: a read-only
/// location (e.g. a Kubernetes ConfigMap mount) surfaces as an `Err` the caller
/// logs before falling back to in-memory defaults, so this never blocks startup.
pub fn ensure_daemon_config_exists(path: &Path) -> anyhow::Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    save_daemon_config(path, &DaemonConfig::default())?;
    Ok(true)
}

pub fn save_daemon_config(path: &Path, config: &DaemonConfig) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }

    let content = toml::to_string_pretty(config)?;

    // The config can carry credential references and OIDC client identifiers;
    // open with restrictive perms before writing so the file is never briefly
    // world-readable. Mirrors `write_secret_file` below.
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .with_context(|| format!("failed to write daemon config at {}", path.display()))?;
        file.write_all(content.as_bytes())
            .with_context(|| format!("failed to write daemon config at {}", path.display()))?;
        Ok(())
    }

    #[cfg(not(unix))]
    {
        std::fs::write(path, content)
            .with_context(|| format!("failed to write daemon config at {}", path.display()))
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WsAuthDiscoveryInfo {
    pub methods: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oidc: Option<OidcDiscoveryInfo>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OidcDiscoveryInfo {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub client_id: String,
    pub scopes: String,
}

pub fn authenticate_os_user_password(username: &str, password: &str) -> anyhow::Result<bool> {
    #[cfg(target_os = "linux")]
    {
        pam_auth::authenticate(username, password)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (username, password);
        Err(anyhow::anyhow!(
            "OS password authentication is only supported on Linux"
        ))
    }
}

/// Process-wide lock serialising every test in the daemon binary that mutates
/// the global `XDG_DATA_HOME` env var. Both the JWT store and the secret store
/// resolve their directory from it, so tests in different modules (config's JWT
/// / `set_api_key` tests and api_surface's `set_connection_secret` tests) must
/// share one lock — separate per-module locks would race on the same global.
#[cfg(test)]
pub(crate) fn xdg_data_home_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn ws_jwt_env_lock() -> std::sync::MutexGuard<'static, ()> {
        super::xdg_data_home_test_lock()
    }

    #[test]
    fn transports_config_absent_table_is_default() {
        // A config with no `[transports]` section deserializes to all defaults.
        let cfg: DaemonConfig = toml::from_str("").unwrap();
        assert_eq!(cfg.transports, TransportsConfig::default());
    }

    #[test]
    fn transports_config_roundtrips_overrides() {
        let src = r#"
[transports]
ws_enabled = true
ws_bind = "0.0.0.0:8080"
uds_enabled = false
uds_socket = "/tmp/adelie.sock"
"#;
        let cfg: DaemonConfig = toml::from_str(src).unwrap();
        let t = &cfg.transports;
        assert!(t.ws_enabled);
        assert_eq!(t.ws_bind, "0.0.0.0:8080");
        assert!(!t.uds_enabled);
        assert_eq!(t.uds_socket.as_deref(), Some("/tmp/adelie.sock"));

        // Non-default => the table is serialized back out.
        let serialized = toml::to_string(&cfg).unwrap();
        assert!(serialized.contains("[transports]"));
        let reparsed: DaemonConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(reparsed.transports, cfg.transports);
    }

    #[test]
    fn transports_config_default_is_skipped_on_serialize() {
        let cfg = DaemonConfig::default();
        let serialized = toml::to_string(&cfg).unwrap();
        assert!(
            !serialized.contains("[transports]"),
            "default transports table must be skipped: {serialized}"
        );
    }

    #[test]
    fn default_path_points_to_daemon_toml() {
        let path = default_daemon_config_path();
        assert!(path.ends_with("desktop-assistant/daemon.toml"));
    }

    #[test]
    fn parse_minimal_toml() {
        let parsed: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "openai"
            model = "gpt-5.4"
            "#,
        )
        .unwrap();

        assert_eq!(parsed.llm.connector, "openai");
        assert_eq!(parsed.llm.model.as_deref(), Some("gpt-5.4"));
    }

    #[test]
    fn parse_keyring_secret_config() {
        let parsed: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "openai"

            [llm.secret]
            backend = "keyring"
            service = "org.desktopAssistant"
            account = "openai_api_key"
            "#,
        )
        .unwrap();

        let secret = parsed.llm.secret.expect("secret config should parse");
        assert_eq!(secret.backend, "keyring");
        assert_eq!(secret.service.as_deref(), Some("org.desktopAssistant"));
        assert_eq!(secret.account.as_deref(), Some("openai_api_key"));
    }

    #[test]
    fn default_secret_backend_is_auto() {
        let secret = SecretConfig::default();
        assert_eq!(secret.backend, "auto");
    }

    #[test]
    fn default_secret_store_dir_points_to_desktop_assistant_secrets() {
        let path = default_secret_store_dir();
        assert!(path.ends_with("desktop-assistant/secrets"));
    }

    #[test]
    fn common_secret_file_path_uses_account_name() {
        let path = common_secret_file_path("openai_api_key");
        assert!(path.ends_with("desktop-assistant/secrets/openai_api_key"));
    }

    #[test]
    fn resolve_defaults_without_config() {
        let resolved = resolve_llm_config(None);
        assert_eq!(resolved.connector, "openai");
        assert!(!resolved.model.is_empty());
        assert!(!resolved.base_url.is_empty());
    }

    #[test]
    fn default_secret_account_depends_on_connector() {
        assert_eq!(default_secret_account("openai"), "openai_api_key");
        assert_eq!(default_secret_account("anthropic"), "anthropic_api_key");
        assert_eq!(default_secret_account("aws-bedrock"), "aws_bedrock_api_key");
    }

    #[test]
    fn default_api_key_env_depends_on_connector() {
        assert_eq!(default_api_key_env("openai"), "OPENAI_API_KEY");
        assert_eq!(default_api_key_env("anthropic"), "ANTHROPIC_API_KEY");
        assert_eq!(default_api_key_env("aws-bedrock"), "AWS_BEDROCK_API_KEY");
    }

    #[test]
    fn default_model_env_depends_on_connector() {
        assert_eq!(default_model_env("openai"), "OPENAI_MODEL");
        assert_eq!(default_model_env("anthropic"), "ANTHROPIC_MODEL");
        assert_eq!(default_model_env("aws-bedrock"), "AWS_BEDROCK_MODEL");
    }

    #[test]
    fn default_base_url_env_depends_on_connector() {
        assert_eq!(default_base_url_env("openai"), "OPENAI_BASE_URL");
        assert_eq!(default_base_url_env("anthropic"), "ANTHROPIC_BASE_URL");
        assert_eq!(default_base_url_env("aws-bedrock"), "AWS_BEDROCK_BASE_URL");
    }

    #[test]
    fn resolve_secret_account_uses_explicit_override() {
        let secret = SecretConfig {
            backend: "keyring".to_string(),
            service: Some("org.desktopAssistant".to_string()),
            account: Some("custom_key_account".to_string()),
            wallet: "kdewallet".to_string(),
            folder: "desktop-assistant".to_string(),
            entry: None,
        };

        assert_eq!(
            resolve_secret_account(&secret, "anthropic"),
            "custom_key_account"
        );
    }

    #[test]
    fn placeholder_secret_values_are_rejected() {
        assert!(is_placeholder_secret_value("file-store-openai-key"));
        assert!(is_placeholder_secret_value("file-sto********-key"));
        assert!(is_placeholder_secret_value(
            "Write-only; leave blank to keep existing"
        ));
        assert!(!is_placeholder_secret_value("sk-test-real-secret-value"));
    }

    #[test]
    fn sanitize_secret_value_discards_empty_and_placeholder_values() {
        assert_eq!(secrets::sanitize_secret_value("  \n\t "), None);
        assert_eq!(
            secrets::sanitize_secret_value("file-store-openai-key"),
            None
        );
        assert_eq!(
            secrets::sanitize_secret_value("  sk-live-abc123  "),
            Some("sk-live-abc123".to_string())
        );
    }

    #[test]
    fn redacted_secret_audit_is_stable_and_trimmed() {
        let (len, fp) = redacted_secret_audit("  sk-test-abc123  ");
        assert_eq!(len, 14);
        assert_eq!(fp, "fnv1a64:6e6d7d2dfdec1dad");

        let (empty_len, empty_fp) = redacted_secret_audit("   ");
        assert_eq!(empty_len, 0);
        assert_eq!(empty_fp, "fnv1a64:cbf29ce484222325");
    }

    #[test]
    fn oidc_require_https_accepts_https() {
        OidcValidator::require_https_or_loopback("https://idp.example.com", "issuer_url")
            .expect("https URL is permitted");
        OidcValidator::require_https_or_loopback(
            "HTTPS://Idp.Example.com/realms/main",
            "issuer_url",
        )
        .expect("scheme check is case-insensitive");
    }

    #[test]
    fn oidc_require_https_accepts_loopback_http() {
        OidcValidator::require_https_or_loopback("http://localhost:8080", "issuer_url")
            .expect("loopback http is permitted for development");
        OidcValidator::require_https_or_loopback("http://127.0.0.1:9090/path", "issuer_url")
            .expect("ipv4 loopback http is permitted");
        OidcValidator::require_https_or_loopback("http://[::1]:9090/path", "issuer_url")
            .expect("ipv6 loopback http is permitted");
    }

    #[test]
    fn oidc_require_https_rejects_non_loopback_http() {
        // Plaintext JWKS lets a network attacker swap the keys and forge
        // tokens — must reject.
        let err = OidcValidator::require_https_or_loopback("http://idp.example.com", "issuer_url")
            .expect_err("plaintext IdP rejected");
        assert!(err.to_string().contains("https://"));

        OidcValidator::require_https_or_loopback("ftp://idp.example.com", "issuer_url")
            .expect_err("non-http(s) scheme rejected");
    }

    #[test]
    fn bucket_secret_len_collapses_into_coarse_buckets() {
        // Hides the precise length so audit logs don't distinguish 32-char
        // OpenAI keys from 51-char Anthropic keys at info level.
        assert_eq!(bucket_secret_len(0), "0");
        assert_eq!(bucket_secret_len(8), "<16");
        assert_eq!(bucket_secret_len(15), "<16");
        assert_eq!(bucket_secret_len(16), "16-31");
        assert_eq!(bucket_secret_len(32), "32-47"); // typical OpenAI sk- key
        assert_eq!(bucket_secret_len(47), "32-47");
        assert_eq!(bucket_secret_len(51), "48-79"); // typical Anthropic key
        assert_eq!(bucket_secret_len(79), "48-79");
        assert_eq!(bucket_secret_len(80), ">=80");
        assert_eq!(bucket_secret_len(2048), ">=80");
    }

    #[test]
    fn embeddings_defaults_from_llm_connector() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "ollama"
            "#,
        )
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.connector, "ollama");
        assert_eq!(view.model, "nomic-embed-text");
        assert_eq!(view.base_url, "http://localhost:11434");
        assert!(view.available);
        assert!(view.is_default);
    }

    #[test]
    fn embeddings_explicit_override() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "anthropic"

            [embeddings]
            connector = "openai"
            model = "text-embedding-3-large"
            "#,
        )
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.connector, "openai");
        assert_eq!(view.model, "text-embedding-3-large");
        assert!(view.available);
        assert!(!view.is_default);
    }

    #[test]
    fn embeddings_unavailable_for_anthropic_without_override() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "anthropic"
            "#,
        )
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.connector, "anthropic");
        assert!(!view.available);
        assert!(view.is_default);
    }

    #[test]
    fn embeddings_defaults_without_config() {
        let view = resolve_embeddings_config(None);
        assert_eq!(view.connector, "openai");
        assert_eq!(view.model, "text-embedding-3-small");
        assert!(view.available);
        assert!(view.is_default);
    }

    #[test]
    fn bedrock_llm_defaults() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "bedrock"
            "#,
        )
        .unwrap();

        let resolved = resolve_llm_config(Some(&config));
        assert_eq!(resolved.connector, "bedrock");
        assert_eq!(resolved.model, "us.anthropic.claude-sonnet-4-6");
        assert_eq!(resolved.base_url, "us-east-1");
    }

    #[test]
    fn bedrock_embedding_defaults() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "bedrock"
            "#,
        )
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.connector, "bedrock");
        assert_eq!(view.model, "amazon.titan-embed-text-v2:0");
        assert_eq!(view.base_url, "us-east-1");
        assert!(view.available);
    }

    #[test]
    fn connector_defaults_openai() {
        let defaults = get_connector_defaults("openai");
        assert_eq!(defaults.llm_model, "gpt-5.4");
        assert_eq!(defaults.llm_base_url, "https://api.openai.com/v1");
        assert_eq!(defaults.embeddings_model, "text-embedding-3-small");
        assert_eq!(defaults.embeddings_base_url, "https://api.openai.com/v1");
        assert!(defaults.embeddings_available);
    }

    #[test]
    fn connector_defaults_anthropic_embeddings_fallback_to_openai() {
        let defaults = get_connector_defaults("anthropic");
        assert_eq!(defaults.llm_model, "claude-sonnet-4-6-20260227");
        assert_eq!(defaults.llm_base_url, "https://api.anthropic.com");
        assert_eq!(defaults.embeddings_model, "text-embedding-3-small");
        assert_eq!(defaults.embeddings_base_url, "https://api.openai.com/v1");
        assert!(!defaults.embeddings_available);
    }

    #[test]
    fn parse_toml_with_embeddings_section() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "anthropic"
            model = "claude-sonnet-4-6-20260227"

            [embeddings]
            connector = "ollama"
            model = "nomic-embed-text"
            "#,
        )
        .unwrap();

        assert_eq!(config.embeddings.connector.as_deref(), Some("ollama"));
        assert_eq!(config.embeddings.model.as_deref(), Some("nomic-embed-text"));
        assert!(config.embeddings.base_url.is_none());
    }

    #[test]
    fn parse_toml_without_embeddings_section() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "ollama"
            "#,
        )
        .unwrap();

        assert!(config.embeddings.connector.is_none());
        assert!(config.embeddings.model.is_none());
        assert!(config.embeddings.base_url.is_none());
    }

    #[test]
    fn set_embeddings_settings_roundtrip() {
        let dir = std::env::temp_dir().join("da-test-emb-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("daemon.toml");

        // Start with an LLM-only config
        let config = DaemonConfig {
            llm: LlmConfig {
                connector: "anthropic".to_string(),
                ..LlmConfig::default()
            },
            ..DaemonConfig::default()
        };
        save_daemon_config(&path, &config).unwrap();

        // Set embeddings override
        set_embeddings_settings(&path, Some("ollama"), Some("nomic-embed-text"), None).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();
        assert_eq!(loaded.embeddings.connector.as_deref(), Some("ollama"));
        assert_eq!(loaded.embeddings.model.as_deref(), Some("nomic-embed-text"));
        assert!(loaded.embeddings.base_url.is_none());

        // Clear override
        set_embeddings_settings(&path, None, None, None).unwrap();
        let loaded = load_daemon_config(&path).unwrap().unwrap();
        assert!(loaded.embeddings.connector.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_toml_with_persistence_section() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [persistence.git]
            enabled = true
            remote_url = "https://example.com/dave/assistant-memory.git"
            remote_name = "backup"
            push_on_update = true
            "#,
        )
        .unwrap();

        assert!(config.persistence.git.enabled);
        assert_eq!(
            config.persistence.git.remote_url.as_deref(),
            Some("https://example.com/dave/assistant-memory.git")
        );
        assert_eq!(config.persistence.git.remote_name, "backup");
        assert!(config.persistence.git.push_on_update);
    }

    #[test]
    fn resolve_persistence_defaults_when_missing() {
        let resolved = resolve_persistence_config(None);
        assert!(!resolved.enabled);
        assert!(resolved.remote_url.is_none());
        assert_eq!(resolved.remote_name, "origin");
        assert!(resolved.push_on_update);
    }

    #[test]
    fn resolve_persistence_trims_remote_url() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [persistence.git]
            enabled = true
            remote_url = "   "
            remote_name = "  "
            push_on_update = false
            "#,
        )
        .unwrap();

        let resolved = resolve_persistence_config(Some(&config));
        assert!(resolved.enabled);
        assert!(resolved.remote_url.is_none());
        assert_eq!(resolved.remote_name, "origin");
        assert!(!resolved.push_on_update);
    }

    #[test]
    fn ws_jwt_generation_allows_multiple_valid_tokens() {
        let _guard = ws_jwt_env_lock();
        let test_dir =
            std::env::temp_dir().join(format!("da-test-ws-jwt-{}", uuid::Uuid::new_v4()));
        let data_home = test_dir.join("data");
        std::fs::create_dir_all(&data_home).unwrap();
        // SAFETY: single-test scope; the temp dir is unique per run
        // (UUID-suffixed); no other test in this binary mutates
        // `XDG_DATA_HOME` concurrently.
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &data_home);
        }

        let token_1 = generate_ws_jwt(Some("tui".to_string())).expect("generate first jwt");
        let token_2 = generate_ws_jwt(Some("plasmoid".to_string())).expect("generate second jwt");

        assert_ne!(token_1, token_2);
        assert!(validate_ws_jwt(&token_1).expect("validate first jwt"));
        assert!(validate_ws_jwt(&token_2).expect("validate second jwt"));
        assert!(!validate_ws_jwt("not-a-jwt").expect("validate invalid token"));

        let claims_1 = jwt::decode_ws_jwt_claims(&token_1).expect("decode first jwt");
        let claims_2 = jwt::decode_ws_jwt_claims(&token_2).expect("decode second jwt");
        assert_eq!(claims_1.sub, "tui");
        assert_eq!(claims_2.sub, "plasmoid");
        assert_eq!(claims_1.iss, jwt::ws_jwt_issuer());
        assert_eq!(claims_1.aud, jwt::ws_jwt_audience());

        // SAFETY: same scope as the matching `set_var` above; clean up
        // before exiting the test so we don't leak state between runs.
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
        std::fs::remove_dir_all(&test_dir).ok();
    }

    // --- Named-connections schema + migration -------------------

    fn unique_test_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("{prefix}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn ensure_config_writes_default_when_missing() {
        let dir = unique_test_dir("da-ensure-missing");
        // A nested, not-yet-created parent dir — save_daemon_config must create it.
        let path = dir.join("nested").join("daemon.toml");
        assert!(!path.exists());

        let wrote = ensure_daemon_config_exists(&path).unwrap();
        assert!(wrote, "should report it wrote a default");
        assert!(path.exists(), "the default config file should now exist");

        // The written file parses back cleanly as a valid config (the daemon can
        // load what it just bootstrapped, and settings writes will round-trip).
        let loaded = load_daemon_config(&path).unwrap();
        assert!(loaded.is_some(), "the written default must load back as valid config");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_config_is_idempotent_and_never_clobbers() {
        let dir = unique_test_dir("da-ensure-existing");
        let path = dir.join("daemon.toml");
        let hand_written = "[connections.mine]\ntype = \"ollama\"\nbase_url = \"http://x:11434\"\n";
        std::fs::write(&path, hand_written).unwrap();

        let wrote = ensure_daemon_config_exists(&path).unwrap();
        assert!(!wrote, "an existing file must not be overwritten");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            hand_written,
            "the existing config must be left byte-for-byte untouched"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_connections_map_preserves_order_and_tags() {
        let content = r#"
[connections.work_openai]
type = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_WORK_KEY"

[connections.home_bedrock]
type = "bedrock"
aws_profile = "home"
region = "us-west-2"

[connections.laptop_ollama]
type = "ollama"
base_url = "http://localhost:11434"
"#;
        let parsed: DaemonConfig = toml::from_str(content).unwrap();
        let validated = parsed.validated_connections().expect("should validate");
        let ids: Vec<_> = validated
            .iter()
            .map(|(id, _)| id.as_str().to_owned())
            .collect();
        assert_eq!(ids, vec!["work_openai", "home_bedrock", "laptop_ollama"]);
        assert_eq!(
            validated
                .get(&ConnectionId::new("work_openai").unwrap())
                .unwrap()
                .connector_type(),
            "openai"
        );
    }

    #[test]
    fn connections_roundtrip_toml() {
        let content = r#"
[connections.work_openai]
type = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_WORK_KEY"

[connections.home_bedrock]
type = "bedrock"
aws_profile = "home"
region = "us-west-2"
"#;
        let parsed: DaemonConfig = toml::from_str(content).unwrap();
        let serialized = toml::to_string_pretty(&parsed).unwrap();
        let reparsed: DaemonConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed.connections, reparsed.connections);
    }

    #[test]
    fn validated_connections_rejects_invalid_slug() {
        let mut cfg = DaemonConfig::default();
        cfg.connections.insert(
            "Bad Id".to_string(),
            ConnectionConfig::OpenAi(Default::default()),
        );
        let err = cfg.validated_connections().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Bad Id"), "error should cite bad id: {msg}");
    }

    #[test]
    fn validated_connections_rejects_empty_table() {
        let cfg = DaemonConfig::default();
        // default `connections` is empty, but `validated_connections` treats empty as error
        let err = cfg.validated_connections().unwrap_err();
        assert_eq!(err, ConnectionsError::Empty);
    }

    #[test]
    fn validated_connections_rejects_duplicates_if_they_appear() {
        // serde + IndexMap silently overwrites on duplicate TOML keys, so we
        // synthesize a duplicate through `ConnectionsMap::from_pairs` to exercise
        // that branch.
        let pairs = vec![
            (
                ConnectionId::new("default").unwrap(),
                ConnectionConfig::OpenAi(Default::default()),
            ),
            (
                ConnectionId::new("default").unwrap(),
                ConnectionConfig::OpenAi(Default::default()),
            ),
        ];
        let err = ConnectionsMap::from_pairs(pairs).unwrap_err();
        assert_eq!(err, ConnectionsError::DuplicateId("default".to_string()));
    }

    #[test]
    fn load_accepts_new_format_without_migration() {
        let dir = unique_test_dir("da-test-connections-new");
        let path = dir.join("daemon.toml");
        let content = r#"
[connections.work_openai]
type = "openai"
base_url = "https://api.openai.com/v1"
"#;
        std::fs::write(&path, content).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();
        assert!(loaded.has_connections());
        assert_eq!(loaded.connections.len(), 1);

        // No migration side-effects
        assert!(!dir.join("daemon.toml.bak").exists());
        // File contents unchanged
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk, content);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_invalid_connection_id_with_clear_error() {
        let dir = unique_test_dir("da-test-connections-bad-id");
        let path = dir.join("daemon.toml");
        let content = r#"
[connections."Bad Id"]
type = "openai"
"#;
        std::fs::write(&path, content).unwrap();

        let err = load_daemon_config(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("Bad Id"), "error should cite bad id: {msg}");
        assert!(
            msg.contains("connection id"),
            "error should mention 'connection id': {msg}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_explicitly_empty_connections_table() {
        let dir = unique_test_dir("da-test-connections-empty-table");
        let path = dir.join("daemon.toml");
        // Explicit `[connections]` header with no entries. This is treated as
        // "user meant to configure connections but made a mistake" — reject
        // so the misconfiguration surfaces at startup, not at request time.
        let content = r#"
[connections]
"#;
        std::fs::write(&path, content).unwrap();

        let err = load_daemon_config(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("at least one"),
            "expected empty-table error, got: {msg}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    fn load_migrates_legacy_for_connector(connector: &str, extra_fields: &str) {
        let dir = unique_test_dir(&format!("da-test-mig-{connector}"));
        let path = dir.join("daemon.toml");
        let legacy = format!(
            r#"[llm]
connector = "{connector}"
{extra_fields}

[backend_tasks]
dreaming_enabled = true
"#
        );
        std::fs::write(&path, &legacy).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();

        // Exactly one synthesized connection called `default`.
        assert_eq!(loaded.connections.len(), 1);
        let (id, conn) = loaded.connections.iter().next().unwrap();
        assert_eq!(id, "default");
        let type_tag = conn.connector_type();
        let expected = match connector {
            "aws-bedrock" => "bedrock",
            other => other,
        };
        assert_eq!(
            type_tag, expected,
            "connector type mismatch for {connector}"
        );

        // Backup written alongside original.
        let bak = dir.join("daemon.toml.bak");
        assert!(bak.exists(), ".bak should exist after migration");
        let backed_up = std::fs::read_to_string(&bak).unwrap();
        assert_eq!(backed_up, legacy, ".bak should be the original content");

        // New form persisted.
        let persisted = std::fs::read_to_string(&path).unwrap();
        assert!(
            persisted.contains("[connections.default]"),
            "rewritten config should contain migrated connection: {persisted}"
        );
        assert!(
            persisted.contains(&format!("type = \"{expected}\"")),
            "rewritten config should declare connector type: {persisted}"
        );

        // Reload is idempotent — no new .bak, no new rewrite, connections still parse.
        let reloaded = load_daemon_config(&path).unwrap().unwrap();
        assert_eq!(reloaded.connections.len(), 1);
        assert!(
            !dir.join("daemon.toml.bak.2").exists(),
            "second load should not create a new backup"
        );

        // backend_tasks preserved (shape unchanged; #10 reshapes it).
        assert!(reloaded.backend_tasks.dreaming_enabled);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migration_openai() {
        load_migrates_legacy_for_connector(
            "openai",
            r#"base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY""#,
        );
    }

    #[test]
    fn migration_anthropic() {
        load_migrates_legacy_for_connector(
            "anthropic",
            r#"base_url = "https://api.anthropic.com""#,
        );
    }

    #[test]
    fn migration_bedrock() {
        load_migrates_legacy_for_connector(
            "bedrock",
            r#"base_url = "us-west-2"
aws_profile = "home""#,
        );
    }

    #[test]
    fn migration_aws_bedrock_alias() {
        // Legacy users of the `aws-bedrock` connector alias migrate to the
        // canonical `bedrock` variant.
        load_migrates_legacy_for_connector("aws-bedrock", r#"base_url = "us-east-1""#);
    }

    #[test]
    fn migration_ollama() {
        load_migrates_legacy_for_connector("ollama", r#"base_url = "http://localhost:11434""#);
    }

    #[test]
    fn migration_picks_bak_dot_n_when_bak_exists() {
        let dir = unique_test_dir("da-test-bak-collision");
        let path = dir.join("daemon.toml");
        // Pre-existing .bak file — migration must not clobber it.
        let existing_bak_content = "# pre-existing backup from a previous migration\n";
        std::fs::write(dir.join("daemon.toml.bak"), existing_bak_content).unwrap();

        let legacy = r#"[llm]
connector = "openai"
api_key_env = "OPENAI_API_KEY"
"#;
        std::fs::write(&path, legacy).unwrap();

        let _loaded = load_daemon_config(&path).unwrap().unwrap();

        // Original .bak preserved as-is.
        let preserved = std::fs::read_to_string(dir.join("daemon.toml.bak")).unwrap();
        assert_eq!(preserved, existing_bak_content);

        // New backup in .bak.2 with original content.
        let bak2 = dir.join("daemon.toml.bak.2");
        assert!(bak2.exists(), ".bak.2 should exist when .bak is taken");
        let bak2_content = std::fs::read_to_string(&bak2).unwrap();
        assert_eq!(bak2_content, legacy);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migration_bumps_to_bak_dot_3_when_bak_and_bak2_exist() {
        let dir = unique_test_dir("da-test-bak-collision-2");
        let path = dir.join("daemon.toml");
        std::fs::write(dir.join("daemon.toml.bak"), "old1").unwrap();
        std::fs::write(dir.join("daemon.toml.bak.2"), "old2").unwrap();

        let legacy = r#"[llm]
connector = "openai"
"#;
        std::fs::write(&path, legacy).unwrap();

        let _loaded = load_daemon_config(&path).unwrap().unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.join("daemon.toml.bak")).unwrap(),
            "old1"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("daemon.toml.bak.2")).unwrap(),
            "old2"
        );
        assert!(dir.join("daemon.toml.bak.3").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migration_reshapes_backend_tasks_llm_into_purposes_same_connector() {
        // Issue #10: when `[backend_tasks.llm]` uses the same connector as
        // `[llm]`, it does not need a new connection — purposes inherit the
        // primary connection and pin their model to the backend-tasks model.
        let dir = unique_test_dir("da-test-bt-llm-same-connector");
        let path = dir.join("daemon.toml");
        let legacy = r#"[llm]
connector = "openai"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5.4"

[backend_tasks]
dreaming_enabled = true

[backend_tasks.llm]
connector = "openai"
model = "gpt-4o-mini"
"#;
        std::fs::write(&path, legacy).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();

        // `backend_tasks.llm` has been absorbed into `[purposes]` and removed.
        assert!(loaded.backend_tasks.llm.is_none());
        assert!(loaded.backend_tasks.dreaming_enabled);

        // Exactly one connection: the primary synthesized by #8's migration.
        assert_eq!(loaded.connections.len(), 1);
        assert!(loaded.connections.contains_key("default"));

        // Interactive → default connection, primary model preserved.
        let interactive = loaded
            .purposes
            .get(PurposeKind::Interactive)
            .expect("interactive");
        assert_eq!(interactive.connection.to_string(), "default");
        assert_eq!(interactive.model.to_string(), "gpt-5.4");

        // Dreaming/titling → primary connection, backend model.
        let dreaming = loaded
            .purposes
            .get(PurposeKind::Dreaming)
            .expect("dreaming");
        assert_eq!(dreaming.connection.to_string(), "primary");
        assert_eq!(dreaming.model.to_string(), "gpt-4o-mini");
        let titling = loaded.purposes.get(PurposeKind::Titling).expect("titling");
        assert_eq!(titling.connection.to_string(), "primary");
        assert_eq!(titling.model.to_string(), "gpt-4o-mini");

        // Embedding always inherits both (the legacy `[llm]` didn't carry an
        // embedding model — that lives in `[embeddings]`, untouched here).
        let embedding = loaded
            .purposes
            .get(PurposeKind::Embedding)
            .expect("embedding");
        assert_eq!(embedding.connection.to_string(), "primary");
        assert_eq!(embedding.model.to_string(), "primary");

        let rewritten = std::fs::read_to_string(&path).unwrap();
        assert!(
            !rewritten.contains("[backend_tasks.llm]"),
            "backend_tasks.llm should be dropped after migration: {rewritten}"
        );
        assert!(rewritten.contains("[purposes.interactive]"));
        assert!(rewritten.contains("[purposes.dreaming]"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migration_reshapes_backend_tasks_llm_into_purposes_different_connector() {
        // When `[backend_tasks.llm]` uses a *different* connector than
        // `[llm]`, we must synthesize a second connection so both work
        // concurrently. The new connection is named `backend` (or
        // `backend_N` if taken) and owns the backend-tasks credentials.
        let dir = unique_test_dir("da-test-bt-llm-diff-connector");
        let path = dir.join("daemon.toml");
        let legacy = r#"[llm]
connector = "openai"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5.4"

[backend_tasks]
dreaming_enabled = true

[backend_tasks.llm]
connector = "anthropic"
model = "claude-haiku-4-5-20251001"
"#;
        std::fs::write(&path, legacy).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();

        assert!(loaded.backend_tasks.llm.is_none());
        assert_eq!(loaded.connections.len(), 2);
        assert!(loaded.connections.contains_key("default"));
        assert!(loaded.connections.contains_key("backend"));
        assert_eq!(
            loaded.connections.get("backend").unwrap().connector_type(),
            "anthropic"
        );

        let interactive = loaded.purposes.get(PurposeKind::Interactive).unwrap();
        assert_eq!(interactive.connection.to_string(), "default");
        assert_eq!(interactive.model.to_string(), "gpt-5.4");

        // Dreaming/titling → named `backend`, with the backend model.
        let dreaming = loaded.purposes.get(PurposeKind::Dreaming).unwrap();
        assert_eq!(dreaming.connection.to_string(), "backend");
        assert_eq!(dreaming.model.to_string(), "claude-haiku-4-5-20251001");
        let titling = loaded.purposes.get(PurposeKind::Titling).unwrap();
        assert_eq!(titling.connection.to_string(), "backend");

        // Embedding → always `primary`/`primary`, because embedding models
        // live in `[embeddings]`, not in `backend_tasks.llm`. Users with a
        // cross-connector embeddings config keep that config; the purpose
        // entry is just there for a uniform lookup point.
        let embedding = loaded.purposes.get(PurposeKind::Embedding).unwrap();
        assert_eq!(embedding.connection.to_string(), "primary");
        assert_eq!(embedding.model.to_string(), "primary");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migration_reshapes_absent_backend_tasks_llm_to_primary() {
        // Legacy `[llm]` alone (no `[backend_tasks.llm]`) still synthesizes
        // purposes — dreaming/titling/embedding inherit everything via
        // `"primary"` since there is no per-backend override to honour.
        let dir = unique_test_dir("da-test-bt-llm-absent");
        let path = dir.join("daemon.toml");
        let legacy = r#"[llm]
connector = "openai"
api_key_env = "OPENAI_API_KEY"
"#;
        std::fs::write(&path, legacy).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();
        assert_eq!(loaded.connections.len(), 1);

        let interactive = loaded.purposes.get(PurposeKind::Interactive).unwrap();
        assert_eq!(interactive.connection.to_string(), "default");
        // Model falls back to the connector default when none was set.
        assert_eq!(interactive.model.to_string(), "gpt-5.4");

        for p in [
            loaded.purposes.get(PurposeKind::Dreaming).unwrap(),
            loaded.purposes.get(PurposeKind::Titling).unwrap(),
            loaded.purposes.get(PurposeKind::Embedding).unwrap(),
        ] {
            assert_eq!(p.connection.to_string(), "primary");
            assert_eq!(p.model.to_string(), "primary");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn connection_migration_skipped_when_connections_already_present() {
        // Hybrid config (both `[llm]` and `[connections]`) must not trigger
        // the connection-synthesis step, because doing so would silently
        // overwrite user-authored connections. Purpose synthesis (#10)
        // still runs because the legacy `[llm]` marker is present and no
        // `[purposes]` table has been authored — interactive is pinned to
        // the first user-authored connection, not a new `default`.
        let dir = unique_test_dir("da-test-hybrid-skip");
        let path = dir.join("daemon.toml");
        let content = r#"[llm]
connector = "openai"

[connections.work]
type = "openai"
api_key_env = "OPENAI_WORK_KEY"
"#;
        std::fs::write(&path, content).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();

        // Connections untouched.
        assert_eq!(loaded.connections.len(), 1);
        assert!(loaded.connections.contains_key("work"));
        // No backup because connection migration was the only thing that
        // writes .bak; purpose migration rewrites the file in place.
        assert!(!dir.join("daemon.toml.bak").exists());

        // Purposes synthesized, pointing at the user-authored connection.
        let interactive = loaded.purposes.get(PurposeKind::Interactive).unwrap();
        assert_eq!(interactive.connection.to_string(), "work");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn purpose_migration_skipped_when_purposes_already_present() {
        // If the user already authored a `[purposes]` block, respect it.
        let dir = unique_test_dir("da-test-purposes-respected");
        let path = dir.join("daemon.toml");
        let content = r#"[connections.work]
type = "openai"
api_key_env = "OPENAI_WORK_KEY"

[purposes.interactive]
connection = "work"
model = "gpt-5.4"
effort = "high"
"#;
        std::fs::write(&path, content).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();
        let interactive = loaded.purposes.get(PurposeKind::Interactive).unwrap();
        assert_eq!(interactive.effort, Some(crate::purposes::Effort::High));
        // No other purposes synthesized.
        assert!(loaded.purposes.get(PurposeKind::Dreaming).is_none());

        // File unchanged (no legacy shape, no purpose migration).
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk, content);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migration_golden_file_purposes_anthropic_backend() {
        // Golden-file test for the purpose-migration shape when
        // `[backend_tasks.llm]` targets a different connector than `[llm]`.
        // Exercises: new `backend` connection synthesis, dreaming/titling
        // pointed at it, backend_tasks.llm removed from serialized form.
        let legacy =
            include_str!("../../tests/fixtures/purposes_migration/legacy_anthropic_backend.toml");
        let expected_new =
            include_str!("../../tests/fixtures/purposes_migration/migrated_anthropic_backend.toml");

        let dir = unique_test_dir("da-test-golden-purposes");
        let path = dir.join("daemon.toml");
        std::fs::write(&path, legacy).unwrap();

        let _loaded = load_daemon_config(&path).unwrap().unwrap();
        let actual = std::fs::read_to_string(&path).unwrap();

        assert_eq!(
            actual.trim_end(),
            expected_new.trim_end(),
            "migrated form differs from golden fixture.\n--- actual ---\n{actual}\n--- expected ---\n{expected_new}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migration_golden_file_openai() {
        // Golden-file test: a representative legacy config migrates to the
        // expected new form byte-for-byte (modulo trailing whitespace).
        let legacy = include_str!("../../tests/fixtures/connections_migration/legacy_openai.toml");
        let expected_new =
            include_str!("../../tests/fixtures/connections_migration/migrated_openai.toml");

        let dir = unique_test_dir("da-test-golden-openai");
        let path = dir.join("daemon.toml");
        std::fs::write(&path, legacy).unwrap();

        let _loaded = load_daemon_config(&path).unwrap().unwrap();
        let actual = std::fs::read_to_string(&path).unwrap();

        assert_eq!(
            actual.trim_end(),
            expected_new.trim_end(),
            "migrated form differs from golden fixture.\n--- actual ---\n{actual}\n--- expected ---\n{expected_new}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pick_backup_path_returns_bak_when_nothing_exists() {
        let dir = unique_test_dir("da-test-pick-bak-fresh");
        let path = dir.join("daemon.toml");
        let picked = migration::pick_backup_path(&path);
        assert_eq!(picked, dir.join("daemon.toml.bak"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pick_backup_path_escalates_to_bak_dot_n() {
        let dir = unique_test_dir("da-test-pick-bak-escalate");
        let path = dir.join("daemon.toml");
        std::fs::write(dir.join("daemon.toml.bak"), "").unwrap();
        std::fs::write(dir.join("daemon.toml.bak.2"), "").unwrap();
        std::fs::write(dir.join("daemon.toml.bak.3"), "").unwrap();
        let picked = migration::pick_backup_path(&path);
        assert_eq!(picked, dir.join("daemon.toml.bak.4"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn file_has_top_level_table_matches_dotted_and_bracketed() {
        let content = r#"
# leading comment
[llm]
x = 1

[backend_tasks.llm]
y = 2
"#;
        assert!(migration::file_has_top_level_table(content, "llm"));
        assert!(migration::file_has_top_level_table(
            content,
            "backend_tasks"
        ));
        assert!(!migration::file_has_top_level_table(content, "connections"));
    }

    #[test]
    fn ws_jwt_rejects_wrong_issuer() {
        let _guard = ws_jwt_env_lock();
        let test_dir =
            std::env::temp_dir().join(format!("da-test-ws-jwt-iss-{}", uuid::Uuid::new_v4()));
        let data_home = test_dir.join("data");
        std::fs::create_dir_all(&data_home).unwrap();
        // SAFETY: serialised against other JWT tests via `ws_jwt_env_lock`;
        // the temp dir is unique per run (UUID-suffixed) so different test
        // executions can't collide.
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &data_home);
        }

        let token = generate_ws_jwt(Some("tui".to_string())).expect("generate jwt");
        let mut claims = jwt::decode_ws_jwt_claims(&token).expect("decode generated jwt");
        claims.iss = "other-issuer".to_string();
        let forged = jwt::encode_ws_jwt(&claims).expect("re-encode forged jwt");

        assert!(!validate_ws_jwt(&forged).expect("validate forged token"));

        // SAFETY: same scope as the matching `set_var` above (see lock guard).
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
        std::fs::remove_dir_all(&test_dir).ok();
    }

    // ─────────────────────────────────────────────────────────────────────
    // DT-3 (#269): saner default TTL + jti revocation
    // ─────────────────────────────────────────────────────────────────────

    /// Run `body` with `XDG_DATA_HOME` pointed at a fresh unique temp dir so
    /// the signing key and revocation list live in isolation. Serialised
    /// against the other JWT tests via `ws_jwt_env_lock`.
    fn with_isolated_jwt_store<F: FnOnce()>(tag: &str, body: F) {
        let _guard = ws_jwt_env_lock();
        let test_dir =
            std::env::temp_dir().join(format!("da-test-ws-jwt-{tag}-{}", uuid::Uuid::new_v4()));
        let data_home = test_dir.join("data");
        std::fs::create_dir_all(&data_home).unwrap();
        // SAFETY: serialised against other JWT tests via `ws_jwt_env_lock`;
        // the temp dir is unique per run (UUID-suffixed).
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &data_home);
        }
        body();
        // SAFETY: same scope as the matching `set_var` above.
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
        std::fs::remove_dir_all(&test_dir).ok();
    }

    #[test]
    fn default_ws_jwt_ttl_is_one_hour() {
        // DT-3: the daemon default must align with the minter's 1h, not the
        // old 30-day window. Clients re-mint on expiry.
        assert_eq!(
            jwt::default_ws_jwt_ttl_seconds(),
            60 * 60,
            "default WS JWT TTL must be 1 hour"
        );
    }

    #[test]
    fn valid_token_is_accepted() {
        with_isolated_jwt_store("valid", || {
            let token = generate_ws_jwt(Some("tui".to_string())).expect("generate jwt");
            assert!(validate_ws_jwt(&token).expect("validate"));
            assert_eq!(ws_jwt_sub(&token).as_deref(), Some("tui"));
        });
    }

    #[test]
    fn expired_token_is_rejected() {
        with_isolated_jwt_store("expired", || {
            // Forge a token whose exp is in the past (re-signed with the real
            // key so only the clock, not the signature, rejects it).
            let token = generate_ws_jwt(Some("tui".to_string())).expect("generate jwt");
            let mut claims = jwt::decode_ws_jwt_claims(&token).expect("decode");
            claims.exp = claims.iat.saturating_sub(3600);
            let expired = jwt::encode_ws_jwt(&claims).expect("re-encode expired");

            assert!(!validate_ws_jwt(&expired).expect("validate expired"));
            assert!(ws_jwt_sub(&expired).is_none());
        });
    }

    #[test]
    fn revoked_token_is_rejected() {
        with_isolated_jwt_store("revoked", || {
            let token = generate_ws_jwt(Some("tui".to_string())).expect("generate jwt");
            // Valid before revocation.
            assert!(validate_ws_jwt(&token).expect("validate pre-revoke"));

            revoke_ws_jwt(&token).expect("revoke token");

            // Rejected by BOTH chokepoints after revocation.
            assert!(
                !validate_ws_jwt(&token).expect("validate post-revoke"),
                "revoked token must not validate"
            );
            assert!(
                ws_jwt_sub(&token).is_none(),
                "revoked token must not yield a sub"
            );
        });
    }

    #[test]
    fn revoking_one_token_does_not_affect_another() {
        with_isolated_jwt_store("revoke-isolation", || {
            let a = generate_ws_jwt(Some("tui".to_string())).expect("token a");
            let b = generate_ws_jwt(Some("plasmoid".to_string())).expect("token b");

            revoke_ws_jwt(&a).expect("revoke a");

            assert!(!validate_ws_jwt(&a).expect("a rejected"));
            assert!(validate_ws_jwt(&b).expect("b still valid"));
        });
    }

    #[test]
    fn revocation_survives_a_fresh_decode_path() {
        // The deny-list must be persisted, not just held in memory: a second
        // logical "process" (a fresh read of the list from disk) still
        // rejects the revoked jti.
        with_isolated_jwt_store("revoke-persist", || {
            let token = generate_ws_jwt(Some("tui".to_string())).expect("generate jwt");
            revoke_ws_jwt(&token).expect("revoke");

            // Read the revocation file straight from disk and confirm the jti
            // is recorded (persistence, not just a cached set).
            let claims =
                jwt::decode_ws_jwt_claims_ignoring_revocation(&token).expect("decode for jti");
            assert!(
                jwt::is_jti_revoked(&claims.jti),
                "revoked jti must be readable from the persisted list"
            );
        });
    }

    #[test]
    fn revoking_unparseable_token_is_an_error_not_a_silent_noop() {
        // Unhappy path: revoking garbage must surface an error so an operator
        // isn't lulled into thinking a bad token id was revoked.
        with_isolated_jwt_store("revoke-malformed", || {
            assert!(revoke_ws_jwt("not-a-jwt").is_err());
            assert!(revoke_ws_jwt("").is_err());
        });
    }

    #[test]
    fn expired_revocation_entries_are_pruned() {
        // The deny-list must self-prune: an entry whose exp has passed is
        // dropped so the file can't grow without bound. We revoke a token,
        // then prove a manually-aged entry is gone after a prune cycle.
        with_isolated_jwt_store("revoke-prune", || {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            // Record a long-dead jti directly, then prune.
            jwt::record_revocation_for_test("dead-jti", now.saturating_sub(10_000));
            jwt::record_revocation_for_test("live-jti", now + 10_000);

            jwt::prune_revocations();

            assert!(!jwt::is_jti_revoked("dead-jti"), "expired entry pruned");
            assert!(jwt::is_jti_revoked("live-jti"), "live entry retained");
        });
    }

    // ─────────────────────────────────────────────────────────────────────
    // Purpose-aware LLM config resolution
    // ─────────────────────────────────────────────────────────────────────

    /// Build a config with an `ollama` interactive connection at the given
    /// id. Used as a base for purpose-resolution tests so they don't have to
    /// repeat the same TOML each time.
    fn config_with_ollama_interactive(connection_id: &str, model: &str) -> DaemonConfig {
        toml::from_str(&format!(
            r#"
            [llm]
            connector = "ollama"

            [connections.{connection_id}]
            type = "ollama"
            base_url = "http://localhost:11434"

            [purposes.interactive]
            connection = "{connection_id}"
            model = "{model}"
            "#
        ))
        .expect("test fixture should parse")
    }

    #[test]
    fn resolve_purpose_returns_none_when_no_purposes_configured() {
        // Bare `[llm]` config with no `[purposes]` table: every kind returns
        // None so callers can fall back to the legacy resolvers.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "openai"
            "#,
        )
        .unwrap();

        for kind in PurposeKind::all() {
            assert!(
                resolve_purpose_llm_config(Some(&config), kind).is_none(),
                "expected None for {kind:?} when no purposes configured"
            );
        }
    }

    #[test]
    fn resolve_purpose_returns_none_when_purpose_kind_absent() {
        // Interactive is set; embedding is not. Asking for embedding must
        // return None — `purposes.embedding` was never authored.
        let config = config_with_ollama_interactive("local", "llama3.2");

        assert!(resolve_purpose_llm_config(Some(&config), PurposeKind::Interactive).is_some());
        assert!(resolve_purpose_llm_config(Some(&config), PurposeKind::Embedding).is_none());
        assert!(resolve_purpose_llm_config(Some(&config), PurposeKind::Dreaming).is_none());
        assert!(resolve_purpose_llm_config(Some(&config), PurposeKind::Titling).is_none());
    }

    #[test]
    fn resolve_purpose_pulls_connector_and_overrides_model() {
        // Purpose pins a *different* model than any connection-level default;
        // we should see the purpose's model flow through.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "ollama"

            [connections.local]
            type = "ollama"
            base_url = "http://localhost:11434"

            [purposes.interactive]
            connection = "local"
            model = "llama3.2"

            [purposes.dreaming]
            connection = "local"
            model = "qwen2.5:14b"
            "#,
        )
        .unwrap();

        let resolved = resolve_purpose_llm_config(Some(&config), PurposeKind::Dreaming)
            .expect("dreaming purpose should resolve");
        assert_eq!(resolved.connector, "ollama");
        assert_eq!(resolved.model, "qwen2.5:14b");
        assert_eq!(resolved.base_url, "http://localhost:11434");
    }

    #[test]
    fn resolve_purpose_inherits_model_from_interactive_via_primary() {
        // `model = "primary"` is the documented inheritance sentinel.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "ollama"

            [connections.local]
            type = "ollama"
            base_url = "http://localhost:11434"

            [purposes.interactive]
            connection = "local"
            model = "llama3.2"

            [purposes.titling]
            connection = "primary"
            model = "primary"
            "#,
        )
        .unwrap();

        let resolved = resolve_purpose_llm_config(Some(&config), PurposeKind::Titling)
            .expect("titling should resolve via primary inheritance");
        assert_eq!(resolved.model, "llama3.2");
        assert_eq!(resolved.connector, "ollama");
    }

    #[test]
    fn resolve_purpose_uses_purpose_connection_when_different_from_interactive() {
        // Two connections; interactive pins one, dreaming pins the other.
        // The dreaming resolver must pick up the second connection's
        // connector / base_url, not the interactive one's.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "ollama"

            [connections.local]
            type = "ollama"
            base_url = "http://localhost:11434"

            [connections.remote]
            type = "ollama"
            base_url = "http://remote.example:11434"

            [purposes.interactive]
            connection = "local"
            model = "llama3.2"

            [purposes.dreaming]
            connection = "remote"
            model = "qwen2.5"
            "#,
        )
        .unwrap();

        let resolved = resolve_purpose_llm_config(Some(&config), PurposeKind::Dreaming)
            .expect("dreaming should resolve");
        assert_eq!(resolved.connector, "ollama");
        assert_eq!(resolved.base_url, "http://remote.example:11434");
        assert_eq!(resolved.model, "qwen2.5");
    }

    #[test]
    fn resolve_purpose_dangling_connection_falls_back_to_interactive() {
        // `purpose.dreaming.connection = "missing"` — `resolve_purpose` warns
        // and falls back to interactive's connection. The model stays as
        // authored (no sensible auto-fallback).
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "ollama"

            [connections.local]
            type = "ollama"
            base_url = "http://localhost:11434"

            [purposes.interactive]
            connection = "local"
            model = "llama3.2"

            [purposes.dreaming]
            connection = "missing"
            model = "qwen2.5"
            "#,
        )
        .unwrap();

        let resolved = resolve_purpose_llm_config(Some(&config), PurposeKind::Dreaming)
            .expect("should fall back rather than error");
        // Connector/base_url come from interactive's `local` connection.
        assert_eq!(resolved.connector, "ollama");
        assert_eq!(resolved.base_url, "http://localhost:11434");
        // Model stays as authored — `purpose.dreaming.model` was never wrong,
        // only its connection ref was.
        assert_eq!(resolved.model, "qwen2.5");
    }

    #[test]
    fn resolve_purpose_returns_none_when_no_config() {
        // Defensive: callers may pass `None` for ambient `daemon_config`.
        for kind in PurposeKind::all() {
            assert!(resolve_purpose_llm_config(None, kind).is_none());
        }
    }

    #[test]
    fn resolve_embeddings_uses_purposes_embedding_when_configured() {
        // A user who has set `[purposes.embedding]` gets *that*
        // connection/model back from `resolve_embeddings_config`, not
        // whatever the legacy `[embeddings]` block (or `[llm]` fallback)
        // would have inferred.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "anthropic"

            [connections.local]
            type = "ollama"
            base_url = "http://localhost:11434"

            [purposes.interactive]
            connection = "local"
            model = "llama3.2"

            [purposes.embedding]
            connection = "local"
            model = "nomic-embed-text"
            "#,
        )
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        // Without the purpose-aware path, this would resolve to `anthropic`
        // (from `[llm].connector`) and `available = false`.
        assert_eq!(view.connector, "ollama");
        assert_eq!(view.model, "nomic-embed-text");
        assert_eq!(view.base_url, "http://localhost:11434");
        assert!(view.available, "ollama embedding must be marked available");
        assert!(
            !view.is_default,
            "is_default should be false when purposes.embedding is explicit"
        );
    }

    #[test]
    fn resolve_embeddings_falls_back_to_legacy_when_no_purpose() {
        // When `[purposes.embedding]` is *not* set, the legacy resolver
        // path runs unchanged: `[embeddings]` overrides win, then the
        // `[llm].connector` default. Installs without a purposes block
        // see no behaviour change.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "ollama"

            [connections.local]
            type = "ollama"
            base_url = "http://localhost:11434"

            [purposes.interactive]
            connection = "local"
            model = "llama3.2"
            "#,
        )
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        // Legacy default for ollama.
        assert_eq!(view.connector, "ollama");
        assert_eq!(view.model, "nomic-embed-text");
        assert!(view.available);
        assert!(view.is_default, "no [embeddings] override → is_default");
    }

    #[test]
    fn resolve_embeddings_purpose_with_primary_model_inherits_interactive() {
        // `purposes.embedding.model = "primary"` inherits interactive's model.
        // Unusual for embeddings (LLM models don't normally double as
        // embedding models) but the resolver should still wire the
        // inheritance correctly — model validity is a deployment concern.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "ollama"

            [connections.local]
            type = "ollama"
            base_url = "http://localhost:11434"

            [purposes.interactive]
            connection = "local"
            model = "nomic-embed-text"

            [purposes.embedding]
            connection = "primary"
            model = "primary"
            "#,
        )
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.connector, "ollama");
        assert_eq!(view.model, "nomic-embed-text");
    }

    #[test]
    fn embeddings_view_carries_api_key_through_legacy_path() {
        // The legacy resolver populates `api_key` from the shared LLM
        // resolver when connectors match. Use a clearly-marked env var so
        // we can assert the value flows end-to-end without depending on
        // ambient OPENAI_API_KEY.
        let env_var = format!(
            "DA_TEST_PURPOSE_LEGACY_KEY_{}",
            uuid::Uuid::new_v4().simple()
        );
        // SAFETY: unique name, single-threaded test scope.
        unsafe {
            std::env::set_var(&env_var, "legacy-secret");
        }

        let config: DaemonConfig = toml::from_str(&format!(
            r#"
            [llm]
            connector = "openai"
            api_key_env = "{env_var}"
            "#
        ))
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.api_key, "legacy-secret");
        assert!(view.has_api_key);

        // SAFETY: same scope as the matching `set_var` above; env var
        // name is unique per run.
        unsafe {
            std::env::remove_var(&env_var);
        }
    }

    #[test]
    fn embeddings_view_carries_api_key_through_purpose_path() {
        // Mirror of the legacy test, but via `purposes.embedding`. Proves
        // the api_key from the purpose's connection's secret/env reaches the
        // view (not just `has_api_key`), so `main.rs` can hand it to the
        // OpenAI-compatible embedding client without an extra round-trip.
        let env_var = format!("DA_TEST_PURPOSE_KEY_{}", uuid::Uuid::new_v4().simple());
        // SAFETY: unique name, single-threaded test scope.
        unsafe {
            std::env::set_var(&env_var, "purpose-secret");
        }

        let config: DaemonConfig = toml::from_str(&format!(
            r#"
            [llm]
            connector = "openai"

            [connections.cloud]
            type = "openai"
            base_url = "https://api.openai.com/v1"
            api_key_env = "{env_var}"

            [purposes.interactive]
            connection = "cloud"
            model = "gpt-4o"

            [purposes.embedding]
            connection = "cloud"
            model = "text-embedding-3-small"
            "#
        ))
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.connector, "openai");
        assert_eq!(view.model, "text-embedding-3-small");
        assert_eq!(view.api_key, "purpose-secret");
        assert!(view.has_api_key);

        // SAFETY: same scope as the matching `set_var` above; env var
        // name is unique per run.
        unsafe {
            std::env::remove_var(&env_var);
        }
    }

    #[test]
    fn purpose_only_config_without_legacy_llm_block_loads_and_resolves() {
        // Hygiene check: a config with `[purposes.*]` + `[connections.*]` and
        // no legacy `[llm]` / `[embeddings]` / `[backend_tasks.llm]` blocks
        // must parse, validate, and produce a working dispatch view for
        // every kind. This is the shape we recommend after PRs #29-31, and
        // we should not regress on it without noticing.
        let toml_str = r#"
            [connections.bedrock]
            type = "bedrock"
            region = "us-east-1"

            [connections.local]
            type = "ollama"
            base_url = "http://localhost:11434"

            [purposes.interactive]
            connection = "bedrock"
            model = "us.anthropic.claude-sonnet-4-6"
            effort = "medium"

            [purposes.dreaming]
            connection = "bedrock"
            model = "anthropic.claude-haiku-4-5"

            [purposes.consolidation]
            connection = "bedrock"
            model = "us.anthropic.claude-sonnet-4-6"

            [purposes.embedding]
            connection = "local"
            model = "mxbai-embed-large:335m"

            [purposes.titling]
            connection = "bedrock"
            model = "anthropic.claude-haiku-4-5"

            [purposes.voice]
            connection = "bedrock"
            model = "us.anthropic.claude-sonnet-4-6"
        "#;

        let config: DaemonConfig = toml::from_str(toml_str).expect("parses cleanly");
        config.purposes.validate().expect("purposes valid");
        let _connections = config.validated_connections().expect("connections valid");

        // Every configured purpose must resolve to a concrete client config.
        for kind in PurposeKind::all() {
            let resolved = resolve_purpose_llm_config(Some(&config), kind)
                .expect("purpose must resolve without legacy fallback");
            assert!(
                !resolved.connector.is_empty() && !resolved.model.is_empty(),
                "{kind:?} → empty connector/model"
            );
        }

        // Embeddings view must reflect the purpose, not synthesize from the
        // (absent) `[llm]` block.
        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.connector, "ollama");
        assert_eq!(view.model, "mxbai-embed-large:335m");
        assert!(
            !view.is_default,
            "purpose-driven view must be marked non-default"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Purpose-aware max_context_tokens resolution
    // ─────────────────────────────────────────────────────────────────────

    // --- resolve_context_budget three-tier resolution -------------------

    #[test]
    fn resolve_context_budget_purpose_override_wins() {
        // Tier 1: an explicit `purpose.max_context_tokens` beats the connector
        // value even when it's known. The user always wins — and for Ollama
        // this same value is read back and provisioned as `num_ctx` (then
        // clamped at the connector to the model ceiling and the per-connection
        // hard cap), so it isn't budgeting past the real window. We deliberately
        // don't clamp the override to `connector_max` here: that value is read
        // before the per-turn budget is installed, so it's the pre-override
        // default — clamping would wrongly cap the override down to it.
        let budget = resolve_context_budget(Some(500_000), Some(200_000));
        assert_eq!(budget.max_input_tokens, 500_000);
        assert_eq!(budget.source, BudgetSource::PurposeOverride);
    }

    #[test]
    fn resolve_context_budget_connector_table_used_when_no_override() {
        // Tier 2: when no purpose override is set, the connector's curated
        // value (e.g. `BedrockClient::max_context_tokens()` returning
        // 200k for Claude 3.x) wins over the universal fallback. This
        // matters when a connector knows the model has *less* than the
        // 200k floor (none of our current curated entries do, but the
        // resolver mustn't pretend a smaller window is bigger). Tagged
        // so we can distinguish it from the silent fallback.
        let budget = resolve_context_budget(None, Some(128_000));
        assert_eq!(budget.max_input_tokens, 128_000);
        assert_eq!(budget.source, BudgetSource::ConnectorTable);
    }

    #[test]
    fn resolve_context_budget_universal_fallback_when_neither() {
        // Tier 3: unknown model + no override → conservative 200K
        // fallback so token-based compaction stays on instead of silently
        // disabling for non-curated providers. Tag explicitly identifies
        // the silent floor so operators can grep logs.
        let budget = resolve_context_budget(None, None);
        assert_eq!(budget.max_input_tokens, DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS);
        assert_eq!(budget.source, BudgetSource::UniversalFallback);
        assert_eq!(budget.max_input_tokens, 200_000);
    }

    // --- apply_learned_cap: snap-down cap + success-floor bracket (#343/#425) --

    use crate::config::resolution::apply_learned_cap;
    use desktop_assistant_core::ports::store::LearnedWindow;

    fn budget(max: u64, source: BudgetSource) -> ContextBudget {
        ContextBudget {
            max_input_tokens: max,
            source,
        }
    }

    /// An overflow-only learned row (no success high-water yet).
    fn overflow(observed: u64, configured: u64) -> LearnedWindow {
        LearnedWindow {
            observed_limit: Some(observed),
            configured_window: Some(configured),
            max_success_input: None,
        }
    }

    #[test]
    fn learned_cap_none_leaves_budget_untouched() {
        let b = budget(8_192, BudgetSource::ConnectorTable);
        assert_eq!(apply_learned_cap(b, None), b);
    }

    #[test]
    fn learned_cap_snaps_overflow_down_to_common_rung() {
        // The incident's derived ceiling (202752 − 8192 = 194560) caps the
        // 200k budget DOWN, snapped to the 192k rung and re-tagged.
        let b = budget(200_000, BudgetSource::ConnectorTable);
        let capped = apply_learned_cap(b, Some(overflow(194_560, 200_000)));
        assert_eq!(capped.max_input_tokens, 192_000);
        assert_eq!(capped.source, BudgetSource::LearnedCap);
    }

    #[test]
    fn learned_cap_never_raises_budget() {
        // A snapped cap at/above the resolved budget is a no-op; original stands.
        let b = budget(8_192, BudgetSource::ConnectorTable);
        let capped = apply_learned_cap(b, Some(overflow(200_000, 8_192)));
        assert_eq!(capped.max_input_tokens, 8_192);
        assert_eq!(capped.source, BudgetSource::ConnectorTable);
    }

    #[test]
    fn learned_cap_invalidated_when_configured_window_changes() {
        // The overflow was observed under an 8192 window; the budget is now
        // 16384, so the observation is stale and ignored — the higher ceiling
        // stands. This is how a config bump escapes a previously-learned cap.
        let b = budget(16_384, BudgetSource::PurposeOverride);
        let capped = apply_learned_cap(b, Some(overflow(4_096, 8_192)));
        assert_eq!(capped.max_input_tokens, 16_384);
        assert_eq!(capped.source, BudgetSource::PurposeOverride);
    }

    #[test]
    fn pathological_observed_is_snapped_away_not_applied() {
        // Issue #425: the 534-token poison (and any value below the smallest
        // ladder rung) snaps to `None`, so it can NEVER pin the budget. This is
        // the regression that bricked the assistant.
        for poison in [0_u64, 1, 534, 4_095] {
            let b = budget(200_000, BudgetSource::ConnectorTable);
            let capped = apply_learned_cap(b, Some(overflow(poison, 200_000)));
            assert_eq!(
                capped.max_input_tokens, 200_000,
                "poison {poison} must never pin the budget"
            );
            assert_eq!(capped.source, BudgetSource::ConnectorTable);
        }
    }

    #[test]
    fn success_high_water_floors_an_overaggressive_cap() {
        // Even a legitimate overflow cap can't drop the budget below a size the
        // model has PROVEN it accepts — the #425 safety net / recovery basis.
        let b = budget(200_000, BudgetSource::ConnectorTable);
        let learned = LearnedWindow {
            observed_limit: Some(100_000),
            configured_window: Some(200_000),
            max_success_input: Some(160_000),
        };
        let capped = apply_learned_cap(b, Some(learned));
        assert_eq!(
            capped.max_input_tokens, 160_000,
            "floored up to proven-good"
        );
        assert_eq!(capped.source, BudgetSource::LearnedCap);
    }

    #[test]
    fn success_floor_never_exceeds_configured_budget() {
        // A stale high-water larger than the (now lower) configured budget is
        // clamped to it — the floor can raise, but never above the user's window.
        let b = budget(100_000, BudgetSource::ConnectorTable);
        let learned = LearnedWindow {
            observed_limit: None,
            configured_window: None,
            max_success_input: Some(180_000),
        };
        let capped = apply_learned_cap(b, Some(learned));
        assert_eq!(capped.max_input_tokens, 100_000);
        assert_eq!(capped.source, BudgetSource::ConnectorTable);
    }

    #[test]
    fn success_only_row_below_budget_is_noop() {
        // A proven-good size under the budget doesn't move anything.
        let b = budget(200_000, BudgetSource::ConnectorTable);
        let learned = LearnedWindow {
            observed_limit: None,
            configured_window: None,
            max_success_input: Some(150_000),
        };
        let capped = apply_learned_cap(b, Some(learned));
        assert_eq!(capped.max_input_tokens, 200_000);
        assert_eq!(capped.source, BudgetSource::ConnectorTable);
    }

    /// Integration (issue #342): a real Ollama connector with a small
    /// configured window must drive a small resolved budget — NOT the 200K
    /// universal fallback. Before #342 Ollama reported `None` here, so the
    /// budget fell through to 200K, the 0.85 proactive-compaction trigger
    /// never fired (170K vs. a real 8k window), and every turn landed in
    /// reactive overflow-recovery. This proves the connector's effective
    /// `num_ctx` reaches tier-2 budget resolution so compaction keys off the
    /// real runtime window.
    #[tokio::test]
    async fn ollama_effective_window_drives_budget_not_200k_fallback() {
        use desktop_assistant_core::ports::llm::LlmClient;
        use httpmock::Method::POST;
        use httpmock::MockServer;

        let server = MockServer::start();
        // Model's architecture ceiling is 32k; the configured num_ctx (4096)
        // is smaller, so the effective runtime window is 4096.
        server.mock(|when, then| {
            when.method(POST).path("/api/show");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"model_info":{"qwen2.context_length":32768}}"#);
        });

        let client = desktop_assistant_llm_ollama::OllamaClient::new(server.url(""), "qwen2.5")
            .with_num_ctx(Some(4_096));
        client.warm_context_length().await;

        // Tier-2 source: the connector's honest effective window.
        let connector_max = client.max_context_tokens();
        assert_eq!(connector_max, Some(4_096));

        // No purpose override → tier 2 wins, NOT the 200K fallback.
        let budget = resolve_context_budget(None, connector_max);
        assert_eq!(budget.source, BudgetSource::ConnectorTable);
        assert_eq!(budget.max_input_tokens, 4_096);
        assert_ne!(budget.max_input_tokens, DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS);

        // The 0.85 proactive-compaction trigger therefore fires at ~3481
        // tokens — well below the real window — instead of at 170K, which
        // the model could never reach. Sanity-check the trigger point sits
        // inside the real window.
        let trigger = (budget.max_input_tokens as f64 * 0.85) as u64;
        assert!(trigger < 4_096 && trigger > 3_000);
    }

    #[test]
    fn max_context_purpose_override_pulls_from_config() {
        // The `purpose_max_context_override` helper extracts the field
        // from the right purpose without exposing the `Purposes` map to
        // every caller.
        let config: DaemonConfig = toml::from_str(
            r#"
            [connections.bedrock]
            type = "bedrock"
            region = "us-east-1"

            [purposes.interactive]
            connection = "bedrock"
            model = "us.amazon.nova-premier-v1:0"
            max_context_tokens = 1000000

            [purposes.dreaming]
            connection = "bedrock"
            model = "anthropic.claude-haiku-4-5"
            "#,
        )
        .unwrap();

        // Interactive carries an explicit override.
        assert_eq!(
            purpose_max_context_override(Some(&config), PurposeKind::Interactive),
            Some(1_000_000)
        );
        // Dreaming has the field absent → None (caller falls through to
        // tier 2/3).
        assert_eq!(
            purpose_max_context_override(Some(&config), PurposeKind::Dreaming),
            None
        );
        // Unconfigured purpose → None.
        assert_eq!(
            purpose_max_context_override(Some(&config), PurposeKind::Embedding),
            None
        );
        // No config at all → None.
        assert_eq!(
            purpose_max_context_override(None, PurposeKind::Interactive),
            None
        );
    }

    #[test]
    fn max_context_purpose_override_roundtrips_through_toml() {
        // Migration check: a config WITHOUT the field deserializes (legacy
        // shape). A config WITH the field round-trips byte-equivalent
        // (modulo whitespace) — `None` on serialize is omitted, `Some`
        // on serialize is preserved.

        // 1. Legacy config — no `max_context_tokens` anywhere.
        let legacy_toml = r#"
[connections.local]
type = "ollama"
base_url = "http://localhost:11434"

[purposes.interactive]
connection = "local"
model = "llama3.2"
"#;
        let legacy: DaemonConfig = toml::from_str(legacy_toml).expect("legacy parses");
        assert_eq!(
            legacy
                .purposes
                .get(PurposeKind::Interactive)
                .unwrap()
                .max_context_tokens,
            None
        );
        let reserialized = toml::to_string(&legacy).unwrap();
        assert!(
            !reserialized.contains("max_context_tokens"),
            "None must not appear on the wire: {reserialized}"
        );

        // 2. Config with an explicit override round-trips.
        let with_override_toml = r#"
[connections.bedrock]
type = "bedrock"
region = "us-east-1"

[purposes.interactive]
connection = "bedrock"
model = "us.amazon.nova-premier-v1:0"
max_context_tokens = 1000000
"#;
        let parsed: DaemonConfig = toml::from_str(with_override_toml).unwrap();
        assert_eq!(
            parsed
                .purposes
                .get(PurposeKind::Interactive)
                .unwrap()
                .max_context_tokens,
            Some(1_000_000)
        );
        let serialized = toml::to_string(&parsed).unwrap();
        assert!(
            serialized.contains("max_context_tokens"),
            "explicit override must be preserved: {serialized}"
        );
        let reparsed: DaemonConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed.purposes, reparsed.purposes);
    }

    // --- [personality] section (#226) --------------------------------------

    #[test]
    fn personality_defaults_when_section_absent() {
        // A config with no `[personality]` block still resolves the
        // Expressive-7 defaults so every install has a disposition.
        let config: DaemonConfig = toml::from_str("").unwrap();
        assert_eq!(config.personality, Personality::default());
        assert_eq!(config.personality.professionalism, PersonalityLevel::Always);
        assert_eq!(config.personality.humor, PersonalityLevel::Sometimes);
    }

    #[test]
    fn personality_section_parses_and_round_trips() {
        let config: DaemonConfig = toml::from_str(
            r#"
            [personality]
            professionalism = "always"
            warmth = "often"
            directness = "often"
            enthusiasm = "sometimes"
            humor = "never"
            sarcasm = "rarely"
            pretentiousness = "rarely"
            "#,
        )
        .unwrap();

        assert_eq!(config.personality.humor, PersonalityLevel::Never);
        assert_eq!(config.personality.warmth, PersonalityLevel::Often);

        // Serialize → reparse is lossless.
        let serialized = toml::to_string(&config).unwrap();
        let reparsed: DaemonConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(config.personality, reparsed.personality);
    }

    // ─────────────────────────────────────────────────────────────────────
    // #438 — resolution.rs decision branches
    // ─────────────────────────────────────────────────────────────────────

    // --- resolve_consolidation_llm_config 3-level fallback (resolution.rs:213) --

    #[test]
    fn consolidation_llm_prefers_own_config() {
        // A dedicated `[backend_tasks.consolidation_llm]` wins over both the
        // backend-tasks LLM and the primary `[llm]` — so consolidation can run
        // on a stronger model than extraction.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "openai"
            model = "primary-model"

            [backend_tasks.llm]
            connector = "openai"
            model = "backend-model"

            [backend_tasks.consolidation_llm]
            connector = "anthropic"
            model = "consolidation-model"
            "#,
        )
        .unwrap();

        let resolved = resolve_consolidation_llm_config(Some(&config));
        assert_eq!(resolved.connector, "anthropic");
        assert_eq!(resolved.model, "consolidation-model");
    }

    #[test]
    fn consolidation_llm_falls_back_to_backend_tasks() {
        // No `[backend_tasks.consolidation_llm]` → fall back to the shared
        // backend-tasks LLM (NOT the primary), so consolidation follows the
        // cheaper extraction model rather than silently landing on the primary.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "openai"
            model = "primary-model"

            [backend_tasks.llm]
            connector = "anthropic"
            model = "backend-model"
            "#,
        )
        .unwrap();

        let resolved = resolve_consolidation_llm_config(Some(&config));
        assert_eq!(resolved.connector, "anthropic");
        assert_eq!(resolved.model, "backend-model");
    }

    #[test]
    fn consolidation_llm_falls_back_to_primary() {
        // Neither `consolidation_llm` nor `backend_tasks.llm` set → the primary
        // `[llm]` block is used. This is the model-drift the dream-cycle overhaul
        // depends on: without an override, everything routes to the primary.
        let config: DaemonConfig = toml::from_str(
            r#"
            [llm]
            connector = "openai"
            model = "primary-model"
            "#,
        )
        .unwrap();

        let resolved = resolve_consolidation_llm_config(Some(&config));
        assert_eq!(resolved.connector, "openai");
        assert_eq!(resolved.model, "primary-model");
    }

    // --- resolve_connection_llm_config connector-mismatch skip (resolution.rs:582-599) --

    #[test]
    fn fallback_llm_model_does_not_leak_across_connectors() {
        use crate::connections::AnthropicConnection;

        let anthropic_conn = ConnectionConfig::Anthropic(AnthropicConnection::default());

        // A top-level `[llm]` for a *different* connector (openai) must NOT leak
        // its model / tuning into an Anthropic connection — its values are wrong
        // for that connector and would 400 at dispatch.
        let openai_fallback = LlmConfig {
            connector: "openai".to_string(),
            model: Some("gpt-5.4".to_string()),
            temperature: Some(1.9),
            top_p: Some(0.3),
            max_tokens: Some(4096),
            ..LlmConfig::default()
        };
        let resolved = resolve_connection_llm_config(&anthropic_conn, Some(&openai_fallback));
        assert_eq!(resolved.connector, "anthropic");
        assert_ne!(
            resolved.model, "gpt-5.4",
            "openai model must not leak into the anthropic connection"
        );
        assert_eq!(resolved.temperature, None, "openai temperature leaked");
        assert_eq!(resolved.top_p, None, "openai top_p leaked");
        assert_eq!(resolved.max_tokens, None, "openai max_tokens leaked");

        // Sanity: a fallback whose connector *matches* IS honored — proving the
        // leak-guard admits when it should, not just "always skip".
        let anthropic_fallback = LlmConfig {
            connector: "anthropic".to_string(),
            model: Some("claude-custom".to_string()),
            temperature: Some(0.5),
            ..LlmConfig::default()
        };
        let resolved = resolve_connection_llm_config(&anthropic_conn, Some(&anthropic_fallback));
        assert_eq!(resolved.model, "claude-custom");
        assert_eq!(resolved.temperature, Some(0.5));
    }

    // --- Bedrock base_url-vs-region (resolution.rs:539-553) --

    #[test]
    fn bedrock_region_used_as_base_when_base_url_absent() {
        use crate::connections::BedrockConnection;

        // No explicit base_url → the region is used as the "base" (the historical
        // Bedrock shape where `base_url` encoded the region).
        let with_region = ConnectionConfig::Bedrock(BedrockConnection {
            region: Some("us-west-2".to_string()),
            base_url: None,
            ..BedrockConnection::default()
        });
        assert_eq!(
            resolve_connection_llm_config(&with_region, None).base_url,
            "us-west-2"
        );

        // Explicit base_url wins over region (private-endpoint proxy case).
        let with_base = ConnectionConfig::Bedrock(BedrockConnection {
            region: Some("us-west-2".to_string()),
            base_url: Some("https://bedrock.proxy.internal".to_string()),
            ..BedrockConnection::default()
        });
        assert_eq!(
            resolve_connection_llm_config(&with_base, None).base_url,
            "https://bedrock.proxy.internal"
        );

        // Whitespace-only region is filtered → falls through to the connector's
        // default HTTP base ("us-east-1"), never an empty string.
        let blank_region = ConnectionConfig::Bedrock(BedrockConnection {
            region: Some("   ".to_string()),
            base_url: None,
            ..BedrockConnection::default()
        });
        assert_eq!(
            resolve_connection_llm_config(&blank_region, None).base_url,
            "us-east-1"
        );
    }

    // --- resolve_database_config env fallback (resolution.rs:135) --

    fn db_env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn database_url_env_used_when_config_absent() {
        // Serialize against any other test touching this process-global env var.
        let _guard = db_env_lock();
        const ENV: &str = "DESKTOP_ASSISTANT_DATABASE_URL";
        // SAFETY: single-threaded test scope, serialized via `db_env_lock`.
        unsafe {
            std::env::remove_var(ENV);
        }

        // Config has no `[database].url` → the env var is used as the fallback,
        // and `max_connections` defaults. Without this branch the daemon would
        // start with no persistence despite an env-configured DB.
        // SAFETY: same scope as above.
        unsafe {
            std::env::set_var(ENV, "postgres://env-host/db");
        }
        let (url, max) = resolve_database_config(None);
        assert_eq!(url.as_deref(), Some("postgres://env-host/db"));
        assert_eq!(max, default_database_max_connections());

        // An explicit config url wins over the env var.
        let cfg: DaemonConfig = toml::from_str(
            r#"
            [database]
            url = "postgres://config-host/db"
            max_connections = 20
            "#,
        )
        .unwrap();
        let (url, max) = resolve_database_config(Some(&cfg));
        assert_eq!(url.as_deref(), Some("postgres://config-host/db"));
        assert_eq!(max, 20);

        // Whitespace-only env value is filtered to None.
        // SAFETY: same scope as above.
        unsafe {
            std::env::set_var(ENV, "   ");
        }
        let (url, _) = resolve_database_config(None);
        assert_eq!(url, None);

        // SAFETY: same scope as above; clean up before releasing the lock.
        unsafe {
            std::env::remove_var(ENV);
        }
    }

    // --- Ollama keep_warm (resolution.rs:625-630) --

    #[test]
    fn keep_warm_only_on_ollama() {
        use crate::connections::{
            AnthropicConnection, BedrockConnection, OllamaConnection, OpenAiConnection,
        };

        // Ollama with keep_warm set → resolved true.
        let warm = ConnectionConfig::Ollama(OllamaConnection {
            keep_warm: Some(true),
            ..OllamaConnection::default()
        });
        assert!(resolve_connection_llm_config(&warm, None).keep_warm);

        // Ollama with keep_warm unset → false.
        let unset = ConnectionConfig::Ollama(OllamaConnection {
            keep_warm: None,
            ..OllamaConnection::default()
        });
        assert!(!resolve_connection_llm_config(&unset, None).keep_warm);

        // keep_warm is Ollama-only: every other connector resolves to false,
        // regardless of any fallback config.
        for conn in [
            ConnectionConfig::Anthropic(AnthropicConnection::default()),
            ConnectionConfig::OpenAi(OpenAiConnection::default()),
            ConnectionConfig::Bedrock(BedrockConnection::default()),
        ] {
            assert!(!resolve_connection_llm_config(&conn, None).keep_warm);
        }
    }

    // --- resolve_embeddings_config cross-connector key (resolution.rs:64-70) --

    #[test]
    fn embeddings_uses_own_env_key_when_connector_differs() {
        // The `else` arm: when the embeddings connector differs from the LLM
        // connector, the api_key must come from the embeddings connector's OWN
        // env key — not be reused from the LLM's secret. Unique connector names
        // keep the derived env-var names unique, so this needs no env lock.
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let emb_connector = format!("testemb{suffix}");
        let emb_env = default_api_key_env(&emb_connector);
        let llm_env = format!("TESTLLMKEY{suffix}");

        // SAFETY: env-var names are unique per run; single-threaded test scope.
        unsafe {
            std::env::set_var(&emb_env, "embeddings-own-secret");
            std::env::set_var(&llm_env, "llm-shared-secret");
        }

        let config: DaemonConfig = toml::from_str(&format!(
            r#"
            [llm]
            connector = "openai"
            api_key_env = "{llm_env}"

            [embeddings]
            connector = "{emb_connector}"
            "#
        ))
        .unwrap();

        let view = resolve_embeddings_config(Some(&config));
        assert_eq!(view.connector, emb_connector);
        assert_eq!(
            view.api_key, "embeddings-own-secret",
            "must read the embeddings connector's own env key"
        );
        assert_ne!(
            view.api_key, "llm-shared-secret",
            "must NOT reuse the LLM's key when connectors differ"
        );
        assert!(view.has_api_key);
        assert!(!view.is_default);

        // SAFETY: same scope as the matching set_var above.
        unsafe {
            std::env::remove_var(&emb_env);
            std::env::remove_var(&llm_env);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // #439 — config views.rs validation + migration edge cases
    // ─────────────────────────────────────────────────────────────────────

    // --- set_llm_settings bounds (views.rs:65-79) --

    #[test]
    fn set_llm_settings_rejects_out_of_range() {
        let dir = unique_test_dir("da-test-setllm-range");
        let path = dir.join("daemon.toml");

        // temperature out of [0.0, 2.0].
        let err =
            set_llm_settings(&path, "openai", None, None, Some(2.5), None, None, None).unwrap_err();
        assert!(err.to_string().contains("temperature"), "{err}");
        assert!(
            set_llm_settings(&path, "openai", None, None, Some(-0.1), None, None, None).is_err()
        );

        // top_p out of [0.0, 1.0].
        let err =
            set_llm_settings(&path, "openai", None, None, None, Some(1.5), None, None).unwrap_err();
        assert!(err.to_string().contains("top_p"), "{err}");
        assert!(
            set_llm_settings(&path, "openai", None, None, None, Some(-0.01), None, None).is_err()
        );

        // max_tokens == 0.
        let err =
            set_llm_settings(&path, "openai", None, None, None, None, Some(0), None).unwrap_err();
        assert!(err.to_string().contains("max_tokens"), "{err}");

        // empty connector.
        assert!(set_llm_settings(&path, "   ", None, None, None, None, None, None).is_err());

        // No rejected call may have persisted anything.
        assert!(
            !path.exists(),
            "rejected settings must not be written to disk"
        );

        // Valid values round-trip through the read view.
        set_llm_settings(
            &path,
            "anthropic",
            Some("claude-x"),
            Some("https://api.anthropic.com"),
            Some(0.7),
            Some(0.9),
            Some(1024),
            Some(true),
        )
        .unwrap();
        let view = get_llm_settings_view(&path).unwrap();
        assert_eq!(view.connector, "anthropic");
        assert_eq!(view.model, "claude-x");
        assert_eq!(view.base_url, "https://api.anthropic.com");
        assert_eq!(view.temperature, Some(0.7));
        assert_eq!(view.top_p, Some(0.9));
        assert_eq!(view.max_tokens, Some(1024));
        assert_eq!(view.hosted_tool_search, Some(true));

        // Boundary values are accepted (inclusive ranges).
        set_llm_settings(
            &path,
            "openai",
            None,
            None,
            Some(0.0),
            Some(0.0),
            Some(1),
            None,
        )
        .unwrap();
        set_llm_settings(
            &path,
            "openai",
            None,
            None,
            Some(2.0),
            Some(1.0),
            Some(1),
            None,
        )
        .unwrap();

        std::fs::remove_dir_all(&dir).ok();
    }

    // --- set_api_key (views.rs:92-140) --

    #[test]
    fn set_api_key_rejects_placeholder() {
        // Empty and masked/placeholder values must be rejected BEFORE any secret
        // write, so a redacted "sk-****" round-tripped from the UI can't wipe the
        // stored key. Point the file-backend at a temp dir so the mutation-check
        // (removing the guard) can't touch the real secret store.
        let _guard = ws_jwt_env_lock();
        let dir = unique_test_dir("da-test-setapikey-placeholder");
        let data_home = dir.join("data");
        std::fs::create_dir_all(&data_home).unwrap();
        // SAFETY: serialized via the shared env lock; unique per-run temp dir.
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &data_home);
        }

        let path = dir.join("daemon.toml");
        std::fs::write(&path, "[llm]\nconnector = \"openai\"\n").unwrap();

        let err = set_api_key(&path, "   ").unwrap_err();
        assert!(err.to_string().contains("must not be empty"), "{err}");

        let err = set_api_key(&path, "sk-****").unwrap_err();
        assert!(err.to_string().contains("placeholder"), "{err}");

        // SAFETY: same scope as the matching set_var above.
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn set_api_key_empty_connector_uses_default() {
        // An explicitly-empty `[llm].connector` must fall back to the default
        // connector before deriving the secret account, so a real key still
        // persists (under the openai account) instead of erroring or landing
        // in a garbage bucket. Uses the file backend under a temp XDG dir.
        let _guard = ws_jwt_env_lock();
        let dir = unique_test_dir("da-test-setapikey-default");
        let data_home = dir.join("data");
        std::fs::create_dir_all(&data_home).unwrap();
        // SAFETY: serialized via the shared env lock; unique per-run temp dir.
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &data_home);
        }

        let path = dir.join("daemon.toml");
        std::fs::write(&path, "[llm]\nconnector = \"\"\n").unwrap();

        set_api_key(&path, "sk-live-empty-connector-xyz").expect("set_api_key should succeed");

        // Written under the default connector's account ("openai_api_key").
        let secret_file = data_home
            .join("desktop-assistant")
            .join("secrets")
            .join("openai_api_key");
        let stored = std::fs::read_to_string(&secret_file).expect("secret file written");
        assert_eq!(stored, "sk-live-empty-connector-xyz");

        // SAFETY: same scope as the matching set_var above.
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- set_ws_auth_settings OIDC merge (views.rs:339-380) --

    #[test]
    fn set_ws_auth_preserves_jwks_and_audience_on_partial_update() {
        let dir = unique_test_dir("da-test-wsauth-oidc");
        let path = dir.join("daemon.toml");
        // Seed a fully-populated OIDC block. `jwks_uri` + `audience` are NOT in
        // the setter's argument list, so a partial update must preserve them.
        let content = r#"
[ws_auth]
methods = ["oidc"]

[ws_auth.oidc]
issuer_url = "https://old.example.com"
authorization_endpoint = "https://old.example.com/auth"
token_endpoint = "https://old.example.com/token"
client_id = "old-client"
scopes = "openid profile"
jwks_uri = "https://old.example.com/jwks"
audience = "adelie-aud"
"#;
        std::fs::write(&path, content).unwrap();

        // Partial update: new issuer/endpoints, empty scopes (→ keep existing).
        set_ws_auth_settings(
            &path,
            &["oidc".to_string()],
            "https://new.example.com",
            "https://new.example.com/auth",
            "https://new.example.com/token",
            "new-client",
            "",
        )
        .unwrap();

        let oidc = get_ws_auth_settings(&path)
            .unwrap()
            .oidc
            .expect("oidc block preserved");
        assert_eq!(oidc.issuer_url, "https://new.example.com");
        assert_eq!(oidc.client_id, "new-client");
        assert_eq!(
            oidc.scopes, "openid profile",
            "empty scopes must preserve the existing value"
        );
        assert_eq!(
            oidc.jwks_uri, "https://old.example.com/jwks",
            "jwks_uri must be preserved across a partial update"
        );
        assert_eq!(
            oidc.audience, "adelie-aud",
            "audience must be preserved across a partial update"
        );

        // Removing "oidc" from methods clears the block entirely.
        set_ws_auth_settings(
            &path,
            &["password".to_string()],
            "https://new.example.com",
            "",
            "",
            "",
            "",
        )
        .unwrap();
        assert!(
            get_ws_auth_settings(&path).unwrap().oidc.is_none(),
            "oidc must be cleared when the method is removed"
        );

        // "oidc" present but empty issuer → no block created.
        set_ws_auth_settings(&path, &["oidc".to_string()], "", "", "", "", "").unwrap();
        assert!(
            get_ws_auth_settings(&path).unwrap().oidc.is_none(),
            "empty issuer must not create an oidc block"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // --- migration Case C: different connector, no model (migration.rs:221-233) --

    #[test]
    fn migration_synthesizes_default_model_for_backend_connector() {
        // Case C with NO explicit backend model: the different-connector branch
        // must fall back to `default_backend_llm_model(bt_connector)`, not leave
        // the purpose model unset. The existing different-connector test always
        // pins a model, so this `(Named, None)` arm was unexecuted.
        let dir = unique_test_dir("da-test-mig-case-c-default-model");
        let path = dir.join("daemon.toml");
        let legacy = r#"[llm]
connector = "openai"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5.4"

[backend_tasks]
dreaming_enabled = true

[backend_tasks.llm]
connector = "anthropic"
"#;
        std::fs::write(&path, legacy).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();

        // A second connection was synthesized for the different connector.
        let backend = loaded
            .connections
            .get("backend")
            .expect("synthesized backend connection");
        assert_eq!(backend.connector_type(), "anthropic");

        // Dreaming/titling point at it with the anthropic *backend* default model.
        let dreaming = loaded.purposes.get(PurposeKind::Dreaming).unwrap();
        assert_eq!(dreaming.connection.to_string(), "backend");
        assert_eq!(dreaming.model.to_string(), "claude-haiku-4-5-20251001");
        let titling = loaded.purposes.get(PurposeKind::Titling).unwrap();
        assert_eq!(titling.model.to_string(), "claude-haiku-4-5-20251001");

        std::fs::remove_dir_all(&dir).ok();
    }

    // --- migration pick_free_connection_id collision (migration.rs:285) --

    #[test]
    fn migration_avoids_clobbering_existing_backend_connection() {
        // A legacy config that already declares `[connections.backend]` must not
        // have it overwritten when purpose-migration synthesizes a backend
        // connection for a different-connector `[backend_tasks.llm]`. The
        // collision resolver picks `backend_2` instead — no data loss.
        let dir = unique_test_dir("da-test-mig-backend-collision");
        let path = dir.join("daemon.toml");
        let legacy = r#"[llm]
connector = "openai"
model = "gpt-5.4"

[connections.backend]
type = "ollama"
base_url = "http://user-authored:11434"

[backend_tasks]
dreaming_enabled = true

[backend_tasks.llm]
connector = "anthropic"
model = "claude-haiku-4-5-20251001"
"#;
        std::fs::write(&path, legacy).unwrap();

        let loaded = load_daemon_config(&path).unwrap().unwrap();

        // Exactly two connections: the user's `backend` + the synthesized one.
        assert_eq!(loaded.connections.len(), 2);

        // The user's original `backend` connection is untouched (still ollama).
        let original = loaded
            .connections
            .get("backend")
            .expect("original backend preserved");
        assert_eq!(
            original.connector_type(),
            "ollama",
            "user-authored backend connection must not be clobbered"
        );

        // The synthesized backend connection landed at `backend_2`.
        let synthesized = loaded
            .connections
            .get("backend_2")
            .expect("synthesized backend_2 connection");
        assert_eq!(synthesized.connector_type(), "anthropic");

        // Dreaming points at the synthesized connection, not the user's.
        let dreaming = loaded.purposes.get(PurposeKind::Dreaming).unwrap();
        assert_eq!(dreaming.connection.to_string(), "backend_2");

        std::fs::remove_dir_all(&dir).ok();
    }
}
