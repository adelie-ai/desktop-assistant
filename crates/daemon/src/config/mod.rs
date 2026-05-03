// Submodules (#41 — split into focused modules).
mod jwt;
mod migration;
mod oidc;
#[cfg(target_os = "linux")]
mod pam_auth;
mod secrets;

// Re-export the JWT + OIDC public API at the `config::` path so existing
// callers (`config::generate_ws_jwt`, `config::OidcValidator`, etc.)
// keep working unchanged.
pub use jwt::{current_username, generate_ws_jwt, validate_ws_jwt};
pub use oidc::OidcValidator;
// Bring the secrets-backend helpers used by non-test code in
// `mod.rs` (settings setters, audit logging) into scope so call
// sites don't need `secrets::…` prefixes. Test-only callers
// reference `secrets::sanitize_secret_value` directly to avoid a
// cfg-gated `use`.
use secrets::{
    bucket_secret_len, is_placeholder_secret_value, read_secret_from_backend,
    redacted_secret_audit, write_secret_to_backend,
};

use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use desktop_assistant_core::ports::llm::{BudgetSource, ContextBudget};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::connections::{
    AnthropicConnection, BedrockConnection, ConnectionConfig, ConnectionId, ConnectionsError,
    ConnectionsMap, Connector, OllamaConnection, OpenAiConnection,
};
use crate::purposes::{PurposeKind, Purposes};

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
}

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
            allowed_origins: vec![],
        }
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

fn default_oidc_scopes() -> String {
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
}

impl Default for BackendTasksConfig {
    fn default() -> Self {
        Self {
            llm: None,
            dreaming_enabled: false,
            dreaming_interval_secs: default_dreaming_interval_secs(),
            archive_after_days: default_archive_after_days(),
        }
    }
}

fn default_archive_after_days() -> u32 {
    7
}

fn default_dreaming_interval_secs() -> u64 {
    3600
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

fn default_database_max_connections() -> u32 {
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

fn default_connector() -> String {
    "openai".to_string()
}

fn default_secret_backend() -> String {
    "auto".to_string()
}

fn default_git_remote_name() -> String {
    "origin".to_string()
}

fn default_push_on_update() -> bool {
    true
}

fn default_secret_service() -> String {
    "org.desktopAssistant".to_string()
}

fn default_secret_account(connector: &str) -> String {
    format!("{}_api_key", normalized_connector_key_prefix(connector))
}

fn default_api_key_env(connector: &str) -> String {
    format!(
        "{}_API_KEY",
        normalized_connector_key_prefix(connector).to_ascii_uppercase()
    )
}

fn default_model_env(connector: &str) -> String {
    format!(
        "{}_MODEL",
        normalized_connector_key_prefix(connector).to_ascii_uppercase()
    )
}

fn default_base_url_env(connector: &str) -> String {
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
        return Ok(());
    }

    #[cfg(not(unix))]
    {
        std::fs::write(path, content)
            .with_context(|| format!("failed to write daemon config at {}", path.display()))
    }
}

pub fn get_llm_settings_view(path: &Path) -> anyhow::Result<LlmSettingsView> {
    let config = load_daemon_config(path)?;
    let resolved = resolve_llm_config(config.as_ref());

    Ok(LlmSettingsView {
        connector: resolved.connector,
        model: resolved.model,
        base_url: resolved.base_url,
        has_api_key: !resolved.api_key.is_empty(),
        temperature: resolved.temperature,
        top_p: resolved.top_p,
        max_tokens: resolved.max_tokens,
        hosted_tool_search: resolved.hosted_tool_search,
    })
}

pub fn set_llm_settings(
    path: &Path,
    connector: &str,
    model: Option<&str>,
    base_url: Option<&str>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    hosted_tool_search: Option<bool>,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    let connector = connector.trim().to_lowercase();
    if connector.is_empty() {
        return Err(anyhow!("connector must not be empty"));
    }

    if let Some(t) = temperature
        && !(0.0..=2.0).contains(&t)
    {
        return Err(anyhow!("temperature must be between 0.0 and 2.0"));
    }
    if let Some(p) = top_p
        && !(0.0..=1.0).contains(&p)
    {
        return Err(anyhow!("top_p must be between 0.0 and 1.0"));
    }
    if let Some(m) = max_tokens
        && m == 0
    {
        return Err(anyhow!("max_tokens must be greater than 0"));
    }

    config.llm.connector = connector;
    config.llm.model = normalize_optional_value(model);
    config.llm.base_url = normalize_optional_value(base_url);
    config.llm.temperature = temperature;
    config.llm.top_p = top_p;
    config.llm.max_tokens = max_tokens;
    config.llm.hosted_tool_search = hosted_tool_search;

    save_daemon_config(path, &config)
}

pub fn set_api_key(path: &Path, api_key: &str) -> anyhow::Result<()> {
    let api_key = api_key.trim();
    let (key_len, key_fingerprint) = redacted_secret_audit(api_key);
    // Logging the precise length narrows the search space for guessing the
    // connector type from logs (e.g. 51 chars ≈ Anthropic, 32 ≈ OpenAI).
    let key_len_bucket = bucket_secret_len(key_len);

    tracing::info!(
        secret_len_bucket = key_len_bucket,
        secret_fingerprint = %key_fingerprint,
        "received SetApiKey request"
    );

    if api_key.is_empty() {
        return Err(anyhow!("api key must not be empty"));
    }

    if is_placeholder_secret_value(api_key) {
        tracing::warn!(
            secret_len_bucket = key_len_bucket,
            secret_fingerprint = %key_fingerprint,
            "rejecting placeholder-like SetApiKey value"
        );
        return Err(anyhow!(
            "api key looks like a placeholder or masked value; provide the real key"
        ));
    }

    let mut config = load_daemon_config(path)?.unwrap_or_default();
    if config.llm.secret.is_none() {
        config.llm.secret = Some(SecretConfig::default());
    }

    let secret = config
        .llm
        .secret
        .clone()
        .unwrap_or_else(SecretConfig::default);

    let connector = config.llm.connector.trim().to_lowercase();
    let connector = if connector.is_empty() {
        default_connector()
    } else {
        connector
    };

    write_secret_to_backend(&secret, api_key, &connector)?;
    save_daemon_config(path, &config)
}

pub fn get_embeddings_settings_view(path: &Path) -> anyhow::Result<EmbeddingsSettingsView> {
    let config = load_daemon_config(path)?;
    let resolved = resolve_embeddings_config(config.as_ref());
    Ok(resolved)
}

pub fn set_embeddings_settings(
    path: &Path,
    connector: Option<&str>,
    model: Option<&str>,
    base_url: Option<&str>,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    config.embeddings.connector = connector
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|v| v.to_lowercase());
    config.embeddings.model = normalize_optional_value(model);
    config.embeddings.base_url = normalize_optional_value(base_url);

    save_daemon_config(path, &config)
}

pub fn get_persistence_settings_view(path: &Path) -> anyhow::Result<ResolvedPersistenceConfig> {
    let config = load_daemon_config(path)?;
    Ok(resolve_persistence_config(config.as_ref()))
}

pub fn set_persistence_settings(
    path: &Path,
    enabled: bool,
    remote_url: Option<&str>,
    remote_name: Option<&str>,
    push_on_update: bool,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    config.persistence.git.enabled = enabled;
    config.persistence.git.remote_url = normalize_optional_value(remote_url);
    config.persistence.git.remote_name = remote_name
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(default_git_remote_name);
    config.persistence.git.push_on_update = push_on_update;

    save_daemon_config(path, &config)
}

pub fn get_database_settings_view(path: &Path) -> anyhow::Result<(String, u32)> {
    let config = load_daemon_config(path)?;
    let (url, max_connections) = resolve_database_config(config.as_ref());
    Ok((url.unwrap_or_default(), max_connections))
}

pub fn set_database_settings(
    path: &Path,
    url: Option<&str>,
    max_connections: u32,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    config.database.url = normalize_optional_value(url);
    config.database.max_connections = max_connections;

    save_daemon_config(path, &config)
}

pub struct BackendTasksSettingsViewConfig {
    pub has_separate_llm: bool,
    pub llm_connector: String,
    pub llm_model: String,
    pub llm_base_url: String,
    pub dreaming_enabled: bool,
    pub dreaming_interval_secs: u64,
    pub archive_after_days: u32,
}

pub fn get_backend_tasks_settings_view(
    path: &Path,
) -> anyhow::Result<BackendTasksSettingsViewConfig> {
    let config = load_daemon_config(path)?;
    let bt = config.as_ref().map(|c| &c.backend_tasks);
    let has_separate_llm = bt.is_some_and(|b| b.llm.is_some());
    let dreaming_enabled = bt.map(|b| b.dreaming_enabled).unwrap_or(false);
    let dreaming_interval_secs = bt
        .map(|b| b.dreaming_interval_secs)
        .unwrap_or_else(default_dreaming_interval_secs);
    let archive_after_days = bt
        .map(|b| b.archive_after_days)
        .unwrap_or_else(default_archive_after_days);

    let resolved = resolve_backend_tasks_llm_config(config.as_ref());

    Ok(BackendTasksSettingsViewConfig {
        has_separate_llm,
        llm_connector: resolved.connector,
        llm_model: resolved.model,
        llm_base_url: resolved.base_url,
        dreaming_enabled,
        dreaming_interval_secs,
        archive_after_days,
    })
}

pub fn set_backend_tasks_settings(
    path: &Path,
    llm_connector: Option<&str>,
    llm_model: Option<&str>,
    llm_base_url: Option<&str>,
    dreaming_enabled: bool,
    dreaming_interval_secs: u64,
    archive_after_days: u32,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    config.backend_tasks.dreaming_enabled = dreaming_enabled;
    config.backend_tasks.dreaming_interval_secs = dreaming_interval_secs;
    config.backend_tasks.archive_after_days = archive_after_days;

    // If connector is provided, configure a separate backend-tasks LLM.
    // If connector is None/empty, clear the override (fall back to primary).
    let connector = llm_connector.map(str::trim).filter(|v| !v.is_empty());

    if let Some(connector) = connector {
        let mut llm = config.backend_tasks.llm.unwrap_or_default();
        llm.connector = connector.to_lowercase();
        llm.model = normalize_optional_value(llm_model);
        llm.base_url = normalize_optional_value(llm_base_url);
        config.backend_tasks.llm = Some(llm);
    } else {
        config.backend_tasks.llm = None;
    }

    save_daemon_config(path, &config)
}

pub fn get_connector_defaults(connector: &str) -> ConnectorDefaultsView {
    let connector = connector.trim().to_lowercase();
    let connector = if connector.is_empty() {
        default_connector()
    } else {
        connector
    };

    let typed = parse_connector_or_openai(&connector);
    let llm_model = default_llm_model(&connector);
    let llm_base_url = default_base_url(&connector);

    let embeddings_available = typed.supports_embeddings();
    // Substitute OpenAI for the embedding lookup when this connector
    // doesn't ship one (Anthropic) — preserves legacy behaviour where
    // `embeddings_model` always resolves to a real value.
    let embeddings_connector = if embeddings_available {
        typed
    } else {
        Connector::OpenAi
    };

    ConnectorDefaultsView {
        llm_model,
        llm_base_url,
        backend_llm_model: default_backend_llm_model(&connector),
        embeddings_model: embeddings_connector.default_embedding_model().to_string(),
        embeddings_base_url: embeddings_connector.default_base_url().to_string(),
        embeddings_available,
        hosted_tool_search_available: typed.supports_hosted_tool_search(),
    }
}

pub fn resolve_embeddings_config(config: Option<&DaemonConfig>) -> EmbeddingsSettingsView {
    // Purpose-driven path: when `[purposes.embedding]` is configured, it wins
    // over the legacy `[embeddings]` block. The daemon API surface
    // (`set_purpose("embedding", ...)`) writes into `[purposes]`, so without
    // this short-circuit user-set purposes silently get ignored at startup.
    if let Some(view) = resolve_purpose_embeddings_view(config) {
        return view;
    }

    let llm_connector = config
        .map(|c| c.llm.connector.trim().to_lowercase())
        .filter(|c| !c.is_empty())
        .unwrap_or_else(default_connector);

    let emb_config = config.map(|c| &c.embeddings);

    let explicit_connector = emb_config
        .and_then(|c| c.connector.as_deref())
        .map(|v| v.trim().to_lowercase())
        .filter(|v| !v.is_empty());

    let is_default = explicit_connector.is_none();
    let connector = explicit_connector.unwrap_or_else(|| llm_connector.clone());
    let available = connector != "anthropic";

    let model = emb_config
        .and_then(|c| c.model.clone())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default_embedding_model(&connector));

    let base_url = emb_config
        .and_then(|c| c.base_url.clone())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default_base_url(&connector));

    // Resolve API key: reuse LLM secret config if connectors match, otherwise use env fallback
    let api_key = if is_default || connector == llm_connector {
        resolve_llm_config(config).api_key
    } else {
        let env_key = default_api_key_env(&connector);
        std::env::var(env_key).unwrap_or_default()
    };
    let has_api_key = !api_key.trim().is_empty();

    EmbeddingsSettingsView {
        connector,
        model,
        base_url,
        api_key,
        has_api_key,
        available,
        is_default,
    }
}

/// Build an `EmbeddingsSettingsView` from `purposes.embedding` if it is
/// configured, otherwise return `None`. Centralises the purpose-aware
/// short-circuit so the legacy resolver can skip the rest of its work.
fn resolve_purpose_embeddings_view(
    config: Option<&DaemonConfig>,
) -> Option<EmbeddingsSettingsView> {
    let resolved = resolve_purpose_llm_config(config, PurposeKind::Embedding)?;
    let available = resolved.connector != "anthropic";
    let has_api_key = !resolved.api_key.trim().is_empty();
    Some(EmbeddingsSettingsView {
        connector: resolved.connector,
        model: resolved.model,
        base_url: resolved.base_url,
        api_key: resolved.api_key,
        has_api_key,
        available,
        // Always `false` for purpose-driven config: the user explicitly chose
        // a connection/model, so this is no longer "the inferred default".
        is_default: false,
    })
}

pub fn resolve_persistence_config(config: Option<&DaemonConfig>) -> ResolvedPersistenceConfig {
    let persistence = config.map(|c| &c.persistence.git);

    let enabled = persistence.map(|p| p.enabled).unwrap_or(false);
    let remote_url = persistence
        .and_then(|p| p.remote_url.as_deref())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string);

    let remote_name = persistence
        .map(|p| p.remote_name.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(default_git_remote_name);

    let push_on_update = persistence
        .map(|p| p.push_on_update)
        .unwrap_or_else(default_push_on_update);

    ResolvedPersistenceConfig {
        enabled,
        remote_url,
        remote_name,
        push_on_update,
    }
}

/// Resolve the database URL from config, then env var fallback.
/// Returns `None` if no database URL is configured anywhere.
pub fn resolve_database_config(config: Option<&DaemonConfig>) -> (Option<String>, u32) {
    let db = config.map(|c| &c.database);
    let url = db
        .and_then(|d| d.url.clone())
        .or_else(|| std::env::var("DESKTOP_ASSISTANT_DATABASE_URL").ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let max_conns = db
        .map(|d| d.max_connections)
        .unwrap_or_else(default_database_max_connections);
    (url, max_conns)
}

/// Resolve `connector` to a typed [`Connector`], falling back to
/// `Connector::OpenAi` for unrecognised values — the historical
/// "default to OpenAI for unknown connector strings" behaviour, now
/// concentrated in one helper instead of repeated as a `_` arm in
/// every match (#47).
fn parse_connector_or_openai(connector: &str) -> Connector {
    Connector::parse(connector).unwrap_or(Connector::OpenAi)
}

fn default_embedding_model(connector: &str) -> String {
    let c = parse_connector_or_openai(connector);
    let model = c.default_embedding_model();
    // Anthropic has no embeddings; the legacy default for that case
    // was `text-embedding-3-small` (the OpenAI default).
    if model.is_empty() {
        Connector::OpenAi.default_embedding_model().to_string()
    } else {
        model.to_string()
    }
}

fn default_base_url(connector: &str) -> String {
    parse_connector_or_openai(connector)
        .default_base_url()
        .to_string()
}

fn default_llm_model(connector: &str) -> String {
    parse_connector_or_openai(connector)
        .default_chat_model()
        .to_string()
}

fn default_backend_llm_model(connector: &str) -> String {
    parse_connector_or_openai(connector)
        .default_backend_chat_model()
        .to_string()
}

fn normalize_optional_value(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

pub fn resolve_llm_config(config: Option<&DaemonConfig>) -> ResolvedLlmConfig {
    resolve_llm_config_from(config.map(|c| &c.llm))
}

/// Resolve backend-tasks LLM config: uses `[backend_tasks.llm]` if set,
/// otherwise falls back to the top-level `[llm]`.
pub fn resolve_backend_tasks_llm_config(config: Option<&DaemonConfig>) -> ResolvedLlmConfig {
    let bt_llm = config.and_then(|c| c.backend_tasks.llm.as_ref());
    if bt_llm.is_some() {
        resolve_llm_config_from(bt_llm)
    } else {
        resolve_llm_config(config)
    }
}

/// Resolve the LLM config for a given [`PurposeKind`] when the user has
/// configured `[purposes.<kind>]`. Returns `None` when no purpose is set
/// (callers fall back to the legacy resolvers — `resolve_embeddings_config`
/// for embedding, `resolve_backend_tasks_llm_config` for dreaming/titling).
///
/// Resolution flow:
/// 1. Look up `cfg.purposes.<kind>`. If absent, return `None`.
/// 2. Validate `[connections]`. If the map fails to validate, log + `None` —
///    the legacy resolver still produces something usable from `[llm]`.
/// 3. Run [`crate::purposes::resolve_purpose`] which handles `"primary"`
///    inheritance (connection and model both fall through to interactive)
///    and dangling-connection warnings.
/// 4. Build a [`ResolvedLlmConfig`] from the purpose's connection via
///    [`resolve_connection_llm_config`], then override the model with the
///    purpose's `model_id`. The connection's resolved `api_key` /
///    `base_url` / connector type are preserved as-is — the purpose layer
///    only chooses *which* connection + model, not credentials.
///
/// Effort threading is handled at the call site (see
/// `api_surface::RoutingConversationHandler::apply_effort_mapping` for the
/// interactive path; backend tasks call the same mapper directly). The
/// effort hint lives on `cfg.purposes.<kind>.effort` and can be read back
/// via `cfg.purposes.get(kind).effort`.
pub fn resolve_purpose_llm_config(
    config: Option<&DaemonConfig>,
    kind: PurposeKind,
) -> Option<ResolvedLlmConfig> {
    let cfg = config?;
    cfg.purposes.get(kind)?;

    let connections = match cfg.validated_connections() {
        Ok(map) => map,
        Err(err) => {
            tracing::warn!(
                purpose = kind.as_key(),
                error = %err,
                "cannot resolve purpose: [connections] failed validation; falling back to legacy resolver"
            );
            return None;
        }
    };

    let resolved = match crate::purposes::resolve_purpose(kind, &cfg.purposes, &connections) {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!(
                purpose = kind.as_key(),
                error = %err,
                "purpose resolution failed; falling back to legacy resolver"
            );
            return None;
        }
    };

    // The connection must exist after `resolve_purpose` — it returns the
    // interactive fallback id for dangling refs, and interactive itself is
    // checked by `expect_interactive_connection`. Map miss here would be a
    // logic bug in `resolve_purpose`, not a config issue.
    let conn = connections.get(&resolved.connection_id)?;
    let mut llm = resolve_connection_llm_config(conn, Some(&cfg.llm));
    llm.model = resolved.model_id;
    Some(llm)
}

/// Universal fallback for purpose-aware context-window resolution.
/// Used when no purpose override is set and the connector's curated
/// table reports nothing for the model. Most modern frontier models
/// meet or exceed this; under-stating is safe (we compact slightly
/// earlier than necessary), over-stating is not (the LLM rejects).
pub const DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS: u64 = 200_000;

/// Three-tier resolution for "what's the context window for this purpose?"
///
/// Resolution order:
///   1. The purpose's `max_context_tokens` override, if explicitly set —
///      the user always wins. Tagged [`BudgetSource::PurposeOverride`].
///   2. The connector's curated table for the configured model, surfaced
///      via `LlmClient::max_context_tokens()` (or any equivalent the
///      caller passes through `connector_max`). Tagged
///      [`BudgetSource::ConnectorTable`].
///   3. [`DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS`] — a conservative universal
///      fallback so token-based compaction stays on for non-curated
///      models instead of silently disabling. Tagged
///      [`BudgetSource::UniversalFallback`].
///
/// `purpose_override` carries tier 1; `connector_max` carries tier 2.
/// Both are optional so callers without a live value can pass `None` and
/// still get the fallback.
///
/// Why a typed [`ContextBudget`]: the previous `u64`-only signature lost
/// the tier provenance, so callers couldn't tell whether the value came
/// from user config, the connector, or the silent fallback. Surfacing
/// the source as a tag lets the dispatch wrapper log which tier won and
/// gives operators a clean signal for "this model's window is unknown,
/// we're guessing 200K".
pub fn resolve_context_budget(
    purpose_override: Option<u64>,
    connector_max: Option<u64>,
) -> ContextBudget {
    if let Some(value) = purpose_override {
        return ContextBudget {
            max_input_tokens: value,
            source: BudgetSource::PurposeOverride,
        };
    }
    if let Some(value) = connector_max {
        return ContextBudget {
            max_input_tokens: value,
            source: BudgetSource::ConnectorTable,
        };
    }
    ContextBudget {
        max_input_tokens: DEFAULT_PURPOSE_MAX_CONTEXT_TOKENS,
        source: BudgetSource::UniversalFallback,
    }
}

/// Convenience: pull `purposes.<kind>.max_context_tokens` from a
/// `DaemonConfig`. Returns `None` when no purpose is configured for `kind`
/// or the override is unset; in that case the caller should drop into
/// tier 2 / tier 3 of [`resolve_context_budget`].
pub fn purpose_max_context_override(
    config: Option<&DaemonConfig>,
    kind: PurposeKind,
) -> Option<u64> {
    config
        .and_then(|cfg| cfg.purposes.get(kind))
        .and_then(|p| p.max_context_tokens)
}

/// Shared resolution logic: takes an optional `LlmConfig` reference and
/// resolves connector, model, base_url, api_key with env-var fallbacks.
fn resolve_llm_config_from(llm_config: Option<&LlmConfig>) -> ResolvedLlmConfig {
    let connector = llm_config
        .map(|c| c.connector.trim().to_lowercase())
        .filter(|c| !c.is_empty())
        .unwrap_or_else(default_connector);

    let default_api_key_env = default_api_key_env(&connector);
    let default_model_env = default_model_env(&connector);
    let default_base_url_env = default_base_url_env(&connector);

    let api_key_env = llm_config
        .and_then(|c| c.api_key_env.as_deref())
        .unwrap_or(default_api_key_env.as_str());

    let mut api_key = llm_config
        .and_then(|c| c.secret.as_ref())
        .and_then(|secret| read_secret_from_backend(secret, &connector))
        .unwrap_or_default();

    if api_key.is_empty() {
        api_key = std::env::var(api_key_env).unwrap_or_default();
    }

    let model = llm_config
        .and_then(|c| c.model.clone())
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var(default_model_env).ok())
        .unwrap_or_else(|| default_llm_model(&connector));

    let base_url = llm_config
        .and_then(|c| c.base_url.clone())
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var(default_base_url_env).ok())
        .unwrap_or_else(|| {
            parse_connector_or_openai(&connector)
                .default_http_base_url()
                .to_string()
        });

    let temperature = llm_config.and_then(|c| c.temperature);
    let top_p = llm_config.and_then(|c| c.top_p);
    let max_tokens = llm_config.and_then(|c| c.max_tokens);
    let hosted_tool_search = llm_config.and_then(|c| c.hosted_tool_search);
    let aws_profile = llm_config.and_then(|c| c.aws_profile.clone());

    ResolvedLlmConfig {
        connector,
        model,
        base_url,
        api_key,
        temperature,
        top_p,
        max_tokens,
        hosted_tool_search,
        aws_profile,
    }
}

/// Resolve a per-connection [`ResolvedLlmConfig`] from a [`ConnectionConfig`].
///
/// Used by the connection registry (#9) to build one client per declared
/// connection. A [`ConnectionConfig`] holds only connector-identity fields
/// (endpoint, credentials, aws profile); it does not carry model, temperature,
/// hosted-tool-search, or `max_tokens` — those belong to purpose configs
/// (#10), which will supply overrides at dispatch time.
///
/// For now, this resolver fills the missing per-purpose fields from
/// `fallback_llm` (the top-level `[llm]` block) when present, then from
/// connector defaults / env vars. That keeps existing single-config installs
/// working until #10 lands.
pub fn resolve_connection_llm_config(
    connection: &ConnectionConfig,
    fallback_llm: Option<&LlmConfig>,
) -> ResolvedLlmConfig {
    let connector = connection.connector_type().to_string();
    let default_api_key_env = default_api_key_env(&connector);
    let default_model_env = default_model_env(&connector);
    let default_base_url_env = default_base_url_env(&connector);

    // Per-connector fields.
    let (conn_base_url, conn_api_key_env, conn_secret, conn_aws_profile): (
        Option<String>,
        Option<String>,
        Option<SecretConfig>,
        Option<String>,
    ) = match connection {
        ConnectionConfig::OpenAi(OpenAiConnection {
            base_url,
            api_key_env,
            secret,
        })
        | ConnectionConfig::Anthropic(AnthropicConnection {
            base_url,
            api_key_env,
            secret,
        }) => (base_url.clone(), api_key_env.clone(), secret.clone(), None),
        ConnectionConfig::Ollama(OllamaConnection { base_url }) => {
            (base_url.clone(), None, None, None)
        }
        ConnectionConfig::Bedrock(BedrockConnection {
            aws_profile,
            region,
            base_url,
        }) => {
            // Bedrock historically used `base_url` to encode the region when
            // no explicit URL was set. Preserve that shape: prefer `base_url`,
            // fall back to `region`.
            let effective_base = base_url
                .clone()
                .or_else(|| region.clone())
                .filter(|v| !v.trim().is_empty());
            (effective_base, None, None, aws_profile.clone())
        }
    };

    // API key: connection secret → connection env var → fallback env var.
    let api_key_env_name = conn_api_key_env
        .as_deref()
        .unwrap_or(default_api_key_env.as_str());
    let mut api_key = conn_secret
        .as_ref()
        .and_then(|secret| read_secret_from_backend(secret, &connector))
        .unwrap_or_default();
    if api_key.is_empty() {
        api_key = std::env::var(api_key_env_name).unwrap_or_default();
    }

    // Base URL resolution.
    let base_url = conn_base_url
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var(&default_base_url_env).ok())
        .unwrap_or_else(|| {
            parse_connector_or_openai(&connector)
                .default_http_base_url()
                .to_string()
        });

    // Model / tuning: not on the connection. Use the legacy `[llm]` block as
    // a placeholder until purpose configs (#10) provide per-request overrides.
    // If the fallback's connector differs from this connection's, its `model`
    // value is wrong for this connector, so we skip it.
    let fallback_model = fallback_llm
        .filter(|c| c.connector.trim().to_lowercase() == connector)
        .and_then(|c| c.model.clone())
        .filter(|v| !v.trim().is_empty());
    let model = fallback_model
        .or_else(|| std::env::var(&default_model_env).ok())
        .unwrap_or_else(|| default_llm_model(&connector));

    let (temperature, top_p, max_tokens, hosted_tool_search) = fallback_llm
        .filter(|c| c.connector.trim().to_lowercase() == connector)
        .map(|c| (c.temperature, c.top_p, c.max_tokens, c.hosted_tool_search))
        .unwrap_or((None, None, None, None));

    let aws_profile = conn_aws_profile.or_else(|| {
        fallback_llm
            .filter(|c| c.connector.trim().to_lowercase() == connector)
            .and_then(|c| c.aws_profile.clone())
    });

    ResolvedLlmConfig {
        connector,
        model,
        base_url,
        api_key,
        temperature,
        top_p,
        max_tokens,
        hosted_tool_search,
        aws_profile,
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

pub fn get_ws_auth_discovery(config_path: &Path) -> anyhow::Result<WsAuthDiscoveryInfo> {
    let config = load_daemon_config(config_path)?.unwrap_or_default();
    let ws_auth = config.ws_auth;

    let oidc_info = if ws_auth.methods.contains(&"oidc".to_string()) {
        ws_auth.oidc.as_ref().map(|oidc| OidcDiscoveryInfo {
            authorization_endpoint: oidc.authorization_endpoint.clone(),
            token_endpoint: oidc.token_endpoint.clone(),
            client_id: oidc.client_id.clone(),
            scopes: oidc.scopes.clone(),
        })
    } else {
        None
    };

    Ok(WsAuthDiscoveryInfo {
        methods: ws_auth.methods,
        oidc: oidc_info,
    })
}

pub fn get_ws_auth_settings(config_path: &Path) -> anyhow::Result<WsAuthConfig> {
    let config = load_daemon_config(config_path)?.unwrap_or_default();
    Ok(config.ws_auth)
}

pub fn set_ws_auth_settings(
    config_path: &Path,
    methods: &[String],
    oidc_issuer: &str,
    oidc_auth_endpoint: &str,
    oidc_token_endpoint: &str,
    oidc_client_id: &str,
    oidc_scopes: &str,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(config_path)?.unwrap_or_default();

    config.ws_auth.methods = methods.to_vec();

    if methods.contains(&"oidc".to_string()) && !oidc_issuer.is_empty() {
        let existing_oidc = config.ws_auth.oidc.unwrap_or_else(|| OidcConfig {
            issuer_url: String::new(),
            authorization_endpoint: String::new(),
            token_endpoint: String::new(),
            client_id: String::new(),
            scopes: default_oidc_scopes(),
            jwks_uri: String::new(),
            audience: String::new(),
        });
        config.ws_auth.oidc = Some(OidcConfig {
            issuer_url: oidc_issuer.to_string(),
            authorization_endpoint: oidc_auth_endpoint.to_string(),
            token_endpoint: oidc_token_endpoint.to_string(),
            client_id: oidc_client_id.to_string(),
            scopes: if oidc_scopes.is_empty() {
                existing_oidc.scopes
            } else {
                oidc_scopes.to_string()
            },
            jwks_uri: existing_oidc.jwks_uri,
            audience: existing_oidc.audience,
        });
    } else {
        config.ws_auth.oidc = None;
    }

    save_daemon_config(config_path, &config)
}

pub fn authenticate_os_user_password(username: &str, password: &str) -> anyhow::Result<bool> {
    #[cfg(target_os = "linux")]
    {
        pam_auth::authenticate(username, password)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (username, password);
        Err(anyhow!(
            "OS password authentication is only supported on Linux"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn ws_jwt_env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
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
        assert_eq!(claims_1.iss, jwt::default_ws_jwt_issuer());
        assert_eq!(claims_1.aud, jwt::default_ws_jwt_audience());

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

            [purposes.embedding]
            connection = "local"
            model = "mxbai-embed-large:335m"

            [purposes.titling]
            connection = "bedrock"
            model = "anthropic.claude-haiku-4-5"
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
        // Tier 1: an explicit `purpose.max_context_tokens` beats the
        // connector's curated table even when the curated value is known.
        // The user always wins. Source tag identifies user config so the
        // dispatch wrapper can log "user said 500K" rather than guessing.
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
}
