use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, anyhow};
use desktop_assistant_llm_anthropic::AnthropicClient;
use desktop_assistant_llm_bedrock::BedrockClient;
use desktop_assistant_llm_ollama::OllamaClient;
use desktop_assistant_llm_openai::OpenAiClient;
use keyring::Entry;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct DaemonConfig {
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub embeddings: EmbeddingsConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
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
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            connector: default_connector(),
            model: None,
            base_url: None,
            api_key_env: None,
            secret: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
}

#[derive(Debug, Clone)]
pub struct LlmSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
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
    pub has_api_key: bool,
    pub available: bool,
    pub is_default: bool,
}

#[derive(Debug, Clone)]
pub struct ConnectorDefaultsView {
    pub llm_model: String,
    pub llm_base_url: String,
    pub embeddings_model: String,
    pub embeddings_base_url: String,
    pub embeddings_available: bool,
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
    Ok(Some(parsed))
}

pub fn save_daemon_config(path: &Path, config: &DaemonConfig) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }

    let content = toml::to_string_pretty(config)?;
    std::fs::write(path, content)
        .with_context(|| format!("failed to write daemon config at {}", path.display()))
}

pub fn get_llm_settings_view(path: &Path) -> anyhow::Result<LlmSettingsView> {
    let config = load_daemon_config(path)?;
    let resolved = resolve_llm_config(config.as_ref());

    Ok(LlmSettingsView {
        connector: resolved.connector,
        model: resolved.model,
        base_url: resolved.base_url,
        has_api_key: !resolved.api_key.is_empty(),
    })
}

pub fn set_llm_settings(
    path: &Path,
    connector: &str,
    model: Option<&str>,
    base_url: Option<&str>,
) -> anyhow::Result<()> {
    let mut config = load_daemon_config(path)?.unwrap_or_default();

    let connector = connector.trim().to_lowercase();
    if connector.is_empty() {
        return Err(anyhow!("connector must not be empty"));
    }

    config.llm.connector = connector;
    config.llm.model = normalize_optional_value(model);
    config.llm.base_url = normalize_optional_value(base_url);

    save_daemon_config(path, &config)
}

pub fn set_api_key(path: &Path, api_key: &str) -> anyhow::Result<()> {
    let api_key = api_key.trim();
    let (key_len, key_fingerprint) = redacted_secret_audit(api_key);

    tracing::info!(
        secret_len = key_len,
        secret_fingerprint = %key_fingerprint,
        "received SetApiKey request"
    );

    if api_key.is_empty() {
        return Err(anyhow!("api key must not be empty"));
    }

    if is_placeholder_secret_value(api_key) {
        tracing::warn!(
            secret_len = key_len,
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

pub fn get_connector_defaults(connector: &str) -> ConnectorDefaultsView {
    let connector = connector.trim().to_lowercase();
    let connector = if connector.is_empty() {
        default_connector()
    } else {
        connector
    };

    let llm_model = default_llm_model(&connector);
    let llm_base_url = default_base_url(&connector);

    let embeddings_available = connector != "anthropic";
    let embeddings_connector = if embeddings_available {
        connector.as_str()
    } else {
        "openai"
    };

    ConnectorDefaultsView {
        llm_model,
        llm_base_url,
        embeddings_model: default_embedding_model(embeddings_connector),
        embeddings_base_url: default_base_url(embeddings_connector),
        embeddings_available,
    }
}

pub fn resolve_embeddings_config(config: Option<&DaemonConfig>) -> EmbeddingsSettingsView {
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
    let has_api_key = if is_default || connector == llm_connector {
        let resolved_llm = resolve_llm_config(config);
        !resolved_llm.api_key.is_empty()
    } else {
        let env_key = default_api_key_env(&connector);
        !std::env::var(env_key).unwrap_or_default().trim().is_empty()
    };

    EmbeddingsSettingsView {
        connector,
        model,
        base_url,
        has_api_key,
        available,
        is_default,
    }
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

fn default_embedding_model(connector: &str) -> String {
    match connector {
        "ollama" => "nomic-embed-text".to_string(),
        "bedrock" | "aws-bedrock" => "amazon.titan-embed-text-v2:0".to_string(),
        _ => "text-embedding-3-small".to_string(),
    }
}

fn default_base_url(connector: &str) -> String {
    match connector {
        "ollama" => OllamaClient::get_default_base_url(),
        "anthropic" => AnthropicClient::get_default_base_url(),
        "bedrock" | "aws-bedrock" => BedrockClient::get_default_base_url(),
        _ => OpenAiClient::get_default_base_url(),
    }
    .unwrap_or_default()
    .to_string()
}

fn default_llm_model(connector: &str) -> String {
    match connector {
        "ollama" => OllamaClient::get_default_model(),
        "anthropic" => AnthropicClient::get_default_model(),
        "bedrock" | "aws-bedrock" => BedrockClient::get_default_model(),
        _ => OpenAiClient::get_default_model(),
    }
    .unwrap_or_default()
    .to_string()
}

fn normalize_optional_value(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

pub fn resolve_llm_config(config: Option<&DaemonConfig>) -> ResolvedLlmConfig {
    let llm_config = config.map(|c| &c.llm);

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
        .unwrap_or_else(|| match connector.as_str() {
            "ollama" => "http://localhost:11434".to_string(),
            "anthropic" => "https://api.anthropic.com".to_string(),
            "bedrock" | "aws-bedrock" => "us-east-1".to_string(),
            _ => "https://api.openai.com/v1".to_string(),
        });

    ResolvedLlmConfig {
        connector,
        model,
        base_url,
        api_key,
    }
}

fn read_secret_from_backend(secret: &SecretConfig, connector: &str) -> Option<String> {
    match secret.backend.trim().to_lowercase().as_str() {
        "auto" => read_auto_secret(secret, connector),
        "systemd" | "systemd-credentials" => read_systemd_credential(secret, connector),
        "keyring" | "libsecret" => read_keyring_secret(secret, connector),
        "kwallet" => read_kwallet_secret(secret, connector),
        other => {
            tracing::warn!("unsupported secret backend '{}', falling back", other);
            None
        }
    }
}

fn read_auto_secret(secret: &SecretConfig, connector: &str) -> Option<String> {
    let account = resolve_secret_account(secret, connector);
    if let Some(value) = read_common_file_secret(&account) {
        return Some(value);
    }

    if let Some(value) = read_systemd_credential(secret, connector) {
        return Some(value);
    }

    if let Some(value) = read_keyring_secret(secret, connector) {
        return Some(value);
    }

    read_kwallet_secret(secret, connector)
}

fn read_common_file_secret(account: &str) -> Option<String> {
    let path = common_secret_file_path(account);
    let value = std::fs::read_to_string(path).ok()?;
    sanitize_secret_value(&value)
}

fn read_systemd_credential(secret: &SecretConfig, connector: &str) -> Option<String> {
    let credentials_dir = std::env::var_os("CREDENTIALS_DIRECTORY")?;
    let account = resolve_secret_account(secret, connector);
    let path = PathBuf::from(credentials_dir).join(account);

    let value = std::fs::read_to_string(path).ok()?;
    sanitize_secret_value(&value)
}

fn read_keyring_secret(secret: &SecretConfig, connector: &str) -> Option<String> {
    let service = secret
        .service
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(default_secret_service);
    let account = resolve_secret_account(secret, connector);

    if let Some(value) = read_secret_tool_secret(&service, &account) {
        return Some(value);
    }

    let entry = Entry::new(&service, &account).ok()?;
    let value = entry.get_password().ok()?;
    sanitize_secret_value(&value)
}

fn write_secret_to_backend(
    secret: &SecretConfig,
    value: &str,
    connector: &str,
) -> anyhow::Result<()> {
    match secret.backend.trim().to_lowercase().as_str() {
        "auto" => write_auto_secret(secret, value, connector),
        "systemd" | "systemd-credentials" => Err(anyhow!(
            "systemd credentials backend is read-only; configure credentials via systemd and use SetLlmSettings only"
        )),
        "keyring" | "libsecret" => write_keyring_secret(secret, value, connector),
        "kwallet" => write_kwallet_secret(secret, value, connector),
        other => Err(anyhow!("unsupported secret backend '{other}'")),
    }
}

fn write_auto_secret(secret: &SecretConfig, value: &str, connector: &str) -> anyhow::Result<()> {
    let account = resolve_secret_account(secret, connector);
    write_common_file_secret(&account, value)
}

fn write_common_file_secret(account: &str, value: &str) -> anyhow::Result<()> {
    let dir = default_secret_store_dir();
    std::fs::create_dir_all(&dir).map_err(|error| {
        anyhow!(
            "failed to create secret store directory {}: {error}",
            dir.display()
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }

    let path = common_secret_file_path(account);
    std::fs::write(&path, value)
        .map_err(|error| anyhow!("failed to write secret file {}: {error}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).map_err(
            |error| {
                anyhow!(
                    "failed to set secret file permissions {}: {error}",
                    path.display()
                )
            },
        )?;
    }

    Ok(())
}

fn write_keyring_secret(secret: &SecretConfig, value: &str, connector: &str) -> anyhow::Result<()> {
    let service = secret
        .service
        .clone()
        .filter(|candidate| !candidate.trim().is_empty())
        .unwrap_or_else(default_secret_service);
    let account = resolve_secret_account(secret, connector);

    if command_exists("secret-tool") {
        write_secret_tool_secret(&service, &account, value)?;
        return Ok(());
    }

    let entry = Entry::new(&service, &account)
        .map_err(|error| anyhow!("failed to initialize keyring entry: {error}"))?;
    entry
        .set_password(value)
        .map_err(|error| anyhow!("failed to write keyring secret: {error}"))
}

fn command_exists(command: &str) -> bool {
    Command::new(command)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn read_secret_tool_secret(service: &str, account: &str) -> Option<String> {
    let output = Command::new("secret-tool")
        .arg("lookup")
        .arg("service")
        .arg(service)
        .arg("account")
        .arg(account)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8_lossy(&output.stdout);
    sanitize_secret_value(value.as_ref())
}

fn sanitize_secret_value(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    if is_placeholder_secret_value(trimmed) {
        tracing::warn!("ignoring placeholder-like secret value from backend");
        return None;
    }

    Some(trimmed.to_string())
}

fn is_placeholder_secret_value(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();

    value.contains('*')
        || normalized.starts_with("file-store")
        || normalized.starts_with("secret-store")
        || normalized.contains("write-only")
        || normalized.contains("leave blank")
}

fn redacted_secret_audit(value: &str) -> (usize, String) {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

    let trimmed = value.trim();
    let mut hash = FNV_OFFSET_BASIS;
    for byte in trimmed.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    (trimmed.len(), format!("fnv1a64:{hash:016x}"))
}

fn write_secret_tool_secret(service: &str, account: &str, value: &str) -> anyhow::Result<()> {
    let mut child = Command::new("secret-tool")
        .arg("store")
        .arg("--label")
        .arg("Desktop Assistant API Key")
        .arg("service")
        .arg(service)
        .arg("account")
        .arg(account)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| anyhow!("failed to invoke secret-tool: {error}"))?;

    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write as _;
        stdin
            .write_all(value.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .map_err(|error| anyhow!("failed to write secret-tool stdin: {error}"))?;
    } else {
        return Err(anyhow!("failed to open secret-tool stdin"));
    }

    let output = child
        .wait_with_output()
        .map_err(|error| anyhow!("failed waiting for secret-tool: {error}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            "secret-tool returned non-zero exit status".to_string()
        };
        Err(anyhow!("failed to write secret-tool secret: {detail}"))
    }
}

fn write_kwallet_secret(secret: &SecretConfig, value: &str, connector: &str) -> anyhow::Result<()> {
    let entry = resolve_wallet_entry(secret, connector);
    let attempts = [
        vec![
            "-f".to_string(),
            secret.folder.clone(),
            "-w".to_string(),
            value.to_string(),
            entry.clone(),
            secret.wallet.clone(),
        ],
        vec![
            "-f".to_string(),
            secret.folder.clone(),
            "-e".to_string(),
            entry,
            "-w".to_string(),
            value.to_string(),
            secret.wallet.clone(),
        ],
    ];

    let mut last_error = String::from("unknown kwallet error");
    for args in attempts {
        let output = Command::new("kwallet-query").args(args).output();

        match output {
            Ok(result) if result.status.success() => return Ok(()),
            Ok(result) => {
                let stderr = String::from_utf8_lossy(&result.stderr).trim().to_string();
                let stdout = String::from_utf8_lossy(&result.stdout).trim().to_string();
                last_error = if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    "kwallet-query returned non-zero exit status".to_string()
                };
            }
            Err(error) => {
                last_error = error.to_string();
            }
        }
    }

    Err(anyhow!("failed to write KWallet secret: {last_error}"))
}

fn read_kwallet_secret(secret: &SecretConfig, connector: &str) -> Option<String> {
    let entry = resolve_wallet_entry(secret, connector);
    let output = Command::new("kwallet-query")
        .arg("-f")
        .arg(&secret.folder)
        .arg("-r")
        .arg(&entry)
        .arg(&secret.wallet)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8_lossy(&output.stdout);
    sanitize_secret_value(value.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            model = "gpt-5.2"
            "#,
        )
        .unwrap();

        assert_eq!(parsed.llm.connector, "openai");
        assert_eq!(parsed.llm.model.as_deref(), Some("gpt-5.2"));
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
        assert_eq!(sanitize_secret_value("  \n\t "), None);
        assert_eq!(sanitize_secret_value("file-store-openai-key"), None);
        assert_eq!(
            sanitize_secret_value("  sk-live-abc123  "),
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
        assert_eq!(resolved.model, "anthropic.claude-3-5-sonnet-20241022-v2:0");
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
        assert_eq!(defaults.llm_model, "gpt-5.2");
        assert_eq!(defaults.llm_base_url, "https://api.openai.com/v1");
        assert_eq!(defaults.embeddings_model, "text-embedding-3-small");
        assert_eq!(defaults.embeddings_base_url, "https://api.openai.com/v1");
        assert!(defaults.embeddings_available);
    }

    #[test]
    fn connector_defaults_anthropic_embeddings_fallback_to_openai() {
        let defaults = get_connector_defaults("anthropic");
        assert_eq!(defaults.llm_model, "claude-sonnet-4-5-20250929");
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
            model = "claude-sonnet-4-5-20250929"

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
}
