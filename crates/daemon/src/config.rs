use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, anyhow};
use keyring::Entry;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct DaemonConfig {
    #[serde(default)]
    pub llm: LlmConfig,
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
    #[serde(default = "default_wallet_entry")]
    pub entry: String,
}

impl Default for SecretConfig {
    fn default() -> Self {
        Self {
            backend: default_secret_backend(),
            service: Some(default_secret_service()),
            account: Some(default_secret_account()),
            wallet: default_wallet_name(),
            folder: default_wallet_folder(),
            entry: default_wallet_entry(),
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

fn default_connector() -> String {
    "openai".to_string()
}

fn default_secret_backend() -> String {
    "keyring".to_string()
}

fn default_secret_service() -> String {
    "org.desktopAssistant".to_string()
}

fn default_secret_account() -> String {
    "openai_api_key".to_string()
}

fn default_wallet_name() -> String {
    "kdewallet".to_string()
}

fn default_wallet_folder() -> String {
    "desktop-assistant".to_string()
}

fn default_wallet_entry() -> String {
    "openai_api_key".to_string()
}

pub fn default_daemon_config_path() -> PathBuf {
    let config_home = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string())).join(".config")
        });

    config_home.join("desktop-assistant").join("daemon.toml")
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
    if api_key.is_empty() {
        return Err(anyhow!("api key must not be empty"));
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

    write_secret_to_backend(&secret, api_key)?;
    save_daemon_config(path, &config)
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

    let default_api_key_env = match connector.as_str() {
        "anthropic" => "ANTHROPIC_API_KEY",
        _ => "OPENAI_API_KEY",
    };

    let api_key_env = llm_config
        .and_then(|c| c.api_key_env.as_deref())
        .unwrap_or(default_api_key_env);

    let mut api_key = llm_config
        .and_then(|c| c.secret.as_ref())
        .and_then(read_secret_from_backend)
        .unwrap_or_default();

    if api_key.is_empty() {
        api_key = std::env::var(api_key_env).unwrap_or_default();
    }

    let model = llm_config
        .and_then(|c| c.model.clone())
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var("OPENAI_MODEL").ok())
        .unwrap_or_else(|| match connector.as_str() {
            "ollama" => "llama3.2".to_string(),
            "anthropic" => "claude-sonnet-4-5-20250929".to_string(),
            _ => "gpt-4o".to_string(),
        });

    let base_url = llm_config
        .and_then(|c| c.base_url.clone())
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| match connector.as_str() {
            "ollama" => "http://localhost:11434".to_string(),
            "anthropic" => "https://api.anthropic.com".to_string(),
            _ => "https://api.openai.com/v1".to_string(),
        });

    ResolvedLlmConfig {
        connector,
        model,
        base_url,
        api_key,
    }
}

fn read_secret_from_backend(secret: &SecretConfig) -> Option<String> {
    match secret.backend.trim().to_lowercase().as_str() {
        "keyring" | "libsecret" => read_keyring_secret(secret),
        "kwallet" => read_kwallet_secret(secret),
        other => {
            tracing::warn!("unsupported secret backend '{}', falling back", other);
            None
        }
    }
}

fn read_keyring_secret(secret: &SecretConfig) -> Option<String> {
    let service = secret
        .service
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(default_secret_service);
    let account = secret
        .account
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(default_secret_account);

    let entry = Entry::new(&service, &account).ok()?;
    let value = entry.get_password().ok()?.trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

fn write_secret_to_backend(secret: &SecretConfig, value: &str) -> anyhow::Result<()> {
    match secret.backend.trim().to_lowercase().as_str() {
        "keyring" | "libsecret" => write_keyring_secret(secret, value),
        "kwallet" => write_kwallet_secret(secret, value),
        other => Err(anyhow!("unsupported secret backend '{other}'")),
    }
}

fn write_keyring_secret(secret: &SecretConfig, value: &str) -> anyhow::Result<()> {
    let service = secret
        .service
        .clone()
        .filter(|candidate| !candidate.trim().is_empty())
        .unwrap_or_else(default_secret_service);
    let account = secret
        .account
        .clone()
        .filter(|candidate| !candidate.trim().is_empty())
        .unwrap_or_else(default_secret_account);

    let entry = Entry::new(&service, &account)
        .map_err(|error| anyhow!("failed to initialize keyring entry: {error}"))?;
    entry
        .set_password(value)
        .map_err(|error| anyhow!("failed to write keyring secret: {error}"))
}

fn write_kwallet_secret(secret: &SecretConfig, value: &str) -> anyhow::Result<()> {
    let attempts = [
        vec![
            "-f".to_string(),
            secret.folder.clone(),
            "-w".to_string(),
            value.to_string(),
            secret.entry.clone(),
            secret.wallet.clone(),
        ],
        vec![
            "-f".to_string(),
            secret.folder.clone(),
            "-e".to_string(),
            secret.entry.clone(),
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

fn read_kwallet_secret(secret: &SecretConfig) -> Option<String> {
    let output = Command::new("kwallet-query")
        .arg("-f")
        .arg(&secret.folder)
        .arg("-r")
        .arg(&secret.entry)
        .arg(&secret.wallet)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() { None } else { Some(value) }
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
            model = "gpt-4o-mini"
            "#,
        )
        .unwrap();

        assert_eq!(parsed.llm.connector, "openai");
        assert_eq!(parsed.llm.model.as_deref(), Some("gpt-4o-mini"));
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
    fn resolve_defaults_without_config() {
        let resolved = resolve_llm_config(None);
        assert_eq!(resolved.connector, "openai");
        assert!(!resolved.model.is_empty());
        assert!(!resolved.base_url.is_empty());
    }
}
