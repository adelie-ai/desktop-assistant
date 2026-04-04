use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use crate::McpError;
use crate::executor::McpServerConfig;

/// Top-level MCP configuration file structure.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct McpConfig {
    #[serde(default)]
    servers: Vec<McpServerConfig>,
}

/// Returns the default path for the MCP servers config file.
/// Uses `$XDG_CONFIG_HOME/desktop-assistant/mcp_servers.toml`,
/// falling back to `~/.config/desktop-assistant/mcp_servers.toml`.
pub fn default_config_path() -> PathBuf {
    let config_dir = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".config")
        });
    config_dir
        .join("desktop-assistant")
        .join("mcp_servers.toml")
}

/// Ensure the config file is owner-only (0600) since it may contain secrets.
fn enforce_permissions(path: &std::path::Path) -> Result<(), McpError> {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|e| {
        McpError::UnexpectedResponse(format!("failed to set config file permissions: {e}"))
    })
}

/// Load MCP server configurations from a TOML file.
/// Returns an empty vec if the file doesn't exist.
pub fn load_mcp_configs(path: &std::path::Path) -> Result<Vec<McpServerConfig>, McpError> {
    if !path.exists() {
        tracing::debug!(
            "MCP config file not found at {}, no servers configured",
            path.display()
        );
        return Ok(Vec::new());
    }

    enforce_permissions(path)?;

    let contents = std::fs::read_to_string(path).map_err(|e| {
        McpError::UnexpectedResponse(format!("failed to read MCP config file: {e}"))
    })?;

    let config: McpConfig = toml::from_str(&contents).map_err(|e| {
        McpError::UnexpectedResponse(format!("failed to parse MCP config file: {e}"))
    })?;

    tracing::info!(
        "loaded {} MCP server config(s) from {}",
        config.servers.len(),
        path.display()
    );
    Ok(config.servers)
}

/// Save MCP server configurations to a TOML file.
pub fn save_mcp_configs(
    path: &std::path::Path,
    configs: &[McpServerConfig],
) -> Result<(), McpError> {
    let config = McpConfig {
        servers: configs.to_vec(),
    };

    let contents = toml::to_string_pretty(&config).map_err(|e| {
        McpError::UnexpectedResponse(format!("failed to serialize MCP config: {e}"))
    })?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            McpError::UnexpectedResponse(format!("failed to create config directory: {e}"))
        })?;
    }

    std::fs::write(path, contents).map_err(|e| {
        McpError::UnexpectedResponse(format!("failed to write MCP config file: {e}"))
    })?;

    enforce_permissions(path)?;

    tracing::info!(
        "saved {} MCP server config(s) to {}",
        configs.len(),
        path.display()
    );
    Ok(())
}

/// Top-level secrets file structure.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SecretsConfig {
    #[serde(default)]
    secrets: HashMap<String, String>,
}

/// Returns the default path for the secrets file.
/// Uses `$XDG_CONFIG_HOME/desktop-assistant/secrets.toml`,
/// falling back to `~/.config/desktop-assistant/secrets.toml`.
pub fn default_secrets_path() -> PathBuf {
    let config_dir = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".config")
        });
    config_dir
        .join("desktop-assistant")
        .join("secrets.toml")
}

/// Load secrets from a TOML file.
/// Returns an empty map if the file doesn't exist.
pub fn load_secrets(path: &std::path::Path) -> Result<HashMap<String, String>, McpError> {
    if !path.exists() {
        tracing::debug!(
            "secrets file not found at {}, no secrets loaded",
            path.display()
        );
        return Ok(HashMap::new());
    }

    enforce_permissions(path)?;

    let contents = std::fs::read_to_string(path).map_err(|e| {
        McpError::UnexpectedResponse(format!("failed to read secrets file: {e}"))
    })?;

    let config: SecretsConfig = toml::from_str(&contents).map_err(|e| {
        McpError::UnexpectedResponse(format!("failed to parse secrets file: {e}"))
    })?;

    tracing::info!(
        "loaded {} secret(s) from {}",
        config.secrets.len(),
        path.display()
    );
    Ok(config.secrets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mcp_config_toml() {
        let toml = r#"
[[servers]]
name = "fileio"
command = "fileio-mcp"

[[servers]]
name = "genmcp"
command = "genmcp"
args = ["--config", "/path/to/config.toml"]
"#;
        let config: McpConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.servers.len(), 2);
        assert_eq!(config.servers[0].name, "fileio");
        assert_eq!(config.servers[0].command, "fileio-mcp");
        assert!(config.servers[0].args.is_empty());
        assert!(config.servers[0].env.is_empty(), "env should default to empty");
        assert_eq!(config.servers[1].name, "genmcp");
        assert_eq!(config.servers[1].args.len(), 2);
    }

    #[test]
    fn parse_mcp_config_with_env() {
        let toml = r#"
[[servers]]
name = "github"
command = "github-mcp-server"
args = ["stdio"]

[servers.env]
GITHUB_PERSONAL_ACCESS_TOKEN = "my-token"
OTHER_VAR = "value"
"#;
        let config: McpConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "github");
        assert_eq!(config.servers[0].env.len(), 2);
        assert_eq!(
            config.servers[0].env.get("GITHUB_PERSONAL_ACCESS_TOKEN").unwrap(),
            "my-token"
        );
        assert_eq!(config.servers[0].env.get("OTHER_VAR").unwrap(), "value");
    }

    #[test]
    fn parse_empty_config() {
        let toml = "";
        let config: McpConfig = toml::from_str(toml).unwrap();
        assert!(config.servers.is_empty());
    }

    #[test]
    fn load_nonexistent_file_returns_empty() {
        let result = load_mcp_configs(std::path::Path::new("/nonexistent/path.toml")).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn default_config_path_is_reasonable() {
        let path = default_config_path();
        assert!(path.to_str().unwrap().contains("mcp_servers.toml"));
        assert!(path.to_str().unwrap().contains("desktop-assistant"));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join("mcp_config_roundtrip_test");
        let path = dir.join("mcp_servers.toml");
        let _ = std::fs::remove_dir_all(&dir);

        let configs = vec![
            McpServerConfig {
                name: "fileio".into(),
                command: "fileio-mcp".into(),
                args: vec![],
                namespace: None,
                enabled: true,
                env: std::collections::HashMap::new(),
                env_secrets: std::collections::HashMap::new(),
            },
            McpServerConfig {
                name: "jira".into(),
                command: "jira-mcp".into(),
                args: vec!["--host".into(), "jira.example.com".into()],
                namespace: Some("jira".into()),
                enabled: false,
                env: std::collections::HashMap::new(),
                env_secrets: std::collections::HashMap::new(),
            },
        ];

        save_mcp_configs(&path, &configs).unwrap();
        let loaded = load_mcp_configs(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "fileio");
        assert!(loaded[0].enabled);
        assert_eq!(loaded[1].name, "jira");
        assert!(!loaded[1].enabled);
        assert_eq!(loaded[1].namespace.as_deref(), Some("jira"));
        assert_eq!(loaded[1].args.len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_mcp_config_with_env_secrets() {
        let toml = r#"
[[servers]]
name = "github"
command = "github-mcp-server"
args = ["stdio"]

[servers.env_secrets]
GITHUB_PERSONAL_ACCESS_TOKEN = "github_pat"
"#;
        let config: McpConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.servers.len(), 1);
        assert_eq!(
            config.servers[0].env_secrets.get("GITHUB_PERSONAL_ACCESS_TOKEN").unwrap(),
            "github_pat"
        );
        assert!(config.servers[0].env.is_empty());
    }

    #[test]
    fn parse_mcp_config_with_both_env_and_env_secrets() {
        let toml = r#"
[[servers]]
name = "github"
command = "github-mcp-server"
args = ["stdio"]

[servers.env]
SOME_PUBLIC_VAR = "public-value"

[servers.env_secrets]
SECRET_VAR = "my_secret_id"
"#;
        let config: McpConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.servers[0].env.get("SOME_PUBLIC_VAR").unwrap(), "public-value");
        assert_eq!(config.servers[0].env_secrets.get("SECRET_VAR").unwrap(), "my_secret_id");
    }

    #[test]
    fn parse_secrets_toml() {
        let toml = r#"
[secrets]
github_pat = "ghp_abc123"
other_key = "secret-value"
"#;
        let config: SecretsConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.secrets.len(), 2);
        assert_eq!(config.secrets.get("github_pat").unwrap(), "ghp_abc123");
        assert_eq!(config.secrets.get("other_key").unwrap(), "secret-value");
    }

    #[test]
    fn parse_empty_secrets_toml() {
        let toml = "";
        let config: SecretsConfig = toml::from_str(toml).unwrap();
        assert!(config.secrets.is_empty());
    }

    #[test]
    fn load_nonexistent_secrets_returns_empty() {
        let result = load_secrets(std::path::Path::new("/nonexistent/secrets.toml")).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn default_secrets_path_is_reasonable() {
        let path = default_secrets_path();
        assert!(path.to_str().unwrap().contains("secrets.toml"));
        assert!(path.to_str().unwrap().contains("desktop-assistant"));
    }
}
