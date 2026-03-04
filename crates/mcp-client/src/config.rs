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

    tracing::info!(
        "saved {} MCP server config(s) to {}",
        configs.len(),
        path.display()
    );
    Ok(())
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
        assert_eq!(config.servers[1].name, "genmcp");
        assert_eq!(config.servers[1].args.len(), 2);
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
            },
            McpServerConfig {
                name: "jira".into(),
                command: "jira-mcp".into(),
                args: vec!["--host".into(), "jira.example.com".into()],
                namespace: Some("jira".into()),
                enabled: false,
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
}
