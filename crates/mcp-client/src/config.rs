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

    // Open with 0600 *before* writing — `std::fs::write` followed by chmod
    // leaves a window where the file (which carries env_secrets references) is
    // world-readable.
    {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| {
                McpError::UnexpectedResponse(format!("failed to open MCP config file: {e}"))
            })?;
        file.write_all(contents.as_bytes()).map_err(|e| {
            McpError::UnexpectedResponse(format!("failed to write MCP config file: {e}"))
        })?;
    }

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
    config_dir.join("desktop-assistant").join("secrets.toml")
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

    let contents = std::fs::read_to_string(path)
        .map_err(|e| McpError::UnexpectedResponse(format!("failed to read secrets file: {e}")))?;

    let config: SecretsConfig = toml::from_str(&contents)
        .map_err(|e| McpError::UnexpectedResponse(format!("failed to parse secrets file: {e}")))?;

    tracing::info!(
        "loaded {} secret(s) from {}",
        config.secrets.len(),
        path.display()
    );
    Ok(config.secrets)
}

/// Save secrets to a TOML file, owner-only (0600). Used by the interactive
/// OAuth login to persist a freshly minted refresh token. Any prior comments
/// or formatting in the file are not preserved.
pub fn save_secrets(
    path: &std::path::Path,
    secrets: &HashMap<String, String>,
) -> Result<(), McpError> {
    let config = SecretsConfig {
        secrets: secrets.clone(),
    };
    let contents = toml::to_string_pretty(&config)
        .map_err(|e| McpError::UnexpectedResponse(format!("failed to serialize secrets: {e}")))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            McpError::UnexpectedResponse(format!("failed to create secrets directory: {e}"))
        })?;
    }

    // Open 0600 before writing so the secrets never touch a world-readable file.
    {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| {
                McpError::UnexpectedResponse(format!("failed to open secrets file: {e}"))
            })?;
        file.write_all(contents.as_bytes()).map_err(|e| {
            McpError::UnexpectedResponse(format!("failed to write secrets file: {e}"))
        })?;
    }

    enforce_permissions(path)?;
    tracing::info!("saved {} secret(s) to {}", secrets.len(), path.display());
    Ok(())
}

/// Set one secret value in `secrets.toml`, preserving every other entry
/// (read-modify-write), and return the resulting map. Used by the settings-UI
/// `set_mcp_secret` so the user can store a bearer token / OAuth client secret
/// without hand-editing files.
///
/// Fails closed on a *parse* error of an existing file rather than clobbering
/// it — silently overwriting with just the new key would drop every other
/// secret. A missing file is fine (starts from empty).
pub fn upsert_secret(
    path: &std::path::Path,
    id: &str,
    value: &str,
) -> Result<HashMap<String, String>, McpError> {
    let mut secrets = if path.exists() {
        load_secrets(path).map_err(|e| {
            McpError::UnexpectedResponse(format!(
                "refusing to overwrite secrets: cannot read existing {}: {e}",
                path.display()
            ))
        })?
    } else {
        HashMap::new()
    };
    secrets.insert(id.to_string(), value.to_string());
    save_secrets(path, &secrets)?;
    Ok(secrets)
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
        assert!(
            config.servers[0].env.is_empty(),
            "env should default to empty"
        );
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
            config.servers[0]
                .env
                .get("GITHUB_PERSONAL_ACCESS_TOKEN")
                .unwrap(),
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
                http: None,
            },
            McpServerConfig {
                name: "jira".into(),
                command: "jira-mcp".into(),
                args: vec!["--host".into(), "jira.example.com".into()],
                namespace: Some("jira".into()),
                enabled: false,
                env: std::collections::HashMap::new(),
                env_secrets: std::collections::HashMap::new(),
                http: None,
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
            config.servers[0]
                .env_secrets
                .get("GITHUB_PERSONAL_ACCESS_TOKEN")
                .unwrap(),
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
        assert_eq!(
            config.servers[0].env.get("SOME_PUBLIC_VAR").unwrap(),
            "public-value"
        );
        assert_eq!(
            config.servers[0].env_secrets.get("SECRET_VAR").unwrap(),
            "my_secret_id"
        );
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
    fn save_secrets_roundtrip_and_upsert() {
        let dir = std::env::temp_dir().join("mcp_secrets_roundtrip_test");
        let path = dir.join("secrets.toml");
        let _ = std::fs::remove_dir_all(&dir);

        let mut secrets = HashMap::new();
        secrets.insert("existing_key".to_string(), "existing".to_string());
        save_secrets(&path, &secrets).unwrap();

        // Reload, add a new secret (as the OAuth login does), save again.
        let mut loaded = load_secrets(&path).unwrap();
        assert_eq!(loaded.get("existing_key").unwrap(), "existing");
        loaded.insert("gmail_refresh".to_string(), "rt-value".to_string());
        save_secrets(&path, &loaded).unwrap();

        let reloaded = load_secrets(&path).unwrap();
        assert_eq!(reloaded.get("existing_key").unwrap(), "existing");
        assert_eq!(reloaded.get("gmail_refresh").unwrap(), "rt-value");

        // File must be owner-only (it holds secrets).
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "secrets file must be 0600");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn upsert_secret_preserves_other_entries_and_starts_from_missing() {
        let dir = std::env::temp_dir().join("mcp_upsert_secret_test");
        let path = dir.join("secrets.toml");
        let _ = std::fs::remove_dir_all(&dir);

        // Missing file: starts from empty, writes the one key at 0600.
        let map = upsert_secret(&path, "gmail_work_token", "tok-1").unwrap();
        assert_eq!(map.get("gmail_work_token").unwrap(), "tok-1");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);

        // Second upsert must NOT clobber the first (read-modify-write).
        let map = upsert_secret(&path, "cal_refresh", "rt-2").unwrap();
        assert_eq!(map.get("gmail_work_token").unwrap(), "tok-1");
        assert_eq!(map.get("cal_refresh").unwrap(), "rt-2");
        let on_disk = load_secrets(&path).unwrap();
        assert_eq!(on_disk.len(), 2);

        // Overwriting an existing id updates just that value.
        let map = upsert_secret(&path, "gmail_work_token", "tok-3").unwrap();
        assert_eq!(map.get("gmail_work_token").unwrap(), "tok-3");
        assert_eq!(map.get("cal_refresh").unwrap(), "rt-2");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_secrets_path_is_reasonable() {
        let path = default_secrets_path();
        assert!(path.to_str().unwrap().contains("secrets.toml"));
        assert!(path.to_str().unwrap().contains("desktop-assistant"));
    }

    #[test]
    fn parse_http_transport_server() {
        // A remote (streamable-HTTP) server: no `command`, an `[servers.http]`
        // table selects the HTTP transport. Mirrors pointing Adele at Google's
        // hosted Gmail endpoint for one account.
        let toml = r#"
[[servers]]
name = "gmail-personal"
namespace = "gmail_personal"

[servers.http]
url = "https://gmailmcp.googleapis.com/mcp/v1"
auth_bearer_secret = "google_personal_token"
"#;
        let config: McpConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.servers.len(), 1);
        let server = &config.servers[0];
        assert_eq!(server.name, "gmail-personal");
        assert!(server.command.is_empty(), "http server needs no command");
        assert_eq!(server.namespace.as_deref(), Some("gmail_personal"));
        let http = server.http.as_ref().expect("http transport table");
        assert_eq!(http.url, "https://gmailmcp.googleapis.com/mcp/v1");
        assert_eq!(
            http.auth_bearer_secret.as_deref(),
            Some("google_personal_token")
        );
    }

    #[test]
    fn parse_http_transport_with_oauth() {
        // A remote server authenticating via OAuth: no static bearer, an
        // `[servers.http.oauth]` table naming secret *references* (not values)
        // plus the non-secret client id / URLs / scopes.
        let toml = r#"
[[servers]]
name = "gmail-work"
namespace = "gmail_work"

[servers.http]
url = "https://gmailmcp.googleapis.com/mcp/v1"

[servers.http.oauth]
client_id = "1234.apps.googleusercontent.com"
token_url = "https://oauth2.googleapis.com/token"
authorize_url = "https://accounts.google.com/o/oauth2/v2/auth"
refresh_token_ref = "gmail_work_refresh"
client_secret_ref = "google_client_secret"
account = "dave@example.com"
scopes = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
]
"#;
        let config: McpConfig = toml::from_str(toml).unwrap();
        let server = &config.servers[0];
        assert!(
            server.command.is_empty(),
            "oauth http server needs no command"
        );
        let http = server.http.as_ref().expect("http table");
        assert!(
            http.auth_bearer_secret.is_none(),
            "oauth server has no static bearer"
        );
        let oauth = http.oauth.as_ref().expect("oauth table");
        assert_eq!(oauth.client_id, "1234.apps.googleusercontent.com");
        assert_eq!(oauth.token_url, "https://oauth2.googleapis.com/token");
        assert_eq!(oauth.refresh_token_ref, "gmail_work_refresh");
        assert_eq!(
            oauth.client_secret_ref.as_deref(),
            Some("google_client_secret")
        );
        assert_eq!(
            oauth.authorize_url.as_deref(),
            Some("https://accounts.google.com/o/oauth2/v2/auth")
        );
        assert_eq!(oauth.account.as_deref(), Some("dave@example.com"));
        assert_eq!(oauth.scopes.len(), 2);
        // Optional numeric knob defaults to absent (⇒ 60s skew at build time).
        assert!(oauth.refresh_skew_seconds.is_none());
    }

    #[test]
    fn oauth_config_survives_save_load_roundtrip() {
        use crate::executor::{HttpTransportConfig, OAuthServerConfig};

        let dir = std::env::temp_dir().join("mcp_config_oauth_roundtrip_test");
        let path = dir.join("mcp_servers.toml");
        let _ = std::fs::remove_dir_all(&dir);

        let configs = vec![McpServerConfig {
            name: "calendar".into(),
            command: String::new(),
            args: vec![],
            namespace: Some("calendar".into()),
            enabled: true,
            env: std::collections::HashMap::new(),
            env_secrets: std::collections::HashMap::new(),
            http: Some(HttpTransportConfig {
                url: "https://calendarmcp.googleapis.com/mcp/v1".into(),
                auth_bearer_secret: None,
                oauth: Some(OAuthServerConfig {
                    client_id: "cid".into(),
                    token_url: "https://oauth2.googleapis.com/token".into(),
                    refresh_token_ref: "cal_refresh".into(),
                    client_secret_ref: None,
                    authorize_url: Some("https://accounts.google.com/o/oauth2/v2/auth".into()),
                    scopes: vec!["https://www.googleapis.com/auth/calendar".into()],
                    account: Some("dave@example.com".into()),
                    refresh_skew_seconds: Some(120),
                }),
            }),
        }];

        save_mcp_configs(&path, &configs).unwrap();
        let loaded = load_mcp_configs(&path).unwrap();
        let oauth = loaded[0]
            .http
            .as_ref()
            .and_then(|h| h.oauth.as_ref())
            .expect("oauth survives roundtrip");
        assert_eq!(oauth.client_id, "cid");
        assert_eq!(oauth.refresh_token_ref, "cal_refresh");
        assert!(oauth.client_secret_ref.is_none());
        assert_eq!(oauth.refresh_skew_seconds, Some(120));
        assert_eq!(oauth.scopes.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stdio_and_http_servers_roundtrip() {
        // A stdio server and an HTTP server survive a save/load cycle with their
        // transport intact.
        let dir = std::env::temp_dir().join("mcp_config_http_roundtrip_test");
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
                http: None,
            },
            McpServerConfig {
                name: "calendar-work".into(),
                command: String::new(),
                args: vec![],
                namespace: Some("calendar_work".into()),
                enabled: true,
                env: std::collections::HashMap::new(),
                env_secrets: std::collections::HashMap::new(),
                http: Some(crate::executor::HttpTransportConfig {
                    url: "https://calendarmcp.googleapis.com/mcp/v1".into(),
                    auth_bearer_secret: Some("google_work_token".into()),
                    oauth: None,
                }),
            },
        ];

        save_mcp_configs(&path, &configs).unwrap();
        let loaded = load_mcp_configs(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "fileio");
        assert!(loaded[0].http.is_none());
        assert_eq!(loaded[0].command, "fileio-mcp");

        assert_eq!(loaded[1].name, "calendar-work");
        assert!(loaded[1].command.is_empty());
        let http = loaded[1].http.as_ref().expect("http survives roundtrip");
        assert_eq!(http.url, "https://calendarmcp.googleapis.com/mcp/v1");
        assert_eq!(
            http.auth_bearer_secret.as_deref(),
            Some("google_work_token")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
