//! Client-side MCP host configuration (`client-mcp.toml`).
//!
//! Mirrors the daemon's server-side `mcp_servers.toml` schema for the server
//! *definitions* (reusing [`McpServerConfig`]), and adds a per-surface enable
//! layer so each client surface (tui/gtk/voice/kde) exposes its own subset of
//! the local MCP servers to the (possibly remote) daemon as client-side tools.
//!
//! **Central per machine.** This file lives at `~/.config/adele/client-mcp.toml`
//! and every Adele client on the box reads the same file, selecting its set via
//! `[surfaces.<name>]`. That is deliberate: which local tools exist is a
//! property of the *machine* (the edge), while which surface exposes them is a
//! per-client choice.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// The server-definition schema is shared verbatim with the daemon so a local
/// server is described identically wherever it runs.
pub use desktop_assistant_mcp_client::executor::McpServerConfig;

/// Name of the fallback surface. Its `enabled` list applies to any surface that
/// has no `[surfaces.<name>]` entry of its own.
pub const DEFAULT_SURFACE: &str = "default";

/// Per-surface tool selection: which defined servers this surface exposes.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SurfaceConfig {
    /// Server names (from `[[servers]]`) this surface hosts.
    #[serde(default)]
    pub enabled: Vec<String>,
}

/// Parsed `client-mcp.toml`: server definitions plus per-surface enable lists.
///
/// `servers` says what *exists* on this machine; `surfaces` says *who gets what*.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ClientMcpConfig {
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
    #[serde(default)]
    pub surfaces: HashMap<String, SurfaceConfig>,
}

/// `$XDG_CONFIG_HOME/adele/client-mcp.toml`, else `~/.config/adele/client-mcp.toml`.
///
/// A shared, per-machine location — distinct from each client's own config dir
/// (`adele-tui/`, `adele-gtk/`, `adele-voice/`) — so every surface reads one file.
pub fn default_client_mcp_path() -> PathBuf {
    let config_dir = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".config")
        });
    config_dir.join("adele").join("client-mcp.toml")
}

impl ClientMcpConfig {
    /// Parse and validate from TOML.
    ///
    /// Fails closed on a **duplicate server name**: a typo'd duplicate must not
    /// silently shadow another definition (the daemon keys tools by name, so a
    /// duplicate would be ambiguous downstream).
    pub fn from_toml(contents: &str) -> Result<Self, String> {
        let config: ClientMcpConfig =
            toml::from_str(contents).map_err(|e| format!("parse error: {e}"))?;
        let mut seen = HashSet::new();
        for server in &config.servers {
            if !seen.insert(server.name.as_str()) {
                return Err(format!("duplicate server name: {}", server.name));
            }
        }
        Ok(config)
    }

    /// Load from a file, tolerantly — matching the clients' established idiom.
    ///
    /// An absent file, an unreadable one, or an invalid one yields [`Default`]
    /// (with a warning) rather than failing the client. A bad config file must
    /// never stop a client from connecting.
    pub fn load(path: &Path) -> Self {
        if !path.exists() {
            tracing::debug!(
                "client-mcp config not found at {}; no client-hosted MCP servers",
                path.display()
            );
            return Self::default();
        }
        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(err) => {
                tracing::warn!(
                    "failed to read client-mcp config {}: {err}; ignoring",
                    path.display()
                );
                return Self::default();
            }
        };
        match Self::from_toml(&contents) {
            Ok(config) => {
                tracing::info!(
                    "loaded {} client MCP server definition(s) from {}",
                    config.servers.len(),
                    path.display()
                );
                config
            }
            Err(err) => {
                tracing::warn!(
                    "invalid client-mcp config {}: {err}; ignoring",
                    path.display()
                );
                Self::default()
            }
        }
    }

    /// The servers a given surface should host.
    ///
    /// Resolution: the names in `[surfaces.<surface>]` — falling back to
    /// `[surfaces.default]` **only when the surface has no entry of its own**
    /// (an explicit empty list means "nothing", not "inherit default") — filtered
    /// to defined, `enabled = true` servers. Order follows the surface's list;
    /// undefined names are skipped with a warning.
    pub fn resolved_servers(&self, surface: &str) -> Vec<&McpServerConfig> {
        let Some(selection) = self
            .surfaces
            .get(surface)
            .or_else(|| self.surfaces.get(DEFAULT_SURFACE))
        else {
            return Vec::new();
        };
        let by_name: HashMap<&str, &McpServerConfig> =
            self.servers.iter().map(|s| (s.name.as_str(), s)).collect();
        let mut resolved = Vec::new();
        for name in &selection.enabled {
            match by_name.get(name.as_str()) {
                Some(server) if server.enabled => resolved.push(*server),
                Some(_) => tracing::debug!(
                    "client MCP server '{name}' is disabled; skipping for surface '{surface}'"
                ),
                None => tracing::warn!(
                    "surface '{surface}' enables undefined client MCP server '{name}'; skipping"
                ),
            }
        }
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[[servers]]
name = "filesystem"
command = "fileio-mcp"
args = ["serve", "--root", "/home/dave"]
namespace = "fs"

[[servers]]
name = "git"
command = "git-mcp"
namespace = "git"

[[servers]]
name = "disabled-one"
command = "nope"
enabled = false

[surfaces.default]
enabled = ["filesystem"]

[surfaces.gtk]
enabled = ["filesystem", "git"]

[surfaces.locked]
enabled = ["filesystem", "disabled-one", "ghost"]
"#;

    fn names(servers: &[&McpServerConfig]) -> Vec<String> {
        servers.iter().map(|s| s.name.clone()).collect()
    }

    #[test]
    fn parses_servers_and_surfaces() {
        let cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        assert_eq!(cfg.servers.len(), 3);
        assert_eq!(cfg.servers[0].name, "filesystem");
        assert_eq!(cfg.servers[0].namespace.as_deref(), Some("fs"));
        assert_eq!(cfg.surfaces["gtk"].enabled, vec!["filesystem", "git"]);
    }

    #[test]
    fn resolved_servers_uses_surface_list() {
        let cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        assert_eq!(names(&cfg.resolved_servers("gtk")), vec!["filesystem", "git"]);
    }

    #[test]
    fn resolved_servers_falls_back_to_default() {
        let cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        // "tui" has no surface entry -> inherits [surfaces.default].
        assert_eq!(names(&cfg.resolved_servers("tui")), vec!["filesystem"]);
    }

    #[test]
    fn no_default_and_unknown_surface_is_empty() {
        let cfg = ClientMcpConfig::from_toml(
            r#"
[[servers]]
name = "a"
command = "a"
[surfaces.gtk]
enabled = ["a"]
"#,
        )
        .unwrap();
        assert!(cfg.resolved_servers("voice").is_empty());
    }

    #[test]
    fn resolved_servers_skips_undefined() {
        let cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        // "locked" lists filesystem (ok), disabled-one (disabled), ghost (undefined).
        assert_eq!(names(&cfg.resolved_servers("locked")), vec!["filesystem"]);
    }

    #[test]
    fn resolved_servers_excludes_disabled() {
        let cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        assert!(
            !names(&cfg.resolved_servers("locked"))
                .iter()
                .any(|n| n == "disabled-one")
        );
    }

    #[test]
    fn empty_surface_does_not_fall_back() {
        let cfg = ClientMcpConfig::from_toml(
            r#"
[[servers]]
name = "a"
command = "a"
[surfaces.default]
enabled = ["a"]
[surfaces.voice]
enabled = []
"#,
        )
        .unwrap();
        // Explicit empty list = "nothing"; a surface with no entry = default.
        assert!(cfg.resolved_servers("voice").is_empty());
        assert_eq!(names(&cfg.resolved_servers("tui")), vec!["a"]);
    }

    #[test]
    fn absent_file_is_default() {
        let cfg = ClientMcpConfig::load(Path::new("/nonexistent/client-mcp.toml"));
        assert!(cfg.servers.is_empty());
        assert!(cfg.surfaces.is_empty());
    }

    #[test]
    fn malformed_file_warns_and_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("client-mcp.toml");
        std::fs::write(&path, "this is : not valid toml [[[").unwrap();
        let cfg = ClientMcpConfig::load(&path);
        assert!(cfg.servers.is_empty());
    }

    #[test]
    fn duplicate_server_name_is_error() {
        let err = ClientMcpConfig::from_toml(
            r#"
[[servers]]
name = "dup"
command = "a"
[[servers]]
name = "dup"
command = "b"
"#,
        )
        .unwrap_err();
        assert!(err.contains("duplicate"), "got: {err}");
    }

    #[test]
    fn default_client_mcp_path_is_shared_adele_dir() {
        let path = default_client_mcp_path();
        let shown = path.to_str().unwrap();
        assert!(shown.contains("adele"), "got: {shown}");
        assert!(shown.ends_with("client-mcp.toml"), "got: {shown}");
    }
}
