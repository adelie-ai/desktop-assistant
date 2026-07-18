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

/// The client surfaces that can host MCP servers. Each corresponds to a
/// `[surfaces.<name>]` section; a surface with no section inherits
/// [`DEFAULT_SURFACE`]. Kept here as the single source of truth so admin UIs and
/// clients agree on the set rather than scattering string literals.
pub const CLIENT_SURFACES: [&str; 4] = ["gtk", "tui", "kde", "voice"];

/// True when `name` is one of the [`CLIENT_SURFACES`]. The inheritance fallback
/// [`DEFAULT_SURFACE`] is deliberately not a client surface.
pub fn is_client_surface(name: &str) -> bool {
    CLIENT_SURFACES.contains(&name)
}

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

    /// Every defined server, including disabled ones and those no surface hosts.
    ///
    /// Unlike [`resolved_servers`](Self::resolved_servers) this is the raw
    /// definition list — the natural backing for an admin table that edits the
    /// full set, not a per-surface projection.
    pub fn list_defined_servers(&self) -> &[McpServerConfig] {
        &self.servers
    }

    /// The **raw** `enabled` list for a surface, exactly as written.
    ///
    /// Unlike [`resolved_servers`](Self::resolved_servers) this does not filter
    /// out disabled or undefined names, and it does **not** fall back to
    /// [`DEFAULT_SURFACE`]: a surface with no entry returns an empty slice. This
    /// is what an editor toggles against; resolution is applied at read time.
    pub fn surface_enabled_names(&self, surface: &str) -> &[String] {
        self.surfaces
            .get(surface)
            .map(|s| s.enabled.as_slice())
            .unwrap_or(&[])
    }

    /// Add a new server definition.
    ///
    /// Rejects an empty name and a duplicate of an existing one — the same
    /// fail-closed rule [`from_toml`](Self::from_toml) enforces, since the daemon
    /// keys client tools by server name.
    pub fn add_server(&mut self, server: McpServerConfig) -> Result<(), String> {
        if server.name.trim().is_empty() {
            return Err("server name must not be empty".to_string());
        }
        if self.servers.iter().any(|s| s.name == server.name) {
            return Err(format!("duplicate server name: {}", server.name));
        }
        self.servers.push(server);
        Ok(())
    }

    /// Insert a server, or replace the existing one with the same name in place.
    ///
    /// Idempotent by name: re-applying the same definition leaves the config
    /// unchanged, and applying an edited definition swaps it without disturbing
    /// ordering or any surface's enable lists.
    pub fn upsert_server(&mut self, server: McpServerConfig) {
        match self.servers.iter_mut().find(|s| s.name == server.name) {
            Some(existing) => *existing = server,
            None => self.servers.push(server),
        }
    }

    /// Remove a server definition and prune its name from every surface.
    ///
    /// Errors if no server by that name exists. On success the name is also
    /// dropped from every `[surfaces.*].enabled` list so no surface is left
    /// pointing at a definition that no longer exists.
    pub fn remove_server(&mut self, name: &str) -> Result<(), String> {
        let before = self.servers.len();
        self.servers.retain(|s| s.name != name);
        if self.servers.len() == before {
            return Err(format!("no such server: {name}"));
        }
        for surface in self.surfaces.values_mut() {
            surface.enabled.retain(|n| n != name);
        }
        Ok(())
    }

    /// Flip a server definition's own `enabled` flag.
    ///
    /// This is the definition-level switch (a disabled definition is hosted by no
    /// surface); it is orthogonal to per-surface selection. Errors if the server
    /// is not defined.
    pub fn set_server_enabled(&mut self, name: &str, on: bool) -> Result<(), String> {
        let server = self
            .servers
            .iter_mut()
            .find(|s| s.name == name)
            .ok_or_else(|| format!("no such server: {name}"))?;
        server.enabled = on;
        Ok(())
    }

    /// Add or remove a server name in one surface's own `enabled` list.
    ///
    /// Materializes a [`SurfaceConfig`] for `surface` if it has none, so editing
    /// a surface that was inheriting [`DEFAULT_SURFACE`] gives it its own explicit
    /// list. Only the named surface is touched — the `default` list is never
    /// mutated as a side effect. Adding is idempotent (no duplicate entry).
    pub fn set_surface_enabled(&mut self, surface: &str, name: &str, on: bool) {
        let entry = self.surfaces.entry(surface.to_string()).or_default();
        if on {
            if !entry.enabled.iter().any(|n| n == name) {
                entry.enabled.push(name.to_string());
            }
        } else {
            entry.enabled.retain(|n| n != name);
        }
    }

    /// Serialize to TOML and write to `path` atomically with `0600` permissions.
    ///
    /// Re-runs the duplicate-name validation first and fails closed without
    /// writing anything if it trips. Creates the parent directory, writes a
    /// private sibling temp file (same directory, so the rename is atomic on one
    /// filesystem), fsyncs it, then renames it over `path` — a reader never sees a
    /// partial or world-readable file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut seen = HashSet::new();
        for server in &self.servers {
            if !seen.insert(server.name.as_str()) {
                return Err(format!("duplicate server name: {}", server.name));
            }
        }

        let contents = toml::to_string_pretty(self).map_err(|e| format!("serialize error: {e}"))?;

        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }

        let file_name = path
            .file_name()
            .ok_or_else(|| format!("invalid config path: {}", path.display()))?;
        let mut tmp = path.to_path_buf();
        tmp.set_file_name(format!(
            ".{}.{}.tmp",
            file_name.to_string_lossy(),
            uuid::Uuid::new_v4()
        ));

        if let Err(err) = write_private(&tmp, contents.as_bytes()) {
            let _ = std::fs::remove_file(&tmp);
            return Err(err);
        }
        if let Err(err) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("failed to write {}: {err}", path.display()));
        }
        Ok(())
    }
}

/// Create `path` (which must not exist) with `0600` permissions on Unix, write
/// `bytes`, and fsync. `create_new` closes a temp-file symlink/pre-seed race.
fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts
        .open(path)
        .map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    file.sync_all()
        .map_err(|e| format!("failed to flush {}: {e}", path.display()))?;
    Ok(())
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

    /// Build a minimal `McpServerConfig` by name, going through the parser so
    /// the test doesn't depend on the (cross-crate) struct's full field set.
    fn server(name: &str) -> McpServerConfig {
        ClientMcpConfig::from_toml(&format!(
            "[[servers]]\nname = \"{name}\"\ncommand = \"cmd\"\n"
        ))
        .expect("valid single-server toml")
        .servers
        .into_iter()
        .next()
        .expect("one server")
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
        assert_eq!(
            names(&cfg.resolved_servers("gtk")),
            vec!["filesystem", "git"]
        );
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

    // ----- Phase-1 edit operations (#532) -----

    #[test]
    fn client_surfaces_helper_recognizes_known() {
        assert_eq!(CLIENT_SURFACES.len(), 4);
        assert!(is_client_surface("gtk"));
        assert!(is_client_surface("tui"));
        assert!(is_client_surface("kde"));
        assert!(is_client_surface("voice"));
        // `default` is the inheritance fallback, not a client surface.
        assert!(!is_client_surface(DEFAULT_SURFACE));
        assert!(!is_client_surface("bogus"));
    }

    #[test]
    fn save_roundtrips_0600() {
        let dir = tempfile::tempdir().unwrap();
        // A nested path exercises parent-dir creation.
        let path = dir.path().join("nested").join("client-mcp.toml");
        let cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        cfg.save(&path).expect("save");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "expected 0600, got {:o}", mode & 0o777);
        }

        // load-modify-save-load preserves data.
        let reloaded = ClientMcpConfig::load(&path);
        assert_eq!(reloaded.servers.len(), cfg.servers.len());
        assert_eq!(reloaded.servers[0].name, "filesystem");
        assert_eq!(reloaded.servers[0].namespace.as_deref(), Some("fs"));
        assert_eq!(
            reloaded.servers[0].args,
            vec!["serve", "--root", "/home/dave"]
        );
        assert_eq!(
            names(&reloaded.resolved_servers("gtk")),
            vec!["filesystem", "git"]
        );
        // A disabled definition and the raw per-surface lists survive the trip.
        assert!(
            reloaded
                .list_defined_servers()
                .iter()
                .any(|s| s.name == "disabled-one" && !s.enabled)
        );
        assert_eq!(
            reloaded.surface_enabled_names("locked"),
            &["filesystem", "disabled-one", "ghost"]
        );
    }

    #[test]
    fn save_rejects_duplicate_names() {
        // Force a duplicate past the parser (which would have rejected it) and
        // confirm save re-validates and fails closed without writing a file.
        let mut cfg = ClientMcpConfig::default();
        cfg.servers.push(server("dup"));
        cfg.servers.push(server("dup"));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("client-mcp.toml");
        let err = cfg.save(&path).unwrap_err();
        assert!(err.contains("duplicate"), "got: {err}");
        assert!(
            !path.exists(),
            "no file should be written on validation failure"
        );
    }

    #[test]
    fn add_rejects_empty_and_duplicate() {
        let mut cfg = ClientMcpConfig::default();
        cfg.add_server(server("a")).expect("first add");

        let dup = cfg.add_server(server("a")).unwrap_err();
        assert!(dup.contains("duplicate"), "got: {dup}");

        let empty = cfg.add_server(server("")).unwrap_err();
        assert!(empty.contains("empty"), "got: {empty}");

        // Neither rejected add mutated the definition list.
        assert_eq!(cfg.list_defined_servers().len(), 1);
    }

    #[test]
    fn upsert_replaces_or_appends() {
        let mut cfg = ClientMcpConfig::default();
        cfg.upsert_server(server("a"));
        cfg.upsert_server(server("b"));
        assert_eq!(cfg.list_defined_servers().len(), 2);

        // Same name replaces in place (no new entry), carrying new fields.
        let mut updated = server("a");
        updated.command = "newcmd".to_string();
        cfg.upsert_server(updated);
        assert_eq!(cfg.list_defined_servers().len(), 2);
        let a = cfg
            .list_defined_servers()
            .iter()
            .find(|s| s.name == "a")
            .unwrap();
        assert_eq!(a.command, "newcmd");
    }

    #[test]
    fn remove_prunes_from_all_surfaces_and_errors_if_absent() {
        let mut cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        // "filesystem" appears in default, gtk, and locked.
        cfg.remove_server("filesystem").expect("remove existing");
        assert!(
            !cfg.list_defined_servers()
                .iter()
                .any(|s| s.name == "filesystem")
        );
        for surface in ["default", "gtk", "locked"] {
            assert!(
                !cfg.surface_enabled_names(surface)
                    .iter()
                    .any(|n| n == "filesystem"),
                "surface '{surface}' still lists filesystem"
            );
        }
        // Sibling names in those surfaces are untouched.
        assert_eq!(cfg.surface_enabled_names("gtk"), &["git"]);

        let err = cfg.remove_server("filesystem").unwrap_err();
        assert!(err.contains("filesystem"), "got: {err}");
    }

    #[test]
    fn set_surface_enabled_materializes_and_never_touches_default() {
        let mut cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        // "voice" has no entry of its own -> it inherits default.
        assert!(cfg.surface_enabled_names("voice").is_empty());

        cfg.set_surface_enabled("voice", "git", true);
        assert_eq!(cfg.surface_enabled_names("voice"), &["git"]);
        // default's own list must be untouched by editing another surface.
        assert_eq!(cfg.surface_enabled_names("default"), &["filesystem"]);

        // Adding is idempotent (no duplicate entry).
        cfg.set_surface_enabled("voice", "git", true);
        assert_eq!(cfg.surface_enabled_names("voice"), &["git"]);

        // Removing operates on the named surface only.
        cfg.set_surface_enabled("voice", "git", false);
        assert!(cfg.surface_enabled_names("voice").is_empty());
        assert_eq!(cfg.surface_enabled_names("default"), &["filesystem"]);
    }

    #[test]
    fn set_server_enabled_flips_definition() {
        let mut cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        cfg.set_server_enabled("filesystem", false)
            .expect("disable");
        assert!(
            !cfg.list_defined_servers()
                .iter()
                .find(|s| s.name == "filesystem")
                .unwrap()
                .enabled
        );
        // A disabled definition drops out of resolution everywhere.
        assert!(
            cfg.resolved_servers("gtk")
                .iter()
                .all(|s| s.name != "filesystem")
        );

        cfg.set_server_enabled("filesystem", true)
            .expect("re-enable");
        assert!(
            cfg.list_defined_servers()
                .iter()
                .find(|s| s.name == "filesystem")
                .unwrap()
                .enabled
        );

        let err = cfg.set_server_enabled("ghost", true).unwrap_err();
        assert!(err.contains("ghost"), "got: {err}");
    }

    #[test]
    fn surface_enabled_names_is_raw_unfiltered() {
        let cfg = ClientMcpConfig::from_toml(SAMPLE).unwrap();
        // Raw list keeps the disabled and undefined names that resolution drops.
        assert_eq!(
            cfg.surface_enabled_names("locked"),
            &["filesystem", "disabled-one", "ghost"]
        );
        assert_eq!(names(&cfg.resolved_servers("locked")), vec!["filesystem"]);

        // A surface with no entry yields an empty slice, with NO default fallback
        // (unlike `resolved_servers`).
        assert!(cfg.surface_enabled_names("voice").is_empty());
    }
}
