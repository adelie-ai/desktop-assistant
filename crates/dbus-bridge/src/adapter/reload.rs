//! D-Bus adapter for `/org/desktopAssistant/Reload` (#222).
//!
//! Mirrors the in-process `org.desktopAssistant.Reload` interface
//! (`crates/dbus-interface/src/reload.rs`) method-for-method so a client (the
//! KCM) gets a working `Reload` regardless of which D-Bus surface — the
//! in-process daemon adapter or this standalone bridge — owns
//! `org.desktopAssistant`.
//!
//! Reload is a daemon-*local* operation (re-read `daemon.toml`, validate, swap
//! the connection registry under its lock). It is deliberately NOT modeled as
//! an `api::Command` on the UDS/WS wire — the daemon already watches its config
//! file. So the bridge implements `Reload` by bumping the mtime of
//! `daemon.toml`, which trips the daemon's file watcher and drives the same
//! validate-classify-swap path the in-process `Reload` method does. This keeps
//! the bridge free of any new wire command while still giving the KCM a single
//! `Reload` call to make after it writes the config.

use std::path::PathBuf;

use zbus::{fdo, interface};

/// Resolve `daemon.toml` the same way the daemon does: `$XDG_CONFIG_HOME`
/// (falling back to `$HOME/.config`) + `desktop-assistant/daemon.toml`. Kept in
/// sync with `desktop_assistant_daemon::config::default_daemon_config_path` (the
/// bridge does not link the daemon crate, by design).
fn default_daemon_config_path() -> PathBuf {
    let config_home = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string())).join(".config")
        });
    config_home.join("desktop-assistant").join("daemon.toml")
}

/// D-Bus adapter exposing `Reload` on `org.desktopAssistant.Reload`.
pub struct DbusReloadAdapter {
    config_path: PathBuf,
}

impl Default for DbusReloadAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl DbusReloadAdapter {
    pub fn new() -> Self {
        Self {
            config_path: default_daemon_config_path(),
        }
    }

    /// Override the config path (used in tests).
    pub fn with_config_path(mut self, path: PathBuf) -> Self {
        self.config_path = path;
        self
    }

    /// Touch the config file's mtime so the daemon's file watcher fires and
    /// applies the (already-written) config. Returns an error if the file does
    /// not exist — there is nothing for the daemon to reload.
    fn touch_config(&self) -> std::io::Result<()> {
        let now = std::time::SystemTime::now();
        let file = std::fs::File::open(&self.config_path)?;
        file.set_modified(now)
    }
}

#[interface(name = "org.desktopAssistant.Reload")]
impl DbusReloadAdapter {
    /// Ask the daemon to re-read `daemon.toml` and apply any changed config
    /// without a restart (#222).
    ///
    /// The KCM calls this after writing the config. The bridge bumps the
    /// config file's mtime to trip the daemon's file watcher, which runs the
    /// state-preserving validate-classify-swap (in-flight turns keep their
    /// clients; a bad config is refused and the last-good config keeps
    /// running).
    async fn reload(&self) -> fdo::Result<()> {
        self.touch_config().map_err(|e| {
            fdo::Error::Failed(format!(
                "failed to signal reload via {}: {e}",
                self.config_path.display()
            ))
        })?;
        tracing::info!("config reload requested over D-Bus (bridge); nudged the daemon watcher");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reload_bumps_config_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("daemon.toml");
        std::fs::write(&path, "[llm]\nconnector = \"ollama\"\n").unwrap();

        // Backdate the mtime so the touch is observable.
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        std::fs::File::open(&path)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let before = std::fs::metadata(&path).unwrap().modified().unwrap();

        let adapter = DbusReloadAdapter::new().with_config_path(path.clone());
        adapter.reload().await.expect("reload touches the config");

        let after = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert!(after > before, "Reload must bump the config file mtime");
    }

    #[tokio::test]
    async fn reload_errors_when_config_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.toml");
        let adapter = DbusReloadAdapter::new().with_config_path(path);
        let err = adapter
            .reload()
            .await
            .expect_err("a missing config must surface a clean D-Bus error");
        assert!(format!("{err}").contains("failed to signal reload"));
    }
}
