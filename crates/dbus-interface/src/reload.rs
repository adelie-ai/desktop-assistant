//! D-Bus `Reload` method for state-preserving config hot-reload (#222).
//!
//! Mirrors the voice daemon's `org.desktopAssistant.Voice.Reload`: the KCM
//! writes `daemon.toml` via the settings setters, then calls `Reload` so the
//! orchestrator re-reads the file and applies the changed knobs without a
//! restart (which would sever every client and kill in-flight turns).
//!
//! The adapter only *pings* a channel; the daemon's reload-consumer task does
//! the validate-classify-swap (see `crates/daemon/src/api_surface.rs`
//! `RegistryHandle::apply_reload`). A bounded channel keeps a flurry of calls
//! from unbounded buffering — the consumer coalesces them.

use tokio::sync::mpsc;
use zbus::{fdo, interface};

/// D-Bus adapter exposing `Reload` on `org.desktopAssistant.Reload`.
pub struct DbusReloadAdapter {
    /// Pings the daemon's reload-consumer task to re-read and apply the config.
    reload_tx: mpsc::Sender<()>,
}

impl DbusReloadAdapter {
    pub fn new(reload_tx: mpsc::Sender<()>) -> Self {
        Self { reload_tx }
    }
}

#[interface(name = "org.desktopAssistant.Reload")]
impl DbusReloadAdapter {
    /// Re-read `daemon.toml` and apply any changed config to the running
    /// daemon without a restart (#222).
    ///
    /// Hot-applies `[connections]` / `[purposes]` / `[llm]` by rebuilding the
    /// connection registry — new turns route through the new clients while
    /// in-flight turns keep the client they already hold alive until they
    /// finish. Subsystems wired once at startup (database, embeddings, TLS,
    /// ws-auth, persistence, profiling) are logged as needing a restart. A
    /// config that fails to parse/validate is refused and the last-good config
    /// keeps running.
    ///
    /// Returns immediately after queuing the request; the apply happens
    /// asynchronously. A file watcher also picks up edits made any other way.
    async fn reload(&self) -> fdo::Result<()> {
        self.reload_tx
            .send(())
            .await
            .map_err(|e| fdo::Error::Failed(format!("failed to trigger reload: {e}")))?;
        tracing::info!("config reload requested over D-Bus");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reload_pings_the_channel() {
        let (tx, mut rx) = mpsc::channel(4);
        let adapter = DbusReloadAdapter::new(tx);
        adapter.reload().await.expect("reload queues a ping");
        assert!(rx.try_recv().is_ok(), "Reload must enqueue a reload ping");
    }

    #[tokio::test]
    async fn reload_errors_when_consumer_is_gone() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx); // consumer task gone
        let adapter = DbusReloadAdapter::new(tx);
        let err = adapter
            .reload()
            .await
            .expect_err("a dropped consumer must surface a clean D-Bus error");
        assert!(format!("{err}").contains("failed to trigger reload"));
    }
}
