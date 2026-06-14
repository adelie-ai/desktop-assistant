//! `adelie-dbus-bridge` — standalone D-Bus bridge binary (issue #106).
//!
//! At startup:
//! 1. Build a UDS [`ConnectionConfig`] pointing at the daemon socket + the local
//!    `adelie-mint` socket, and open a client-common [`Connector`] (#316). The
//!    Connector mints a JWT from the minter, performs the handshake, and from
//!    then on owns reconnect + re-minting on every drop.
//! 2. Stand up zbus adapters at the canonical object paths over a transport that
//!    forwards each command through the Connector.
//! 3. Forward the daemon's signal stream to D-Bus signals (auto-resubscribing
//!    across reconnects).
//! 4. Wait for SIGTERM / SIGINT; tear down cleanly.
//!
//! The well-known bus name is configurable (`--name`); it defaults to
//! `org.desktopAssistant` — the bridge is the live D-Bus surface as of the
//! cutover name flip (#318), and the daemon no longer claims the name (its
//! in-process surface is off by default). Pass `--name org.desktopAssistant.Bridge`
//! or `.Dev` to run a side-by-side instance for QA without colliding.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use desktop_assistant_client_common::minter::default_minter_socket_path;
use desktop_assistant_client_common::{
    ConnectionConfig, Connector, TransportMode, default_desktop_socket_path,
};
use desktop_assistant_dbus_bridge::adapter::{
    DBUS_SERVICE_NAME, DbusBackgroundTasksAdapter, DbusCommandsAdapter, DbusConnectionsAdapter,
    DbusConversationsAdapter, DbusKnowledgeAdapter, DbusReloadAdapter, DbusSettingsAdapter,
    event_forwarder, paths,
};
use desktop_assistant_dbus_bridge::transport::ConnectorBridgeTransport;
use tokio::signal::unix::{SignalKind, signal};

// The bridge is the live D-Bus surface as of the cutover's name flip (#318):
// it claims `org.desktopAssistant` and the daemon steps off the name (its
// in-process surface is off by default; `DESKTOP_ASSISTANT_DBUS_INPROCESS=true`
// re-enables it as a revert). Use `--name org.desktopAssistant.Bridge`/`.Dev`
// to run a side-by-side instance for QA without colliding.
const DEFAULT_BRIDGE_NAME: &str = "org.desktopAssistant";

#[derive(Debug, Parser)]
#[command(
    name = "adelie-dbus-bridge",
    about = "Per-user D-Bus bridge: translates org.desktopAssistant calls into UDS+JWT requests to the daemon.",
    version
)]
struct Cli {
    /// Path to the local JWT minter socket. Defaults to
    /// `$XDG_RUNTIME_DIR/adelie/mint.sock`.
    #[arg(long, env = "ADELIE_BRIDGE_MINTER_SOCKET")]
    minter_socket: Option<PathBuf>,

    /// Path to the daemon's UDS socket. Defaults to
    /// `$XDG_RUNTIME_DIR/adelie/sock`.
    #[arg(long, env = "ADELIE_BRIDGE_DAEMON_SOCKET")]
    daemon_socket: Option<PathBuf>,

    /// D-Bus well-known name to bind. Defaults to `org.desktopAssistant` — the
    /// bridge is the live D-Bus surface (#318). Use `org.desktopAssistant.Bridge`
    /// or `.Dev` to run a side-by-side instance for QA.
    #[arg(long, env = "ADELIE_BRIDGE_NAME", default_value = DEFAULT_BRIDGE_NAME)]
    name: String,

    /// JWT TTL in seconds requested from the minter on every (re)connect.
    #[arg(long, default_value_t = 60 * 60)]
    token_ttl_seconds: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let minter_socket = cli
        .minter_socket
        .or_else(default_minter_socket_path)
        .context("XDG_RUNTIME_DIR not set; pass --minter-socket explicitly")?;
    let daemon_socket = cli
        .daemon_socket
        .or_else(default_desktop_socket_path)
        .context("XDG_RUNTIME_DIR not set; pass --daemon-socket explicitly")?;

    // 1. Connect via the shared Connector (mints a JWT from the local minter,
    //    handshakes, and owns reconnect + re-minting from here on — #316).
    tracing::info!(
        daemon = %daemon_socket.display(),
        minter = %minter_socket.display(),
        "connecting to daemon UDS via the client-common Connector",
    );
    let config = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(daemon_socket.clone()),
        minter_socket: Some(minter_socket.clone()),
        minter_ttl_seconds: Some(cli.token_ttl_seconds),
        ws_jwt: None,
        ..ConnectionConfig::default()
    };
    let connector = Connector::connect(&config).await.with_context(|| {
        format!(
            "failed to connect to the daemon at {} (minter {}) — are \
             desktop-assistant-daemon and adelie-mint running?",
            daemon_socket.display(),
            minter_socket.display(),
        )
    })?;
    let connector = Arc::new(connector);

    // 2. Stand up adapters over a Connector-backed transport + bind the name.
    tracing::info!(name = %cli.name, "binding D-Bus name");
    let _ = DBUS_SERVICE_NAME; // referenced for symmetry; CLI flag overrides
    let transport = Arc::new(ConnectorBridgeTransport::new(Arc::clone(&connector)));
    // Generic command channel (#213 / #315 G1): the JSON-in/JSON-out surface
    // tui/gtk use on `--transport dbus`.
    let commands = DbusCommandsAdapter::new(Arc::clone(&transport));
    let conversations = DbusConversationsAdapter::new(Arc::clone(&transport));
    let settings = DbusSettingsAdapter::new(Arc::clone(&transport));
    let connections = DbusConnectionsAdapter::new(Arc::clone(&transport));
    let knowledge = DbusKnowledgeAdapter::new(Arc::clone(&transport));
    let background_tasks = DbusBackgroundTasksAdapter::new(Arc::clone(&transport));
    // Config hot-reload (#222): nudges the daemon's file watcher so the KCM can
    // apply config changes without a daemon restart.
    let reload = DbusReloadAdapter::new();

    let connection = zbus::connection::Builder::session()
        .context("failed to connect to D-Bus session bus")?
        .name(cli.name.as_str())?
        .serve_at(paths::COMMANDS, commands)?
        .serve_at(paths::CONVERSATIONS, conversations)?
        .serve_at(paths::SETTINGS, settings)?
        .serve_at(paths::CONNECTIONS, connections)?
        .serve_at(paths::KNOWLEDGE, knowledge)?
        .serve_at(paths::BACKGROUND_TASKS, background_tasks)?
        .serve_at(paths::RELOAD, reload)?
        .build()
        .await
        .context("failed to build D-Bus connection")?;
    tracing::info!(name = %cli.name, "D-Bus bridge ready");

    // 3. Event forwarder. It issues the initial `SubscribeBackgroundTasks` and
    //    re-issues it (and re-subscribes the stream) across daemon restarts.
    let forwarder_shutdown = build_shutdown_signal()?;
    let forwarder = tokio::spawn(event_forwarder::run(
        Arc::clone(&connector),
        connection.clone(),
        forwarder_shutdown,
    ));

    // 4. Wait for SIGTERM/SIGINT.
    let main_shutdown = build_shutdown_signal()?;
    main_shutdown.await;
    tracing::info!("shutdown signal received; tearing down");

    // Dropping the connection stops serving; the forwarder exits on its own
    // shutdown signal. Give it a moment to drain, then drop the Connector
    // (which aborts its reconnect supervisor).
    let _ = tokio::time::timeout(Duration::from_secs(2), forwarder).await;
    drop(connection);
    drop(connector);
    Ok(())
}

fn build_shutdown_signal() -> anyhow::Result<impl std::future::Future<Output = ()> + Send + 'static>
{
    let mut sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let mut sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    Ok(async move {
        tokio::select! {
            _ = sigterm.recv() => {
                tracing::info!("received SIGTERM");
            }
            _ = sigint.recv() => {
                tracing::info!("received SIGINT");
            }
        }
    })
}
