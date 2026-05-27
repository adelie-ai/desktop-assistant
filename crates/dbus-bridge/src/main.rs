//! `adelie-dbus-bridge` — standalone D-Bus bridge binary (issue #106).
//!
//! At startup:
//! 1. Fetch a JWT from the local minter (`$XDG_RUNTIME_DIR/adelie/mint.sock`
//!    by default; overridable via `--minter-socket`).
//! 2. Open a UDS connection to the daemon (`$XDG_RUNTIME_DIR/adelie/sock`,
//!    overridable via `--daemon-socket`), perform the JWT handshake.
//! 3. Stand up zbus adapters at the four canonical object paths.
//! 4. Forward incoming wire events to D-Bus signals.
//! 5. Wait for SIGTERM / SIGINT; tear down cleanly.
//!
//! The well-known bus name is configurable (`--name`, default
//! `org.desktopAssistant.Bridge`). This default deliberately differs
//! from the daemon's in-process name (`org.desktopAssistant`) so the
//! bridge can run alongside the daemon during the transition (PR #106
//! ships Option A — see PR body). A follow-up issue switches the
//! default to `org.desktopAssistant` and removes the daemon's
//! in-process surface.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use desktop_assistant_dbus_bridge::adapter::{
    DBUS_SERVICE_NAME, DbusConnectionsAdapter, DbusConversationsAdapter, DbusKnowledgeAdapter,
    DbusSettingsAdapter, event_forwarder, paths,
};
use desktop_assistant_dbus_bridge::minter::{MintRequest, default_minter_socket_path, fetch_jwt};
use desktop_assistant_dbus_bridge::transport::{
    BridgeTransport, UdsBridgeConfig, UdsBridgeTransport, default_daemon_socket_path,
};
use tokio::signal::unix::{SignalKind, signal};

const DEFAULT_BRIDGE_NAME: &str = "org.desktopAssistant.Bridge";

#[derive(Debug, Parser)]
#[command(
    name = "adelie-dbus-bridge",
    about = "Per-user D-Bus bridge: translates org.desktopAssistant calls into UDS+JWT requests to the daemon.",
    version,
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

    /// D-Bus well-known name to bind. Defaults to
    /// `org.desktopAssistant.Bridge` so the bridge can run alongside
    /// the daemon's in-process surface during the Option-A
    /// transition. Set to `org.desktopAssistant` (and remove the
    /// daemon's surface) to flip the cutover.
    #[arg(long, env = "ADELIE_BRIDGE_NAME", default_value = DEFAULT_BRIDGE_NAME)]
    name: String,

    /// JWT TTL in seconds requested from the minter.
    #[arg(long, default_value_t = 60 * 60)]
    token_ttl_seconds: u64,

    /// Per-request timeout (seconds) for the UDS dispatch.
    #[arg(long, default_value_t = 30)]
    request_timeout_seconds: u64,
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
        .or_else(default_daemon_socket_path)
        .context("XDG_RUNTIME_DIR not set; pass --daemon-socket explicitly")?;

    // 1. Mint.
    tracing::info!(
        minter = %minter_socket.display(),
        "requesting JWT from local minter",
    );
    let jwt = fetch_jwt(
        &minter_socket,
        MintRequest {
            ttl_seconds: Some(cli.token_ttl_seconds),
            audience: None,
        },
        Duration::from_secs(10),
    )
    .await
    .with_context(|| {
        format!(
            "failed to fetch JWT from minter at {} — is adelie-mint.service running?",
            minter_socket.display()
        )
    })?;

    // 2. Connect.
    tracing::info!(
        daemon = %daemon_socket.display(),
        "connecting to daemon UDS",
    );
    let transport_config = UdsBridgeConfig {
        socket_path: daemon_socket.clone(),
        request_timeout: Duration::from_secs(cli.request_timeout_seconds),
        event_buffer: 256,
    };
    let transport = UdsBridgeTransport::connect(transport_config, &jwt)
        .await
        .with_context(|| {
            format!(
                "failed to handshake with daemon at {} — is the daemon running with the UDS frontend enabled?",
                daemon_socket.display()
            )
        })?;
    let transport = Arc::new(transport);

    // 3. Stand up adapters + bind D-Bus name.
    tracing::info!(name = %cli.name, "binding D-Bus name");
    let _ = DBUS_SERVICE_NAME; // referenced for symmetry; CLI flag overrides
    let conversations = DbusConversationsAdapter::new(Arc::clone(&transport));
    let settings = DbusSettingsAdapter::new(Arc::clone(&transport));
    let connections = DbusConnectionsAdapter::new(Arc::clone(&transport));
    let knowledge = DbusKnowledgeAdapter::new(Arc::clone(&transport));

    let connection = zbus::connection::Builder::session()
        .context("failed to connect to D-Bus session bus")?
        .name(cli.name.as_str())?
        .serve_at(paths::CONVERSATIONS, conversations)?
        .serve_at(paths::SETTINGS, settings)?
        .serve_at(paths::CONNECTIONS, connections)?
        .serve_at(paths::KNOWLEDGE, knowledge)?
        .build()
        .await
        .context("failed to build D-Bus connection")?;
    tracing::info!(name = %cli.name, "D-Bus bridge ready");

    // 4. Event forwarder.
    let events = transport.subscribe_events();
    let forwarder_shutdown = build_shutdown_signal()?;
    let forwarder_connection = connection.clone();
    let forwarder = tokio::spawn(event_forwarder::run(
        events,
        forwarder_connection,
        forwarder_shutdown,
    ));

    // 5. Wait for SIGTERM/SIGINT.
    let main_shutdown = build_shutdown_signal()?;
    main_shutdown.await;
    tracing::info!("shutdown signal received; tearing down");

    transport.shutdown();
    // Best effort: give the forwarder a moment to drain.
    let _ = tokio::time::timeout(Duration::from_secs(2), forwarder).await;
    drop(connection);
    Ok(())
}

fn build_shutdown_signal()
-> anyhow::Result<impl std::future::Future<Output = ()> + Send + 'static> {
    let mut sigterm =
        signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let mut sigint =
        signal(SignalKind::interrupt()).context("install SIGINT handler")?;
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
