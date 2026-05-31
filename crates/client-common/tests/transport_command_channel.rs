//! adele-gtk#49: the generic command channel must be reachable on every
//! socket transport, so `TransportClient::as_commands` yields a
//! `&dyn AssistantCommands` for the WebSocket *and* Unix-domain-socket
//! variants and `None` for D-Bus (which speaks a separate typed zbus
//! interface).
//!
//! The `Uds` arm and the round-trip of the two promoted command methods over
//! UDS are exercised in `uds_transport.rs` against the real UDS server. This
//! file covers the `Ws` arm with a minimal in-process WebSocket accept server
//! (enough to complete the upgrade and own the socket) and documents the
//! `Dbus` arm, which can't be constructed without a live session bus.

use desktop_assistant_client_common::TransportClient;
use desktop_assistant_client_common::ws_client::WsClient;
use tokio::net::TcpListener;
use tokio::time::{Duration, timeout};

/// A bare `ws://` server: accept one TCP connection, complete the WebSocket
/// upgrade, and hold the socket open. It never replies to commands — this test
/// only asserts the transport-capability mapping, not request round-trips.
async fn spawn_ws_accept_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Ok((stream, _)) = listener.accept().await
            && let Ok(ws) = tokio_tungstenite::accept_async(stream).await
        {
            // Keep the connection (and the upgraded socket) alive for the
            // lifetime of the test so the client side stays connected.
            let _ws = ws;
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    });
    format!("ws://{addr}/ws")
}

#[tokio::test]
async fn as_commands_returns_some_for_ws_transport() {
    let ws_url = spawn_ws_accept_server().await;

    // No TLS for plain `ws://`; the accept server ignores the bearer token.
    let (client, _signals) = timeout(
        Duration::from_secs(5),
        WsClient::connect(&ws_url, "test-token", None),
    )
    .await
    .expect("ws connect did not complete in time")
    .expect("ws connect");

    let transport = TransportClient::Ws(client);
    assert!(
        transport.as_commands().is_some(),
        "as_commands must be Some for a WebSocket transport"
    );
    // `as_ws` is retained alongside `as_commands` for now.
    assert!(transport.as_ws().is_some());
}

/// The D-Bus transport speaks a separate typed zbus interface and does not
/// implement `AssistantCommands`, so `as_commands` must be `None` for it.
///
/// zbus proxies build lazily, so `DbusClient::connect` succeeds against the
/// session bus without the daemon's service being present. If no session bus
/// is reachable at all (some CI sandboxes), the capability mapping is still
/// guaranteed by the type system — the `Dbus` match arm cannot return
/// `Some(client)` because `DbusClient: !AssistantCommands` — so we skip rather
/// than fail.
#[cfg(feature = "dbus")]
#[tokio::test]
async fn as_commands_returns_none_for_dbus_transport() {
    let client = match desktop_assistant_client_common::dbus_client::DbusClient::connect().await {
        Ok(client) => client,
        Err(e) => {
            eprintln!("skipping: no usable session bus ({e})");
            return;
        }
    };

    let transport = TransportClient::Dbus(client);
    assert!(
        transport.as_commands().is_none(),
        "as_commands must be None for a D-Bus transport"
    );
    assert!(transport.as_ws().is_none());
}
