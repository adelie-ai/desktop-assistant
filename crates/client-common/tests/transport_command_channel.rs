//! adele-gtk#49 / #213: the generic command channel must be reachable on
//! *every* transport, so `TransportClient::as_commands` yields a
//! `&dyn AssistantCommands` for the WebSocket, Unix-domain-socket *and* D-Bus
//! variants — there is no longer any connector-specific command surface and
//! the WS-only `as_ws` accessor has been retired.
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
    let (client, _signals, _drop) = timeout(
        Duration::from_secs(5),
        WsClient::connect(&ws_url, "test-token", None, None, None, None),
    )
    .await
    .expect("ws connect did not complete in time")
    .expect("ws connect");

    let transport = TransportClient::Ws(client);
    assert!(
        transport.as_commands().is_some(),
        "as_commands must be Some for a WebSocket transport"
    );
}

/// #213: the D-Bus transport now implements `AssistantCommands` (round-tripping
/// `api::Command`/`api::CommandResult` as JSON over `org.desktopAssistant.
/// Commands`), so `as_commands` must be `Some` for it too — no transport is
/// left without the management command channel.
///
/// zbus proxies build lazily, so `DbusClient::connect` succeeds against the
/// session bus without the daemon's service being present. If no session bus
/// is reachable at all (some CI sandboxes), the capability mapping is still
/// guaranteed by the type system — the `Dbus` match arm returns
/// `Some(client)` because `DbusClient: AssistantCommands` — so we skip rather
/// than fail.
#[cfg(feature = "dbus")]
#[tokio::test]
async fn as_commands_returns_some_for_dbus_transport() {
    let client = match desktop_assistant_client_common::dbus_client::DbusClient::connect().await {
        Ok(client) => client,
        Err(e) => {
            eprintln!("skipping: no usable session bus ({e})");
            return;
        }
    };

    let transport = TransportClient::Dbus(client);
    assert!(
        transport.as_commands().is_some(),
        "as_commands must be Some for a D-Bus transport (#213)"
    );
}
