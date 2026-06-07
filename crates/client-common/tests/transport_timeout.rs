//! Transport robustness regression tests (#221).
//!
//! The socket transports used to wait forever on a silent server: a command
//! awaited its response with no deadline and the event fan-out only noticed a
//! *closed* connection, never an open-but-silent one. These tests stand up a
//! loopback UDS server that accepts the connection (and the handshake) but then
//! goes silent, and assert that:
//!
//!   1. command dispatch times out with a clear transport error (doesn't hang),
//!   2. the open-but-silent connection surfaces a terminal `Disconnected`
//!      through the high-level `Connector` (the stall is detected).
//!
//! Unlike `uds_transport.rs` / `uds_streaming.rs`, which drive the *real*
//! `desktop-assistant-uds` server (which always replies), the silent behaviour
//! we need here can't come from a well-behaved server — so we hand-roll a
//! minimal loopback that speaks just enough of the wire format (the same 4-byte
//! little-endian length-prefixed framing) to accept the handshake and then do
//! nothing.

use std::path::PathBuf;
use std::time::Duration;

use desktop_assistant_client_common::uds_client::UdsClient;
use desktop_assistant_client_common::{
    AssistantCommands, ConnectionConfig, Connector, SignalEvent, TransportMode,
};
use tempfile::TempDir;
use tokio::io::AsyncReadExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::time::timeout;

/// A loopback server that accepts every connection, drains whatever each client
/// writes (handshake + any command frames), and never writes a single byte back
/// — modelling a wedged/silent server. It accepts in a loop (so the
/// `wait_for_socket` probe and the real client are both handled) and holds each
/// connection open until its peer goes away, so the client observes *silence*,
/// not closure/EOF.
async fn spawn_silent_server(path: PathBuf) {
    let listener = UnixListener::bind(&path).expect("bind silent uds");
    tokio::spawn(async move {
        while let Ok((mut stream, _addr)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buf = [0u8; 1024];
                loop {
                    match stream.read(&mut buf).await {
                        Ok(0) => break,    // peer closed
                        Ok(_) => continue, // swallow bytes, stay silent
                        Err(_) => break,
                    }
                }
            });
        }
    });
}

async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..100 {
        if path.exists() && UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("silent uds socket {path:?} did not appear");
}

fn uds_config(socket_path: PathBuf) -> ConnectionConfig {
    ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(socket_path),
        // The silent server never validates the token, so any non-empty JWT is
        // fine — the handshake frame just has to be writable.
        ws_jwt: Some("silent-server-no-validation".to_string()),
        ..ConnectionConfig::default()
    }
}

/// A command issued to a connected-but-silent server must time out with a clear
/// transport error rather than hanging forever (#221, part 1).
#[tokio::test]
async fn uds_command_dispatch_times_out_on_a_silent_server() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("silent.sock");
    spawn_silent_server(path.clone()).await;
    wait_for_socket(&path).await;

    // Connect at the raw-client level so we can shorten the dispatch deadline;
    // the production default is 30s, far too long for a unit test.
    let (mut client, _signals, _drop) =
        UdsClient::connect(&path, "silent-server-no-validation", None, None)
            .await
            .expect("connect to silent uds");
    client.set_dispatch_timeout(Duration::from_millis(150));

    // The outer `timeout` is a test guard: if the dispatch timeout regressed,
    // the call would hang and we'd fail here instead of looping forever.
    let result = timeout(Duration::from_secs(2), client.list_conversations())
        .await
        .expect("the dispatch timeout must resolve the call, not hang");
    let err = result.expect_err("a silent server must surface a timeout error");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("timed out"),
        "expected a clear timeout error, got: {err}"
    );
}

/// A second command after a timeout must still time out (not hang on a leaked
/// pending slot) — proves the timed-out request was removed from the pending
/// map (#221, part 1: "drop the pending request").
#[tokio::test]
async fn uds_pending_slot_is_reclaimed_after_a_timeout() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("silent2.sock");
    spawn_silent_server(path.clone()).await;
    wait_for_socket(&path).await;

    let (mut client, _signals, _drop) =
        UdsClient::connect(&path, "silent-server-no-validation", None, None)
            .await
            .expect("connect to silent uds");
    client.set_dispatch_timeout(Duration::from_millis(100));

    for attempt in 0..2 {
        let result = timeout(Duration::from_secs(2), client.list_conversations())
            .await
            .unwrap_or_else(|_| panic!("attempt {attempt} hung instead of timing out"));
        assert!(
            result.is_err(),
            "attempt {attempt} should time out against a silent server"
        );
    }
}

/// End-to-end stall detection (#221, part 2): a `Connector` over an
/// open-but-silent UDS connection must surface a terminal `Disconnected` to its
/// subscribers once the stall window elapses, instead of hanging forever. The
/// old fan-out only detected *closure*, so this connection (open, never EOF,
/// never an event) would have hung a subscriber's `recv()` indefinitely.
#[tokio::test]
async fn connector_over_silent_uds_surfaces_a_stall() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("silent3.sock");
    spawn_silent_server(path.clone()).await;
    wait_for_socket(&path).await;

    // Short stall window so the test doesn't wait the production 90s.
    let connector =
        Connector::connect_with_stall_timeout(&uds_config(path), Duration::from_millis(150))
            .await
            .expect("connector over silent uds");
    let mut events = connector.subscribe();

    let event = timeout(Duration::from_secs(2), events.recv())
        .await
        .expect("a stalled connection must produce a terminal event, not hang");
    match event {
        Some(SignalEvent::Disconnected { reason }) => {
            assert!(
                reason.contains("stalled"),
                "expected a stall reason distinguishable from a clean close, got: {reason}"
            );
        }
        other => panic!("expected SignalEvent::Disconnected on stall, got {other:?}"),
    }
}
