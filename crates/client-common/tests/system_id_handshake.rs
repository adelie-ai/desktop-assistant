//! System-id co-location handshake threading (#248).
//!
//! These tests assert the per-machine **system id** the Connector stamps onto
//! the connection config is carried in the connect handshake AND re-sent on a
//! reconnect (#246/#247) — the explicit requirement that any new handshake field
//! survive a daemon restart.
//!
//! We hand-roll a minimal loopback UDS server (the same 4-byte little-endian
//! length-prefixed framing the real server uses) that parses the first frame as
//! an [`api::UdsHandshake`], reports it back over a channel, then **drops the
//! connection** to force the Connector's reconnect supervisor to re-handshake.
//! Driving the real daemon stack isn't needed to observe the handshake bytes.

use std::path::PathBuf;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_client_common::{ConnectionConfig, Connector, TransportMode};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::time::timeout;

/// Read one 4-byte LE length-prefixed frame.
async fn read_frame(stream: &mut UnixStream) -> std::io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).await?;
    let n = u32::from_le_bytes(len) as usize;
    let mut body = vec![0u8; n];
    if n > 0 {
        stream.read_exact(&mut body).await?;
    }
    Ok(body)
}

/// A loopback server that, on each accepted connection, reads the handshake
/// frame, parses it as a [`api::UdsHandshake`], sends it over `tx`, then shuts
/// the socket down. The shutdown makes the client's reader see EOF, which fires
/// the drop-notifier and drives the Connector's reconnect — so the next accepted
/// connection re-handshakes and we capture a SECOND handshake.
fn spawn_handshake_capture_server(path: PathBuf, tx: mpsc::UnboundedSender<api::UdsHandshake>) {
    tokio::spawn(async move {
        let listener = UnixListener::bind(&path).expect("bind capture uds");
        while let Ok((mut stream, _addr)) = listener.accept().await {
            let tx = tx.clone();
            tokio::spawn(async move {
                if let Ok(frame) = read_frame(&mut stream).await
                    && let Ok(handshake) = serde_json::from_slice::<api::UdsHandshake>(&frame)
                {
                    let _ = tx.send(handshake);
                }
                // Drop the connection so the client reconnects (#246).
                let _ = stream.shutdown().await;
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
    panic!("capture uds socket {path:?} did not appear");
}

#[tokio::test]
async fn system_id_is_sent_on_connect_and_resent_on_reconnect() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("capture.sock");
    let (tx, mut rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(path.clone(), tx);
    wait_for_socket(&path).await;

    // Inject a known system id + host label on the config. The Connector
    // respects a pre-set id (it only fills one in when absent), so the test is
    // hermetic — it doesn't depend on the host's /etc/machine-id.
    let config = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(path.clone()),
        ws_jwt: Some("test-token".to_string()),
        system_id: Some("machine-under-test".to_string()),
        host_label: Some("test-laptop".to_string()),
        ..ConnectionConfig::default()
    };

    // The first accepted connection is `wait_for_socket`'s probe (no handshake
    // written — it connects and drops), so it never reaches the handshake
    // parse. The Connector's real connect produces the first captured handshake;
    // the forced drop produces the second on reconnect.
    let _connector = Connector::connect(&config)
        .await
        .expect("connector connects");

    // First handshake (initial connect) must carry the id + label.
    let first = timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("first handshake captured")
        .expect("handshake present");
    assert_eq!(first.jwt.as_deref(), Some("test-token"));
    assert_eq!(first.system_id.as_deref(), Some("machine-under-test"));
    assert_eq!(first.host_label.as_deref(), Some("test-laptop"));

    // Second handshake (after the forced drop → reconnect) must re-send the SAME
    // id + label — the #248 field survives a reconnect (#246/#247).
    let second = timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("reconnect handshake captured (id must be re-sent)")
        .expect("handshake present");
    assert_eq!(
        second.system_id.as_deref(),
        Some("machine-under-test"),
        "the system id must be re-sent on reconnect, not dropped"
    );
    assert_eq!(second.host_label.as_deref(), Some("test-laptop"));
}

#[tokio::test]
async fn no_system_id_yields_legacy_handshake_shape() {
    // A config whose id is explicitly blank-out by a host with no /etc/machine-id
    // would normally fall back to a generated id; to test the *no-id* wire shape
    // deterministically we connect over the raw client with `None`/`None` and
    // assert the captured frame is the bare `{"jwt": "…"}` an older client sends.
    use desktop_assistant_client_common::uds_client::UdsClient;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("legacy.sock");
    let (tx, mut rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(path.clone(), tx);
    wait_for_socket(&path).await;

    let (_client, _signals, _drop) = UdsClient::connect(&path, Some("legacy-token"), None, None)
        .await
        .expect("raw uds connect");

    let frame = timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("handshake captured")
        .expect("handshake present");
    assert_eq!(frame.jwt.as_deref(), Some("legacy-token"));
    assert_eq!(frame.system_id, None, "no-id client must omit system_id");
    assert_eq!(frame.host_label, None);
}

#[tokio::test]
async fn peer_cred_handshake_omits_jwt() {
    // The local peer-cred path (#407) sends no bearer token: the daemon
    // authenticates the connection by its kernel `SO_PEERCRED`. Assert the
    // handshake frame omits `jwt` entirely so the wire shape is honest.
    use desktop_assistant_client_common::uds_client::UdsClient;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("peercred.sock");
    let (tx, mut rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(path.clone(), tx);
    wait_for_socket(&path).await;

    let (_client, _signals, _drop) = UdsClient::connect(&path, None, None, None)
        .await
        .expect("raw uds connect");

    let frame = timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("handshake captured")
        .expect("handshake present");
    assert_eq!(frame.jwt, None, "peer-cred client must omit jwt");
    assert_eq!(frame.system_id, None);
    assert_eq!(frame.host_label, None);
}
