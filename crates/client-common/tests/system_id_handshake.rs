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

    let (_client, _signals, _drop) =
        UdsClient::connect(&path, Some("legacy-token"), None, None, None, true)
            .await
            .expect("raw uds connect");

    let frame = timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("handshake captured")
        .expect("handshake present");
    assert_eq!(frame.jwt.as_deref(), Some("legacy-token"));
    assert_eq!(frame.system_id, None, "no-id client must omit system_id");
    assert_eq!(frame.host_label, None);
    assert_eq!(frame.client_context, None, "no-context client must omit it");
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

    let (_client, _signals, _drop) = UdsClient::connect(&path, None, None, None, None, true)
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

/// #549 Phase 2a: with the default-on `share_client_context` setting, the
/// Connector must attach the resolved client context to the handshake on the
/// initial connect AND re-attach it on reconnect. The expected value is the
/// gated resolution itself (dropped to `None` when the host resolves nothing),
/// so the assertion is deterministic and host-independent.
#[tokio::test]
async fn client_context_is_sent_on_connect_and_resent_on_reconnect_when_enabled() {
    use desktop_assistant_client_common::resolve_client_context;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ctx.sock");
    let (tx, mut rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(path.clone(), tx);
    wait_for_socket(&path).await;

    let config = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(path.clone()),
        ws_jwt: Some("test-token".to_string()),
        // Default is on, but set it explicitly so the test states its premise.
        share_client_context: true,
        ..ConnectionConfig::default()
    };

    // What the client is expected to send: the resolved context, or `None` when
    // nothing resolves (an empty context collapses to no field on the wire).
    let resolved = resolve_client_context();
    let expected = (!resolved.is_empty()).then_some(resolved);

    let _connector = Connector::connect(&config)
        .await
        .expect("connector connects");

    let first = timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("first handshake captured")
        .expect("handshake present");
    assert_eq!(
        first.client_context, expected,
        "the resolved client context must ride the initial handshake"
    );

    let second = timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("reconnect handshake captured")
        .expect("handshake present");
    assert_eq!(
        second.client_context, expected,
        "the client context must be re-sent on reconnect, not dropped"
    );
}

/// #549 Phase 2a: with the setting off, the client attaches no context at all —
/// the handshake `client_context` field is absent. Fully deterministic.
#[tokio::test]
async fn client_context_is_omitted_when_sharing_disabled() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("noctx.sock");
    let (tx, mut rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(path.clone(), tx);
    wait_for_socket(&path).await;

    let config = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(path.clone()),
        ws_jwt: Some("test-token".to_string()),
        share_client_context: false,
        ..ConnectionConfig::default()
    };

    let _connector = Connector::connect(&config)
        .await
        .expect("connector connects");

    let first = timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("first handshake captured")
        .expect("handshake present");
    assert_eq!(
        first.client_context, None,
        "sharing disabled must attach no client context"
    );
}

/// #783: the `share_client_context` setting being off is a refusal, and the
/// daemon cannot infer it from an absent context - a client that resolved
/// nothing sends an absent context too, and keeps its peer-identity grounding.
/// So the client states the refusal on the handshake, and re-states it on
/// reconnect. A sharing client states nothing, which keeps its handshake bytes
/// identical to the pre-#783 shape.
#[tokio::test]
async fn the_client_context_refusal_is_stated_on_connect_and_restated_on_reconnect() {
    let dir = TempDir::new().unwrap();

    let declining_path = dir.path().join("declining.sock");
    let (declining_tx, mut declining_rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(declining_path.clone(), declining_tx);
    wait_for_socket(&declining_path).await;

    let sharing_path = dir.path().join("sharing.sock");
    let (sharing_tx, mut sharing_rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(sharing_path.clone(), sharing_tx);
    wait_for_socket(&sharing_path).await;

    let base = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        ws_jwt: Some("test-token".to_string()),
        ..ConnectionConfig::default()
    };

    let _declining = Connector::connect(&ConnectionConfig {
        socket_path: Some(declining_path.clone()),
        share_client_context: false,
        ..base.clone()
    })
    .await
    .expect("connector connects");

    let first = timeout(Duration::from_secs(5), declining_rx.recv())
        .await
        .expect("first handshake captured")
        .expect("handshake present");
    assert_eq!(
        first.share_client_context,
        Some(false),
        "sharing disabled must state the refusal, not merely omit the context"
    );

    let second = timeout(Duration::from_secs(10), declining_rx.recv())
        .await
        .expect("reconnect handshake captured")
        .expect("handshake present");
    assert_eq!(
        second.share_client_context,
        Some(false),
        "the refusal must survive a reconnect, like every other handshake field"
    );

    let _sharing = Connector::connect(&ConnectionConfig {
        socket_path: Some(sharing_path.clone()),
        share_client_context: true,
        ..base
    })
    .await
    .expect("connector connects");

    let sharing_handshake = timeout(Duration::from_secs(5), sharing_rx.recv())
        .await
        .expect("sharing handshake captured")
        .expect("handshake present");
    assert_eq!(
        sharing_handshake.share_client_context, None,
        "a sharing client must send no declaration at all"
    );
}

#[tokio::test]
async fn a_caller_set_host_label_is_still_cleared_when_sharing_is_off() {
    // #782: `host_label` is the machine's hostname under another name - the same
    // `local_hostname()` that fills the client context's `hostname` - so it has
    // to go when the user withholds device context, or the fact leaves anyway.
    //
    // `stamp_system_id` otherwise respects a caller-supplied label. It does not
    // here: a privacy decision must not be overridable by a config field the
    // user cannot see. This is the only test that sets BOTH, so it is the only
    // one exercising the clearing branch; the others leave `host_label` unset,
    // where the pre-existing "fill it in if absent" path was already correct.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("labelled-optout.sock");
    let (tx, mut rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(path.clone(), tx);
    wait_for_socket(&path).await;

    let config = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(path.clone()),
        host_label: Some("a-machine-name".to_string()),
        share_client_context: false,
        ..ConnectionConfig::default()
    };
    let _connector = Connector::connect(&config)
        .await
        .expect("connector connects");

    let frame = timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("handshake captured")
        .expect("handshake present");
    assert_eq!(
        frame.host_label, None,
        "an explicitly-configured host label must not survive the opt-out"
    );
    assert!(
        frame.system_id.is_some(),
        "the per-machine co-location id is not the hostname and must still be \
         sent, or the daemon mis-routes tools for a user who only asked for privacy"
    );
}

#[tokio::test]
async fn a_caller_set_host_label_survives_when_sharing_is_on() {
    // The other half of the same branch: with sharing on, a caller's own label is
    // respected rather than overwritten by the resolved hostname. Without this, a
    // clearing bug that also broke the respect-the-caller path would look correct.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("labelled-optin.sock");
    let (tx, mut rx) = mpsc::unbounded_channel();
    spawn_handshake_capture_server(path.clone(), tx);
    wait_for_socket(&path).await;

    let config = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(path.clone()),
        host_label: Some("a-machine-name".to_string()),
        share_client_context: true,
        ..ConnectionConfig::default()
    };
    let _connector = Connector::connect(&config)
        .await
        .expect("connector connects");

    let frame = timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("handshake captured")
        .expect("handshake present");
    assert_eq!(frame.host_label.as_deref(), Some("a-machine-name"));
}
