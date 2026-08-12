//! The client-context privacy preference over D-Bus (#782 / #778).
//!
//! The KDE "Share device info" control is only real if turning it off stops the
//! daemon from receiving the user's identity. Every test here therefore asserts
//! at the **daemon** - the stub daemon's recorded handshake frame - and never at
//! the client. A test that proves the client sent the preference is exactly what
//! makes this class of defect look fixed when it is not.
//!
//! The path under test is the production one: the `Commands` interface method
//! body records the caller's declaration, the [`SessionRegistry`] resolves it,
//! and the real [`ConnectorSessionFactory`] builds that sender's daemon session
//! from it. Only zbus's own header extraction is bypassed (the adapter's
//! `declare_client_context_sharing` core takes the caller directly, exactly as
//! `run_command` does), because a p2p connection carries no sender name.

mod common;

use std::sync::{Arc, OnceLock};

use common::{DaemonScript, socket_tempdir, spawn_stub_daemon, unique_socket_path};
use desktop_assistant_api_model as api;
use desktop_assistant_dbus_bridge::adapter::DbusCommandsAdapter;
use desktop_assistant_dbus_bridge::session::{ConnectorSessionFactory, SessionRegistry};
use desktop_assistant_dbus_bridge::transport::{
    BridgeTransport, BridgeTransportError, ConnectorBridgeTransport, bridge_daemon_config,
};
use serde_json::Value;
use tokio::net::UnixStream;

const SENDER: &str = ":1.7";

/// A transport that must never be reached: every test here routes through a
/// per-sender session, so a command landing on the shared fallback is a routing
/// bug, not a passing test.
struct UnusedTransport;

#[async_trait::async_trait]
impl BridgeTransport for UnusedTransport {
    async fn request(
        &self,
        _command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError> {
        panic!("the shared fallback transport must not be used for a session-pinned command");
    }
}

/// A live `zbus::Connection` with no bus daemon, for the per-sender unicast
/// forwarder to emit on. The forwarder's own delivery is not what these tests
/// assert; the session must simply be constructible the production way.
async fn p2p_connection() -> zbus::Connection {
    let (s0, s1) = UnixStream::pair().expect("unix stream pair");
    let guid = zbus::Guid::generate();
    let server = zbus::connection::Builder::unix_stream(s0)
        .server(guid)
        .expect("server builder")
        .p2p();
    let client = zbus::connection::Builder::unix_stream(s1).p2p();
    let (server, client) = futures_util::try_join!(server.build(), client.build())
        .expect("p2p connection pair builds");
    // The client end must outlive the server end or the server tears down.
    Box::leak(Box::new(client));
    server
}

/// Stand the production wiring up against a stub daemon at `socket`.
async fn wiring(
    socket: std::path::PathBuf,
) -> (Arc<SessionRegistry>, DbusCommandsAdapter<UnusedTransport>) {
    let connection: Arc<OnceLock<zbus::Connection>> = Arc::new(OnceLock::new());
    let _ = connection.set(p2p_connection().await);
    let factory = Arc::new(ConnectorSessionFactory::new(
        bridge_daemon_config(socket),
        connection,
    ));
    let registry = Arc::new(SessionRegistry::new(factory));
    let adapter =
        DbusCommandsAdapter::new(Arc::new(UnusedTransport)).with_sessions(Arc::clone(&registry));
    (registry, adapter)
}

/// Open `sender`'s daemon session and drive one round trip on it, so the daemon
/// has certainly read the handshake before the test asserts on it. Opening the
/// socket only proves the client wrote; the reply proves the daemon read.
async fn open_and_settle(registry: &SessionRegistry, sender: &str) {
    let session = registry
        .session_for(sender)
        .await
        .expect("the sender's daemon session opens");
    session
        .request(api::Command::Ping)
        .await
        .expect("the session round-trips a command");
}

/// The handshake frame the daemon read for the most recent session.
fn daemon_handshake(frames: &[Value]) -> Value {
    frames
        .last()
        .cloned()
        .expect("the daemon must have read a handshake")
}

#[tokio::test]
async fn declared_withhold_reaches_the_daemon_as_an_explicit_refusal() {
    let dir = socket_tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let (registry, adapter) = wiring(socket).await;
    adapter
        .declare_client_context_sharing(Some(SENDER), false)
        .expect("declaring the preference succeeds");
    open_and_settle(&registry, SENDER).await;

    let frames = handle.handshake_frames.lock().await.clone();
    let handshake = daemon_handshake(&frames);
    assert_eq!(
        handshake.get("share_client_context"),
        Some(&Value::Bool(false)),
        "the daemon must be told the client refuses; saw {handshake}"
    );
    assert!(
        handshake.get("client_context").is_none(),
        "a refusing client must send no context at all; saw {handshake}"
    );
    assert!(
        handshake.get("host_label").is_none(),
        "`host_label` is the hostname under another name and must go too; saw {handshake}"
    );
    assert!(
        handshake.get("system_id").is_some(),
        "the opaque co-location id names nobody and must survive, or tools mis-route; \
         saw {handshake}"
    );

    let _ = handle.stop_tx.send(());
}

#[tokio::test]
async fn an_undeclared_caller_reaches_the_daemon_as_an_explicit_refusal() {
    // Fail-closed: a D-Bus caller that never declared - including one too old to
    // know the method - must not be treated as consenting.
    let dir = socket_tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let (registry, _adapter) = wiring(socket).await;
    open_and_settle(&registry, SENDER).await;

    let frames = handle.handshake_frames.lock().await.clone();
    let handshake = daemon_handshake(&frames);
    assert_eq!(
        handshake.get("share_client_context"),
        Some(&Value::Bool(false)),
        "an undeclared caller must reach the daemon as a refusal; saw {handshake}"
    );
    assert!(
        handshake.get("client_context").is_none(),
        "an undeclared caller must send no context; saw {handshake}"
    );

    let _ = handle.stop_tx.send(());
}

#[tokio::test]
async fn declared_sharing_reaches_the_daemon_as_a_context_and_no_refusal() {
    let dir = socket_tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let (registry, adapter) = wiring(socket).await;
    adapter
        .declare_client_context_sharing(Some(SENDER), true)
        .expect("declaring the preference succeeds");
    open_and_settle(&registry, SENDER).await;

    let frames = handle.handshake_frames.lock().await.clone();
    let handshake = daemon_handshake(&frames);
    // #783 states only the REFUSAL on the wire: a sharing client omits the field
    // entirely, keeping its handshake byte-identical to the pre-#783 shape. So
    // consent is "a context arrived and no refusal did", not a `true`.
    assert_ne!(
        handshake.get("share_client_context"),
        Some(&Value::Bool(false)),
        "a consenting caller must not reach the daemon as a refusal; saw {handshake}"
    );
    assert!(
        handshake
            .get("client_context")
            .and_then(Value::as_object)
            .is_some_and(|c| !c.is_empty()),
        "a consenting caller must carry a non-empty context; saw {handshake}"
    );
    assert!(
        handshake.get("host_label").is_some(),
        "a consenting caller still labels its host for the tool note; saw {handshake}"
    );

    let _ = handle.stop_tx.send(());
}

#[tokio::test]
async fn changing_the_declaration_rebuilds_the_session_so_the_daemon_sees_the_new_state() {
    // A user who turns the control off after a session is already open must not
    // keep the consenting session. The declaration change drops it, and the next
    // session-scoped call opens a refusing one.
    let dir = socket_tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let (registry, adapter) = wiring(socket).await;
    adapter
        .declare_client_context_sharing(Some(SENDER), true)
        .expect("declaring the preference succeeds");
    open_and_settle(&registry, SENDER).await;

    adapter
        .declare_client_context_sharing(Some(SENDER), false)
        .expect("re-declaring the preference succeeds");
    open_and_settle(&registry, SENDER).await;

    let frames = handle.handshake_frames.lock().await.clone();
    assert_eq!(frames.len(), 2, "the session must be rebuilt, not reused");
    let handshake = daemon_handshake(&frames);
    assert_eq!(
        handshake.get("share_client_context"),
        Some(&Value::Bool(false)),
        "the daemon must see the new refusal; saw {handshake}"
    );

    let _ = handle.stop_tx.send(());
}

#[tokio::test]
async fn a_turn_driven_by_a_withholding_caller_reaches_the_daemon_with_no_context() {
    // The whole control, end to end: the caller declares, then drives a TURN the
    // way `SendPrompt` does - through `SessionRegistry::route`, which is what
    // decides a turn runs on the caller's own daemon session. The daemon must see
    // that session's handshake refuse.
    let dir = socket_tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let (registry, adapter) = wiring(socket).await;
    adapter
        .declare_client_context_sharing(Some(SENDER), false)
        .expect("declaring the preference succeeds");

    registry
        .route(
            Some(SENDER),
            api::Command::SendMessage {
                conversation_id: "c1".to_string(),
                content: "hello".to_string(),
                override_selection: None,
                system_refinement: String::new(),
                // The D-Bus bridge never originates a per-turn context (#557 is
                // for the browser-multiplexed web BFF), so the connection's
                // preference is the only thing that decides this turn.
                client_context: None,
                idempotency_key: None,
                turn_id: None,
                traceparent: None,
            },
            &UnusedTransport,
        )
        .await
        .expect("the turn routes to the caller's own session");

    let frames = handle.handshake_frames.lock().await.clone();
    let handshake = daemon_handshake(&frames);
    assert_eq!(
        handshake.get("share_client_context"),
        Some(&Value::Bool(false)),
        "the session a turn runs on must refuse; saw {handshake}"
    );
    assert!(
        handshake.get("client_context").is_none(),
        "the session a turn runs on must carry no context; saw {handshake}"
    );

    let _ = handle.stop_tx.send(());
}

#[tokio::test]
async fn the_bridges_own_shared_session_withholds_the_client_context() {
    // The bridge's shared connection serves stateless commands on behalf of no
    // declared caller. It runs as the user, on the user's machine, so resolving
    // its own environment would hand the daemon the very facts the control is
    // meant to withhold - by a route no caller ever consented to.
    let dir = socket_tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let config = bridge_daemon_config(socket);
    let connector = Arc::new(
        desktop_assistant_client_common::Connector::connect(&config)
            .await
            .expect("the bridge's shared connection opens"),
    );
    ConnectorBridgeTransport::new(connector)
        .request(api::Command::Ping)
        .await
        .expect("the shared connection round-trips a command");

    let frames = handle.handshake_frames.lock().await.clone();
    let handshake = daemon_handshake(&frames);
    assert_eq!(
        handshake.get("share_client_context"),
        Some(&Value::Bool(false)),
        "the bridge's own session must refuse; saw {handshake}"
    );
    assert!(
        handshake.get("client_context").is_none(),
        "the bridge must not resolve the user's environment for its shared session; saw {handshake}"
    );
    assert!(
        handshake.get("host_label").is_none(),
        "`host_label` is the machine's hostname by another name, and must go with the \
         context; saw {handshake}"
    );

    let _ = handle.stop_tx.send(());
}
