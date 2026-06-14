//! Integration tests for the D-Bus bridge (issue #106, #316).
//!
//! Each acceptance criterion is a named `#[tokio::test]` so the output reads as
//! the spec. Since #316 the bridge talks to the daemon through the shared
//! client-common `Connector`; the bespoke UDS client (and its JWT/minter/framing
//! tests) is gone — that path is now covered in `client-common`. What's
//! exercised here:
//!
//! - End-to-end command dispatch through `ConnectorBridgeTransport` against a
//!   stub minter + stub daemon (the Connector mints, handshakes, and forwards).
//! - **Reconnect (#316):** the bridge survives a daemon restart, re-mints a
//!   fresh token, and resumes serving — the acceptance criterion for this step.
//! - The event translator (`event_forwarder::translate`) over each
//!   `SignalEvent` variant.
//! - The frozen D-Bus object paths + well-known name.
//!
//! Tests don't bind a real D-Bus session bus (none in CI); D-Bus signal
//! translation is asserted at the `translate` boundary, and the wiring loop that
//! calls `zbus::Connection::emit_signal` is exercised in manual smoke-testing.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{
    DaemonScript, MinterScript, StubDaemonHandle, spawn_stub_daemon, spawn_stub_minter,
    unique_socket_path,
};
use desktop_assistant_api_model as api;
use desktop_assistant_client_common::{ConnectionConfig, Connector, SignalEvent, TransportMode};
use desktop_assistant_dbus_bridge::adapter::event_forwarder::{ForwardAction, translate};
use desktop_assistant_dbus_bridge::transport::{BridgeTransport, ConnectorBridgeTransport};
use tempfile::TempDir;

const TEST_TOKEN: &str = "test.jwt.token";

fn tempdir() -> TempDir {
    tempfile::tempdir().expect("tempdir")
}

/// Build a bridge Connector pointed at the given daemon + minter sockets (UDS,
/// no static JWT — it mints from the stub minter on every (re)connect).
async fn connect_bridge(
    daemon_socket: std::path::PathBuf,
    minter_socket: std::path::PathBuf,
) -> Arc<Connector> {
    let config = ConnectionConfig {
        transport_mode: TransportMode::Uds,
        socket_path: Some(daemon_socket),
        minter_socket: Some(minter_socket),
        minter_ttl_seconds: Some(3600),
        ws_jwt: None,
        ..ConnectionConfig::default()
    };
    Arc::new(
        Connector::connect(&config)
            .await
            .expect("bridge Connector connects"),
    )
}

fn drop_handle(handle: StubDaemonHandle) {
    let _ = handle.stop_tx.send(());
}

// ---------------------------------------------------------------------------
// End-to-end command dispatch through the Connector
// ---------------------------------------------------------------------------

#[tokio::test]
async fn connector_request_reaches_daemon_and_returns_result() {
    let dir = tempdir();
    let daemon_socket = unique_socket_path(dir.path(), "daemon");
    let minter_socket = unique_socket_path(dir.path(), "mint");
    let (minted, _minter_stop) = spawn_stub_minter(
        &minter_socket,
        MinterScript::Success {
            token: TEST_TOKEN.to_string(),
        },
    )
    .await;
    let handle = spawn_stub_daemon(&daemon_socket, DaemonScript::EchoAck).await;

    let connector = connect_bridge(daemon_socket, minter_socket).await;
    let transport = ConnectorBridgeTransport::new(Arc::clone(&connector));

    let cmd = api::Command::CreateConversation {
        title: "lunch plans".to_string(),
    };
    let result = transport.request(cmd.clone()).await.expect("request ok");
    assert_eq!(result, api::CommandResult::Ack);

    // The daemon saw the command verbatim...
    let requests = handle.requests.lock().await.clone();
    assert!(
        requests.iter().any(|r| r.command == cmd),
        "the daemon must receive the dispatched command; saw {requests:?}"
    );
    // ...and the Connector minted a token to authenticate.
    assert_eq!(minted.lock().await.len(), 1, "exactly one mint at connect");

    drop_handle(handle);
}

#[tokio::test]
async fn bridge_survives_daemon_restart_and_resumes() {
    // The #316 acceptance test: a daemon restart drops the bridge's connection;
    // the Connector must reconnect, *re-mint a fresh token*, and resume serving
    // — so KDE stays live without restarting the bridge.
    let dir = tempdir();
    let daemon_socket = unique_socket_path(dir.path(), "daemon");
    let minter_socket = unique_socket_path(dir.path(), "mint");
    let (minted, _minter_stop) = spawn_stub_minter(
        &minter_socket,
        MinterScript::Success {
            token: TEST_TOKEN.to_string(),
        },
    )
    .await;

    // First daemon instance — bridge connects and serves.
    let daemon1 = spawn_stub_daemon(&daemon_socket, DaemonScript::EchoAck).await;
    let connector = connect_bridge(daemon_socket.clone(), minter_socket.clone()).await;
    let transport = ConnectorBridgeTransport::new(Arc::clone(&connector));
    assert_eq!(
        transport
            .request(api::Command::Ping)
            .await
            .expect("ping ok"),
        api::CommandResult::Ack,
        "the bridge serves before the restart"
    );
    let mints_before = minted.lock().await.len();
    assert_eq!(mints_before, 1, "one mint at the initial connect");

    // Restart: kill daemon1 (closes the connection → Connector sees the drop),
    // let the socket free, then bring a fresh daemon up on the same path.
    drop_handle(daemon1);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let daemon2 = spawn_stub_daemon(&daemon_socket, DaemonScript::EchoAck).await;

    // The bridge resumes serving once the Connector reconnects (bounded retry —
    // reconnect backs off, so poll rather than assume the first call lands).
    let mut resumed = false;
    for _ in 0..50 {
        if let Ok(Ok(api::CommandResult::Ack)) = tokio::time::timeout(
            Duration::from_millis(500),
            transport.request(api::Command::Ping),
        )
        .await
        {
            resumed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        resumed,
        "the bridge must resume serving methods after a daemon restart"
    );

    // And it minted a *fresh* token to re-authenticate — the core #316 fix
    // (a static token would have stranded it on an expired credential).
    assert!(
        minted.lock().await.len() > mints_before,
        "reconnect must re-mint a token (saw {} mints, started at {mints_before})",
        minted.lock().await.len()
    );

    drop_handle(daemon2);
}

// ---------------------------------------------------------------------------
// Event translator coverage — each SignalEvent variant → known ForwardAction
// ---------------------------------------------------------------------------

#[test]
fn translate_covers_streaming_response_variants() {
    assert!(matches!(
        translate(SignalEvent::Chunk {
            conversation_id: "c".into(),
            request_id: "r".into(),
            chunk: "x".into(),
        }),
        ForwardAction::ResponseChunk { .. }
    ));
    assert!(matches!(
        translate(SignalEvent::Complete {
            conversation_id: "c".into(),
            request_id: "r".into(),
            full_response: "yo".into(),
        }),
        ForwardAction::ResponseComplete { .. }
    ));
    assert!(matches!(
        translate(SignalEvent::Error {
            conversation_id: "c".into(),
            request_id: "r".into(),
            error: "boom".into(),
        }),
        ForwardAction::ResponseError { .. }
    ));
}

#[test]
fn translate_marks_unforwarded_variants_explicitly() {
    // Variants with no D-Bus signal must hit `Ignored` with a stable kind, so a
    // future parity step is deliberate rather than silent drift. (UserMessageAdded
    // + ConversationListChanged ARE now forwarded — see the next test. Task*
    // variants are forwarded and covered in `tests/background_tasks.rs`.)
    let cases: Vec<(SignalEvent, &str)> = vec![
        (
            SignalEvent::Status {
                conversation_id: "c".into(),
                request_id: "r".into(),
                message: "...".into(),
            },
            "assistant_status",
        ),
        (
            SignalEvent::TitleChanged {
                conversation_id: "c".into(),
                title: "t".into(),
            },
            "conversation_title_changed",
        ),
        (
            SignalEvent::ContextUsage {
                conversation_id: "c".into(),
                request_id: "r".into(),
                used_tokens: 12_000,
                budget_tokens: 32_000,
                compaction_active: false,
            },
            "context_usage",
        ),
    ];
    for (event, expected_kind) in cases {
        match translate(event) {
            ForwardAction::Ignored { kind } => assert_eq!(kind, expected_kind),
            other => panic!("expected Ignored {expected_kind}, got {other:?}"),
        }
    }
}

#[test]
fn translate_forwards_user_message_added_and_conversation_list_changed() {
    // #367: both are now real D-Bus signals on Conversations, carrying their
    // payload through verbatim.
    assert!(matches!(
        translate(SignalEvent::UserMessageAdded {
            conversation_id: "c".into(),
            request_id: "r".into(),
            content: "hi".into(),
        }),
        ForwardAction::UserMessageAdded { conversation_id, request_id, content }
            if conversation_id == "c" && request_id == "r" && content == "hi"
    ));
    assert!(matches!(
        translate(SignalEvent::ConversationListChanged {
            conversation_id: "c".into(),
        }),
        ForwardAction::ConversationListChanged { conversation_id } if conversation_id == "c"
    ));
}

// ---------------------------------------------------------------------------
// Frozen D-Bus surface (object paths + well-known name)
// ---------------------------------------------------------------------------

#[test]
fn dbus_surface_object_paths_match_legacy() {
    use desktop_assistant_dbus_bridge::adapter::paths;
    // These paths are the public D-Bus contract; any change breaks tui/gtk/KDE.
    assert_eq!(paths::COMMANDS, "/org/desktopAssistant/Commands");
    assert_eq!(paths::CONVERSATIONS, "/org/desktopAssistant/Conversations");
    assert_eq!(paths::SETTINGS, "/org/desktopAssistant/Settings");
    assert_eq!(paths::CONNECTIONS, "/org/desktopAssistant/Connections");
    assert_eq!(paths::KNOWLEDGE, "/org/desktopAssistant/Knowledge");
    assert_eq!(paths::RELOAD, "/org/desktopAssistant/Reload");
}

#[test]
fn dbus_service_name_is_canonical() {
    use desktop_assistant_dbus_bridge::adapter::DBUS_SERVICE_NAME;
    assert_eq!(DBUS_SERVICE_NAME, "org.desktopAssistant");
}
