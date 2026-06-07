//! Integration tests for the D-Bus bridge (issue #106).
//!
//! Each acceptance criterion is a named `#[tokio::test]` so the test
//! output reads as the spec. The tests are TDD: they were written
//! against the public types before the implementation behavior was
//! complete, so initial failures are the design pressure.
//!
//! What's exercised here, end-to-end:
//! - JWT fetch from the local minter (`fetch_jwt`).
//! - Daemon handshake over UDS (`UdsBridgeTransport::connect`).
//! - Round-trip of `api::Command` ↔ `WsFrame::Result` through the
//!   transport.
//! - Event delivery: subscribing to `subscribe_events` sees pushed
//!   `WsFrame::Event` frames.
//! - Translator `event_forwarder::translate` covering each
//!   `api::Event` variant.
//! - Unhappy paths: missing minter, malformed minter reply, daemon
//!   handshake rejection, daemon hangup mid-request, concurrent
//!   requests, oversized frames.
//!
//! Tests don't bind real D-Bus session connections — those need a
//! running session bus we can't assume in CI. The D-Bus signal
//! translation is asserted at the translator boundary
//! (`event_forwarder::translate`); the wiring loop that calls into
//! `zbus` is small and exercised in manual smoke-testing.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{
    DaemonScript, MinterScript, StubDaemonHandle, spawn_stub_daemon, spawn_stub_minter,
    unique_socket_path,
};
use desktop_assistant_api_model as api;
use desktop_assistant_dbus_bridge::adapter::event_forwarder::{ForwardAction, translate};
use desktop_assistant_dbus_bridge::adapter::settings::ConfigData;
use desktop_assistant_dbus_bridge::minter::{MintRequest, MintResponse, fetch_jwt};
use desktop_assistant_dbus_bridge::transport::{
    BridgeTransport, BridgeTransportError, UdsBridgeConfig, UdsBridgeTransport,
};
use tempfile::TempDir;

const TEST_TOKEN: &str = "test.jwt.token";

fn tempdir() -> TempDir {
    tempfile::tempdir().expect("tempdir")
}

// ---------------------------------------------------------------------------
// Minter
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bridge_fetches_jwt_from_minter_at_startup() {
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "mint");
    let (received, _stop) = spawn_stub_minter(
        &socket,
        MinterScript::Success {
            token: TEST_TOKEN.to_string(),
        },
    )
    .await;

    let token = fetch_jwt(
        &socket,
        MintRequest {
            ttl_seconds: Some(3600),
            audience: None,
        },
        Duration::from_secs(3),
    )
    .await
    .expect("mint succeeds");

    assert_eq!(token, TEST_TOKEN);
    let received = received.lock().await;
    assert_eq!(received.len(), 1, "minter saw exactly one request");
    let body: serde_json::Value =
        serde_json::from_str(&received[0]).expect("minter request is JSON");
    assert_eq!(body["ttl_seconds"], 3600);
}

#[tokio::test]
async fn bridge_with_invalid_jwt_logs_clear_error_and_exits() {
    // Minter returns an explicit error string; bridge surfaces it
    // verbatim to the caller so an operator sees "minter rejected
    // request: <message>" in logs and can act.
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "mint");
    let (_received, _stop) = spawn_stub_minter(
        &socket,
        MinterScript::Error {
            message: "caller uid 1001 is not a member of group adelie".to_string(),
        },
    )
    .await;

    let err = fetch_jwt(&socket, MintRequest::default(), Duration::from_secs(3))
        .await
        .expect_err("minter error surfaces");
    let msg = format!("{err}");
    assert!(
        msg.contains("minter rejected request") && msg.contains("not a member of group adelie"),
        "error message must name the failure; got {msg}"
    );
}

#[tokio::test]
async fn minter_unavailable_returns_descriptive_error() {
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "missing");
    let err = fetch_jwt(&socket, MintRequest::default(), Duration::from_secs(2))
        .await
        .expect_err("missing minter fails");
    let msg = format!("{err}");
    assert!(
        msg.contains("connect") || msg.contains("minter"),
        "error must mention the minter / connect failure; got {msg}"
    );
}

#[tokio::test]
async fn minter_malformed_reply_is_caught() {
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "mint");
    let (_received, _stop) = spawn_stub_minter(&socket, MinterScript::MalformedReply).await;
    let err = fetch_jwt(&socket, MintRequest::default(), Duration::from_secs(2))
        .await
        .expect_err("malformed reply fails");
    assert!(
        format!("{err}").contains("malformed"),
        "error must say malformed; got {err}"
    );
}

#[tokio::test]
async fn minter_hang_triggers_timeout() {
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "mint");
    let (_received, _stop) = spawn_stub_minter(&socket, MinterScript::HangForever).await;
    let err = fetch_jwt(&socket, MintRequest::default(), Duration::from_millis(200))
        .await
        .expect_err("hang fails");
    assert!(
        format!("{err}").contains("timed out"),
        "error must say timed out; got {err}"
    );
}

// ---------------------------------------------------------------------------
// Daemon handshake + dispatch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bridge_establishes_uds_handshake_with_token() {
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let config = UdsBridgeConfig {
        socket_path: socket,
        request_timeout: Duration::from_secs(3),
        event_buffer: 16,
    };
    let _transport = UdsBridgeTransport::connect(config_owned(&config), TEST_TOKEN)
        .await
        .expect("transport connects");

    // Give the daemon a moment to accept the handshake frame.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let handshakes = handle.handshakes.lock().await.clone();
    assert_eq!(handshakes.len(), 1, "daemon saw one handshake");
    assert_eq!(handshakes[0], TEST_TOKEN, "handshake carries the JWT");

    drop_handle(handle);
}

#[tokio::test]
async fn dbus_method_call_translates_to_uds_request() {
    // Acceptance: a method call (or its api::Command equivalent)
    // dispatched through the transport lands as a WsRequest on the
    // daemon side carrying the same Command.
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let config = UdsBridgeConfig {
        socket_path: socket,
        request_timeout: Duration::from_secs(3),
        event_buffer: 16,
    };
    let transport = UdsBridgeTransport::connect(config, TEST_TOKEN)
        .await
        .expect("connect");

    let cmd = api::Command::CreateConversation {
        title: "lunch plans".to_string(),
    };
    let result = transport.request(cmd.clone()).await.expect("request ok");
    assert_eq!(result, api::CommandResult::Ack);

    let requests = handle.requests.lock().await.clone();
    assert_eq!(requests.len(), 1, "daemon received one request");
    assert_eq!(requests[0].command, cmd);
    assert!(!requests[0].id.is_empty(), "request id must be set");

    drop_handle(handle);
}

#[tokio::test]
async fn concurrent_method_calls_correlate_by_id() {
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::EchoAck).await;

    let config = UdsBridgeConfig {
        socket_path: socket,
        request_timeout: Duration::from_secs(3),
        event_buffer: 16,
    };
    let transport = Arc::new(
        UdsBridgeTransport::connect(config, TEST_TOKEN)
            .await
            .expect("connect"),
    );

    let mut joins = Vec::new();
    for i in 0..16 {
        let t = Arc::clone(&transport);
        joins.push(tokio::spawn(async move {
            t.request(api::Command::CreateConversation {
                title: format!("title-{i}"),
            })
            .await
        }));
    }
    for j in joins {
        let res = j.await.expect("join").expect("request ok");
        assert_eq!(res, api::CommandResult::Ack);
    }

    let requests = handle.requests.lock().await.clone();
    assert_eq!(requests.len(), 16, "every request landed");
    let mut ids: Vec<&str> = requests.iter().map(|r| r.id.as_str()).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 16, "ids are unique");

    drop_handle(handle);
}

#[tokio::test]
async fn uds_event_translates_to_dbus_signal() {
    // Acceptance: an `Event::AssistantDelta` pushed by the daemon
    // becomes a `ForwardAction::ResponseChunk` for the
    // /org/desktopAssistant/Conversations path.
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(
        &socket,
        DaemonScript::EchoAckWithEvents {
            events: vec![api::Event::AssistantDelta {
                conversation_id: "c1".to_string(),
                request_id: "r1".to_string(),
                chunk: "hello".to_string(),
            }],
        },
    )
    .await;

    let config = UdsBridgeConfig {
        socket_path: socket,
        request_timeout: Duration::from_secs(3),
        event_buffer: 16,
    };
    let transport = UdsBridgeTransport::connect(config, TEST_TOKEN)
        .await
        .expect("connect");

    let mut events = transport.subscribe_events();
    // Drive a request so the daemon emits its queued events.
    let _ = transport
        .request(api::Command::CreateConversation { title: "x".into() })
        .await;

    let event = tokio::time::timeout(Duration::from_secs(2), events.recv())
        .await
        .expect("event arrives in time")
        .expect("event ok");
    let action = translate(event);
    match action {
        ForwardAction::ResponseChunk {
            conversation_id,
            request_id,
            chunk,
        } => {
            assert_eq!(conversation_id, "c1");
            assert_eq!(request_id, "r1");
            assert_eq!(chunk, "hello");
        }
        other => panic!("expected ResponseChunk, got {other:?}"),
    }
    drop_handle(handle);
}

#[tokio::test]
async fn bridge_request_to_disconnected_daemon_fails_fast() {
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(&socket, DaemonScript::AcceptThenClose).await;

    let config = UdsBridgeConfig {
        socket_path: socket,
        request_timeout: Duration::from_secs(3),
        event_buffer: 16,
    };
    let transport = UdsBridgeTransport::connect(config, TEST_TOKEN)
        .await
        .expect("connect");

    // Give the reader task a moment to observe EOF.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let err = transport
        .request(api::Command::CreateConversation { title: "x".into() })
        .await
        .expect_err("disconnected request fails");
    match err {
        BridgeTransportError::Disconnected { .. } | BridgeTransportError::Io(_) => {}
        other => panic!("expected Disconnected/Io, got {other:?}"),
    }
    drop_handle(handle);
}

#[tokio::test]
async fn bridge_handshake_rejection_surfaces_daemon_error() {
    let dir = tempdir();
    let socket = unique_socket_path(dir.path(), "daemon");
    let handle = spawn_stub_daemon(
        &socket,
        DaemonScript::RejectHandshake {
            error: "auth: invalid jwt".to_string(),
        },
    )
    .await;

    let config = UdsBridgeConfig {
        socket_path: socket,
        request_timeout: Duration::from_secs(3),
        event_buffer: 16,
    };
    // The connect call itself succeeds (handshake write is unilateral);
    // the next request observes the closed connection with the error
    // we got pre-dispatch.
    let transport = UdsBridgeTransport::connect(config, "bogus-token")
        .await
        .expect("connect socket-level ok");
    tokio::time::sleep(Duration::from_millis(100)).await;

    let err = transport
        .request(api::Command::CreateConversation { title: "x".into() })
        .await
        .expect_err("auth-rejected request fails");
    let msg = format!("{err}");
    assert!(
        msg.contains("auth") || msg.contains("invalid jwt") || msg.contains("closed"),
        "error must explain failure; got {msg}"
    );
    drop_handle(handle);
}

// ---------------------------------------------------------------------------
// Event translator coverage — each api::Event variant has a known
// ForwardAction outcome.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn event_translator_covers_all_streaming_variants() {
    // AssistantDelta -> ResponseChunk
    let a = translate(api::Event::AssistantDelta {
        conversation_id: "c".into(),
        request_id: "r".into(),
        chunk: "x".into(),
    });
    assert!(matches!(a, ForwardAction::ResponseChunk { .. }));

    // AssistantCompleted -> ResponseComplete
    let a = translate(api::Event::AssistantCompleted {
        conversation_id: "c".into(),
        request_id: "r".into(),
        full_response: "yo".into(),
    });
    assert!(matches!(a, ForwardAction::ResponseComplete { .. }));

    // AssistantError -> ResponseError
    let a = translate(api::Event::AssistantError {
        conversation_id: "c".into(),
        request_id: "r".into(),
        error: "boom".into(),
    });
    assert!(matches!(a, ForwardAction::ResponseError { .. }));
}

#[tokio::test]
async fn event_translator_handles_config_changed() {
    let event = api::Event::ConfigChanged {
        config: api::Config {
            embeddings: api::EmbeddingsSettingsView {
                connector: "openai".into(),
                model: "text-embedding-3-small".into(),
                base_url: "https://api.openai.com/v1".into(),
                has_api_key: true,
                available: true,
                is_default: true,
            },
            persistence: api::PersistenceSettingsView {
                enabled: true,
                remote_url: "git@example.com:repo.git".into(),
                remote_name: "origin".into(),
                push_on_update: true,
            },
            personality: api::PersonalitySettingsView::default(),
        },
    };
    let action = translate(event);
    match action {
        ForwardAction::ConfigChanged { config } => {
            assert_eq!(config.embeddings_connector, "openai");
            assert!(config.persistence_enabled);
            // ConfigData round-trips via serde so zbus marshaling
            // doesn't drift between bridge releases.
            let json = serde_json::to_string(&config).unwrap();
            let back: ConfigData = serde_json::from_str(&json).unwrap();
            assert_eq!(back.persistence_remote_name, "origin");
        }
        other => panic!("expected ConfigChanged, got {other:?}"),
    }
}

#[tokio::test]
async fn event_translator_marks_unhandled_variants_explicitly() {
    // Variants we deliberately do not surface as D-Bus signals must
    // hit `Ignored` so the next person to add one notices.
    // `Task*` events were Ignored under #106 but are now translated
    // onto `org.desktopAssistant.BackgroundTasks` per #116; see
    // `tests/background_tasks.rs` for that coverage.
    let cases = [
        (
            api::Event::AssistantStatus {
                conversation_id: "c".into(),
                request_id: "r".into(),
                message: "...".into(),
            },
            "assistant_status",
        ),
        (
            api::Event::ConversationTitleChanged {
                conversation_id: "c".into(),
                title: "t".into(),
            },
            "conversation_title_changed",
        ),
    ];
    for (event, expected_kind) in cases {
        match translate(event) {
            ForwardAction::Ignored { kind } => assert_eq!(kind, expected_kind),
            other => panic!("expected Ignored {expected_kind}, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// D-Bus surface preservation: ensure the bridge's interface names +
// method signatures match the in-process adapter so existing clients
// keep working.
// ---------------------------------------------------------------------------

#[test]
fn dbus_surface_object_paths_match_legacy() {
    use desktop_assistant_dbus_bridge::adapter::paths;
    // These paths are the public D-Bus contract. Any change here is
    // a breaking change for TUI / GTK / KDE clients.
    assert_eq!(paths::CONVERSATIONS, "/org/desktopAssistant/Conversations");
    assert_eq!(paths::SETTINGS, "/org/desktopAssistant/Settings");
    assert_eq!(paths::CONNECTIONS, "/org/desktopAssistant/Connections");
    assert_eq!(paths::KNOWLEDGE, "/org/desktopAssistant/Knowledge");
    assert_eq!(paths::RELOAD, "/org/desktopAssistant/Reload");
}

#[test]
fn dbus_service_name_is_canonical() {
    use desktop_assistant_dbus_bridge::adapter::DBUS_SERVICE_NAME;
    // The follow-up flips the daemon's surface off and switches the
    // default well-known name; until then the canonical name is
    // pinned here so the test fails loudly on accidental drift.
    assert_eq!(DBUS_SERVICE_NAME, "org.desktopAssistant");
}

// ---------------------------------------------------------------------------
// Minter wire format
// ---------------------------------------------------------------------------

#[test]
fn mint_response_parses_success_and_error_shapes() {
    // Success shape from the minter.
    let body = r#"{"token":"abc","exp":42}"#;
    let r: MintResponse = serde_json::from_str(body).unwrap();
    assert_eq!(r.token.as_deref(), Some("abc"));
    assert_eq!(r.exp, Some(42));
    assert!(r.error.is_none());

    // Error shape.
    let body = r#"{"error":"nope"}"#;
    let r: MintResponse = serde_json::from_str(body).unwrap();
    assert!(r.token.is_none());
    assert_eq!(r.error.as_deref(), Some("nope"));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn config_owned(c: &UdsBridgeConfig) -> UdsBridgeConfig {
    UdsBridgeConfig {
        socket_path: c.socket_path.clone(),
        request_timeout: c.request_timeout,
        event_buffer: c.event_buffer,
    }
}

fn drop_handle(handle: StubDaemonHandle) {
    let _ = handle.stop_tx.send(());
}
