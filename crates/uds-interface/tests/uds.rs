//! Behavioral tests for the UDS frontend.
//!
//! These wire the same `AssistantApiHandler` test doubles used by the WS
//! adapter into the UDS listener, mint a JWT with `auth-jwt`, and drive
//! commands over a local Unix socket. The framing under test is the same
//! one production clients (D-Bus bridge, CLI) will speak.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_uds::{
    UdsAuthValidator, UdsServer, UdsServerConfig, read_frame, write_frame,
};
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::time::{Duration, timeout};

const ISS: &str = "test-uds-iss";
const AUD: &str = "test-uds-aud";

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn mint_test_jwt(signing_key: &str, subject: &str) -> String {
    let now = unix_now();
    let claims = jwt::Claims {
        iss: ISS.into(),
        sub: subject.into(),
        aud: AUD.into(),
        exp: now + 600,
        iat: now,
        nbf: now.saturating_sub(1),
        jti: uuid::Uuid::new_v4().to_string(),
    };
    jwt::encode(&claims, signing_key).expect("encode jwt")
}

struct PingHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for PingHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::Ping => Ok(api::CommandResult::Pong {
                value: "pong".into(),
            }),
            _ => Err(ApiError::Unsupported),
        }
    }

    async fn handle_send_message(
        &self,
        _conversation_id: String,
        _content: String,
        _request_id: String,
        _sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        Ok(())
    }
}

/// JWT auth validator backed by an in-memory signing key.
struct StaticJwtAuth {
    signing_key: String,
}

#[async_trait::async_trait]
impl UdsAuthValidator for StaticJwtAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        jwt::decode(token, &self.signing_key, ISS, AUD).is_ok()
    }
}

fn socket_path(dir: &TempDir) -> PathBuf {
    dir.path().join("adelie.sock")
}

fn start_server(
    socket_path: PathBuf,
    signing_key: String,
) -> (
    Arc<dyn AssistantApiHandler>,
    tokio::task::JoinHandle<anyhow::Result<()>>,
    tokio::sync::oneshot::Sender<()>,
) {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth { signing_key });
    let config = UdsServerConfig::new(socket_path.clone());
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(Arc::clone(&handler), auth, config);
    let join = tokio::spawn(async move {
        server
            .serve_with_shutdown(async move {
                let _ = rx.await;
            })
            .await
    });
    (handler, join, tx)
}

async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..100 {
        if path.exists() {
            // The path might be a stale regular file the listener
            // hasn't yet replaced. Confirm we can actually connect.
            if UnixStream::connect(path).await.is_ok() {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("uds socket {path:?} did not appear");
}

#[tokio::test]
async fn dispatcher_serves_uds_command_through_neutral_handler() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);
    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    let token = mint_test_jwt(&signing_key, "alice");
    let handshake = serde_json::json!({ "jwt": token });
    write_frame(&mut stream, &serde_json::to_vec(&handshake).unwrap())
        .await
        .unwrap();

    let req = api::WsRequest {
        id: "1".into(),
        command: api::Command::Ping,
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).unwrap())
        .await
        .unwrap();

    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .expect("no response within 2s")
        .expect("io error on read_frame");
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    match frame {
        api::WsFrame::Result { id, result } => {
            assert_eq!(id, "1");
            assert_eq!(
                result,
                api::CommandResult::Pong {
                    value: "pong".into()
                }
            );
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    let _ = shutdown.send(());
}

#[tokio::test]
async fn uds_connection_without_jwt_is_rejected() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);
    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key);
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    // Send a handshake frame missing the jwt field.
    let bad_handshake = serde_json::json!({ "hello": "world" });
    write_frame(&mut stream, &serde_json::to_vec(&bad_handshake).unwrap())
        .await
        .unwrap();

    // Server should emit an error frame and close.
    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .expect("expected error frame")
        .expect("expected error frame");
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    match frame {
        api::WsFrame::Error { error, .. } => {
            assert!(error.to_lowercase().contains("auth"));
        }
        other => panic!("expected Error frame, got {other:?}"),
    }

    // Subsequent reads should return EOF / 0 bytes (socket closed).
    let next = read_frame(&mut stream).await;
    assert!(
        next.is_err() || next.as_ref().map(|b| b.is_empty()).unwrap_or(false),
        "expected socket close after auth rejection, got {next:?}"
    );

    let _ = shutdown.send(());
}

#[tokio::test]
async fn uds_connection_with_invalid_jwt_is_rejected() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);
    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key);
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    let handshake = serde_json::json!({ "jwt": "not-a-real-jwt" });
    write_frame(&mut stream, &serde_json::to_vec(&handshake).unwrap())
        .await
        .unwrap();

    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .expect("expected error frame")
        .expect("expected error frame");
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    assert!(matches!(frame, api::WsFrame::Error { .. }));

    let _ = shutdown.send(());
}

#[tokio::test]
async fn uds_connection_with_valid_jwt_proceeds() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);
    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    let token = mint_test_jwt(&signing_key, "dave");
    let handshake = serde_json::json!({ "jwt": token });
    write_frame(&mut stream, &serde_json::to_vec(&handshake).unwrap())
        .await
        .unwrap();

    let req = api::WsRequest {
        id: "ping-1".into(),
        command: api::Command::Ping,
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).unwrap())
        .await
        .unwrap();

    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .unwrap()
        .unwrap();
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    match frame {
        api::WsFrame::Result {
            id,
            result: api::CommandResult::Pong { value },
        } => {
            assert_eq!(id, "ping-1");
            assert_eq!(value, "pong");
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    let _ = shutdown.send(());
}

#[tokio::test]
async fn uds_framing_handles_partial_reads() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);
    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    let token = mint_test_jwt(&signing_key, "dave");
    let handshake = serde_json::to_vec(&serde_json::json!({ "jwt": token })).unwrap();
    write_frame(&mut stream, &handshake).await.unwrap();

    // Write a Ping request in three chunks.
    let req = api::WsRequest {
        id: "split".into(),
        command: api::Command::Ping,
    };
    let body = serde_json::to_vec(&req).unwrap();
    let len = body.len() as u32;
    let mut frame = Vec::with_capacity(4 + body.len());
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(&body);

    // Write 2 bytes, sleep, 2 bytes, sleep, the rest.
    stream.write_all(&frame[..2]).await.unwrap();
    stream.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    stream.write_all(&frame[2..4]).await.unwrap();
    stream.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    stream.write_all(&frame[4..]).await.unwrap();
    stream.flush().await.unwrap();

    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .unwrap()
        .unwrap();
    let response: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    match response {
        api::WsFrame::Result { id, result } => {
            assert_eq!(id, "split");
            assert_eq!(
                result,
                api::CommandResult::Pong {
                    value: "pong".into()
                }
            );
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    let _ = shutdown.send(());
}

#[tokio::test]
async fn uds_socket_file_is_removed_on_shutdown() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);
    let (_handler, join, shutdown) = start_server(path.clone(), signing_key);
    wait_for_socket(&path).await;
    assert!(path.exists());

    let _ = shutdown.send(());
    // Allow the listener to clean up.
    let _ = timeout(Duration::from_secs(2), join).await;

    assert!(
        !path.exists(),
        "socket file should be removed on graceful shutdown"
    );
}

#[tokio::test]
async fn uds_listener_removes_existing_socket_on_startup() {
    // Behavior choice: the listener unlinks an existing socket file on
    // bind so the daemon can restart cleanly. (The alternative —
    // refusing to start — pushes a stale-socket recovery problem onto
    // operators after an unclean shutdown.) This test pins that choice.
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);

    // Pre-create a stale file at the socket path.
    std::fs::write(&path, b"stale").unwrap();
    assert!(path.exists());

    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    // The listener should have replaced the stale file with a live socket.
    let mut stream = UnixStream::connect(&path).await.unwrap();
    let token = mint_test_jwt(&signing_key, "dave");
    write_frame(
        &mut stream,
        &serde_json::to_vec(&serde_json::json!({ "jwt": token })).unwrap(),
    )
    .await
    .unwrap();
    let req = api::WsRequest {
        id: "p".into(),
        command: api::Command::Ping,
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).unwrap())
        .await
        .unwrap();
    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .unwrap()
        .unwrap();
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    assert!(matches!(frame, api::WsFrame::Result { .. }));

    let _ = shutdown.send(());
}

#[tokio::test]
async fn same_assistant_api_handler_services_both_transports() {
    // Wire ONE handler instance; drive a Ping over UDS and a Ping over
    // WS (via the dispatcher directly, since spinning a full axum server
    // here would duplicate the WS test coverage). Both must yield the
    // same Pong without per-transport branching.
    use desktop_assistant_transport_dispatch::{AuthContext, dispatch_loop};
    use futures::stream;

    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);

    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);

    // UDS leg.
    let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth {
        signing_key: signing_key.clone(),
    });
    let config = UdsServerConfig::new(path.clone());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(Arc::clone(&handler), auth, config);
    let join = tokio::spawn(async move {
        server
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    let token = mint_test_jwt(&signing_key, "dave");
    write_frame(
        &mut stream,
        &serde_json::to_vec(&serde_json::json!({ "jwt": token })).unwrap(),
    )
    .await
    .unwrap();
    let req = api::WsRequest {
        id: "uds-ping".into(),
        command: api::Command::Ping,
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).unwrap())
        .await
        .unwrap();
    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .unwrap()
        .unwrap();
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    let uds_result = match frame {
        api::WsFrame::Result { id, result } => {
            assert_eq!(id, "uds-ping");
            result
        }
        other => panic!("unexpected frame: {other:?}"),
    };

    // Dispatcher leg (in-process; the WS adapter is a thin wrapper
    // over this).
    use futures::channel::mpsc;
    let req = api::WsRequest {
        id: "ws-ping".into(),
        command: api::Command::Ping,
    };
    let inbound = stream::iter(vec![Ok::<_, anyhow::Error>(req)]);
    let (out_tx, mut out_rx) = mpsc::channel::<api::WsFrame>(8);
    let dispatcher_handler = Arc::clone(&handler);
    let dispatcher_join = tokio::spawn(dispatch_loop(
        dispatcher_handler,
        AuthContext::anonymous(),
        inbound,
        out_tx,
    ));
    use futures::StreamExt;
    let frame = timeout(Duration::from_secs(2), out_rx.next())
        .await
        .unwrap()
        .unwrap();
    let ws_result = match frame {
        api::WsFrame::Result { id, result } => {
            assert_eq!(id, "ws-ping");
            result
        }
        other => panic!("unexpected frame: {other:?}"),
    };

    assert_eq!(uds_result, ws_result);

    let _ = shutdown_tx.send(());
    let _ = timeout(Duration::from_secs(2), join).await;
    dispatcher_join.abort();
}

// ---------------------------------------------------------------------
// Code review 2026-06-09 — protocol robustness (DT-5, DT-6, DT-7, DT-13)
// ---------------------------------------------------------------------

/// DT-7: a client that connects and never sends its JWT handshake must be
/// disconnected after the handshake timeout instead of pinning a connection
/// task forever.
#[tokio::test]
async fn handshake_times_out_when_client_sends_nothing() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);

    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth { signing_key });
    let config =
        UdsServerConfig::new(path.clone()).with_handshake_timeout(Duration::from_millis(200));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(handler, auth, config);
    let _join = tokio::spawn(async move {
        server
            .serve_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    // Send nothing. The server must close the connection: the next read
    // observes EOF (or an error frame followed by EOF) well before our
    // 2-second ceiling.
    let result = timeout(Duration::from_secs(2), read_frame(&mut stream)).await;
    match result {
        Ok(Err(_)) | Ok(Ok(_)) => {
            // Error frame or EOF-as-io-error: either way the server acted.
            // If we got a frame, the connection must close right after.
            let followup = timeout(Duration::from_secs(2), read_frame(&mut stream))
                .await
                .expect("server must close a silent connection after the handshake timeout");
            assert!(
                followup.is_err() || followup.map(|b| b.is_empty()).unwrap_or(false),
                "expected EOF after the handshake timeout"
            );
        }
        Err(_) => panic!("server kept a silent connection open past the handshake timeout"),
    }

    let _ = shutdown_tx.send(());
}

/// DT-5: a syntactically invalid request frame (post-handshake) must yield
/// an explicit error frame with an empty id, and the connection must keep
/// serving afterwards.
#[tokio::test]
async fn invalid_request_json_gets_error_frame_with_empty_id() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);
    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    let token = mint_test_jwt(&signing_key, "dave");
    write_frame(
        &mut stream,
        &serde_json::to_vec(&serde_json::json!({ "jwt": token })).unwrap(),
    )
    .await
    .unwrap();

    // A frame that is not valid WsRequest JSON.
    write_frame(&mut stream, b"{this is not json")
        .await
        .unwrap();

    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .expect("expected an error frame for malformed JSON, got silence")
        .expect("io error reading the error frame");
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    match frame {
        api::WsFrame::Error { id, error } => {
            assert_eq!(id, "", "no request id is known for a malformed frame");
            assert!(
                error.to_lowercase().contains("json") || error.to_lowercase().contains("invalid"),
                "error should describe the parse failure: {error}"
            );
        }
        other => panic!("expected an Error frame, got {other:?}"),
    }

    // The connection survives: a valid request still round-trips.
    let req = api::WsRequest {
        id: "after".into(),
        command: api::Command::Ping,
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).unwrap())
        .await
        .unwrap();
    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .expect("connection must keep serving after a malformed frame")
        .unwrap();
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    assert!(matches!(frame, api::WsFrame::Result { .. }));

    let _ = shutdown.send(());
}

/// DT-6: replies already queued when the client half-closes its write side
/// must still be delivered — the old `writer_task.abort()` could drop (or
/// tear mid-write) final outbound frames.
#[tokio::test]
async fn queued_replies_drain_after_client_half_close() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = socket_path(&dir);
    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    let token = mint_test_jwt(&signing_key, "dave");
    write_frame(
        &mut stream,
        &serde_json::to_vec(&serde_json::json!({ "jwt": token })).unwrap(),
    )
    .await
    .unwrap();

    const N: usize = 30;
    for i in 0..N {
        let req = api::WsRequest {
            id: format!("req-{i}"),
            command: api::Command::Ping,
        };
        write_frame(&mut stream, &serde_json::to_vec(&req).unwrap())
            .await
            .unwrap();
    }
    // Half-close: we are done sending but still want all replies.
    stream.shutdown().await.unwrap();

    let mut results = 0usize;
    loop {
        match timeout(Duration::from_secs(5), read_frame(&mut stream)).await {
            Ok(Ok(raw)) if raw.is_empty() => break,
            Ok(Ok(raw)) => {
                if matches!(
                    serde_json::from_slice::<api::WsFrame>(&raw),
                    Ok(api::WsFrame::Result { .. })
                ) {
                    results += 1;
                }
            }
            _ => break,
        }
    }
    assert_eq!(
        results, N,
        "all queued replies must be flushed before the connection is torn down"
    );

    let _ = shutdown.send(());
}

/// DT-13: the socket file must be 0600 and its parent directory 0700, so
/// no other user on the host can connect (or even stat the socket). Also
/// pins that permission tightening happens before the listener serves.
#[tokio::test]
async fn socket_file_and_parent_dir_permissions_are_tightened() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    // Use a nested parent dir so the listener itself creates it.
    let path = dir.path().join("nested").join("adelie.sock");
    let (_handler, _join, shutdown) = start_server(path.clone(), signing_key);
    wait_for_socket(&path).await;

    let sock_mode = std::fs::metadata(&path).unwrap().permissions().mode();
    assert_eq!(
        sock_mode & 0o777,
        0o600,
        "socket must be 0600, got {:o}",
        sock_mode & 0o777
    );

    let parent_mode = std::fs::metadata(path.parent().unwrap())
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(
        parent_mode & 0o777,
        0o700,
        "socket parent dir must be 0700, got {:o}",
        parent_mode & 0o777
    );

    let _ = shutdown.send(());
}

// --- Peer-credential auth (#407) ---------------------------------------------

/// A local-trust validator that mirrors the daemon's `PeerCredUdsAuth`:
/// authenticate by the kernel peer identity, no bearer token required.
struct PeerCredAuth;

#[async_trait::async_trait]
impl UdsAuthValidator for PeerCredAuth {
    async fn validate_bearer_token(&self, _token: &str) -> bool {
        // This validator never accepts tokens — peer-cred is the only path.
        false
    }

    async fn authenticate(
        &self,
        _token: Option<&str>,
        peer: Option<&desktop_assistant_uds::PeerIdentity>,
    ) -> desktop_assistant_uds::UdsAuth {
        match peer {
            Some(p) => desktop_assistant_uds::UdsAuth::Allow(
                desktop_assistant_application::UserId::from(p.username.clone()),
            ),
            None => desktop_assistant_uds::UdsAuth::Reject("auth: no peer credentials".to_string()),
        }
    }
}

fn start_server_with(
    socket_path: PathBuf,
    auth: Arc<dyn UdsAuthValidator>,
) -> (
    tokio::task::JoinHandle<anyhow::Result<()>>,
    tokio::sync::oneshot::Sender<()>,
) {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let config = UdsServerConfig::new(socket_path);
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(handler, auth, config);
    let join = tokio::spawn(async move {
        server
            .serve_with_shutdown(async move {
                let _ = rx.await;
            })
            .await
    });
    (join, tx)
}

/// With a peer-cred validator, a handshake carrying **no** jwt is accepted —
/// the kernel-attested peer identity is the authentication (#407). The
/// connection then services commands normally.
#[tokio::test]
async fn uds_connection_without_jwt_proceeds_under_peer_cred() {
    let dir = TempDir::new().unwrap();
    let path = socket_path(&dir);
    let (_join, shutdown) = start_server_with(path.clone(), Arc::new(PeerCredAuth));
    wait_for_socket(&path).await;

    let mut stream = UnixStream::connect(&path).await.unwrap();
    // Tokenless handshake — not even a `jwt` field.
    let handshake = serde_json::json!({});
    write_frame(&mut stream, &serde_json::to_vec(&handshake).unwrap())
        .await
        .unwrap();

    let req = api::WsRequest {
        id: "1".into(),
        command: api::Command::Ping,
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).unwrap())
        .await
        .unwrap();

    let raw = timeout(Duration::from_secs(2), read_frame(&mut stream))
        .await
        .expect("no response within 2s — tokenless peer-cred handshake should proceed")
        .expect("io error on read_frame");
    let frame: api::WsFrame = serde_json::from_slice(&raw).unwrap();
    match frame {
        api::WsFrame::Result { id, result } => {
            assert_eq!(id, "1");
            assert_eq!(
                result,
                api::CommandResult::Pong {
                    value: "pong".into()
                }
            );
        }
        other => panic!("expected Pong, got {other:?}"),
    }

    let _ = shutdown.send(());
}
