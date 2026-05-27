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
    JwtUdsAuth, UdsAuthValidator, UdsServer, UdsServerConfig, read_frame, write_frame,
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
            return;
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
    let _ = tokio::spawn(dispatch_loop(
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
}
