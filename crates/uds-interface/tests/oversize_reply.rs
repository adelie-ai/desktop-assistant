//! An outbound reply past the frame cap must fail the ONE request that
//! produced it, not the connection (issue #1303).
//!
//! The daemon bounds `GetConversation` and `GetMessages` in bytes before it
//! answers, so a reply past the cap should never be built. This file holds the
//! backstop under that. It drives the raw framing rather than a client, because
//! the two cases here need a request the typed client cannot make: one whose
//! reply is oversize, and one whose request id is so large that even the
//! failure frame naming it would be oversize.

use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_auth_jwt as jwt;
use desktop_assistant_frame_codec::MAX_FRAME_LEN;
use desktop_assistant_uds::{
    UdsAuthValidator, UdsServer, UdsServerConfig, read_frame, write_frame,
};
use tempfile::TempDir;
use tokio::net::UnixStream;
use tokio::time::{Duration, timeout};

const ISS: &str = "test-oversize-iss";
const AUD: &str = "test-oversize-aud";

fn mint_test_jwt(signing_key: &str, subject: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
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

/// Answers `Ping` with a payload past the frame cap and everything else with a
/// small one, so one connection can show a failed request beside a served one.
struct OversizeHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for OversizeHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::Ping => Ok(api::CommandResult::Pong {
                value: "p".repeat(MAX_FRAME_LEN as usize + 1024),
            }),
            _ => Ok(api::CommandResult::Ack),
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

struct StaticJwtAuth {
    signing_key: String,
}

#[async_trait::async_trait]
impl UdsAuthValidator for StaticJwtAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        jwt::decode(token, &self.signing_key, ISS, AUD).is_ok()
    }

    async fn extract_user_id(&self, token: &str) -> Option<jwt::UserId> {
        jwt::decode(token, &self.signing_key, ISS, AUD)
            .ok()
            .map(|claims| jwt::UserId::new(claims.sub))
    }
}

fn start_server(socket_path: PathBuf, signing_key: String) -> tokio::sync::oneshot::Sender<()> {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(OversizeHandler);
    let auth: Arc<dyn UdsAuthValidator> = Arc::new(StaticJwtAuth { signing_key });
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = UdsServer::new(handler, auth, UdsServerConfig::new(socket_path));
    tokio::spawn(async move {
        let _ = server
            .serve_with_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    tx
}

async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..100 {
        if path.exists() && UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("uds socket did not appear");
}

/// Connect, hand over the handshake, and return the stream.
async fn connect(path: &std::path::Path, signing_key: &str) -> UnixStream {
    let mut stream = UnixStream::connect(path).await.unwrap();
    let handshake = serde_json::json!({ "jwt": mint_test_jwt(signing_key, "alice") });
    write_frame(&mut stream, &serde_json::to_vec(&handshake).unwrap())
        .await
        .unwrap();
    stream
}

async fn send(stream: &mut UnixStream, id: &str, command: api::Command) {
    let req = api::WsRequest {
        id: id.into(),
        command,
    };
    write_frame(stream, &serde_json::to_vec(&req).unwrap())
        .await
        .expect("the request itself is inside the cap and must send");
}

/// Acceptance (#1303): a reply the codec refuses becomes a failure of the one
/// request. The caller gets an error frame naming its own request id, and the
/// connection keeps serving.
#[tokio::test]
async fn an_oversize_reply_fails_the_one_request_and_keeps_the_connection() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;
    let mut stream = connect(&path, &signing_key).await;

    send(&mut stream, "too-big", api::Command::Ping).await;
    let raw = timeout(Duration::from_secs(5), read_frame(&mut stream))
        .await
        .expect("the server must answer rather than go silent")
        .expect("the connection must not be torn down");
    assert!(
        raw.len() <= MAX_FRAME_LEN as usize,
        "no frame past the cap may reach the wire: {} bytes",
        raw.len()
    );
    match serde_json::from_slice::<api::WsFrame>(&raw).expect("a valid frame") {
        api::WsFrame::Error { id, .. } => assert_eq!(
            id, "too-big",
            "the failure must name the request that caused it"
        ),
        other => panic!("expected an error for the one request, got {other:?}"),
    }

    send(&mut stream, "after", api::Command::ClearAllHistory).await;
    let raw = timeout(Duration::from_secs(5), read_frame(&mut stream))
        .await
        .expect("the connection must keep serving after an oversize reply")
        .expect("the connection must still be open");
    match serde_json::from_slice::<api::WsFrame>(&raw).expect("a valid frame") {
        api::WsFrame::Result { id, .. } => assert_eq!(id, "after"),
        other => panic!("expected the second request to succeed, got {other:?}"),
    }

    let _ = shutdown.send(());
}

/// #1303: when even the FAILURE frame cannot fit, the connection still
/// survives. The failure echoes the request id, so a request whose id is
/// itself near the cap has no answer that both fits and names it. That request
/// goes unanswered and times out at the caller, while every other request on
/// the connection keeps working.
#[tokio::test]
async fn a_request_with_no_answer_that_fits_does_not_take_the_connection_down() {
    let dir = TempDir::new().unwrap();
    let signing_key = "deadbeef".repeat(8);
    let path = dir.path().join("adelie.sock");
    let shutdown = start_server(path.clone(), signing_key.clone());
    wait_for_socket(&path).await;
    let mut stream = connect(&path, &signing_key).await;

    // An id that fills the inbound frame, so any frame echoing it is oversize.
    let envelope = serde_json::to_vec(&api::WsRequest {
        id: String::new(),
        command: api::Command::Ping,
    })
    .unwrap()
    .len();
    let huge = serde_json::to_vec(&api::WsRequest {
        id: "x".repeat(MAX_FRAME_LEN as usize - 1 - envelope),
        command: api::Command::Ping,
    })
    .unwrap();
    assert_eq!(huge.len(), MAX_FRAME_LEN as usize - 1);
    write_frame(&mut stream, &huge)
        .await
        .expect("the request is inside the inbound cap and must send");

    // An ordinary request behind it. Its answer can only arrive if the
    // connection survived the unanswerable one.
    send(&mut stream, "after", api::Command::ClearAllHistory).await;
    let raw = timeout(Duration::from_secs(5), read_frame(&mut stream))
        .await
        .expect("the connection must keep serving after an unanswerable request")
        .expect("the connection must still be open");
    assert!(
        raw.len() <= MAX_FRAME_LEN as usize,
        "no frame past the cap may reach the wire: {} bytes",
        raw.len()
    );
    match serde_json::from_slice::<api::WsFrame>(&raw).expect("a valid frame") {
        api::WsFrame::Result { id, .. } => assert_eq!(
            id, "after",
            "the unanswerable request must be skipped, not answered"
        ),
        other => panic!("expected the ordinary request to succeed, got {other:?}"),
    }

    let _ = shutdown.send(());
}
