//! An outbound reply past the WebSocket message cap must fail the ONE request
//! that produced it, not the connection (issue #1303).
//!
//! The daemon bounds `GetConversation` and `GetMessages` in bytes before it
//! answers, so a reply past the cap should never be built. This file holds the
//! backstop under that: a handler that answers with an oversize payload must
//! not be able to break the socket. Before #1303 the writer sent whatever it
//! was given, the peer rejected it, and every outstanding request on that
//! connection failed with it.
//!
//! The handler here answers `Ping` with a deliberately oversize payload,
//! because `Ping` is the one command that needs no service fakes at all.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_api_model as api_model;
use desktop_assistant_application::{ApiResult, AssistantApiHandler, EventSink, UserId};
use desktop_assistant_ws::{MAX_WS_MESSAGE_BYTES, WsAuthValidator, WsFrame, WsRequest, router};
use futures_util::{SinkExt, StreamExt};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

/// Answers `Ping` with a payload past the WebSocket message cap, and every
/// other command with a small one. Two requests on one connection then tell
/// apart "this request failed" from "the connection died".
struct OversizeHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for OversizeHandler {
    async fn handle_command(&self, cmd: api_model::Command) -> ApiResult<api_model::CommandResult> {
        match cmd {
            api_model::Command::Ping => Ok(api_model::CommandResult::Pong {
                value: "p".repeat(MAX_WS_MESSAGE_BYTES + 1024),
            }),
            _ => Ok(api_model::CommandResult::Ack),
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

struct StaticJwtAuth;

#[async_trait::async_trait]
impl WsAuthValidator for StaticJwtAuth {
    async fn validate_bearer_token(&self, token: &str) -> bool {
        token == "test-jwt"
    }

    async fn extract_user_id(&self, token: &str) -> Option<UserId> {
        self.validate_bearer_token(token)
            .await
            .then(|| UserId::from("test-user"))
    }
}

fn ws_request(url: &str) -> tokio_tungstenite::tungstenite::http::Request<()> {
    let mut request = url.into_client_request().unwrap();
    request.headers_mut().insert(
        tokio_tungstenite::tungstenite::http::header::AUTHORIZATION,
        "Bearer test-jwt".parse().unwrap(),
    );
    request
}

async fn spawn_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let app = router(Arc::new(OversizeHandler), Arc::new(StaticJwtAuth));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, server)
}

/// Acceptance (#1303): a reply the transport cannot carry becomes a failure of
/// the one request. The caller gets an error frame it can match to its request
/// id, and the connection keeps serving - a second request on the same socket
/// still gets its answer.
#[tokio::test]
async fn an_oversize_reply_fails_the_one_request_and_keeps_the_connection() {
    let (addr, server) = spawn_server().await;
    let url = format!("ws://{addr}/ws");
    // The same cap the daemon's own clients read with. Without it a permissive
    // client would swallow the oversize reply and hide the defect.
    let client_config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(MAX_WS_MESSAGE_BYTES))
        .max_frame_size(Some(MAX_WS_MESSAGE_BYTES));
    let (mut ws, _) =
        tokio_tungstenite::connect_async_with_config(ws_request(&url), Some(client_config), false)
            .await
            .unwrap();

    // Request 1: the handler answers with an oversize payload.
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&WsRequest {
            id: "too-big".into(),
            command: api_model::Command::Ping,
        })
        .unwrap()
        .into(),
    ))
    .await
    .expect("the request itself is small and must send");

    let frame = timeout(Duration::from_secs(10), ws.next())
        .await
        .expect("the server must answer the oversize request rather than go silent")
        .expect("the stream must yield a frame, not end")
        .expect("the frame must not be a transport error");
    let text = frame
        .into_text()
        .expect("the answer must be a text frame, not a close");
    assert!(
        text.len() <= MAX_WS_MESSAGE_BYTES,
        "the server must never put a message past its own cap on the wire: {} bytes",
        text.len()
    );
    let parsed: WsFrame = serde_json::from_str(&text).expect("the answer must be a valid WsFrame");
    match parsed {
        WsFrame::Error { id, .. } => assert_eq!(
            id, "too-big",
            "the failure must be matched to the request that caused it"
        ),
        other => panic!("expected an error for the one request, got {other:?}"),
    }

    // Request 2 on the SAME connection: proves the socket survived.
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&WsRequest {
            id: "after".into(),
            command: api_model::Command::ClearAllHistory,
        })
        .unwrap()
        .into(),
    ))
    .await
    .expect("the connection must still accept a request");

    let frame = timeout(Duration::from_secs(10), ws.next())
        .await
        .expect("the connection must keep serving after an oversize reply")
        .expect("the stream must yield")
        .expect("the frame must not be a transport error");
    let parsed: WsFrame = serde_json::from_str(&frame.into_text().unwrap()).expect("valid WsFrame");
    match parsed {
        WsFrame::Result { id, .. } => assert_eq!(id, "after"),
        other => panic!("expected the second request to succeed, got {other:?}"),
    }

    server.abort();
}

/// #1303: when even the FAILURE frame cannot fit, the connection still
/// survives. The failure echoes the request id, so a request whose id is
/// itself near the cap has no answer that both fits and names it. The one
/// request then goes unanswered and times out at the caller, and every other
/// request on the connection keeps working - which is the outcome the whole
/// change exists to protect.
#[tokio::test]
async fn a_request_with_no_answer_that_fits_does_not_take_the_connection_down() {
    let (addr, server) = spawn_server().await;
    let url = format!("ws://{addr}/ws");
    let client_config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(MAX_WS_MESSAGE_BYTES))
        .max_frame_size(Some(MAX_WS_MESSAGE_BYTES));
    let (mut ws, _) =
        tokio_tungstenite::connect_async_with_config(ws_request(&url), Some(client_config), false)
            .await
            .unwrap();

    // An id that fills the inbound message. Any frame echoing it is oversize.
    let base = serde_json::to_string(&WsRequest {
        id: String::new(),
        command: api_model::Command::Ping,
    })
    .unwrap()
    .len();
    let huge = serde_json::to_string(&WsRequest {
        id: "x".repeat(MAX_WS_MESSAGE_BYTES - 1 - base),
        command: api_model::Command::Ping,
    })
    .unwrap();
    assert_eq!(huge.len(), MAX_WS_MESSAGE_BYTES - 1);

    ws.send(tokio_tungstenite::tungstenite::Message::Text(huge.into()))
        .await
        .expect("the request is inside the inbound cap and must send");

    // A second, ordinary request. Its answer must arrive, which can only
    // happen if the connection survived the unanswerable one.
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&WsRequest {
            id: "after".into(),
            command: api_model::Command::ClearAllHistory,
        })
        .unwrap()
        .into(),
    ))
    .await
    .expect("the connection must still accept a request");

    let frame = timeout(Duration::from_secs(10), ws.next())
        .await
        .expect("the connection must keep serving after an unanswerable request")
        .expect("the stream must yield, not end")
        .expect("the frame must not be a transport error");
    let parsed: WsFrame = serde_json::from_str(&frame.into_text().unwrap()).expect("valid WsFrame");
    match parsed {
        WsFrame::Result { id, .. } => assert_eq!(id, "after"),
        other => panic!("expected the ordinary request to succeed, got {other:?}"),
    }

    server.abort();
}
