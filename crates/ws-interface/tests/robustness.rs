//! Protocol-robustness tests for the WS frontend (code review 2026-06-09,
//! findings DT-5 and DT-6).
//!
//! - DT-5: a malformed inbound JSON text frame must produce an explicit
//!   `WsFrame::Error` with an empty id (there is no request id to correlate
//!   to) instead of being warn-logged and silently dropped, which left the
//!   client hanging forever on its reply.
//! - DT-6: when the client closes the websocket, replies already queued on
//!   the outbound channel must still be flushed before teardown — the old
//!   `writer.abort()` could drop final frames for a client that half-closes
//!   and waits for its replies.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_ws::{WsAuthValidator, WsFrame, WsRequest, router};
use futures_util::{SinkExt, StreamExt};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

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

struct AcceptAllAuth;

#[async_trait::async_trait]
impl WsAuthValidator for AcceptAllAuth {
    async fn validate_bearer_token(&self, _token: &str) -> bool {
        true
    }
}

fn ws_request(url: &str, bearer: &str) -> tokio_tungstenite::tungstenite::http::Request<()> {
    let mut request = url.into_client_request().unwrap();
    request.headers_mut().insert(
        tokio_tungstenite::tungstenite::http::header::AUTHORIZATION,
        format!("Bearer {bearer}").parse().unwrap(),
    );
    request
}

async fn spawn_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let app = router(handler, Arc::new(AcceptAllAuth));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, server)
}

/// DT-5: a text frame that isn't valid `WsRequest` JSON yields an error
/// frame with an empty id, and the connection keeps serving afterwards.
#[tokio::test]
async fn invalid_ws_json_yields_error_frame_with_empty_id() {
    let (addr, server) = spawn_server().await;
    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_request(&url, "t"))
        .await
        .unwrap();

    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        "{this is not json".to_string().into(),
    ))
    .await
    .unwrap();

    let msg = timeout(Duration::from_secs(2), ws.next())
        .await
        .expect("expected an error frame for malformed JSON, got silence")
        .unwrap()
        .unwrap();
    let frame: WsFrame = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    match frame {
        WsFrame::Error { id, error } => {
            assert_eq!(id, "", "no request id is known for a malformed frame");
            assert!(
                error.to_lowercase().contains("json") || error.to_lowercase().contains("invalid"),
                "error should describe the parse failure: {error}"
            );
        }
        other => panic!("expected an Error frame, got {other:?}"),
    }

    // The connection must survive a malformed frame: a valid request after
    // it still round-trips.
    let req = WsRequest {
        id: "after".into(),
        command: api::Command::Ping,
    };
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        serde_json::to_string(&req).unwrap().into(),
    ))
    .await
    .unwrap();
    let msg = timeout(Duration::from_secs(2), ws.next())
        .await
        .expect("connection must keep serving after a malformed frame")
        .unwrap()
        .unwrap();
    let frame: WsFrame = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    match frame {
        WsFrame::Result { id, .. } => assert_eq!(id, "after"),
        other => panic!("expected a Result frame, got {other:?}"),
    }

    server.abort();
}

/// DT-6: replies already queued when the client sends Close must still be
/// delivered, not torn down by aborting the writer task.
#[tokio::test]
async fn queued_replies_drain_after_client_close() {
    let (addr, server) = spawn_server().await;
    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = tokio_tungstenite::connect_async(ws_request(&url, "t"))
        .await
        .unwrap();

    const N: usize = 30;
    for i in 0..N {
        let req = WsRequest {
            id: format!("req-{i}"),
            command: api::Command::Ping,
        };
        ws.send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::to_string(&req).unwrap().into(),
        ))
        .await
        .unwrap();
    }
    // Half-close: tell the server we're done sending, then keep reading.
    ws.send(tokio_tungstenite::tungstenite::Message::Close(None))
        .await
        .unwrap();

    let mut results = 0usize;
    loop {
        let Ok(item) = timeout(Duration::from_secs(5), ws.next()).await else {
            break;
        };
        match item {
            Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
                if matches!(
                    serde_json::from_str::<WsFrame>(&text),
                    Ok(WsFrame::Result { .. })
                ) {
                    results += 1;
                }
            }
            Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) | None => break,
            Some(Ok(_)) => {}
            Some(Err(_)) => break,
        }
    }

    assert_eq!(
        results, N,
        "all queued replies must be flushed before the connection is torn down"
    );

    server.abort();
}
