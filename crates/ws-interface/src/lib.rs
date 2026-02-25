use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    Router,
    extract::{State, ws::Message, ws::WebSocket, ws::WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, AssistantApiHandler, EventSink};
use futures::{sink::SinkExt, stream::StreamExt};
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// WebSocket request envelope.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WsRequest {
    pub id: String,
    pub command: api::Command,
}

/// WebSocket frames sent from server to client.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WsFrame {
    Result {
        id: String,
        result: api::CommandResult,
    },
    Error {
        id: String,
        error: String,
    },
    Event {
        event: api::Event,
    },
}

#[derive(Clone)]
pub struct WsServerState {
    handler: Arc<dyn AssistantApiHandler>,
}

pub fn router(handler: Arc<dyn AssistantApiHandler>) -> Router {
    let state = WsServerState { handler };

    Router::new()
        .route("/ws", get(ws_handler))
        .with_state(state)
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<WsServerState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

struct ChannelSink {
    tx: mpsc::UnboundedSender<WsFrame>,
}

#[async_trait::async_trait]
impl EventSink for ChannelSink {
    async fn emit(&self, event: api::Event) {
        let _ = self.tx.send(WsFrame::Event { event });
    }
}

async fn handle_socket(socket: WebSocket, state: WsServerState) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Channel for outbound frames (results + events)
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<WsFrame>();

    // Writer task: serialize WsFrame -> ws text message
    let writer = tokio::spawn(async move {
        while let Some(frame) = out_rx.recv().await {
            let Ok(text) = serde_json::to_string(&frame) else {
                continue;
            };
            if ws_tx.send(Message::Text(text.into())).await.is_err() {
                break;
            }
        }
    });

    // Reader loop: handle inbound requests
    while let Some(Ok(msg)) = ws_rx.next().await {
        match msg {
            Message::Text(text) => {
                let req: WsRequest = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("invalid ws json: {e}");
                        continue;
                    }
                };

                debug!(id = %req.id, "ws command received");

                match req.command {
                    api::Command::SendMessage {
                        conversation_id,
                        content,
                    } => {
                        // Stream via events; acknowledge immediately.
                        let request_id = uuid::Uuid::new_v4().to_string();
                        let sink = Arc::new(ChannelSink { tx: out_tx.clone() });

                        let _ = out_tx.send(WsFrame::Result {
                            id: req.id.clone(),
                            result: api::CommandResult::Ack,
                        });

                        let handler = Arc::clone(&state.handler);
                        tokio::spawn(async move {
                            let _ = handler
                                .handle_send_message(conversation_id, content, request_id, sink)
                                .await;
                        });
                    }

                    other => {
                        let res = state.handler.handle_command(other).await;
                        match res {
                            Ok(result) => {
                                let _ = out_tx.send(WsFrame::Result { id: req.id, result });
                            }
                            Err(ApiError::Core(e)) => {
                                let _ = out_tx.send(WsFrame::Error {
                                    id: req.id,
                                    error: e,
                                });
                            }
                            Err(e) => {
                                let _ = out_tx.send(WsFrame::Error {
                                    id: req.id,
                                    error: e.to_string(),
                                });
                            }
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    writer.abort();
}

pub async fn serve(handler: Arc<dyn AssistantApiHandler>, bind: SocketAddr) -> anyhow::Result<()> {
    let app = router(handler);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
