use std::net::SocketAddr;
use std::sync::Arc;
use std::{future::Future, future::pending};

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

const WS_OUTBOUND_BUFFER: usize = 64;

/// WebSocket request envelope.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WsRequest {
    pub id: String,
    pub command: api::Command,
}

/// WebSocket frames sent from server to client.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    tx: mpsc::Sender<WsFrame>,
}

#[async_trait::async_trait]
impl EventSink for ChannelSink {
    async fn emit(&self, event: api::Event) -> bool {
        self.tx.send(WsFrame::Event { event }).await.is_ok()
    }
}

async fn handle_socket(socket: WebSocket, state: WsServerState) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Channel for outbound frames (results + events)
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(WS_OUTBOUND_BUFFER);

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

                        if out_tx
                            .send(WsFrame::Result {
                                id: req.id.clone(),
                                result: api::CommandResult::Ack,
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }

                        let handler = Arc::clone(&state.handler);
                        tokio::spawn(async move {
                            let _ = handler
                                .handle_send_message(conversation_id, content, request_id, sink)
                                .await;
                        });
                    }

                    api::Command::SetConfig { changes } => {
                        let res = state
                            .handler
                            .handle_command(api::Command::SetConfig { changes })
                            .await;
                        match res {
                            Ok(api::CommandResult::Config(config)) => {
                                if out_tx
                                    .send(WsFrame::Result {
                                        id: req.id.clone(),
                                        result: api::CommandResult::Config(config.clone()),
                                    })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                                if out_tx
                                    .send(WsFrame::Event {
                                        event: api::Event::ConfigChanged { config },
                                    })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Ok(result) => {
                                if out_tx
                                    .send(WsFrame::Result { id: req.id, result })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Err(ApiError::Core(e)) => {
                                if out_tx
                                    .send(WsFrame::Error {
                                        id: req.id,
                                        error: e,
                                    })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Err(e) => {
                                if out_tx
                                    .send(WsFrame::Error {
                                        id: req.id,
                                        error: e.to_string(),
                                    })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                    }

                    other => {
                        let res = state.handler.handle_command(other).await;
                        match res {
                            Ok(result) => {
                                if out_tx
                                    .send(WsFrame::Result { id: req.id, result })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Err(ApiError::Core(e)) => {
                                if out_tx
                                    .send(WsFrame::Error {
                                        id: req.id,
                                        error: e,
                                    })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Err(e) => {
                                if out_tx
                                    .send(WsFrame::Error {
                                        id: req.id,
                                        error: e.to_string(),
                                    })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
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
    serve_with_shutdown(handler, bind, pending::<()>()).await
}

pub async fn serve_with_shutdown<F>(
    handler: Arc<dyn AssistantApiHandler>,
    bind: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let app = router(handler);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}
