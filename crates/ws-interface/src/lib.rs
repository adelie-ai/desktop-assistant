use std::net::SocketAddr;
use std::sync::Arc;
use std::{future::Future, future::pending};

use axum::{
    Router,
    extract::{State, ws::Message, ws::WebSocket, ws::WebSocketUpgrade},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use base64::Engine;
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
}

#[async_trait::async_trait]
pub trait WsAuthValidator: Send + Sync {
    async fn validate_bearer_token(&self, token: &str) -> bool;
}

#[async_trait::async_trait]
pub trait WsLoginService: Send + Sync {
    async fn authenticate_basic(&self, username: &str, password: &str) -> bool;
    async fn issue_token_for_subject(&self, subject: &str) -> Result<String, String>;
}

pub fn router(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
) -> Router {
    router_with_login(handler, auth_validator, None)
}

pub fn router_with_login(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
) -> Router {
    let state = WsServerState {
        handler,
        auth_validator,
        login_service,
    };

    Router::new()
        .route("/ws", get(ws_handler))
        .route("/login", post(login_handler))
        .with_state(state)
}

fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get("authorization")?.to_str().ok()?.trim();
    let (scheme, token) = raw.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

fn extract_basic_credentials(headers: &HeaderMap) -> Option<(String, String)> {
    let raw = headers.get("authorization")?.to_str().ok()?.trim();
    let (scheme, encoded) = raw.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("basic") {
        return None;
    }

    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .ok()?;
    let decoded = String::from_utf8(decoded).ok()?;
    let (username, password) = decoded.split_once(':')?;
    Some((username.to_string(), password.to_string()))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsServerState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let Some(token) = extract_bearer_token(&headers) else {
        return (StatusCode::UNAUTHORIZED, "missing bearer token").into_response();
    };

    if !state.auth_validator.validate_bearer_token(&token).await {
        return (StatusCode::UNAUTHORIZED, "invalid bearer token").into_response();
    }

    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

#[derive(Debug, Clone, serde::Serialize)]
struct LoginResponse {
    token: String,
    token_type: &'static str,
    subject: String,
}

async fn login_handler(
    State(state): State<WsServerState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let Some(login_service) = state.login_service else {
        return (StatusCode::NOT_FOUND, "login is not enabled").into_response();
    };

    let Some((username, password)) = extract_basic_credentials(&headers) else {
        return (StatusCode::UNAUTHORIZED, "missing basic auth").into_response();
    };

    if !login_service.authenticate_basic(&username, &password).await {
        return (StatusCode::UNAUTHORIZED, "invalid credentials").into_response();
    }

    match login_service.issue_token_for_subject(&username).await {
        Ok(token) => (
            StatusCode::OK,
            axum::Json(LoginResponse {
                token,
                token_type: "bearer",
                subject: username,
            }),
        )
            .into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error).into_response(),
    }
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

pub async fn serve(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    bind: SocketAddr,
) -> anyhow::Result<()> {
    serve_with_shutdown(handler, auth_validator, bind, pending::<()>()).await
}

pub async fn serve_with_shutdown<F>(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    bind: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    serve_with_shutdown_and_login(handler, auth_validator, None, bind, shutdown).await
}

pub async fn serve_with_shutdown_and_login<F>(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    bind: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let app = router_with_login(handler, auth_validator, login_service);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}
