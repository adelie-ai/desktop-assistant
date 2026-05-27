use std::net::SocketAddr;
use std::sync::Arc;
use std::{future::Future, future::pending};

use axum::{
    Json, Router,
    extract::{State, ws::Message, ws::WebSocket, ws::WebSocketUpgrade},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use base64::Engine;
use desktop_assistant_api_model as api;
use desktop_assistant_application::{AssistantApiHandler, UserId};
use desktop_assistant_transport_dispatch::{AuthContext, dispatch_loop};
use futures::{SinkExt, StreamExt};
#[cfg(feature = "tls")]
use tracing::debug;
use tracing::warn;

pub use api::{WsFrame, WsRequest};

#[derive(Clone)]
pub struct WsServerState {
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Arc<Vec<String>>,
}

#[async_trait::async_trait]
pub trait WsAuthValidator: Send + Sync {
    async fn validate_bearer_token(&self, token: &str) -> bool;

    /// Extract the user id ([JWT `sub`]) from a bearer token that
    /// [`Self::validate_bearer_token`] already accepted (#105).
    ///
    /// The default returns `None` — meaning "validator opted out of
    /// identity extraction"; the WS handler then falls back to
    /// [`UserId::default`] (the schema sentinel `"default"`). That
    /// covers single-tenant deploys that don't need multi-tenancy
    /// without forcing every existing validator implementation to
    /// change.
    ///
    /// Multi-tenant correctness REQUIRES returning `Some(user_id)`.
    /// The current mapping rule is `sub` → `user_id` verbatim;
    /// future revisions may consult a different claim.
    async fn extract_user_id(&self, token: &str) -> Option<UserId> {
        let _ = token;
        None
    }
}

#[async_trait::async_trait]
pub trait WsLoginService: Send + Sync {
    async fn authenticate_basic(&self, username: &str, password: &str) -> bool;
    async fn issue_token_for_subject(&self, subject: &str) -> Result<String, String>;
}

/// Provides auth discovery information for `GET /auth/config`.
#[async_trait::async_trait]
pub trait WsAuthDiscovery: Send + Sync {
    /// Returns JSON-serializable auth discovery info.
    async fn auth_config(&self) -> serde_json::Value;
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
    router_with_auth(handler, auth_validator, login_service, None)
}

pub fn router_with_auth(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
) -> Router {
    router_full(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        vec![],
    )
}

pub fn router_full(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Vec<String>,
) -> Router {
    let state = WsServerState {
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins: Arc::new(allowed_origins),
    };

    Router::new()
        .route("/ws", get(ws_handler))
        .route("/login", post(login_handler))
        .route("/auth/config", get(auth_config_handler))
        .with_state(state)
}

/// Validates the `Origin` header against the allowed origins list.
///
/// - No `Origin` header: always allowed (native clients like tui/gtk don't send one).
/// - `Origin` present + empty allowlist: rejected (no browser clients permitted by default).
/// - `Origin` present + matches an entry: allowed.
/// - `Origin` present + no match: rejected.
fn validate_origin(headers: &HeaderMap, allowed_origins: &[String]) -> Result<(), StatusCode> {
    let Some(origin) = headers.get("origin") else {
        return Ok(());
    };
    let origin = origin.to_str().map_err(|_| StatusCode::FORBIDDEN)?;
    if allowed_origins.iter().any(|a| a == origin) {
        Ok(())
    } else {
        Err(StatusCode::FORBIDDEN)
    }
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
    if let Err(status) = validate_origin(&headers, &state.allowed_origins) {
        return (status, "origin not allowed").into_response();
    }

    let Some(token) = extract_bearer_token(&headers) else {
        return (StatusCode::UNAUTHORIZED, "missing bearer token").into_response();
    };

    if !state.auth_validator.validate_bearer_token(&token).await {
        return (StatusCode::UNAUTHORIZED, "invalid bearer token").into_response();
    }

    // Resolve the per-connection user identity once at upgrade time
    // (#105). The validator returns `Some(sub)` for multi-tenant
    // tokens; in the single-tenant fallback path it returns `None`
    // and we collapse to the schema sentinel `"default"`. Identity
    // is per-connection: one WS upgrade carries one bearer token,
    // so every command on this socket runs as the same user.
    let user_id = state
        .auth_validator
        .extract_user_id(&token)
        .await
        .unwrap_or_default();

    ws.on_upgrade(move |socket| handle_socket(socket, state, user_id))
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
    if let Err(status) = validate_origin(&headers, &state.allowed_origins) {
        return (status, "origin not allowed").into_response();
    }

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

async fn auth_config_handler(State(state): State<WsServerState>) -> impl IntoResponse {
    match state.auth_discovery {
        Some(discovery) => {
            let config = discovery.auth_config().await;
            (StatusCode::OK, Json(config)).into_response()
        }
        None => {
            // Default: password-only (backward-compatible)
            let default = serde_json::json!({ "methods": ["password"] });
            (StatusCode::OK, Json(default)).into_response()
        }
    }
}

/// Adapts axum's `WebSocket` into the transport-neutral stream + sink
/// pair expected by [`dispatch_loop`]. The receive half is filtered to
/// `Message::Text` frames parsed into `WsRequest`; the send half is
/// driven from an mpsc by a small writer task that serializes
/// `WsFrame` values into `Message::Text`. A `tokio_util::PollSender`
/// gives us a `Sink<WsFrame>` over the mpsc so the dispatcher's
/// generic bound is satisfied without a hand-rolled adapter.
async fn handle_socket(socket: WebSocket, state: WsServerState, user_id: UserId) {
    use tokio::sync::mpsc;
    use tokio_util::sync::PollSender;

    let (mut ws_tx, ws_rx) = socket.split();

    // Outbound: writer task owns the WS sender and serializes
    // `WsFrame` values pulled from a channel into `Message::Text`.
    let (outbound_tx, mut outbound_rx) = mpsc::channel::<WsFrame>(64);
    let writer = tokio::spawn(async move {
        while let Some(frame) = outbound_rx.recv().await {
            let Ok(text) = serde_json::to_string(&frame) else {
                continue;
            };
            if ws_tx.send(Message::Text(text.into())).await.is_err() {
                break;
            }
        }
    });

    // Inbound: axum's `Message` stream down to text frames parsed as
    // `WsRequest`. Close frames terminate the stream; ping / pong /
    // binary frames are silently dropped (matches pre-refactor).
    let inbound = ws_rx.filter_map(|item| async move {
        match item {
            Ok(Message::Text(text)) => match serde_json::from_str::<WsRequest>(&text) {
                Ok(req) => Some(Ok::<_, anyhow::Error>(req)),
                Err(e) => {
                    warn!("invalid ws json: {e}");
                    None
                }
            },
            Ok(Message::Close(_)) | Err(_) => None,
            Ok(_) => None,
        }
    });
    futures::pin_mut!(inbound);

    let sink = PollSender::new(outbound_tx.clone());

    // Per-connection identity resolved in `ws_handler` (#105): the
    // bearer token's `sub` if the validator extracted one, otherwise
    // the schema sentinel for single-tenant fallback. The dispatcher
    // installs this into the per-task task-local on every dispatched
    // command so storage queries scope correctly.
    let auth = AuthContext::new(user_id.into_inner());

    dispatch_loop(Arc::clone(&state.handler), auth, inbound, sink).await;

    // Mirror the pre-refactor cleanup: kill the writer when the
    // dispatcher returns so any in-flight `SendMessage` task observes
    // a closed channel on its next `emit` and shuts down. This is the
    // cancellation path exercised by
    // `ws_send_message_cancels_when_client_disconnects`.
    drop(outbound_tx);
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
    serve_with_shutdown_and_auth(handler, auth_validator, login_service, None, bind, shutdown).await
}

pub async fn serve_with_shutdown_and_auth<F>(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    bind: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    serve_full(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        vec![],
        bind,
        shutdown,
    )
    .await
}

pub async fn serve_full<F>(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Vec<String>,
    bind: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let app = router_full(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins,
    );
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

#[cfg(feature = "tls")]
pub async fn serve_full_tls<F>(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Vec<String>,
    tls_acceptor: tokio_rustls::TlsAcceptor,
    bind: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder as ConnBuilder;
    use hyper_util::service::TowerToHyperService;

    let app = router_full(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins,
    );
    let listener = tokio::net::TcpListener::bind(bind).await?;

    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            result = listener.accept() => {
                let (tcp_stream, _remote_addr) = result?;
                let acceptor = tls_acceptor.clone();
                let app = app.clone();

                tokio::spawn(async move {
                    let tls_stream = match acceptor.accept(tcp_stream).await {
                        Ok(s) => s,
                        Err(e) => {
                            debug!("TLS handshake failed: {e}");
                            return;
                        }
                    };

                    let io = TokioIo::new(tls_stream);
                    let service = TowerToHyperService::new(app.into_service());
                    if let Err(e) = ConnBuilder::new(TokioExecutor::new())
                        .serve_connection_with_upgrades(io, service)
                        .await
                    {
                        debug!("connection error: {e}");
                    }
                });
            }
            _ = &mut shutdown => break,
        }
    }

    Ok(())
}
