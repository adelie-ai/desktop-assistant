//! WebSocket frontend for the assistant API (axum-based, with optional TLS).

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
use desktop_assistant_transport_dispatch::{AuthContext, TransportKind, dispatch_loop};
use futures::SinkExt;
#[cfg(feature = "tls")]
use tracing::debug;
use tracing::warn;

pub use api::{WsFrame, WsRequest};

/// Inbound WebSocket message / frame size cap. Mirrors the
/// length-prefixed frame caps in `crates/uds-interface/src/lib.rs` and
/// `crates/dbus-bridge/src/transport.rs` so every transport into the
/// daemon has the same `4 * 1024 * 1024 == 4_194_304`-byte ceiling.
///
/// We cap *both* `max_message_size` and `max_frame_size`: the former
/// keeps an attacker from assembling a giant message out of many
/// in-cap fragments, and the latter rejects an oversize single frame
/// before its body is even allocated. Without these, axum defaults to
/// a 64 MiB message limit (closes SECURITY_AUDIT.md #7).
pub const MAX_WS_MESSAGE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
pub struct WsServerState {
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Arc<Vec<String>>,
    /// The daemon's own per-machine system id (#248), read once at startup.
    /// Compared against the client's `x-adelie-system-id` upgrade header to
    /// decide co-location exactly — same machine ⇒ co-located even over
    /// WebSocket. `None` ⇒ co-location falls back to the transport heuristic
    /// (WS ⇒ remote), preserving Phase-1 behaviour.
    daemon_system_id: Arc<Option<String>>,
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
    router_full_with_system_id(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins,
        None,
    )
}

/// Like [`router_full`] but also wires the daemon's own per-machine system id
/// (#248) so the `/ws` handler can compute exact tool-locality co-location from
/// the client's `x-adelie-system-id` upgrade header. `None` reproduces the
/// transport-heuristic-only behaviour of [`router_full`].
// One distinct collaborator per argument; a config struct would be an
// out-of-scope refactor of the public router API.
#[allow(clippy::too_many_arguments)]
pub fn router_full_with_system_id(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Vec<String>,
    daemon_system_id: Option<String>,
) -> Router {
    let state = WsServerState {
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins: Arc::new(allowed_origins),
        daemon_system_id: Arc::new(daemon_system_id),
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

/// Read a header value as a trimmed, non-empty `String`, or `None` if absent /
/// non-ASCII / blank. Used for the #248 system-id + host-label upgrade headers.
fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)?
        .to_str()
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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

    // System-id co-location (#248): the client reports its per-machine id (and
    // optionally a host label) in custom upgrade headers. Compare it to the
    // daemon's own id; a match co-locates even over WebSocket. `None` (no header
    // or unresolved daemon id) defers to the transport heuristic (WS ⇒ remote),
    // preserving Phase-1 behaviour. The id is a routing HINT, not a trust
    // boundary — it is self-reported and no privilege is gated on it (auth is
    // the bearer token validated above).
    let client_system_id = header_string(&headers, api::WS_SYSTEM_ID_HEADER);
    let client_label = header_string(&headers, api::WS_HOST_LABEL_HEADER);
    let co_located = desktop_assistant_core::system_id::co_location_from_ids(
        state.daemon_system_id.as_deref(),
        client_system_id.as_deref(),
    );

    ws.max_message_size(MAX_WS_MESSAGE_BYTES)
        .max_frame_size(MAX_WS_MESSAGE_BYTES)
        .on_upgrade(move |socket| handle_socket(socket, state, user_id, co_located, client_label))
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
async fn handle_socket(
    socket: WebSocket,
    state: WsServerState,
    user_id: UserId,
    co_located: Option<bool>,
    client_label: Option<String>,
) {
    use axum::extract::ws::CloseFrame;
    use futures::stream::StreamExt;
    use tokio::sync::mpsc;
    use tokio_util::sync::PollSender;

    let (mut ws_tx, ws_rx) = socket.split();

    // Outbound items: either a dispatcher-produced application frame
    // (the common case — serialized to text) or a server-initiated
    // close (used when the inbound stream errors, most notably for the
    // 4 MiB message-size cap). Keeping both kinds on one channel lets
    // the writer task own `ws_tx` as before without an extra mutex.
    // `WsFrame` is comparatively large (a few hundred bytes), so we
    // box it to keep the enum compact.
    enum Outbound {
        Frame(Box<WsFrame>),
        Close(CloseFrame),
    }

    let (outbound_tx, mut outbound_rx) = mpsc::channel::<Outbound>(64);
    let writer = tokio::spawn(async move {
        while let Some(item) = outbound_rx.recv().await {
            match item {
                Outbound::Frame(frame) => {
                    let Ok(text) = serde_json::to_string(&*frame) else {
                        continue;
                    };
                    if ws_tx.send(Message::Text(text.into())).await.is_err() {
                        break;
                    }
                }
                Outbound::Close(close) => {
                    // Best-effort: any send error means the peer is
                    // gone, so we just exit.
                    let _ = ws_tx.send(Message::Close(Some(close))).await;
                    break;
                }
            }
        }
    });

    // Inbound: walk the axum `Message` stream in a side task, parse
    // text frames into `WsRequest`, and push them onto an mpsc for the
    // dispatcher. The two termination paths worth distinguishing:
    //
    //   - Peer closed gracefully (`Ok(Message::Close(_))`) or the
    //     stream simply ended — drop the channel, dispatcher sees the
    //     stream end.
    //   - Inbound `Err(_)` — most commonly tungstenite's
    //     `Capacity(MessageTooLong)` from our 4 MiB cap; the message
    //     was never delivered to user code, so we owe the client an
    //     explicit close. We send RFC 6455 code 1009 ("Message Too
    //     Big") so well-behaved clients can surface the reason instead
    //     of guessing from a bare TCP RST.
    let close_tx = outbound_tx.clone();
    let (inbound_tx, inbound_rx) = mpsc::channel::<anyhow::Result<WsRequest>>(16);
    let reader = tokio::spawn(async move {
        let mut ws_rx = ws_rx;
        while let Some(item) = ws_rx.next().await {
            match item {
                Ok(Message::Text(text)) => match serde_json::from_str::<WsRequest>(&text) {
                    Ok(req) => {
                        if inbound_tx.send(Ok(req)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => warn!("invalid ws json: {e}"),
                },
                Ok(Message::Close(_)) => break,
                Ok(_) => {}
                Err(e) => {
                    let reason = format!("{e}");
                    warn!("ws inbound error, closing: {reason}");
                    // RFC 6455 §7.4.1: 1009 = "Message Too Big". We
                    // use it for any inbound error because in practice
                    // the only error tungstenite surfaces here is the
                    // capacity overrun from our cap.
                    let _ = close_tx
                        .send(Outbound::Close(CloseFrame {
                            code: 1009,
                            reason: "message exceeds 4 MiB cap".into(),
                        }))
                        .await;
                    break;
                }
            }
        }
    });
    let inbound = futures::stream::unfold(inbound_rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    futures::pin_mut!(inbound);

    // The dispatcher writes `WsFrame` values; wrap them into our
    // `Outbound` enum at the sink boundary so the writer task can
    // continue to own the WS sender.
    let (frame_tx, mut frame_rx) = mpsc::channel::<WsFrame>(64);
    let frame_bridge_tx = outbound_tx.clone();
    let bridge = tokio::spawn(async move {
        while let Some(frame) = frame_rx.recv().await {
            if frame_bridge_tx
                .send(Outbound::Frame(Box::new(frame)))
                .await
                .is_err()
            {
                break;
            }
        }
    });
    let sink = PollSender::new(frame_tx);

    // Per-connection identity resolved in `ws_handler` (#105): the
    // bearer token's `sub` if the validator extracted one, otherwise
    // the schema sentinel for single-tenant fallback. The dispatcher
    // installs this into the per-task task-local on every dispatched
    // command so storage queries scope correctly.
    // A WebSocket connection may terminate on a different host, so by the
    // transport heuristic its client-registered tools are treated as remote
    // (#243). When the client reported a system id that matches the daemon's,
    // the #248 co-location result overrides that — co-located even over WS — and
    // an optional client host label makes the remote tool note friendlier.
    let auth = AuthContext::new(user_id.into_inner(), TransportKind::WebSocket)
        .with_co_location(co_located)
        .with_client_label(client_label);

    dispatch_loop(Arc::clone(&state.handler), auth, inbound, sink).await;

    // Mirror the pre-refactor cleanup: kill the writer when the
    // dispatcher returns so any in-flight `SendMessage` task observes
    // a closed channel on its next `emit` and shuts down. This is the
    // cancellation path exercised by
    // `ws_send_message_cancels_when_client_disconnects`.
    //
    // Drop our local outbound handle and tear down the helper tasks.
    // The reader owns its own `close_tx` clone, but `reader.abort()`
    // forces it to drop. The bridge task owns the dispatcher-facing
    // `frame_bridge_tx`; aborting it likewise lets the writer observe
    // a closed channel.
    drop(outbound_tx);
    bridge.abort();
    reader.abort();
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
    serve_full_with_system_id(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins,
        None,
        bind,
        shutdown,
    )
    .await
}

/// Like [`serve_full`] but also wires the daemon's own per-machine system id
/// (#248) so co-location can be computed from the client's upgrade header.
// One distinct collaborator per argument; a config struct would be an
// out-of-scope refactor of the public serve API.
#[allow(clippy::too_many_arguments)]
pub async fn serve_full_with_system_id<F>(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Vec<String>,
    daemon_system_id: Option<String>,
    bind: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let app = router_full_with_system_id(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins,
        daemon_system_id,
    );
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

// Server wiring entry point: each argument is a distinct collaborator
// (handler, validators, origins, TLS acceptor, bind, shutdown). Bundling
// them into a config struct would be an out-of-scope refactor of the
// public serve API.
#[allow(clippy::too_many_arguments)]
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
    serve_full_tls_with_system_id(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins,
        None,
        tls_acceptor,
        bind,
        shutdown,
    )
    .await
}

/// Like [`serve_full_tls`] but also wires the daemon's own per-machine system id
/// (#248) for co-location from the client's upgrade header.
// One distinct collaborator per argument; a config struct would be an
// out-of-scope refactor of the public serve API.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "tls")]
pub async fn serve_full_tls_with_system_id<F>(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Vec<String>,
    daemon_system_id: Option<String>,
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

    let app = router_full_with_system_id(
        handler,
        auth_validator,
        login_service,
        auth_discovery,
        allowed_origins,
        daemon_system_id,
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
