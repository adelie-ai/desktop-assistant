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

/// Builder for the WebSocket server's router and `serve` entry points.
///
/// Replaces the old `router_*` / `serve_*` parameter ladders (#279 item 1):
/// every optional collaborator — the login service, auth discovery, the
/// browser-origin allowlist, and the daemon's per-machine system id (#248) —
/// is a `with_*` setter that defaults to off. The two required collaborators
/// (the API `handler` and the bearer-token `auth_validator`) are passed to
/// [`WsServeConfig::new`].
///
/// ```ignore
/// WsServeConfig::new(handler, auth_validator)
///     .with_allowed_origins(origins)
///     .with_daemon_system_id(Some(system_id))
///     .serve(bind, shutdown)
///     .await?;
/// ```
pub struct WsServeConfig {
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Vec<String>,
    daemon_system_id: Option<String>,
}

impl WsServeConfig {
    /// Start from the two required collaborators; everything else defaults off.
    pub fn new(
        handler: Arc<dyn AssistantApiHandler>,
        auth_validator: Arc<dyn WsAuthValidator>,
    ) -> Self {
        Self {
            handler,
            auth_validator,
            login_service: None,
            auth_discovery: None,
            allowed_origins: Vec::new(),
            daemon_system_id: None,
        }
    }

    /// Wire a password-login service backing `POST /login`.
    pub fn with_login_service(mut self, login_service: Option<Arc<dyn WsLoginService>>) -> Self {
        self.login_service = login_service;
        self
    }

    /// Wire an auth-discovery provider backing `GET /auth/config`.
    pub fn with_auth_discovery(mut self, auth_discovery: Option<Arc<dyn WsAuthDiscovery>>) -> Self {
        self.auth_discovery = auth_discovery;
        self
    }

    /// Set the browser-`Origin` allowlist. Empty (the default) rejects every
    /// request that carries an `Origin` header; native clients send none.
    pub fn with_allowed_origins(mut self, allowed_origins: Vec<String>) -> Self {
        self.allowed_origins = allowed_origins;
        self
    }

    /// Wire the daemon's own per-machine system id (#248) so the `/ws` handler
    /// can compute exact tool-locality co-location from the client's
    /// `x-adelie-system-id` upgrade header. `None` reproduces the
    /// transport-heuristic-only behaviour.
    pub fn with_daemon_system_id(mut self, daemon_system_id: Option<String>) -> Self {
        self.daemon_system_id = daemon_system_id;
        self
    }

    /// Build the axum [`Router`] for these settings.
    pub fn into_router(self) -> Router {
        let state = WsServerState {
            handler: self.handler,
            auth_validator: self.auth_validator,
            login_service: self.login_service,
            auth_discovery: self.auth_discovery,
            allowed_origins: Arc::new(self.allowed_origins),
            daemon_system_id: Arc::new(self.daemon_system_id),
        };

        Router::new()
            .route("/ws", get(ws_handler))
            .route("/login", post(login_handler))
            .route("/auth/config", get(auth_config_handler))
            .with_state(state)
    }

    /// Bind `bind` and serve until `shutdown` resolves (plaintext).
    pub async fn serve<F>(self, bind: SocketAddr, shutdown: F) -> anyhow::Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let app = self.into_router();
        let listener = tokio::net::TcpListener::bind(bind).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await?;
        Ok(())
    }

    /// Bind `bind` and serve over TLS until `shutdown` resolves.
    #[cfg(feature = "tls")]
    pub async fn serve_tls<F>(
        self,
        tls_acceptor: tokio_rustls::TlsAcceptor,
        bind: SocketAddr,
        shutdown: F,
    ) -> anyhow::Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let app = self.into_router();
        let listener = tokio::net::TcpListener::bind(bind).await?;
        serve_tls_accept_loop(listener, tls_acceptor, app, shutdown).await
    }
}

/// Build a router from just the required collaborators (no login, no auth
/// discovery, no origin allowlist, transport-heuristic co-location). Thin
/// shim over [`WsServeConfig`] kept for the test suite's many call sites.
pub fn router(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
) -> Router {
    WsServeConfig::new(handler, auth_validator).into_router()
}

/// Build a router with an optional login service. Thin shim over
/// [`WsServeConfig`].
pub fn router_with_login(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
) -> Router {
    WsServeConfig::new(handler, auth_validator)
        .with_login_service(login_service)
        .into_router()
}

/// Build a router with optional login service, auth discovery, and an origin
/// allowlist. Thin shim over [`WsServeConfig`].
pub fn router_full(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    login_service: Option<Arc<dyn WsLoginService>>,
    auth_discovery: Option<Arc<dyn WsAuthDiscovery>>,
    allowed_origins: Vec<String>,
) -> Router {
    WsServeConfig::new(handler, auth_validator)
        .with_login_service(login_service)
        .with_auth_discovery(auth_discovery)
        .with_allowed_origins(allowed_origins)
        .into_router()
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
/// pair expected by [`dispatch_loop`].
///
/// The dispatcher runs as a spawned task, reading parsed `WsRequest`s from
/// an inbound mpsc and writing `WsFrame`s into an outbound mpsc. This
/// function owns BOTH halves of the (split) socket in one **biased select
/// loop** so we can enforce a critical ordering invariant for DT-6:
///
/// > Always flush already-queued outbound frames to the wire *before*
/// > reading the next inbound frame.
///
/// Why this matters: tungstenite couples the two directions. The instant we
/// read a peer `Close` frame off the socket, its write state flips to
/// `ClosedByPeer` and every subsequent `send` returns `SendAfterclosing` —
/// so any reply still sitting in our buffers when the `Close` is read is
/// lost. With a separate reader task (the old design) the reader raced
/// ahead, read the `Close`, and the writer's queued replies were dropped.
/// By draining outbound first in a biased select, every reply for a request
/// that arrived before the peer's `Close` is on the wire before we ever read
/// that `Close`.
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

    let (mut ws_tx, mut ws_rx) = socket.split();

    // Dispatcher → socket: the dispatcher writes `WsFrame` values into this
    // channel via a `PollSender` sink; our select loop drains it to the wire.
    let (frame_tx, mut frame_rx) = mpsc::channel::<WsFrame>(64);
    // Socket → dispatcher: parsed requests (or DT-5 decode errors) flow here.
    // Capacity 1 is deliberate (DT-6): it serializes request→reply so the next
    // inbound frame isn't read until the dispatcher has taken — and, since it
    // emits the reply before pulling the next item, replied to — the previous
    // request. Combined with the biased outbound-first drain, that guarantees a
    // request's reply is on the wire before any later frame (e.g. a Close) is
    // read and flips tungstenite's write state.
    let (inbound_tx, inbound_rx) = mpsc::channel::<anyhow::Result<WsRequest>>(1);

    let sink = PollSender::new(frame_tx);

    // Per-connection identity resolved in `ws_handler` (#105): the bearer
    // token's `sub` if the validator extracted one, otherwise the schema
    // sentinel for single-tenant fallback. The dispatcher installs this into
    // the per-task task-local on every dispatched command so storage queries
    // scope correctly.
    // A WebSocket connection may terminate on a different host, so by the
    // transport heuristic its client-registered tools are treated as remote
    // (#243). When the client reported a system id that matches the daemon's,
    // the #248 co-location result overrides that — co-located even over WS —
    // and an optional client host label makes the remote tool note friendlier.
    let auth = AuthContext::new(user_id.into_inner(), TransportKind::WebSocket)
        .with_co_location(co_located)
        .with_client_label(client_label);

    let handler = Arc::clone(&state.handler);
    let mut dispatcher = tokio::spawn(async move {
        // The inbound stream is built inside the spawn so it owns `inbound_rx`
        // and satisfies the `'static` bound (`dispatch_loop` requires
        // `Stream + Unpin`, hence the `pin_mut!`).
        let inbound = futures::stream::unfold(inbound_rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        futures::pin_mut!(inbound);
        dispatch_loop(handler, auth, inbound, sink).await;
    });

    // Helper: serialize+write one frame; `Ok(false)` ⇒ the peer's write side
    // is gone (stop writing).
    async fn write_frame(
        ws_tx: &mut futures::stream::SplitSink<WebSocket, Message>,
        frame: WsFrame,
    ) -> bool {
        let Ok(text) = serde_json::to_string(&frame) else {
            return true; // unserializable: skip, keep going
        };
        ws_tx.send(Message::Text(text.into())).await.is_ok()
    }

    // Set when an inbound error (e.g. the 4 MiB cap) means we owe the client
    // an explicit RFC 6455 close before tearing down.
    let mut pending_close: Option<CloseFrame> = None;
    // Cleared once the peer has stopped sending (graceful Close / stream end /
    // fatal inbound error). After that we only flush remaining outbound.
    let mut inbound_open = true;
    // Becomes false once a write fails: the peer's read side is gone.
    let mut writable = true;

    // Concurrent read/write loop. Outbound is drained continuously (subscription
    // and SendMessage-stream events have no corresponding inbound frame to gate
    // on) while inbound is read concurrently. The `biased` outbound-first
    // ordering is what makes this DT-6-safe: tungstenite couples the directions
    // — the instant we read a peer `Close` its write state flips and any reply
    // still buffered is lost. Draining `frame_rx` before ever polling `ws_rx`
    // means a queued reply reaches the wire first. The cap-1 `inbound_tx`
    // serializes request/response: the dispatcher emits a request's reply
    // BEFORE pulling the next inbound item, so a burst of N requests followed by
    // a Close flushes all N replies before the Close is read.
    while inbound_open {
        tokio::select! {
            biased;

            // Drain dispatcher output to the socket.
            frame = frame_rx.recv() => {
                match frame {
                    Some(frame) => {
                        if writable && !write_frame(&mut ws_tx, frame).await {
                            // Peer's read side is gone: stop. The dispatcher
                            // will observe the closed sink on teardown and
                            // cancel any in-flight turn.
                            writable = false;
                            break;
                        }
                    }
                    None => break, // dispatcher finished and dropped its sink
                }
            }

            // Read the next inbound frame (only while the peer is sending).
            item = ws_rx.next(), if inbound_open => {
                match item {
                    Some(Ok(Message::Text(text))) => {
                        let parsed = serde_json::from_str::<WsRequest>(&text).map_err(|e| {
                            // DT-5: forward the decode failure so the dispatcher
                            // emits an empty-id error frame instead of leaving
                            // the client hanging on a silently dropped request.
                            warn!("invalid ws json: {e}");
                            anyhow::anyhow!("invalid request json: {e}")
                        });
                        // cap-1 channel: this awaits until the dispatcher has
                        // taken (and replied to) the previous request, which —
                        // with the biased drain above — is the DT-6 serializer.
                        if inbound_tx.send(parsed).await.is_err() {
                            inbound_open = false; // dispatcher gone
                        }
                        // Flush the reply for the request we just took before
                        // reading the next frame (which may be a Close that
                        // flips tungstenite's write state). The reply was
                        // emitted into the dispatcher's internal channel before
                        // it pulled this request, but still has to hop through
                        // the dispatcher's writer task into `frame_rx`. Drain
                        // what's buffered, then catch the in-transit reply with
                        // ONE short bounded wait. We do NOT loop the wait, so an
                        // open SendMessage stream isn't chased (its deltas drain
                        // continuously via the select arm above instead).
                        if writable {
                            loop {
                                match frame_rx.try_recv() {
                                    Ok(frame) => {
                                        if !write_frame(&mut ws_tx, frame).await {
                                            writable = false;
                                            break;
                                        }
                                    }
                                    Err(mpsc::error::TryRecvError::Empty) => {
                                        if let Ok(Some(frame)) = tokio::time::timeout(
                                            std::time::Duration::from_millis(20),
                                            frame_rx.recv(),
                                        )
                                        .await
                                        {
                                            writable = write_frame(&mut ws_tx, frame).await;
                                        }
                                        break;
                                    }
                                    Err(mpsc::error::TryRecvError::Disconnected) => {
                                        inbound_open = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        // Graceful peer close (or stream end). Replies for
                        // already-received requests were drained above.
                        inbound_open = false;
                    }
                    Some(Ok(_)) => {} // Ping/Pong/Binary: ignored (axum auto-pongs).
                    Some(Err(e)) => {
                        // Most commonly tungstenite's `Capacity(MessageTooLong)`
                        // from our 4 MiB cap; the message never reached user
                        // code, so we owe the client an explicit close.
                        warn!("ws inbound error, closing: {e}");
                        pending_close = Some(CloseFrame {
                            // RFC 6455 §7.4.1: 1009 = "Message Too Big".
                            code: 1009,
                            reason: "message exceeds 4 MiB cap".into(),
                        });
                        inbound_open = false;
                    }
                }
            }
        }
    }

    // Inbound is done. Drop our `inbound_tx` so the dispatcher's stream ends.
    drop(inbound_tx);

    // Teardown drain. Flush frames that are *already buffered* at this instant —
    // do NOT block for more. A still-running SendMessage turn (the cancellation
    // path) emits chunks continuously; chasing them would keep the turn's
    // `on_chunk` succeeding so it never cancels. We drain what's buffered, then
    // drop `frame_rx`, which closes the dispatcher's sink and makes the turn's
    // next `emit` return the cancellation signal
    // (`ws_send_message_cancels_when_client_disconnects`).
    while writable {
        match frame_rx.try_recv() {
            Ok(frame) => writable = write_frame(&mut ws_tx, frame).await,
            Err(_) => break, // empty or disconnected
        }
    }

    // Send the deferred protocol close (oversize-frame path) if we owe one.
    if let Some(close) = pending_close
        && writable
    {
        let _ = ws_tx.send(Message::Close(Some(close))).await;
    }

    // Drop `frame_rx` BEFORE awaiting the dispatcher so a still-running turn's
    // next `emit` observes the closed sink and cancels.
    drop(frame_rx);
    if tokio::time::timeout(std::time::Duration::from_secs(5), &mut dispatcher)
        .await
        .is_err()
    {
        dispatcher.abort();
    }
}

/// Serve with the minimal config until the process is killed. Thin shim over
/// [`WsServeConfig::serve`].
pub async fn serve(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    bind: SocketAddr,
) -> anyhow::Result<()> {
    serve_with_shutdown(handler, auth_validator, bind, pending::<()>()).await
}

/// Serve with the minimal config until `shutdown` resolves. Thin shim over
/// [`WsServeConfig::serve`] kept for the test suite.
pub async fn serve_with_shutdown<F>(
    handler: Arc<dyn AssistantApiHandler>,
    auth_validator: Arc<dyn WsAuthValidator>,
    bind: SocketAddr,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    WsServeConfig::new(handler, auth_validator)
        .serve(bind, shutdown)
        .await
}

/// Source of accepted TCP connections for [`serve_tls_accept_loop`].
///
/// Exists purely as a test seam: production passes a bound
/// `tokio::net::TcpListener`; tests inject accept errors to pin down the
/// loop's survival behaviour (review finding DT-1).
#[cfg(feature = "tls")]
trait TcpAcceptSource: Send {
    fn accept(&mut self) -> impl Future<Output = std::io::Result<tokio::net::TcpStream>> + Send;
}

#[cfg(feature = "tls")]
impl TcpAcceptSource for tokio::net::TcpListener {
    async fn accept(&mut self) -> std::io::Result<tokio::net::TcpStream> {
        tokio::net::TcpListener::accept(self).await.map(|(s, _)| s)
    }
}

/// TLS accept loop, factored out of [`WsServeConfig::serve_tls`] so its
/// error handling is unit-testable (DT-1).
///
/// A failed `accept()` (`ECONNABORTED`, `EMFILE`, …) is transient: it is
/// logged and the loop keeps serving after a short pause, matching the UDS
/// listener and the non-TLS axum path, which both survive accept errors.
/// Before the DT-1 fix a single accept error returned from this function and
/// permanently killed the WS transport until a daemon restart.
#[cfg(feature = "tls")]
async fn serve_tls_accept_loop<A, F>(
    mut listener: A,
    tls_acceptor: tokio_rustls::TlsAcceptor,
    app: Router,
    shutdown: F,
) -> anyhow::Result<()>
where
    A: TcpAcceptSource,
    F: Future<Output = ()> + Send + 'static,
{
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder as ConnBuilder;
    use hyper_util::service::TowerToHyperService;

    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            result = listener.accept() => {
                let tcp_stream = match result {
                    Ok(stream) => stream,
                    Err(e) => {
                        // Transient accept failures (ECONNABORTED, EMFILE, …)
                        // must not end the loop (DT-1). The brief pause keeps
                        // a persistent condition like fd exhaustion from
                        // spinning the loop hot.
                        warn!("TLS accept failed; continuing: {e}");
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        continue;
                    }
                };
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

#[cfg(test)]
mod client_context_header_tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderName, HeaderValue};

    fn header_map(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static(api::WS_CLIENT_CONTEXT_HEADER),
            HeaderValue::from_str(value).expect("valid header value"),
        );
        headers
    }

    #[test]
    fn client_context_header_round_trips() {
        // Acceptance (f): a full ClientContext survives the base64(JSON) header
        // encode/decode the WS upgrade uses (#549).
        let ctx = api::ClientContext {
            real_name: Some("Ada Lovelace".into()),
            username: Some("ada".into()),
            home_dir: Some("/home/ada".into()),
            hostname: Some("analytical-engine".into()),
            timezone: Some("Europe/London".into()),
            os: Some("Ubuntu 24.04".into()),
        };
        let encoded = encode_client_context_header(&ctx).expect("encode");
        let decoded = decode_client_context_header(&header_map(&encoded)).expect("decode");
        assert_eq!(decoded, ctx);
    }

    #[test]
    fn absent_header_yields_no_context() {
        assert_eq!(decode_client_context_header(&HeaderMap::new()), None);
    }

    #[test]
    fn malformed_header_is_fail_closed_none() {
        // Not base64 at all.
        assert_eq!(
            decode_client_context_header(&header_map("not valid base64 !!")),
            None
        );
        // Valid base64, but not JSON for a ClientContext.
        let junk = base64::engine::general_purpose::STANDARD.encode("this is not json");
        assert_eq!(decode_client_context_header(&header_map(&junk)), None);
    }
}

#[cfg(all(test, feature = "tls"))]
mod tls_accept_tests {
    use super::*;
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Accept source that yields one transient error, then pends forever.
    /// Counts calls so the test can observe whether the loop retried.
    struct FlakyAccept {
        yielded_error: bool,
        calls: StdArc<AtomicUsize>,
    }

    impl TcpAcceptSource for FlakyAccept {
        async fn accept(&mut self) -> std::io::Result<tokio::net::TcpStream> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if !self.yielded_error {
                self.yielded_error = true;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionAborted,
                    "transient accept failure",
                ));
            }
            std::future::pending().await
        }
    }

    fn test_tls_acceptor() -> tokio_rustls::TlsAcceptor {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let key =
            rustls_pki_types::PrivateKeyDer::try_from(cert.signing_key.serialize_der()).unwrap();
        let config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert.cert.der().clone()], key)
            .unwrap();
        tokio_rustls::TlsAcceptor::from(StdArc::new(config))
    }

    /// DT-1: one transient accept error must not end the serve loop. The
    /// loop should log, pause briefly, and call `accept` again — only the
    /// shutdown future may end it.
    #[tokio::test]
    async fn tls_accept_error_does_not_kill_the_server() {
        let calls = StdArc::new(AtomicUsize::new(0));
        let listener = FlakyAccept {
            yielded_error: false,
            calls: StdArc::clone(&calls),
        };
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = shutdown_rx.await;
        };

        let loop_task = tokio::spawn(serve_tls_accept_loop(
            listener,
            test_tls_acceptor(),
            Router::new(),
            shutdown,
        ));

        // Give the loop time to observe the accept error and retry.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        assert!(
            !loop_task.is_finished(),
            "a single transient accept error must not end the TLS serve loop"
        );
        assert!(
            calls.load(Ordering::SeqCst) >= 2,
            "the loop must call accept again after a transient error"
        );

        // Clean shutdown still works.
        let _ = shutdown_tx.send(());
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), loop_task)
            .await
            .expect("loop must exit on shutdown")
            .expect("loop task must not panic");
        assert!(result.is_ok(), "shutdown exit must be Ok: {result:?}");
    }
}
