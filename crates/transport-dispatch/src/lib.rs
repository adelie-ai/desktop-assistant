//! Transport-agnostic dispatcher for the assistant API.
//!
//! This crate factors the per-connection request/event loop out of the
//! WebSocket adapter so both the WS and UDS transports can share it
//! verbatim. Per `docs/architecture-evolution.md` rule #5, the transport
//! is pluggable and `AssistantApiHandler` + `EventSink` are the only
//! interfaces that touch core logic.
//!
//! ## Shape
//!
//! [`dispatch_loop`] consumes a [`futures::Stream`] of [`WsRequest`] frames
//! (one per inbound request, already deserialized) and writes [`WsFrame`]
//! values into a [`futures::Sink`]. Authentication has already happened by
//! the time `dispatch_loop` runs — the caller passes a pre-validated
//! [`AuthContext`] that the dispatcher attaches to spawned tasks and
//! threads through into per-user state lookups (issue #105).
//!
//! Behaviorally identical to the old `handle_socket` in `ws-interface`.
//! Transports differ only in how they produce the stream/sink and how
//! they validate the JWT.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, AssistantApiHandler, EventSink};
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use desktop_assistant_core::ports::session::{SessionId, with_session_id};
use desktop_assistant_core::ports::transport::{
    with_client_label, with_co_location, with_transport_kind,
};
use futures::sink::{Sink, SinkExt};
use futures::stream::{Stream, StreamExt};
use tokio::sync::mpsc;
use tracing::{debug, warn};

pub use api::{WsFrame, WsRequest};
// Re-exported so transport adapters can name their kind without each taking a
// direct dependency on `desktop-assistant-core` (#243).
pub use desktop_assistant_core::domain::TransportKind;

/// Outbound frame buffer; matches the WS adapter's pre-refactor sizing.
const OUTBOUND_BUFFER: usize = 64;

/// Monotonic source of per-connection session ids (#261). A process-local
/// counter is sufficient: session ids are ephemeral, never persisted, and
/// never leave the process — they only need to be unique among live
/// connections so one client's registered tools don't bleed into another's.
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

/// Mint a fresh, process-unique session id for a newly-accepted connection.
fn mint_session_id() -> String {
    format!("sess-{}", NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed))
}

/// Pre-validated identity for the connection.
///
/// Carries the JWT `sub` (`user_id`), the connection's [`TransportKind`], and
/// (issue #248) the authoritative per-machine **system-id co-location** result
/// plus an optional client-reported host label. The dispatcher uses these to
/// scope per-user state (#105) and tag tool co-location (#243/#248). Structured
/// as a type so further per-connection fields can be added without changing the
/// dispatcher signature.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// JWT `sub` claim.
    pub user_id: String,
    /// How this connection reaches the daemon — the **fallback** co-location
    /// signal (UDS/D-Bus ⇒ same machine, WebSocket ⇒ possibly remote), used only
    /// when [`Self::co_located`] is `None` (#243).
    pub transport: TransportKind,
    /// Authoritative co-location from the per-machine system-id handshake
    /// (#248): `Some(true)` when the client's reported id equals the daemon's
    /// own (same machine — even over WebSocket), `Some(false)` when they differ,
    /// `None` when the client reported no id (an older client ⇒ fall back to
    /// [`Self::transport`]).
    pub co_located: Option<bool>,
    /// A client-reported host label (#248) for a friendlier remote tool note
    /// (e.g. `your device 'laptop'`); `None` when the client sent none.
    pub client_label: Option<String>,
    /// Unique id for this login session / connection (#261), minted by the
    /// constructors. Installed as a task-local around every request so
    /// per-connection state (client-local tool registration) keys on it
    /// instead of the user — two windows of the same user stay independent.
    pub session_id: String,
}

impl AuthContext {
    /// Build a context for a connection on `transport`, with no system-id
    /// co-location result (older-client / no-id path — co-location falls back to
    /// the transport heuristic). Each transport adapter passes its own kind so
    /// the per-turn tool note can tag tool localities.
    pub fn new(user_id: impl Into<String>, transport: TransportKind) -> Self {
        Self {
            user_id: user_id.into(),
            transport,
            co_located: None,
            client_label: None,
            session_id: mint_session_id(),
        }
    }

    /// Attach the authoritative system-id co-location result (#248): `Some(true)`
    /// / `Some(false)` when the client reported an id the daemon could compare,
    /// `None` to defer to the transport heuristic.
    pub fn with_co_location(mut self, co_located: Option<bool>) -> Self {
        self.co_located = co_located;
        self
    }

    /// Attach a client-reported host label (#248) for the remote tool note.
    pub fn with_client_label(mut self, label: Option<String>) -> Self {
        self.client_label = label.filter(|l| !l.trim().is_empty());
        self
    }

    /// Convenience for tests that don't need a real subject. Defaults to the
    /// co-located UDS transport with no system-id result.
    pub fn anonymous() -> Self {
        Self {
            user_id: "anonymous".to_string(),
            transport: TransportKind::Uds,
            co_located: None,
            client_label: None,
            session_id: mint_session_id(),
        }
    }
}

/// Sink-side adapter: forwards canonical events as `WsFrame::Event`.
///
/// The dispatcher hands a clone of this to every spawned streaming task
/// so events flow back through the same outbound channel as command
/// results.
pub struct ChannelEventSink {
    tx: mpsc::Sender<WsFrame>,
}

impl ChannelEventSink {
    pub fn new(tx: mpsc::Sender<WsFrame>) -> Self {
        Self { tx }
    }
}

#[async_trait::async_trait]
impl EventSink for ChannelEventSink {
    async fn emit(&self, event: api::Event) -> bool {
        self.tx.send(WsFrame::Event { event }).await.is_ok()
    }
}

/// Best-effort sink registered in [`ConversationSubscriptions`] for fanning a
/// turn's events to OTHER connections viewing the conversation (#1). Uses
/// `try_send`, so a slow or backed-up viewer drops events (and resyncs on its
/// next reload) instead of backpressuring the *originating* turn — the live
/// render is a convenience, never a guarantee, and the sender's own stream
/// stays reliable on its separate [`ChannelEventSink`].
struct FanoutTargetSink {
    tx: mpsc::Sender<WsFrame>,
}

#[async_trait::async_trait]
impl EventSink for FanoutTargetSink {
    async fn emit(&self, event: api::Event) -> bool {
        // Non-blocking: a full or closed channel drops rather than awaiting.
        self.tx.try_send(WsFrame::Event { event }).is_ok()
    }
}

/// Run the dispatcher loop on `inbound` / `outbound` until the inbound
/// stream ends or the outbound sink errors.
///
/// Behavior preserved from the old WS `handle_socket`:
///
/// - Each [`WsRequest::command`] is dispatched through `handler`.
/// - `SendMessage` is streamed: an immediate `SendMessageAck` result frame
///   carrying the turn `request_id` (and, when a registry is attached, the
///   `task_id`), then `AssistantDelta` / `AssistantCompleted` /
///   `AssistantError` events stamped with that same `request_id`, forwarded
///   through a [`ChannelEventSink`] into the outbound channel. The ack's
///   `request_id` is what a socket client correlates its response stream by,
///   mirroring the D-Bus `SendPrompt` reply (voice#49).
/// - `SetConfig` emits both a `Result` and a `ConfigChanged` event on
///   success.
/// - Other commands round-trip through `handler.handle_command` and
///   come back as a single `Result` (or `Error`).
///
/// When the outbound sink errors (transport closed mid-write) the loop
/// exits and any in-flight streaming task's next `emit` call will fail —
/// this is the cancellation channel used today by both transports.
pub async fn dispatch_loop<R, W>(
    handler: Arc<dyn AssistantApiHandler>,
    auth: AuthContext,
    mut inbound: R,
    outbound: W,
) where
    R: Stream<Item = anyhow::Result<WsRequest>> + Unpin,
    W: Sink<WsFrame> + Unpin + Send + 'static,
    W::Error: std::fmt::Debug + Send,
{
    // Forward outbound frames through an mpsc so multiple producers
    // (the inbound loop itself, plus per-request spawned tasks for
    // SendMessage streaming) can all write into the same transport.
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(OUTBOUND_BUFFER);

    let mut outbound_sink = outbound;
    let writer = tokio::spawn(async move {
        while let Some(frame) = out_rx.recv().await {
            if let Err(e) = outbound_sink.send(frame).await {
                debug!("outbound sink closed: {e:?}");
                break;
            }
        }
    });

    // `auth` is stable for the connection; we clone the user id into
    // every handler invocation so the per-task `with_user_id` scope
    // (#105) is established before any storage call. Spawned
    // `SendMessage` tasks get their own clone so the scope still
    // applies after the spawn boundary (`task_local` doesn't inherit
    // across `tokio::spawn`).
    let user_id = UserId::new(auth.user_id.clone());

    // The connection's login-session id (#261): installed as a task-local
    // around every request so client-local tool registration keys on the
    // connection, not the user. Like `user_id` it is re-installed on spawned
    // send tasks (task-locals don't cross `tokio::spawn`).
    let session_id = SessionId::new(auth.session_id.clone());

    // The connection's transport (#243): installed around the send-message
    // dispatch so the turn loop can infer tool co-location. Like `user_id` it
    // is re-installed on spawned send tasks because `task_local`s don't cross
    // `tokio::spawn`. Only the send-message paths run the LLM turn, so the
    // command/subscribe paths don't need it.
    let transport = auth.transport;
    // The authoritative per-machine system-id co-location result and an
    // optional client-reported host label (#248), installed alongside the
    // transport so the turn loop prefers the id match over the heuristic.
    let co_located = auth.co_located;
    let client_label = auth.client_label.clone();

    // Per-connection state for `SubscribeBackgroundTasks` (#114).
    // At most one forwarder runs per connection — Subscribe is
    // idempotent (a second one observes the existing handle and Acks
    // without spawning a duplicate). Unsubscribe aborts the handle;
    // when the connection drops we abort it as part of the loop's
    // teardown so the broadcast receiver is dropped cleanly.
    let mut bg_subscription: Option<tokio::task::JoinHandle<()>> = None;

    // Per-connection conversation subscriptions (#1 live multi-client sync).
    // When the handler provides the registry, register this connection's
    // best-effort fan-out sink under its session id so a turn in any
    // conversation it later subscribes to is delivered here. The set of
    // subscribed conversations is applied by `SubscribeConversations`; the
    // registration is torn down on disconnect. `None` keeps the prior behaviour
    // (turn events reach only the initiating connection).
    let conv_subs = handler.conversation_subscriptions();
    if let Some(ref subs) = conv_subs {
        subs.register(
            &auth.session_id,
            &auth.user_id,
            Arc::new(FanoutTargetSink { tx: out_tx.clone() }),
        );
    }

    while let Some(item) = inbound.next().await {
        let req = match item {
            Ok(req) => req,
            Err(e) => {
                // DT-5: a frame that failed to decode used to be warn-logged
                // and silently dropped, leaving a request/response client
                // hanging forever on its reply. Emit an explicit error frame.
                // The id is empty because the request id is unknowable from
                // an unparseable frame; the loop keeps serving.
                warn!("inbound frame decode error: {e}");
                if out_tx
                    .send(WsFrame::Error {
                        id: String::new(),
                        error: format!("invalid request json: {e}"),
                    })
                    .await
                    .is_err()
                {
                    break;
                }
                continue;
            }
        };

        debug!(id = %req.id, "request received");

        match req.command {
            api::Command::SendMessage {
                conversation_id,
                content,
                override_selection,
                system_refinement,
                idempotency_key,
            } => {
                // Per-request id for event correlation (matches old WS path).
                let request_id = uuid::Uuid::new_v4().to_string();
                // Reliable delivery to THIS connection. The handler additionally
                // fans each turn event to other connections viewing the
                // conversation (#1) — done there, not here, so the fan-out is
                // transport-agnostic and also covers voice turns over D-Bus.
                let sink: Arc<dyn EventSink> = Arc::new(ChannelEventSink::new(out_tx.clone()));

                // Prefer the new `start_send_message` registration
                // path so the ack can carry the registered task id
                // (#114). When the handler opts out (no registry
                // attached — single-tenant tests and the like) we
                // fall back to the inline-streaming path that pre-#114
                // transports used. Either way the ack is a
                // `SendMessageAck` carrying the turn `request_id` that
                // streamed events are stamped with, so a socket client
                // can correlate the response (voice#49); the legacy path
                // simply leaves `task_id` empty.
                let task_id = with_co_location(
                    co_located,
                    with_client_label(
                        client_label.clone(),
                        with_transport_kind(
                            transport,
                            with_user_id(
                                user_id.clone(),
                                with_session_id(
                                    session_id.clone(),
                                    handler.start_send_message(
                                        conversation_id.clone(),
                                        content.clone(),
                                        override_selection.clone(),
                                        system_refinement.clone(),
                                        request_id.clone(),
                                        idempotency_key.clone(),
                                        Arc::clone(&sink),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
                .await;

                match task_id {
                    Ok(Some(id)) => {
                        // Handler registered the turn via the registry
                        // and the body is already running in the
                        // background — emit the new typed ack and
                        // continue the loop. The ack carries BOTH the
                        // turn `request_id` (which every streamed
                        // `Assistant*` event is stamped with) and the
                        // registry `task_id` (which the `Task*` events
                        // carry), so a socket client can correlate the
                        // streamed RESPONSE back to its send the same way
                        // the D-Bus `SendPrompt` reply does (voice#49).
                        if out_tx
                            .send(WsFrame::Result {
                                id: req.id.clone(),
                                result: api::CommandResult::SendMessageAck {
                                    request_id: request_id.clone(),
                                    task_id: id.0,
                                },
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(None) => {
                        // Legacy path (no registry attached — single-tenant
                        // tests and the like): ack first, then drive the
                        // streaming send on a spawned task so the dispatcher
                        // remains responsive. The ack still carries the turn
                        // `request_id` (with an empty `task_id`, since no task
                        // was registered) so a socket client can correlate the
                        // streamed response even on this path (voice#49).
                        if out_tx
                            .send(WsFrame::Result {
                                id: req.id.clone(),
                                result: api::CommandResult::SendMessageAck {
                                    request_id: request_id.clone(),
                                    task_id: String::new(),
                                },
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                        let handler = Arc::clone(&handler);
                        let user_id_for_task = user_id.clone();
                        let session_id_for_task = session_id.clone();
                        let transport_for_task = transport;
                        // Re-install the #248 co-location result + client label
                        // inside the spawn, like the transport, since task-locals
                        // don't cross `tokio::spawn`.
                        let co_located_for_task = co_located;
                        let client_label_for_task = client_label.clone();
                        tokio::spawn(async move {
                            let _ = with_co_location(
                                co_located_for_task,
                                with_client_label(
                                    client_label_for_task,
                                    with_transport_kind(
                                        transport_for_task,
                                        with_user_id(
                                            user_id_for_task,
                                            with_session_id(
                                                session_id_for_task,
                                                handler.handle_send_message_with_override(
                                                    conversation_id,
                                                    content,
                                                    override_selection,
                                                    system_refinement,
                                                    request_id,
                                                    idempotency_key,
                                                    sink,
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            )
                            .await;
                        });
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

            // Subscribe/Unsubscribe for background-task events (#114).
            // The handler's command arm only Acks at the protocol
            // level; the actual broadcast→connection forwarder is a
            // dispatcher concern because it owns the outbound channel
            // and the per-connection lifetime.
            api::Command::SubscribeBackgroundTasks => {
                if bg_subscription.is_some() {
                    // Idempotent: a second subscribe on a connection
                    // that already has a live forwarder just Acks.
                    if out_tx
                        .send(WsFrame::Result {
                            id: req.id,
                            result: api::CommandResult::Ack,
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
                let receiver = with_user_id(
                    user_id.clone(),
                    with_session_id(session_id.clone(), handler.subscribe_user_events()),
                )
                .await;
                let Some(mut receiver) = receiver else {
                    // No-op handler → no event source. Surface as a
                    // clean error frame so the client knows
                    // subscription is unsupported here.
                    if out_tx
                        .send(WsFrame::Error {
                            id: req.id,
                            error: "background-task subscription not supported by this handler"
                                .to_string(),
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                    continue;
                };
                let out_tx_for_forwarder = out_tx.clone();
                let handle = tokio::spawn(async move {
                    use tokio::sync::broadcast::error::RecvError;
                    loop {
                        match receiver.recv().await {
                            Ok(event) => {
                                if out_tx_for_forwarder
                                    .send(WsFrame::Event { event })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            Err(RecvError::Lagged(n)) => {
                                // DT-11: the gap is invisible to the client
                                // unless we say so — forward a synthetic
                                // connection-level (empty-id) error frame so
                                // the client knows to refetch state instead
                                // of trusting an event stream with a hole in
                                // it. The loop then resumes at the oldest
                                // surviving event.
                                warn!(
                                    "background-task event subscriber lagged \
                                     by {n} events; oldest dropped"
                                );
                                if out_tx_for_forwarder
                                    .send(WsFrame::Error {
                                        id: String::new(),
                                        error: format!(
                                            "{n} events dropped (subscriber lagged); \
                                             refetch state to resync"
                                        ),
                                    })
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                                continue;
                            }
                            Err(RecvError::Closed) => break,
                        }
                    }
                });
                bg_subscription = Some(handle);
                if out_tx
                    .send(WsFrame::Result {
                        id: req.id,
                        result: api::CommandResult::Ack,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }
            api::Command::UnsubscribeBackgroundTasks => {
                if let Some(handle) = bg_subscription.take() {
                    handle.abort();
                }
                if out_tx
                    .send(WsFrame::Result {
                        id: req.id,
                        result: api::CommandResult::Ack,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }

            api::Command::SubscribeConversations { conversation_ids } => {
                // Set-replace the conversations this connection is viewing (#1).
                // When the registry is attached, the connection's fan-out sink
                // (registered on connect) now receives turn events for these
                // conversations from other connections. A no-registry handler
                // just Acks — the feature is simply off, the client unaffected.
                if let Some(ref subs) = conv_subs {
                    subs.set_subscriptions(&auth.session_id, conversation_ids);
                }
                if out_tx
                    .send(WsFrame::Result {
                        id: req.id,
                        result: api::CommandResult::Ack,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }

            api::Command::SetConfig { changes } => {
                let res = with_user_id(
                    user_id.clone(),
                    with_session_id(
                        session_id.clone(),
                        handler.handle_command(api::Command::SetConfig { changes }),
                    ),
                )
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
                let res = with_user_id(
                    user_id.clone(),
                    with_session_id(session_id.clone(), handler.handle_command(other)),
                )
                .await;
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

    // Abort the background-task forwarder (if any) so the broadcast
    // receiver is dropped synchronously rather than waiting on a
    // future Lagged/Closed observation. Without this the receiver
    // would buffer events nobody is listening for until the registry
    // shuts down (#114).
    if let Some(handle) = bg_subscription.take() {
        handle.abort();
    }

    // Drop this connection's conversation subscriptions + fan-out sink (#1) so
    // the registry stays bounded by live connections and no turn is routed to a
    // dead channel.
    if let Some(ref subs) = conv_subs {
        subs.unregister(&auth.session_id);
    }

    // Connection closed: evict any client-local tools this session
    // registered (#261) so a long-lived daemon doesn't accumulate stale
    // per-session buckets across reconnects. Run inside the session scope
    // so the handler's coordinator can read which session ended.
    with_session_id(session_id.clone(), handler.on_session_end()).await;

    // Drop the inbound side's `out_tx` so the writer can finish
    // draining whatever is buffered. Spawned `SendMessage` tasks
    // still hold their own clones; they continue to emit until the
    // outbound sink errors (peer hung up) or the streaming handler
    // finishes. When the outbound sink errors the writer exits,
    // `out_rx` drops, and the next `emit` from any in-flight task
    // returns `false` — the cancellation signal core relies on.
    drop(out_tx);
    let _ = writer.await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_context_defaults_have_no_co_location_result() {
        // #248: a plain `new`/`anonymous` context carries no authoritative
        // system-id result, so co-location falls back to the transport
        // heuristic (the Phase-1, #243, behaviour for older clients).
        let a = AuthContext::new("dave", TransportKind::WebSocket);
        assert_eq!(a.co_located, None);
        assert_eq!(a.client_label, None);
        assert_eq!(a.client_context, None);
        assert_eq!(AuthContext::anonymous().co_located, None);
        assert_eq!(AuthContext::anonymous().client_context, None);
    }

    #[test]
    fn auth_context_builders_attach_co_location_and_label() {
        let a = AuthContext::new("dave", TransportKind::WebSocket)
            .with_co_location(Some(true))
            .with_client_label(Some("laptop".to_string()));
        assert_eq!(a.co_located, Some(true));
        assert_eq!(a.client_label.as_deref(), Some("laptop"));
    }

    #[test]
    fn auth_context_builder_attaches_client_context() {
        // #549: the handshake's client context threads onto the AuthContext so
        // the dispatcher can install it as a task-local for the turn.
        let ctx = desktop_assistant_core::ports::transport::ClientContext {
            username: Some("dave".to_string()),
            ..Default::default()
        };
        let a = AuthContext::new("dave", TransportKind::Uds).with_client_context(Some(ctx.clone()));
        assert_eq!(a.client_context, Some(ctx));
        // Default carries none.
        assert_eq!(
            AuthContext::new("dave", TransportKind::Uds).client_context,
            None
        );
    }

    #[test]
    fn auth_context_blank_label_is_dropped() {
        // A blank/whitespace label is treated as absent so it never produces a
        // `your device ''` in the tool note.
        let a =
            AuthContext::new("dave", TransportKind::Uds).with_client_label(Some("   ".to_string()));
        assert_eq!(a.client_label, None);
    }
}
