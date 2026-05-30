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

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, AssistantApiHandler, EventSink};
use desktop_assistant_core::ports::auth::{UserId, with_user_id};
use futures::sink::{Sink, SinkExt};
use futures::stream::{Stream, StreamExt};
use tokio::sync::mpsc;
use tracing::{debug, warn};

pub use api::{WsFrame, WsRequest};

/// Outbound frame buffer; matches the WS adapter's pre-refactor sizing.
const OUTBOUND_BUFFER: usize = 64;

/// Pre-validated identity for the connection.
///
/// Today only `user_id` is used; structured as a type so #105 (per-user
/// state lookup) can extend it without changing the dispatcher signature.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// JWT `sub` claim.
    pub user_id: String,
}

impl AuthContext {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
        }
    }

    /// Convenience for tests that don't need a real subject.
    pub fn anonymous() -> Self {
        Self {
            user_id: "anonymous".to_string(),
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

/// Run the dispatcher loop on `inbound` / `outbound` until the inbound
/// stream ends or the outbound sink errors.
///
/// Behavior preserved from the old WS `handle_socket`:
///
/// - Each [`WsRequest::command`] is dispatched through `handler`.
/// - `SendMessage` is streamed: an immediate `Ack` result frame, then
///   `AssistantDelta` / `AssistantCompleted` / `AssistantError` events,
///   forwarded through a [`ChannelEventSink`] into the outbound channel.
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

    // Per-connection state for `SubscribeBackgroundTasks` (#114).
    // At most one forwarder runs per connection — Subscribe is
    // idempotent (a second one observes the existing handle and Acks
    // without spawning a duplicate). Unsubscribe aborts the handle;
    // when the connection drops we abort it as part of the loop's
    // teardown so the broadcast receiver is dropped cleanly.
    let mut bg_subscription: Option<tokio::task::JoinHandle<()>> = None;

    while let Some(item) = inbound.next().await {
        let req = match item {
            Ok(req) => req,
            Err(e) => {
                warn!("inbound frame decode error: {e}");
                continue;
            }
        };

        debug!(id = %req.id, "request received");

        match req.command {
            api::Command::SendMessage {
                conversation_id,
                content,
                override_selection,
            } => {
                // Per-request id for event correlation (matches old WS path).
                let request_id = uuid::Uuid::new_v4().to_string();
                let sink: Arc<dyn EventSink> = Arc::new(ChannelEventSink::new(out_tx.clone()));

                // Prefer the new `start_send_message` registration
                // path so the ack can carry the registered task id
                // (#114). When the handler opts out (no registry
                // attached — single-tenant tests and the like) we
                // fall back to the legacy bare-`Ack` + inline
                // streaming path that pre-#114 transports used.
                let task_id = with_user_id(
                    user_id.clone(),
                    handler.start_send_message(
                        conversation_id.clone(),
                        content.clone(),
                        override_selection.clone(),
                        request_id.clone(),
                        Arc::clone(&sink),
                    ),
                )
                .await;

                match task_id {
                    Ok(Some(id)) => {
                        // Handler registered the turn via the registry
                        // and the body is already running in the
                        // background — emit the new typed ack and
                        // continue the loop.
                        if out_tx
                            .send(WsFrame::Result {
                                id: req.id.clone(),
                                result: api::CommandResult::SendMessageAck { task_id: id.0 },
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(None) => {
                        // Legacy path: ack first, then drive the
                        // streaming send on a spawned task so the
                        // dispatcher remains responsive.
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
                        let handler = Arc::clone(&handler);
                        let user_id_for_task = user_id.clone();
                        tokio::spawn(async move {
                            let _ = with_user_id(
                                user_id_for_task,
                                handler.handle_send_message_with_override(
                                    conversation_id,
                                    content,
                                    override_selection,
                                    request_id,
                                    sink,
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
                let receiver = with_user_id(user_id.clone(), handler.subscribe_user_events()).await;
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
                                warn!(
                                    "background-task event subscriber lagged \
                                     by {n} events; oldest dropped"
                                );
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

            api::Command::SetConfig { changes } => {
                let res = with_user_id(
                    user_id.clone(),
                    handler.handle_command(api::Command::SetConfig { changes }),
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
                let res = with_user_id(user_id.clone(), handler.handle_command(other)).await;
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
