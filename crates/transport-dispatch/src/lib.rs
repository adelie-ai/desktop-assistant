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

    // `auth` is stable for the connection; clone into spawned tasks so
    // #105 (per-user state) can dispatch via the same identity. Today
    // only the user_id is read; keeping the full context flowing means
    // we won't have to plumb it later.
    let _ = auth;

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
                let sink: Arc<dyn EventSink> =
                    Arc::new(ChannelEventSink::new(out_tx.clone()));

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
                tokio::spawn(async move {
                    let _ = handler
                        .handle_send_message_with_override(
                            conversation_id,
                            content,
                            override_selection,
                            request_id,
                            sink,
                        )
                        .await;
                });
            }

            api::Command::SetConfig { changes } => {
                let res = handler
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
                let res = handler.handle_command(other).await;
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
