//! Behavioral tests for the transport-agnostic dispatcher.
//!
//! These wire a stub `AssistantApiHandler` into `dispatch_loop` via
//! synthetic in-memory streams/sinks. The goal is to nail down the
//! behavior the WS and UDS adapters now share so a future refactor
//! that breaks the contract has a single place to break.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_transport_dispatch::{AuthContext, WsFrame, WsRequest, dispatch_loop};
use futures::channel::mpsc;
use futures::stream;

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
        conversation_id: String,
        _content: String,
        request_id: String,
        sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        sink.emit(api::Event::AssistantDelta {
            conversation_id: conversation_id.clone(),
            request_id: request_id.clone(),
            chunk: "hi".into(),
        })
        .await;
        sink.emit(api::Event::AssistantCompleted {
            conversation_id,
            request_id,
            full_response: "hi".into(),
        })
        .await;
        Ok(())
    }
}

#[tokio::test]
async fn dispatcher_serves_ws_command_through_neutral_handler() {
    let req = WsRequest {
        id: "1".into(),
        command: api::Command::Ping,
    };
    let inbound = stream::iter(vec![Ok::<_, anyhow::Error>(req)]);
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(8);

    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let dispatch = tokio::spawn(dispatch_loop(
        handler,
        AuthContext::anonymous(),
        inbound,
        out_tx,
    ));

    use futures::StreamExt;
    let frame = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .expect("dispatcher produced no frame")
        .expect("outbound channel closed before frame");
    match frame {
        WsFrame::Result { id, result } => {
            assert_eq!(id, "1");
            assert_eq!(
                result,
                api::CommandResult::Pong {
                    value: "pong".into()
                }
            );
        }
        other => panic!("unexpected frame: {other:?}"),
    }

    // Loop should exit when the inbound stream ends.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), dispatch)
        .await
        .expect("dispatch task did not exit after inbound ended");
}

#[tokio::test]
async fn dispatcher_streams_send_message_ack_then_events() {
    let req = WsRequest {
        id: "send-1".into(),
        command: api::Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hello".into(),
            override_selection: None,
            system_refinement: String::new(),
            idempotency_key: None,
        },
    };
    let inbound = stream::iter(vec![Ok::<_, anyhow::Error>(req)]);
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(16);

    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let dispatch = tokio::spawn(dispatch_loop(
        handler,
        AuthContext::anonymous(),
        inbound,
        out_tx,
    ));

    use futures::StreamExt;

    // The send is acked with `SendMessageAck` carrying the turn request_id
    // (voice#49); the streamed events are stamped with that same id.
    let f1 = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .unwrap()
        .unwrap();
    let ack_request_id = match f1 {
        WsFrame::Result {
            result: api::CommandResult::SendMessageAck { request_id, .. },
            ..
        } => {
            assert!(!request_id.is_empty(), "ack must carry the turn request_id");
            request_id
        }
        other => panic!("expected SendMessageAck, got {other:?}"),
    };

    let f2 = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .unwrap()
        .unwrap();
    match f2 {
        WsFrame::Event {
            event: api::Event::AssistantDelta { request_id, .. },
        } => assert_eq!(request_id, ack_request_id, "delta must match the ack id"),
        other => panic!("expected AssistantDelta, got {other:?}"),
    }

    let f3 = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .unwrap()
        .unwrap();
    match f3 {
        WsFrame::Event {
            event: api::Event::AssistantCompleted { request_id, .. },
        } => assert_eq!(
            request_id, ack_request_id,
            "completed must match the ack id"
        ),
        other => panic!("expected AssistantCompleted, got {other:?}"),
    }

    dispatch.abort();
}

#[tokio::test]
async fn dispatcher_exits_when_outbound_sink_closes() {
    // A trivial scenario: outbound sink is dropped, dispatcher should exit.
    let req = WsRequest {
        id: "1".into(),
        command: api::Command::Ping,
    };
    let inbound = stream::iter(vec![Ok::<_, anyhow::Error>(req)]);
    // The sink type used in production is forwarded through an mpsc;
    // here we simulate a closed downstream by dropping the receiver
    // immediately. The dispatcher should still exit when the inbound
    // stream ends.
    let (out_tx, out_rx) = mpsc::channel::<WsFrame>(1);
    drop(out_rx);
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let dispatch = tokio::spawn(dispatch_loop(
        handler,
        AuthContext::anonymous(),
        inbound,
        out_tx,
    ));

    // Should not hang.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), dispatch)
        .await
        .expect("dispatcher should exit when outbound is closed and inbound ended");
}

// ---------------------------------------------------------------------
// Code review 2026-06-09 — DT-5 (error reply for malformed frames) and
// DT-11 (lagged broadcast subscription resync signal).
// ---------------------------------------------------------------------

/// DT-5: an inbound item that failed to decode must produce an explicit
/// `WsFrame::Error` with an empty id (there is no request id to correlate
/// to), and the loop must keep serving subsequent requests.
#[tokio::test]
async fn invalid_inbound_json_yields_error_frame_with_empty_id_and_loop_survives() {
    let inbound = stream::iter(vec![
        Err(anyhow::anyhow!("invalid request json: oops")),
        Ok(WsRequest {
            id: "after".into(),
            command: api::Command::Ping,
        }),
    ]);
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(8);

    let handler: Arc<dyn AssistantApiHandler> = Arc::new(PingHandler);
    let dispatch = tokio::spawn(dispatch_loop(
        handler,
        AuthContext::anonymous(),
        inbound,
        out_tx,
    ));

    use futures::StreamExt;
    let first = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .expect("expected an error frame for the malformed item, got silence")
        .expect("outbound closed early");
    match first {
        WsFrame::Error { id, error } => {
            assert_eq!(id, "", "no request id is known for a malformed frame");
            assert!(
                error.to_lowercase().contains("json") || error.to_lowercase().contains("invalid"),
                "error should describe the decode failure: {error}"
            );
        }
        other => panic!("expected an Error frame, got {other:?}"),
    }

    let second = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .expect("dispatcher must keep serving after a malformed frame")
        .expect("outbound closed early");
    match second {
        WsFrame::Result { id, .. } => assert_eq!(id, "after"),
        other => panic!("expected a Result frame, got {other:?}"),
    }

    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), dispatch).await;
}

/// Handler whose background-task subscription is already lagged when
/// handed out: capacity-1 broadcast with three events published, so the
/// first `recv()` observes `Lagged(2)` and the next yields the surviving
/// event.
struct LaggedSubscriptionHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for LaggedSubscriptionHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::SubscribeBackgroundTasks => Ok(api::CommandResult::Ack),
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

    async fn subscribe_user_events(&self) -> Option<tokio::sync::broadcast::Receiver<api::Event>> {
        let (tx, rx) = tokio::sync::broadcast::channel(1);
        for n in 0..3 {
            let _ = tx.send(api::Event::ScratchpadChanged {
                conversation_id: format!("conv-{n}"),
            });
        }
        Some(rx)
    }
}

/// DT-11: when the per-connection broadcast subscription lags (events
/// were dropped server-side), the client must receive a synthetic
/// empty-id error frame telling it to resync — not just a server-side
/// log line it can never see.
#[tokio::test]
async fn lagged_event_subscription_emits_resync_error_frame() {
    let req = WsRequest {
        id: "sub-1".into(),
        command: api::Command::SubscribeBackgroundTasks,
    };
    // Keep the connection open after the subscribe so the forwarder can run.
    let pending_inbound = stream::pending::<anyhow::Result<WsRequest>>();
    let inbound = stream::iter(vec![Ok::<_, anyhow::Error>(req)]).chain(pending_inbound);
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(16);

    let handler: Arc<dyn AssistantApiHandler> = Arc::new(LaggedSubscriptionHandler);
    let dispatch = tokio::spawn(dispatch_loop(
        handler,
        AuthContext::anonymous(),
        inbound,
        out_tx,
    ));

    use futures::StreamExt;
    // Collect frames until we see both the dropped-events notice and the
    // surviving forwarded event (ordering between the subscribe Ack and the
    // forwarder's frames is racy, so match by shape).
    let mut saw_resync_notice = false;
    let mut saw_surviving_event = false;
    for _ in 0..4 {
        let Ok(Some(frame)) =
            tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next()).await
        else {
            break;
        };
        match frame {
            WsFrame::Error { id, error } => {
                assert_eq!(
                    id, "",
                    "the resync notice is connection-level, not per-request"
                );
                assert!(
                    error.to_lowercase().contains("dropped")
                        || error.to_lowercase().contains("lagged"),
                    "the notice must say events were dropped: {error}"
                );
                saw_resync_notice = true;
            }
            WsFrame::Event {
                event: api::Event::ScratchpadChanged { conversation_id },
            } => {
                assert_eq!(conversation_id, "conv-2", "only the newest event survives");
                saw_surviving_event = true;
            }
            _ => {}
        }
        if saw_resync_notice && saw_surviving_event {
            break;
        }
    }

    assert!(
        saw_resync_notice,
        "a lagged subscription must surface a client-visible resync notice (DT-11)"
    );
    assert!(
        saw_surviving_event,
        "events after the gap must still be forwarded"
    );

    dispatch.abort();
}
