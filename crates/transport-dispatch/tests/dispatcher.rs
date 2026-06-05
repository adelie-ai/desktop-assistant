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

    let f1 = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        f1,
        WsFrame::Result {
            result: api::CommandResult::Ack,
            ..
        }
    ));

    let f2 = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        f2,
        WsFrame::Event {
            event: api::Event::AssistantDelta { .. }
        }
    ));

    let f3 = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        f3,
        WsFrame::Event {
            event: api::Event::AssistantCompleted { .. }
        }
    ));

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
