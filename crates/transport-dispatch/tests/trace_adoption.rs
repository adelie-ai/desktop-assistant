//! Who mints the turn's id, and what the daemon does with what a client sent.
//!
//! A turn starts when a person commits an input, which happens in the client,
//! so the client is the top of the trace and mints the id. The daemon adopts
//! it. That is what lets one identifier be pasted from a client's own event
//! stream into a pod log and into a trace backend.
//!
//! Everything here is asserted at the dispatcher, because that is where the
//! decision is made and it is the one place all three socket transports pass
//! through. The value that comes back on `SendMessageAck` is the value the
//! daemon stamps on every streamed event, so the ack is what these tests read.

use std::sync::{Arc, Mutex};

use desktop_assistant_api_model as api;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink};
use desktop_assistant_core::domain::TransportKind;
use desktop_assistant_core::ports::turn_telemetry::{TurnTrace, current_turn_trace};
use desktop_assistant_transport_dispatch::{
    AuthContext, Capability, WsFrame, WsRequest, dispatch_loop,
};
use futures::StreamExt;
use futures::channel::mpsc;
use futures::stream;

/// A well-formed, non-nil uuid, as a client would mint one.
const CLIENT_TURN_ID: &str = "11111111-2222-4333-8444-555555555555";

/// A trace the caller is already inside. Its trace id is not the one
/// [`CLIENT_TURN_ID`] spells, so continuing it is visibly different from
/// minting.
const INCOMING_TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

/// A handler that answers a send with nothing but the ack the dispatcher
/// builds, and records the trace the dispatcher installed around it.
///
/// Reading the trace is what makes the `traceparent` tests mean anything. The
/// ack carries only `request_id`, which is computed from `turn_id` alone and
/// therefore cannot depend on `traceparent` whatever the dispatcher does with
/// it - so a test that reads only the ack passes for every input, including
/// one where the header is never read at all.
#[derive(Default)]
struct QuietHandler {
    seen: Mutex<Option<TurnTrace>>,
}

impl QuietHandler {
    fn seen(&self) -> Option<TurnTrace> {
        self.seen
            .lock()
            .expect("the lock is held to store one value")
            .clone()
    }
}

#[async_trait::async_trait]
impl AssistantApiHandler for QuietHandler {
    async fn handle_command(&self, _cmd: api::Command) -> ApiResult<api::CommandResult> {
        Err(ApiError::Unsupported)
    }

    async fn handle_send_message(
        &self,
        _conversation_id: String,
        _content: String,
        _request_id: String,
        _sink: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        // The dispatcher installs the trace around this call, which is the
        // same scope `RequestScope::capture` reads on the real path.
        *self
            .seen
            .lock()
            .expect("the lock is held to store one value") = current_turn_trace();
        Ok(())
    }
}

/// A handler that registers the turn, so the dispatcher takes the **registry**
/// path rather than the legacy inline-streaming one.
///
/// Both paths install the trace, and they do it in different places - one
/// around the registration call, one inside a `tokio::spawn`. A test that only
/// ever drove one of them would leave the other free to drop the trace
/// silently, which is the whole failure mode `RequestScope` exists to prevent.
#[derive(Default)]
struct RegisteringHandler {
    seen: Mutex<Option<TurnTrace>>,
}

#[async_trait::async_trait]
impl AssistantApiHandler for RegisteringHandler {
    async fn handle_command(&self, _cmd: api::Command) -> ApiResult<api::CommandResult> {
        Err(ApiError::Unsupported)
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

    async fn start_send_message(
        &self,
        _conversation_id: String,
        _content: String,
        _override_selection: Option<api::SendPromptOverride>,
        _system_refinement: String,
        _request_id: String,
        _idempotency_key: Option<String>,
        _sink: Arc<dyn EventSink>,
    ) -> ApiResult<Option<api::TaskId>> {
        // Where `RequestScope::capture` runs on the real path, so this is the
        // scope the spawned turn body inherits.
        *self
            .seen
            .lock()
            .expect("the lock is held to store one value") = current_turn_trace();
        Ok(Some(api::TaskId("task-1".to_string())))
    }
}

fn send(turn_id: Option<&str>, traceparent: Option<&str>) -> WsRequest {
    WsRequest {
        id: "send-1".into(),
        command: api::Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hello".into(),
            override_selection: None,
            system_refinement: String::new(),
            client_context: None,
            idempotency_key: None,
            turn_id: turn_id.map(str::to_string),
            traceparent: traceparent.map(str::to_string),
        },
    }
}

/// Drive one send through the dispatcher and return the `request_id` the ack
/// carried.
async fn acked_request_id(request: WsRequest) -> String {
    dispatched(request).await.0
}

/// Drive one send through the dispatcher and return the `request_id` the ack
/// carried, together with the trace the handler saw installed around it.
async fn dispatched(request: WsRequest) -> (String, Option<TurnTrace>) {
    let inbound = stream::iter(vec![Ok::<_, anyhow::Error>(request)]);
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(16);
    let observer = Arc::new(QuietHandler::default());
    let handler: Arc<dyn AssistantApiHandler> =
        Arc::clone(&observer) as Arc<dyn AssistantApiHandler>;
    let dispatch = tokio::spawn(dispatch_loop(
        handler,
        AuthContext::anonymous(),
        inbound,
        out_tx,
    ));

    let frame = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .expect("the dispatcher produced no frame")
        .expect("the outbound channel closed before a frame arrived");
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), dispatch).await;

    let request_id = match frame {
        WsFrame::Result {
            result: api::CommandResult::SendMessageAck { request_id, .. },
            ..
        } => request_id,
        other => panic!("expected a send ack, got {other:?}"),
    };
    (request_id, observer.seen())
}

#[tokio::test]
async fn client_supplied_turn_id_is_adopted_by_the_daemon() {
    assert_eq!(
        acked_request_id(send(Some(CLIENT_TURN_ID), None)).await,
        CLIENT_TURN_ID,
        "the daemon must adopt the id the client minted, or the client's own \
         log and the daemon's log name the same turn differently"
    );
}

#[tokio::test]
async fn missing_client_id_falls_back_to_daemon_minting() {
    // An older client sends nothing. This is a supported configuration, not a
    // degraded one: no error, no warning, and a correlatable turn.
    let request_id = acked_request_id(send(None, None)).await;
    let parsed = uuid::Uuid::parse_str(&request_id)
        .unwrap_or_else(|e| panic!("the daemon must mint a uuid, got {request_id:?}: {e}"));
    assert!(
        !parsed.is_nil(),
        "the nil uuid spells the trace id the W3C spec reserves as invalid"
    );
}

#[tokio::test]
async fn malformed_or_nil_client_id_falls_back_to_minting() {
    // Each of these is a value a client can put on the wire today. None may
    // fail the turn, and none may become the trace id.
    let unusable = [
        "",
        "not-a-uuid",
        "00000000-0000-0000-0000-000000000000",
        "11111111-2222-4333-8444-55555555555",
    ];
    for supplied in unusable {
        let request_id = acked_request_id(send(Some(supplied), None)).await;
        assert_ne!(
            request_id, supplied,
            "{supplied:?} is unusable and must not be adopted"
        );
        let parsed = uuid::Uuid::parse_str(&request_id).unwrap_or_else(|e| {
            panic!("{supplied:?} must fall back to a minted uuid, got {request_id:?}: {e}")
        });
        assert!(!parsed.is_nil());
    }
}

#[tokio::test]
async fn a_client_supplied_turn_id_does_not_survive_a_refusal() {
    // A turn id is a correlation id, not a capability. The capability check
    // runs on the command kind alone and is reached before the id is read, so
    // a client cannot spend one to change what it is allowed to do. A
    // connection holding a level this build does not recognise is refused
    // `send_message` whatever id it sends.
    let inbound = stream::iter(vec![Ok::<_, anyhow::Error>(send(
        Some(CLIENT_TURN_ID),
        None,
    ))]);
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(16);
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(QuietHandler::default());
    let dispatch = tokio::spawn(dispatch_loop(
        handler,
        AuthContext::new("subject", TransportKind::WebSocket)
            .with_capability(Capability::Other("unknown".into())),
        inbound,
        out_tx,
    ));

    let frame = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next())
        .await
        .expect("the dispatcher produced no frame")
        .expect("the outbound channel closed before a frame arrived");
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), dispatch).await;

    match frame {
        WsFrame::Error { error, .. } => assert!(
            !format!("{error:?}").contains(CLIENT_TURN_ID),
            "the refusal must not quote the client's own value back: {error:?}"
        ),
        other => panic!(
            "a connection without the capability must be refused, got {other:?}; \
             if this now succeeds, a correlation id has reached an authorization \
             decision"
        ),
    }
}

#[tokio::test]
async fn turn_id_and_idempotency_key_stay_separate() {
    // Two fields, two purposes. The retry path reads one; the trace reads the
    // other. A turn carrying both must adopt the turn id and leave the key
    // alone, or one silently starts standing in for the other.
    let request = WsRequest {
        id: "send-1".into(),
        command: api::Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hello".into(),
            override_selection: None,
            system_refinement: String::new(),
            client_context: None,
            idempotency_key: Some("key-for-the-retry-path".into()),
            turn_id: Some(CLIENT_TURN_ID.into()),
            traceparent: None,
        },
    };
    assert_eq!(acked_request_id(request).await, CLIENT_TURN_ID);
}

#[tokio::test]
async fn an_incoming_traceparent_does_not_replace_the_correlation_id() {
    // The two answer different questions. `traceparent` says which trace to
    // join; `turn_id` says what the client will read its own event stream by.
    // A caller sending both keeps both.
    assert_eq!(
        acked_request_id(send(Some(CLIENT_TURN_ID), Some(INCOMING_TRACEPARENT))).await,
        CLIENT_TURN_ID
    );
}

#[tokio::test]
async fn the_dispatcher_resolves_a_trace_and_installs_it_for_the_turn() {
    // The turn body reads the trace from the scope the dispatcher installs
    // around it, and everything downstream - every child span's conversation
    // id, every outbound MCP call - reads it from there. Nothing else in this
    // file observes that install, and the ack cannot: `request_id` is computed
    // from `turn_id` alone, so it is the same value whether or not the trace
    // was ever resolved.
    let (_, trace) = dispatched(send(Some(CLIENT_TURN_ID), None)).await;
    let trace = trace.expect("the dispatcher must install a trace for the turn");
    assert_eq!(
        trace.trace.trace_id().to_hex(),
        CLIENT_TURN_ID.replace('-', ""),
        "a minted trace takes its id from the turn's own correlation id"
    );
    assert_eq!(
        trace.conversation_id, "c1",
        "the conversation rides the same scope, so every span in the turn \
         carries it"
    );
}

#[tokio::test]
async fn incoming_traceparent_is_continued_not_replaced() {
    // The web BFF's case, and the only reason this field exists: a browser
    // starts the turn, the BFF forwards it, and the daemon serves it. That is
    // one trace across three processes rather than three that share a
    // timestamp. Read from the scope the turn body sees, because the ack
    // cannot show it.
    let (request_id, trace) =
        dispatched(send(Some(CLIENT_TURN_ID), Some(INCOMING_TRACEPARENT))).await;
    let trace = trace.expect("the dispatcher must install a trace for the turn");

    assert_eq!(
        trace.trace.trace_id().to_hex(),
        "4bf92f3577b34da6a3ce929d0e0e4736",
        "the daemon must join the caller's trace rather than mint its own"
    );
    assert_eq!(
        request_id, CLIENT_TURN_ID,
        "continuing a trace must not disturb the correlation id, which is the \
         separate identifier the client reads its own stream by"
    );
    assert!(
        trace.trace.parent_span_id().is_some(),
        "a continued trace carries the caller's span, so the daemon's work \
         hangs where it belongs rather than under an id nobody exported"
    );
}

#[tokio::test]
async fn the_registry_path_installs_the_trace_too() {
    // The other of the dispatcher's two send paths. A handler that registers
    // the turn never reaches the legacy inline branch, so nothing else in this
    // file observes this install.
    let inbound = stream::iter(vec![Ok::<_, anyhow::Error>(send(
        Some(CLIENT_TURN_ID),
        Some(INCOMING_TRACEPARENT),
    ))]);
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(16);
    let observer = Arc::new(RegisteringHandler::default());
    let handler: Arc<dyn AssistantApiHandler> =
        Arc::clone(&observer) as Arc<dyn AssistantApiHandler>;
    let dispatch = tokio::spawn(dispatch_loop(
        handler,
        AuthContext::anonymous(),
        inbound,
        out_tx,
    ));
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), out_rx.next()).await;
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), dispatch).await;

    let trace = observer
        .seen
        .lock()
        .expect("the lock is held to store one value")
        .clone()
        .expect("the registry path must install a trace for the turn");
    assert_eq!(
        trace.trace.trace_id().to_hex(),
        "4bf92f3577b34da6a3ce929d0e0e4736",
        "the registry path must join the caller's trace like the other one"
    );
}

#[tokio::test]
async fn a_discarded_traceparent_leaves_the_turn_on_its_own_trace() {
    // A client that sends a bad header must not be able to stop the daemon
    // doing the work, and must not land the turn in a trace nobody started.
    let (_, trace) = dispatched(send(Some(CLIENT_TURN_ID), Some("garbage"))).await;
    let trace = trace.expect("the dispatcher must install a trace for the turn");
    assert_eq!(
        trace.trace.trace_id().to_hex(),
        CLIENT_TURN_ID.replace('-', ""),
        "an unusable header falls back to the turn's own id"
    );
}

#[tokio::test]
async fn a_malformed_traceparent_never_fails_the_turn() {
    // The header comes from a client, so a client that sends a bad one must
    // not be able to stop the daemon doing the work.
    let long = format!("00-{}-00f067aa0ba902b7-01", "a".repeat(512));
    for header in [
        "garbage",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        long.as_str(),
    ] {
        assert_eq!(
            acked_request_id(send(Some(CLIENT_TURN_ID), Some(header))).await,
            CLIENT_TURN_ID,
            "{header:?} must be discarded, not fatal"
        );
    }
}
