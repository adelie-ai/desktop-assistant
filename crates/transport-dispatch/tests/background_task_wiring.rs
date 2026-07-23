//! Dispatcher-level wiring tests for the background-task command surface
//! (#114). These exercise the transport-agnostic `dispatch_loop` against a
//! stub handler that exposes a real `BackgroundTaskRegistry` so the tests
//! pin down the wire behaviour both the WS and UDS adapters inherit:
//!
//! - Subscribe spawns a forwarder that pumps the registry's per-user
//!   broadcast channel into the outbound `WsFrame::Event` sink, until
//!   Unsubscribe arrives or the connection drops.
//! - Subscribe is idempotent at the per-connection level — a second
//!   Subscribe on a connection that already has a live forwarder Acks
//!   without leaking a second receiver.
//! - SendMessage's response carries `SendMessageAck { request_id, task_id }`:
//!   the `task_id` when the handler registered the turn with the registry, and
//!   the turn `request_id` (which streamed events are stamped with) on both the
//!   registry and the legacy paths so a socket client can correlate the
//!   response stream (voice#49).
//! - When the handler exposes no broadcast hook (no registry attached)
//!   Subscribe surfaces a clean error frame rather than spawning a dead
//!   forwarder.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_application::background_tasks::BackgroundTaskRegistry;
use desktop_assistant_application::{ApiError, ApiResult, AssistantApiHandler, EventSink, UserId};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_transport_dispatch::{AuthContext, WsFrame, WsRequest, dispatch_loop};
use futures::channel::mpsc;
use futures::stream;
use tokio::sync::broadcast;

// ---------- Stub handler ----------

/// Stub that routes the new background-task arms through a real
/// `BackgroundTaskRegistry`. Other commands return `Unsupported` — the
/// dispatcher tests only exercise the new surface.
struct RegistryHandler {
    registry: Arc<BackgroundTaskRegistry>,
    /// Set when `start_send_message` is called, so tests can assert
    /// whether the dispatcher routed through the new entry point.
    start_send_calls: Mutex<u32>,
}

impl RegistryHandler {
    fn new(registry: Arc<BackgroundTaskRegistry>) -> Self {
        Self {
            registry,
            start_send_calls: Mutex::new(0),
        }
    }
}

#[async_trait::async_trait]
impl AssistantApiHandler for RegistryHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        let user_id = current_user_id();
        match cmd {
            api::Command::ListBackgroundTasks {
                include_finished,
                limit,
            } => {
                let tasks = self.registry.list(&user_id, include_finished, limit);
                Ok(api::CommandResult::BackgroundTasks(tasks))
            }
            api::Command::GetBackgroundTask { id } => self
                .registry
                .get(&user_id, &api::TaskId(id))
                .map(api::CommandResult::BackgroundTask)
                .ok_or_else(|| ApiError::Core("task not found".into())),
            api::Command::CancelBackgroundTask { id } => self
                .registry
                .cancel(&user_id, &api::TaskId(id))
                .map(|_| api::CommandResult::Ack)
                .map_err(|e| ApiError::Core(e.to_string())),
            api::Command::GetBackgroundTaskLogs {
                id,
                after_seq,
                limit,
            } => self
                .registry
                .logs(
                    &user_id,
                    &api::TaskId(id),
                    after_seq.unwrap_or(0),
                    limit.unwrap_or(200),
                )
                .map(
                    |(entries, next_seq)| api::CommandResult::BackgroundTaskLogs {
                        entries,
                        next_seq,
                    },
                )
                .map_err(|e| ApiError::Core(e.to_string())),
            api::Command::SubscribeBackgroundTasks => Ok(api::CommandResult::Ack),
            api::Command::UnsubscribeBackgroundTasks => Ok(api::CommandResult::Ack),
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

    async fn subscribe_user_events(&self) -> Option<broadcast::Receiver<api::Event>> {
        let user = current_user_id();
        Some(self.registry.subscribe(&user))
    }

    async fn start_send_message(
        &self,
        conversation_id: String,
        content: String,
        _override_selection: Option<api::SendPromptOverride>,
        _system_refinement: String,
        _request_id: String,
        _idempotency_key: Option<String>,
        _sink: Arc<dyn EventSink>,
    ) -> ApiResult<Option<api::TaskId>> {
        *self.start_send_calls.lock().unwrap() += 1;
        let user = current_user_id();
        let id = self.registry.spawn(
            user,
            api::TaskKind::Conversation {
                conversation_id: conversation_id.clone(),
            },
            format!("Conversation: {conversation_id}"),
            move |ctx| async move {
                let _ = content;
                // Stay Running until cancelled: terminal tasks are evicted
                // from the registry on finalize (#158), so a body that
                // returned immediately would race its own eviction against
                // the SendMessageAck assertion that the id points to a
                // live row. Nothing cancels it; the task is torn down with
                // the test runtime.
                ctx.token.cancelled().await;
                Ok(())
            },
        );
        Ok(Some(id))
    }
}

/// Handler with no event-subscription hook (mirrors a single-tenant
/// deploy without an attached registry).
struct NoSubscribeHandler;

#[async_trait::async_trait]
impl AssistantApiHandler for NoSubscribeHandler {
    async fn handle_command(&self, cmd: api::Command) -> ApiResult<api::CommandResult> {
        match cmd {
            api::Command::SubscribeBackgroundTasks | api::Command::UnsubscribeBackgroundTasks => {
                Ok(api::CommandResult::Ack)
            }
            _ => Err(ApiError::Unsupported),
        }
    }
    async fn handle_send_message(
        &self,
        _c: String,
        _t: String,
        _r: String,
        _s: Arc<dyn EventSink>,
    ) -> ApiResult<()> {
        Ok(())
    }
    // Default impl of subscribe_user_events returns None.
}

// ---------- Helpers ----------

fn user(label: &str) -> AuthContext {
    AuthContext::new(label, desktop_assistant_core::domain::TransportKind::Uds)
}

/// Drive `dispatch_loop` against an in-memory stream of requests; return
/// a receiver of outbound frames and the join handle for the loop.
fn drive(
    handler: Arc<dyn AssistantApiHandler>,
    auth: AuthContext,
    requests: Vec<WsRequest>,
) -> (mpsc::Receiver<WsFrame>, tokio::task::JoinHandle<()>) {
    let inbound = stream::iter(requests.into_iter().map(Ok::<_, anyhow::Error>));
    let (out_tx, out_rx) = mpsc::channel::<WsFrame>(64);
    let handle = tokio::spawn(dispatch_loop(handler, auth, inbound, out_tx));
    (out_rx, handle)
}

async fn next_frame(rx: &mut mpsc::Receiver<WsFrame>) -> WsFrame {
    use futures::StreamExt;
    tokio::time::timeout(Duration::from_secs(2), rx.next())
        .await
        .expect("no frame within 2s")
        .expect("outbound closed")
}

async fn try_next_frame(rx: &mut mpsc::Receiver<WsFrame>) -> Option<WsFrame> {
    use futures::StreamExt;
    tokio::time::timeout(Duration::from_millis(100), rx.next())
        .await
        .ok()
        .flatten()
}

// ---------- Tests ----------

/// Open-ended inbound stream backed by an mpsc — keeps the dispatcher
/// loop alive across multiple sends so we can trigger registry events
/// out-of-band. Returns `'static`-friendly types so the resulting future
/// can be `tokio::spawn`ed.
type InboundStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = anyhow::Result<WsRequest>> + Send + 'static>>;

fn open_inbound() -> (mpsc::Sender<anyhow::Result<WsRequest>>, InboundStream) {
    let (tx, rx) = mpsc::channel::<anyhow::Result<WsRequest>>(8);
    let stream = futures::stream::unfold(rx, |mut rx| async move {
        use futures::StreamExt;
        rx.next().await.map(|item| (item, rx))
    });
    (tx, Box::pin(stream))
}

/// Subscribe immediately Acks; subsequent registry events fan out to the
/// connection as `WsFrame::Event` frames.
#[tokio::test]
async fn subscribe_then_registry_event_streams_to_connection() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let handler: Arc<dyn AssistantApiHandler> =
        Arc::new(RegistryHandler::new(Arc::clone(&registry)));

    let (mut in_tx, inbound) = open_inbound();
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(64);

    let dispatch = tokio::spawn(dispatch_loop(handler, user("alice"), inbound, out_tx));

    // Subscribe.
    use futures::SinkExt;
    in_tx
        .send(Ok(WsRequest {
            id: "sub-1".into(),
            command: api::Command::SubscribeBackgroundTasks,
        }))
        .await
        .unwrap();

    let ack = next_frame(&mut out_rx).await;
    match ack {
        WsFrame::Result { id, result } => {
            assert_eq!(id, "sub-1");
            assert!(matches!(result, api::CommandResult::Ack));
        }
        other => panic!("unexpected: {other:?}"),
    }

    // Trigger a TaskStarted by spawning a task under the same user
    // identity as the connection.
    let _id = registry.spawn(
        UserId::new("alice"),
        api::TaskKind::Standalone {
            name: "x".into(),
            conversation_id: "c".into(),
        },
        "x".into(),
        |_ctx| async move { Ok(()) },
    );

    // The forwarder should push at least one Task* event onto the
    // connection.
    let mut saw_task_event = false;
    for _ in 0..10 {
        if let Some(WsFrame::Event { event }) = try_next_frame(&mut out_rx).await
            && matches!(
                event,
                api::Event::TaskStarted { .. }
                    | api::Event::TaskCompleted { .. }
                    | api::Event::TaskLogAppended { .. }
            )
        {
            saw_task_event = true;
            break;
        }
    }
    assert!(saw_task_event, "expected a Task* event on the connection");

    drop(in_tx);
    let _ = dispatch.await;
}

/// Unsubscribe stops the forwarder; further registry events do not
/// arrive on the connection.
#[tokio::test]
async fn unsubscribe_stops_event_stream() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let handler: Arc<dyn AssistantApiHandler> =
        Arc::new(RegistryHandler::new(Arc::clone(&registry)));

    let (mut in_tx, inbound) = open_inbound();
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(64);

    let dispatch = tokio::spawn(dispatch_loop(handler, user("alice"), inbound, out_tx));

    use futures::SinkExt;
    in_tx
        .send(Ok(WsRequest {
            id: "sub-1".into(),
            command: api::Command::SubscribeBackgroundTasks,
        }))
        .await
        .unwrap();
    let _ = next_frame(&mut out_rx).await; // subscribe ack

    in_tx
        .send(Ok(WsRequest {
            id: "unsub-1".into(),
            command: api::Command::UnsubscribeBackgroundTasks,
        }))
        .await
        .unwrap();
    let unsub_ack = next_frame(&mut out_rx).await;
    match unsub_ack {
        WsFrame::Result { id, result } => {
            assert_eq!(id, "unsub-1");
            assert!(matches!(result, api::CommandResult::Ack));
        }
        other => panic!("unexpected: {other:?}"),
    }

    // Spawn a task AFTER unsubscribe — its events must not arrive on this
    // connection. Give the dispatcher time to process the Unsubscribe.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let _id = registry.spawn(
        UserId::new("alice"),
        api::TaskKind::Standalone {
            name: "post-unsub".into(),
            conversation_id: "c".into(),
        },
        "post-unsub".into(),
        |_ctx| async move { Ok(()) },
    );

    // Poll for a window — there should be no Task* events.
    for _ in 0..10 {
        if let Some(WsFrame::Event { event }) = try_next_frame(&mut out_rx).await {
            assert!(
                !matches!(
                    event,
                    api::Event::TaskStarted { .. }
                        | api::Event::TaskCompleted { .. }
                        | api::Event::TaskLogAppended { .. }
                ),
                "no Task* event should arrive after Unsubscribe; got {event:?}"
            );
        }
    }

    drop(in_tx);
    let _ = dispatch.await;
}

/// Dropping the connection cleanly tears down the forwarder. We assert
/// this indirectly by verifying that subsequent registry broadcasts to
/// the same user observe no broadcast lag (i.e. the per-connection
/// receiver has been dropped — otherwise it would buffer forever and
/// eventually overflow the channel).
#[tokio::test]
async fn connection_drop_cancels_event_subscription_cleanly() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let handler: Arc<dyn AssistantApiHandler> =
        Arc::new(RegistryHandler::new(Arc::clone(&registry)));

    let (mut in_tx, inbound) = open_inbound();
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(64);

    let dispatch = tokio::spawn(dispatch_loop(handler, user("alice"), inbound, out_tx));

    use futures::SinkExt;
    in_tx
        .send(Ok(WsRequest {
            id: "sub-1".into(),
            command: api::Command::SubscribeBackgroundTasks,
        }))
        .await
        .unwrap();
    let _ = next_frame(&mut out_rx).await;

    // Drop the inbound sender so the dispatcher exits cleanly; drop
    // the outbound receiver to force the writer to bail.
    drop(in_tx);
    drop(out_rx);

    // The dispatcher should exit within the timeout; if it hangs, the
    // forwarder is leaking and holding the loop open.
    tokio::time::timeout(Duration::from_secs(2), dispatch)
        .await
        .expect("dispatcher should exit after the connection drops")
        .expect("dispatcher join error");
}

/// SendMessage with a handler that registers via `start_send_message`
/// produces `SendMessageAck { request_id, task_id }` — the `task_id`
/// matches a row in the registry, and the `request_id` (the id streamed
/// `Assistant*` events are stamped with) is present so a socket client can
/// correlate the response stream (voice#49).
#[tokio::test]
async fn send_message_ack_carries_a_real_task_id() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let handler: Arc<dyn AssistantApiHandler> =
        Arc::new(RegistryHandler::new(Arc::clone(&registry)));

    let (mut out_rx, handle) = drive(
        handler,
        user("alice"),
        vec![WsRequest {
            id: "send-1".into(),
            command: api::Command::SendMessage {
                conversation_id: "conv-1".into(),
                content: "hello".into(),
                override_selection: None,
                system_refinement: String::new(),
                client_context: None,
                idempotency_key: None,
            },
        }],
    );

    let frame = next_frame(&mut out_rx).await;
    let task_id = match frame {
        WsFrame::Result {
            id,
            result:
                api::CommandResult::SendMessageAck {
                    request_id,
                    task_id,
                },
        } => {
            assert_eq!(id, "send-1");
            // The ack must carry a non-empty turn correlation id (voice#49):
            // it is what every streamed `Assistant*` event is keyed on, so a
            // socket client matches its response stream by this id.
            assert!(
                !request_id.is_empty(),
                "registry-path ack must carry the turn request_id"
            );
            api::TaskId(task_id)
        }
        other => panic!("expected SendMessageAck, got {other:?}"),
    };

    // The id must point to a real row in the registry.
    assert!(registry.get(&UserId::new("alice"), &task_id).is_some());

    let _ = handle.await;
}

/// When the handler does NOT opt into `start_send_message` (default impl
/// returns `None`), the dispatcher falls back to the legacy inline-streaming
/// flow so existing transports keep working. The ack still carries the turn
/// `request_id` (with an empty `task_id`, since no task was registered) so a
/// socket client can correlate the streamed response on this path too
/// (voice#49).
#[tokio::test]
async fn send_message_falls_back_to_legacy_ack_when_handler_opts_out() {
    /// Stub that only implements `handle_send_message` (no
    /// `start_send_message` override) — the dispatcher should not panic
    /// and should send a `SendMessageAck` carrying the turn request_id and
    /// an empty task_id.
    struct LegacyHandler;
    #[async_trait::async_trait]
    impl AssistantApiHandler for LegacyHandler {
        async fn handle_command(&self, _cmd: api::Command) -> ApiResult<api::CommandResult> {
            Err(ApiError::Unsupported)
        }
        async fn handle_send_message(
            &self,
            _c: String,
            _t: String,
            _r: String,
            _s: Arc<dyn EventSink>,
        ) -> ApiResult<()> {
            Ok(())
        }
    }
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(LegacyHandler);
    let (mut out_rx, handle) = drive(
        handler,
        user("alice"),
        vec![WsRequest {
            id: "send-2".into(),
            command: api::Command::SendMessage {
                conversation_id: "conv-2".into(),
                content: "hello".into(),
                override_selection: None,
                system_refinement: String::new(),
                client_context: None,
                idempotency_key: None,
            },
        }],
    );

    let frame = next_frame(&mut out_rx).await;
    match frame {
        WsFrame::Result {
            id,
            result:
                api::CommandResult::SendMessageAck {
                    request_id,
                    task_id,
                },
        } => {
            assert_eq!(id, "send-2");
            // Legacy path: a turn request_id is present (so the response stream
            // can be correlated) but no background task was registered.
            assert!(
                !request_id.is_empty(),
                "legacy-path ack must still carry the turn request_id"
            );
            assert!(
                task_id.is_empty(),
                "legacy path registers no task, so task_id is empty"
            );
        }
        other => panic!("expected SendMessageAck on the legacy path, got {other:?}"),
    }

    let _ = handle.await;
}

/// A handler with no event-subscription support surfaces Subscribe as
/// an error frame rather than a silent Ack that wouldn't stream
/// anything. This protects clients that rely on the Ack-meaning-"I'll
/// stream events".
#[tokio::test]
async fn subscribe_returns_error_when_handler_opts_out() {
    let handler: Arc<dyn AssistantApiHandler> = Arc::new(NoSubscribeHandler);
    let (mut out_rx, handle) = drive(
        handler,
        user("alice"),
        vec![WsRequest {
            id: "sub-1".into(),
            command: api::Command::SubscribeBackgroundTasks,
        }],
    );
    let frame = next_frame(&mut out_rx).await;
    match frame {
        WsFrame::Error { id, .. } => assert_eq!(id, "sub-1"),
        other => panic!("expected Error frame for unsupported subscribe, got {other:?}"),
    }
    let _ = handle.await;
}

/// A second Subscribe on a connection that already has a live forwarder
/// is idempotent — it Acks without spawning a duplicate forwarder. Tests
/// that no duplicate event arrives downstream when a single registry
/// broadcast fires.
#[tokio::test]
async fn subscribe_is_idempotent_per_connection() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let handler: Arc<dyn AssistantApiHandler> =
        Arc::new(RegistryHandler::new(Arc::clone(&registry)));

    let (mut in_tx, inbound) = open_inbound();
    let (out_tx, mut out_rx) = mpsc::channel::<WsFrame>(64);

    let dispatch = tokio::spawn(dispatch_loop(handler, user("alice"), inbound, out_tx));

    use futures::SinkExt;
    in_tx
        .send(Ok(WsRequest {
            id: "sub-1".into(),
            command: api::Command::SubscribeBackgroundTasks,
        }))
        .await
        .unwrap();
    let _ = next_frame(&mut out_rx).await; // first ack

    in_tx
        .send(Ok(WsRequest {
            id: "sub-2".into(),
            command: api::Command::SubscribeBackgroundTasks,
        }))
        .await
        .unwrap();
    let frame = next_frame(&mut out_rx).await;
    match frame {
        WsFrame::Result { id, result } => {
            assert_eq!(id, "sub-2");
            assert!(matches!(result, api::CommandResult::Ack));
        }
        other => panic!("unexpected: {other:?}"),
    }

    // Spawn a single task and count Task* events arriving on the
    // connection — there must be exactly one TaskStarted (not two).
    let _id = registry.spawn(
        UserId::new("alice"),
        api::TaskKind::Standalone {
            name: "once".into(),
            conversation_id: "c".into(),
        },
        "once".into(),
        |_ctx| async move { Ok(()) },
    );

    let mut started_count = 0;
    for _ in 0..20 {
        if let Some(WsFrame::Event {
            event: api::Event::TaskStarted { .. },
        }) = try_next_frame(&mut out_rx).await
        {
            started_count += 1;
        }
    }
    assert_eq!(
        started_count, 1,
        "second Subscribe must not duplicate the forwarder; expected 1 TaskStarted, got {started_count}"
    );

    drop(in_tx);
    let _ = dispatch.await;
}

/// voice#49 regression at the dispatcher level: the `request_id` in the
/// `SendMessageAck` MUST equal the `request_id` stamped on the streamed
/// `AssistantDelta` / `AssistantCompleted` events. A socket client correlates
/// its response stream by the ack's id, so if these diverge (as they did when
/// the ack returned the registry `task_id` instead) every event is dropped and
/// the turn hangs.
#[tokio::test]
async fn send_message_ack_request_id_matches_streamed_event_request_id() {
    /// Streams two deltas + a completed event, each stamped with the
    /// dispatcher-supplied `request_id`, from a background task — mirroring
    /// the production registry path.
    struct StreamingHandler;
    #[async_trait::async_trait]
    impl AssistantApiHandler for StreamingHandler {
        async fn handle_command(&self, _cmd: api::Command) -> ApiResult<api::CommandResult> {
            Err(ApiError::Unsupported)
        }
        async fn handle_send_message(
            &self,
            _c: String,
            _t: String,
            _r: String,
            _s: Arc<dyn EventSink>,
        ) -> ApiResult<()> {
            Ok(())
        }
        async fn start_send_message(
            &self,
            conversation_id: String,
            _content: String,
            _override_selection: Option<api::SendPromptOverride>,
            _system_refinement: String,
            request_id: String,
            _idempotency_key: Option<String>,
            sink: Arc<dyn EventSink>,
        ) -> ApiResult<Option<api::TaskId>> {
            // task_id is deliberately a different uuid than request_id.
            let task_id = api::TaskId(uuid::Uuid::new_v4().to_string());
            tokio::spawn(async move {
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
            });
            Ok(Some(task_id))
        }
    }

    let handler: Arc<dyn AssistantApiHandler> = Arc::new(StreamingHandler);
    let (mut out_rx, handle) = drive(
        handler,
        user("alice"),
        vec![WsRequest {
            id: "send-corr".into(),
            command: api::Command::SendMessage {
                conversation_id: "conv-1".into(),
                content: "hello".into(),
                override_selection: None,
                system_refinement: String::new(),
                client_context: None,
                idempotency_key: None,
            },
        }],
    );

    // First frame: the ack carrying the correlation request_id.
    let ack_request_id = match next_frame(&mut out_rx).await {
        WsFrame::Result {
            id,
            result: api::CommandResult::SendMessageAck { request_id, .. },
        } => {
            assert_eq!(id, "send-corr");
            request_id
        }
        other => panic!("expected SendMessageAck, got {other:?}"),
    };
    assert!(!ack_request_id.is_empty(), "ack must carry a request_id");

    // Every streamed response event must carry that same request_id.
    let mut saw_delta = false;
    let mut saw_completed = false;
    for _ in 0..5 {
        match next_frame(&mut out_rx).await {
            WsFrame::Event {
                event: api::Event::AssistantDelta { request_id, .. },
            } => {
                assert_eq!(request_id, ack_request_id, "delta must match the ack id");
                saw_delta = true;
            }
            WsFrame::Event {
                event: api::Event::AssistantCompleted { request_id, .. },
            } => {
                assert_eq!(
                    request_id, ack_request_id,
                    "completed must match the ack id"
                );
                saw_completed = true;
                break;
            }
            other => panic!("unexpected frame while streaming: {other:?}"),
        }
    }
    assert!(saw_delta, "expected an AssistantDelta event");
    assert!(saw_completed, "expected an AssistantCompleted event");

    let _ = handle.await;
}
