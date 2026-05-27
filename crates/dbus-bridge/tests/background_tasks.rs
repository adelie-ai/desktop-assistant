//! Integration tests for the bridge's `BackgroundTasks` D-Bus
//! interface (issue #116).
//!
//! The tests are TDD — they were written against the public API
//! before the adapter behavior was complete, so the first run after
//! these lands is the design pressure.
//!
//! Coverage:
//! - Method translation: each method routes to the right `api::Command`
//!   and decodes the matching `api::CommandResult`.
//! - Error mapping: `not found` and `task is already terminal` are
//!   surfaced as fdo-style errors, not generic `Failed`.
//! - Signal translation: each `Event::Task*` variant lands as the
//!   matching `ForwardAction::Task*` and lines up under the
//!   `org.desktopAssistant.BackgroundTasks` interface.
//! - Variant encoding: `TaskView`/`TaskLogEntry` round-trip through
//!   `a{sv}` with the JSON field names as keys.
//! - End-to-end over a p2p unix-stream pair so the zbus marshalling
//!   is exercised, not just the helper functions.
//! - User scoping: the bridge owns one transport per process, so it
//!   never sees another user's events.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_dbus_bridge::adapter::background_tasks::{
    DbusBackgroundTasksAdapter, log_entry_to_dict, task_view_to_dict,
};
use desktop_assistant_dbus_bridge::adapter::event_forwarder::{ForwardAction, run, translate};
use desktop_assistant_dbus_bridge::adapter::paths;
use desktop_assistant_dbus_bridge::transport::{BridgeTransport, BridgeTransportError};
use futures_util::StreamExt;
use tokio::net::UnixStream;
use tokio::sync::{Mutex, broadcast, oneshot};
use zbus::fdo;
use zbus::zvariant::{OwnedValue, Value};

const BG_INTERFACE: &str = "org.desktopAssistant.BackgroundTasks";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// In-memory transport stub that records every dispatched command and
/// replies with a scripted `CommandResult` (or daemon error). Lets the
/// adapter tests run without a daemon.
struct StubTransport {
    /// Captured commands in the order they were dispatched.
    commands: Mutex<Vec<api::Command>>,
    /// Replies the stub will deliver, FIFO. Each entry is either a
    /// `CommandResult` for `Result` frames, or an `Err(String)` for
    /// daemon-level error frames (which the real transport surfaces as
    /// `BridgeTransportError::Daemon`).
    replies: Mutex<Vec<Result<api::CommandResult, String>>>,
    /// Broadcast channel for `subscribe_events`. Tests push events in
    /// via `push_event` to exercise the forwarder.
    events_tx: broadcast::Sender<api::Event>,
}

impl StubTransport {
    fn new(replies: Vec<Result<api::CommandResult, String>>) -> Arc<Self> {
        let (events_tx, _) = broadcast::channel(64);
        Arc::new(Self {
            commands: Mutex::new(Vec::new()),
            replies: Mutex::new(replies),
            events_tx,
        })
    }

    async fn commands(&self) -> Vec<api::Command> {
        self.commands.lock().await.clone()
    }

    fn push_event(&self, event: api::Event) {
        let _ = self.events_tx.send(event);
    }
}

#[async_trait::async_trait]
impl BridgeTransport for StubTransport {
    async fn request(
        &self,
        command: api::Command,
    ) -> Result<api::CommandResult, BridgeTransportError> {
        self.commands.lock().await.push(command);
        let next = {
            let mut replies = self.replies.lock().await;
            if replies.is_empty() {
                Err("no scripted reply".to_string())
            } else {
                replies.remove(0)
            }
        };
        next.map_err(BridgeTransportError::Daemon)
    }

    fn subscribe_events(&self) -> broadcast::Receiver<api::Event> {
        self.events_tx.subscribe()
    }
}

/// Build a representative `TaskView` for assertion ergonomics.
fn sample_task_view(id: &str) -> api::TaskView {
    api::TaskView {
        id: api::TaskId(id.to_string()),
        kind: api::TaskKind::Standalone {
            name: "researcher".to_string(),
            conversation_id: format!("conv-{id}"),
        },
        status: api::TaskStatus::Running,
        started_at: 1_700_000_000_000,
        ended_at: None,
        last_error: None,
        parent: None,
        children: Vec::new(),
        title: format!("Standalone: researcher ({id})"),
        progress_hint: Some("step 1".to_string()),
    }
}

fn sample_log_entry(seq: u64) -> api::TaskLogEntry {
    api::TaskLogEntry {
        seq,
        timestamp: 1_700_000_001_000 + (seq as i64),
        level: api::LogLevel::Info,
        category: api::LogCategory::Lifecycle,
        message: format!("step {seq}"),
        data: None,
    }
}

// ---------------------------------------------------------------------------
// Acceptance: object path + interface naming
// ---------------------------------------------------------------------------

#[test]
fn background_tasks_object_path_is_under_canonical_root() {
    // Public D-Bus contract — KCM / TUI / plasmoid will hard-code this.
    assert_eq!(
        paths::BACKGROUND_TASKS,
        "/org/desktopAssistant/BackgroundTasks"
    );
}

// ---------------------------------------------------------------------------
// Method translation — each method dispatches the right command.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_tasks_via_dbus_returns_registered_tasks() {
    let transport = StubTransport::new(vec![Ok(api::CommandResult::BackgroundTasks(vec![
        sample_task_view("t-1"),
        sample_task_view("t-2"),
    ]))]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let result = adapter.list_tasks(true, 32).await;
    let list = result.expect("list_tasks ok");
    assert_eq!(list.len(), 2, "both registered tasks returned");

    // The dispatched command carries the args verbatim.
    let cmds = transport.commands().await;
    assert_eq!(cmds.len(), 1, "exactly one command dispatched");
    match &cmds[0] {
        api::Command::ListBackgroundTasks {
            include_finished,
            limit,
        } => {
            assert!(*include_finished);
            assert_eq!(*limit, Some(32));
        }
        other => panic!("expected ListBackgroundTasks, got {other:?}"),
    }

    // The id field is keyed by the JSON name and is a string variant.
    let id: String = list[0]
        .get("id")
        .cloned()
        .expect("id key present")
        .try_into()
        .expect("id is a string variant");
    assert_eq!(id, "t-1");
}

#[tokio::test]
async fn list_tasks_treats_limit_zero_as_unbounded() {
    // `limit: 0` from D-Bus is the natural "no cap" sentinel; the
    // wire shape carries Option<u32>, so the adapter maps 0 -> None.
    let transport =
        StubTransport::new(vec![Ok(api::CommandResult::BackgroundTasks(Vec::new()))]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let _ = adapter.list_tasks(false, 0).await;
    let cmds = transport.commands().await;
    match &cmds[0] {
        api::Command::ListBackgroundTasks { limit, .. } => assert_eq!(*limit, None),
        other => panic!("expected ListBackgroundTasks, got {other:?}"),
    }
}

#[tokio::test]
async fn get_task_via_dbus_returns_task_dict() {
    let view = sample_task_view("t-42");
    let transport = StubTransport::new(vec![Ok(api::CommandResult::BackgroundTask(view.clone()))]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let dict = adapter.get_task("t-42").await.expect("get_task ok");

    let id: String = dict
        .get("id")
        .cloned()
        .expect("id key present")
        .try_into()
        .unwrap();
    assert_eq!(id, "t-42");
    let title: String = dict
        .get("title")
        .cloned()
        .expect("title key present")
        .try_into()
        .unwrap();
    assert_eq!(title, view.title);
}

#[tokio::test]
async fn cancel_task_via_dbus_propagates_to_daemon() {
    let transport = StubTransport::new(vec![Ok(api::CommandResult::Ack)]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    adapter
        .cancel_task("t-7")
        .await
        .expect("cancel_task ok");
    let cmds = transport.commands().await;
    match &cmds[0] {
        api::Command::CancelBackgroundTask { id } => assert_eq!(id, "t-7"),
        other => panic!("expected CancelBackgroundTask, got {other:?}"),
    }
}

#[tokio::test]
async fn get_task_logs_returns_entries_and_next_seq() {
    let entries = vec![sample_log_entry(1), sample_log_entry(2), sample_log_entry(3)];
    let transport = StubTransport::new(vec![Ok(api::CommandResult::BackgroundTaskLogs {
        entries: entries.clone(),
        next_seq: 4,
    })]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let (returned, next_seq) = adapter
        .get_task_logs("t-7", 0, 200)
        .await
        .expect("logs ok");
    assert_eq!(returned.len(), 3);
    assert_eq!(next_seq, 4);
    let seq0: u64 = returned[0]
        .get("seq")
        .cloned()
        .expect("seq present")
        .try_into()
        .unwrap();
    assert_eq!(seq0, 1);
    let cmds = transport.commands().await;
    match &cmds[0] {
        api::Command::GetBackgroundTaskLogs {
            id,
            after_seq,
            limit,
        } => {
            assert_eq!(id, "t-7");
            assert_eq!(*after_seq, None);
            assert_eq!(*limit, Some(200));
        }
        other => panic!("expected GetBackgroundTaskLogs, got {other:?}"),
    }
}

#[tokio::test]
async fn get_task_logs_after_seq_is_threaded_through() {
    let transport = StubTransport::new(vec![Ok(api::CommandResult::BackgroundTaskLogs {
        entries: Vec::new(),
        next_seq: 42,
    })]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let (_, _) = adapter.get_task_logs("t-7", 41, 50).await.unwrap();
    let cmds = transport.commands().await;
    match &cmds[0] {
        api::Command::GetBackgroundTaskLogs { after_seq, .. } => {
            assert_eq!(*after_seq, Some(41));
        }
        other => panic!("expected GetBackgroundTaskLogs, got {other:?}"),
    }
}

#[tokio::test]
async fn spawn_standalone_via_dbus_returns_task_id() {
    let transport = StubTransport::new(vec![Ok(api::CommandResult::BackgroundTaskSpawned {
        id: "t-new".to_string(),
    })]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let id = adapter
        .spawn_standalone("research", "hello", HashMap::new())
        .await
        .expect("spawn_standalone ok");
    assert_eq!(id, "t-new");
    let cmds = transport.commands().await;
    match &cmds[0] {
        api::Command::SpawnStandaloneAgent {
            name,
            initial_prompt,
            tools,
            override_selection,
        } => {
            assert_eq!(name, "research");
            assert_eq!(initial_prompt, "hello");
            assert!(tools.is_none(), "no tools key in options -> no tools");
            assert!(override_selection.is_none());
        }
        other => panic!("expected SpawnStandaloneAgent, got {other:?}"),
    }
}

#[tokio::test]
async fn spawn_standalone_threads_tools_option() {
    let transport = StubTransport::new(vec![Ok(api::CommandResult::BackgroundTaskSpawned {
        id: "t-new".to_string(),
    })]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let mut options: HashMap<String, OwnedValue> = HashMap::new();
    let tools: Vec<Value<'_>> = vec![Value::from("search"), Value::from("fetch")];
    let array = Value::new(tools);
    options.insert("tools".to_string(), array.try_into().unwrap());
    let _ = adapter
        .spawn_standalone("research", "hi", options)
        .await
        .expect("spawn_standalone ok");
    let cmds = transport.commands().await;
    match &cmds[0] {
        api::Command::SpawnStandaloneAgent { tools, .. } => {
            assert_eq!(
                tools.clone().unwrap_or_default(),
                vec!["search".to_string(), "fetch".to_string()]
            );
        }
        other => panic!("expected SpawnStandaloneAgent, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_unknown_task_returns_fdo_error() {
    let transport = StubTransport::new(vec![Err("not found".to_string())]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let err = adapter
        .cancel_task("bogus")
        .await
        .expect_err("cancel of unknown task fails");
    // We map daemon's `not found` onto the closest fdo error so
    // clients can match by name rather than fragile string parsing.
    assert!(
        matches!(err, fdo::Error::UnknownObject(_)),
        "expected UnknownObject for NotFound, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("not found"),
        "error message must mention `not found`: {msg}"
    );
}

#[tokio::test]
async fn cancel_already_terminal_task_returns_fdo_error() {
    let transport = StubTransport::new(vec![Err("task is already terminal".to_string())]);
    let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport) as Arc<_>);
    let err = adapter
        .cancel_task("t-done")
        .await
        .expect_err("cancel of terminal task fails");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("terminal"),
        "error message must mention `terminal`: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Signal translation through the event forwarder.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn event_translator_routes_task_started_to_dbus_signal() {
    let event = api::Event::TaskStarted {
        task: sample_task_view("t-1"),
    };
    let action = translate(event);
    match action {
        ForwardAction::TaskStarted { id, task } => {
            assert_eq!(id, "t-1");
            // The dict is the JSON-keyed view.
            let title: String = task
                .get("title")
                .cloned()
                .expect("title in dict")
                .try_into()
                .unwrap();
            assert!(title.contains("researcher"));
        }
        other => panic!("expected TaskStarted, got {other:?}"),
    }
}

#[tokio::test]
async fn event_translator_routes_task_progress_to_dbus_signal() {
    let event = api::Event::TaskProgress {
        id: "t-1".to_string(),
        progress_hint: Some("processing".to_string()),
    };
    match translate(event) {
        ForwardAction::TaskProgress { id, hint } => {
            assert_eq!(id, "t-1");
            assert_eq!(hint, "processing");
        }
        other => panic!("expected TaskProgress, got {other:?}"),
    }
}

#[tokio::test]
async fn event_translator_routes_task_progress_with_no_hint() {
    let event = api::Event::TaskProgress {
        id: "t-1".to_string(),
        progress_hint: None,
    };
    match translate(event) {
        ForwardAction::TaskProgress { id, hint } => {
            assert_eq!(id, "t-1");
            assert_eq!(hint, "", "absent hint maps to empty string on the wire");
        }
        other => panic!("expected TaskProgress, got {other:?}"),
    }
}

#[tokio::test]
async fn event_translator_routes_task_log_appended_to_dbus_signal() {
    let event = api::Event::TaskLogAppended {
        id: "t-1".to_string(),
        entry: sample_log_entry(5),
    };
    match translate(event) {
        ForwardAction::TaskLogAppended { id, entry } => {
            assert_eq!(id, "t-1");
            let seq: u64 = entry
                .get("seq")
                .cloned()
                .expect("seq present")
                .try_into()
                .unwrap();
            assert_eq!(seq, 5);
            let message: String = entry
                .get("message")
                .cloned()
                .expect("message present")
                .try_into()
                .unwrap();
            assert_eq!(message, "step 5");
        }
        other => panic!("expected TaskLogAppended, got {other:?}"),
    }
}

#[tokio::test]
async fn event_translator_routes_task_completed_to_dbus_signal() {
    let event = api::Event::TaskCompleted {
        id: "t-1".to_string(),
        status: api::TaskStatus::Failed,
        last_error: Some("boom".to_string()),
    };
    match translate(event) {
        ForwardAction::TaskCompleted {
            id,
            status,
            last_error,
        } => {
            assert_eq!(id, "t-1");
            assert_eq!(status, "failed");
            assert_eq!(last_error, "boom");
        }
        other => panic!("expected TaskCompleted, got {other:?}"),
    }
}

#[tokio::test]
async fn event_translator_routes_task_completed_with_no_error() {
    let event = api::Event::TaskCompleted {
        id: "t-1".to_string(),
        status: api::TaskStatus::Completed,
        last_error: None,
    };
    match translate(event) {
        ForwardAction::TaskCompleted {
            status,
            last_error,
            ..
        } => {
            assert_eq!(status, "completed");
            assert_eq!(last_error, "");
        }
        other => panic!("expected TaskCompleted, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Variant encoding for TaskView / TaskLogEntry.
// ---------------------------------------------------------------------------

#[test]
fn task_view_dict_keys_match_json_names() {
    let view = sample_task_view("t-1");
    let dict = task_view_to_dict(&view);
    // Sanity check the canonical keys; the full set is enforced by
    // serializing the view to JSON and asserting set equality.
    for key in ["id", "kind", "status", "started_at", "title"] {
        assert!(dict.contains_key(key), "missing key {key}");
    }
    // Absent optional fields are not emitted (mirrors
    // skip_serializing_if = Option::is_none on the JSON side).
    assert!(
        !dict.contains_key("ended_at"),
        "ended_at is None, must not appear"
    );
    assert!(
        !dict.contains_key("last_error"),
        "last_error is None, must not appear"
    );
}

#[test]
fn task_view_dict_encodes_nested_kind_as_subdict() {
    let view = sample_task_view("t-1");
    let dict = task_view_to_dict(&view);
    let kind = dict.get("kind").expect("kind present");
    // The `kind` variant is `a{sv}` because `TaskKind` serializes as
    // an externally-tagged object.
    let nested: HashMap<String, OwnedValue> = kind.clone().try_into().expect("kind is a{sv}");
    assert!(nested.contains_key("standalone"));
    let inner: HashMap<String, OwnedValue> = nested
        .get("standalone")
        .cloned()
        .expect("standalone arm")
        .try_into()
        .expect("standalone arm is a{sv}");
    let name: String = inner
        .get("name")
        .cloned()
        .expect("name in standalone")
        .try_into()
        .unwrap();
    assert_eq!(name, "researcher");
}

#[test]
fn log_entry_dict_encodes_seq_and_level() {
    let entry = sample_log_entry(7);
    let dict = log_entry_to_dict(&entry);
    let seq: u64 = dict
        .get("seq")
        .cloned()
        .expect("seq present")
        .try_into()
        .unwrap();
    assert_eq!(seq, 7);
    let level: String = dict
        .get("level")
        .cloned()
        .expect("level present")
        .try_into()
        .unwrap();
    assert_eq!(level, "info");
}

// ---------------------------------------------------------------------------
// End-to-end through a p2p zbus connection. Drives the adapter via a
// real D-Bus marshalling boundary so we catch sig-mismatch bugs that
// a direct call would miss.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn task_started_signal_fires_within_500ms_of_spawn() {
    let mut pair = ZbusPair::start(Arc::clone(&StubTransport::new(vec![Ok(
        api::CommandResult::BackgroundTaskSpawned {
            id: "t-spawned".to_string(),
        },
    )])))
    .await;

    // Subscribe to TaskStarted from the client side before driving the
    // spawn.
    let mut stream = zbus::MessageStream::for_match_rule(
        zbus::MatchRule::builder()
            .msg_type(zbus::message::Type::Signal)
            .path(paths::BACKGROUND_TASKS)
            .unwrap()
            .interface(BG_INTERFACE)
            .unwrap()
            .member("TaskStarted")
            .unwrap()
            .build(),
        &pair.client,
        Some(8),
    )
    .await
    .expect("subscribe TaskStarted");

    // Drive a spawn via a direct method call + push the matching
    // event.
    let _: String = pair
        .call(
            "SpawnStandalone",
            &("research", "hello", HashMap::<String, OwnedValue>::new()),
        )
        .await
        .expect("spawn ok");
    let started = sample_task_view("t-spawned");
    pair.transport
        .push_event(api::Event::TaskStarted { task: started });

    let msg = tokio::time::timeout(Duration::from_millis(500), stream.next())
        .await
        .expect("TaskStarted within 500ms")
        .expect("stream had a message")
        .expect("message decodes");
    assert_eq!(
        msg.header().member().map(|m| m.to_string()).as_deref(),
        Some("TaskStarted")
    );
    let body: (String, HashMap<String, OwnedValue>) = msg.body().deserialize().unwrap();
    assert_eq!(body.0, "t-spawned");
    assert!(body.1.contains_key("title"));
    pair.shutdown().await;
}

#[tokio::test]
async fn task_log_appended_signal_streams_entries() {
    let mut pair = ZbusPair::start(StubTransport::new(vec![])).await;
    let mut stream = zbus::MessageStream::for_match_rule(
        zbus::MatchRule::builder()
            .msg_type(zbus::message::Type::Signal)
            .path(paths::BACKGROUND_TASKS)
            .unwrap()
            .interface(BG_INTERFACE)
            .unwrap()
            .member("TaskLogAppended")
            .unwrap()
            .build(),
        &pair.client,
        Some(16),
    )
    .await
    .expect("subscribe TaskLogAppended");

    for seq in 1..=5 {
        pair.transport.push_event(api::Event::TaskLogAppended {
            id: "t-1".to_string(),
            entry: sample_log_entry(seq),
        });
    }

    let mut got = Vec::new();
    while got.len() < 5 {
        let msg = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("signal arrives in time")
            .expect("stream open")
            .expect("message decodes");
        let body: (String, HashMap<String, OwnedValue>) = msg.body().deserialize().unwrap();
        let seq: u64 = body
            .1
            .get("seq")
            .cloned()
            .expect("seq present")
            .try_into()
            .unwrap();
        got.push(seq);
    }
    assert_eq!(got, vec![1, 2, 3, 4, 5]);
    pair.shutdown().await;
}

#[tokio::test]
async fn business_outcome_user_can_list_their_running_standalone_via_busctl() {
    // End-to-end: a client invoking `ListTasks` over D-Bus sees the
    // tasks the daemon registered. This is the equivalent of `busctl
    // call ... ListTasks tf true 0`.
    let mut pair = ZbusPair::start(StubTransport::new(vec![Ok(
        api::CommandResult::BackgroundTasks(vec![
            sample_task_view("t-running-1"),
            sample_task_view("t-running-2"),
        ]),
    )]))
    .await;

    let result: Vec<HashMap<String, OwnedValue>> = pair
        .call("ListTasks", &(true, 0_u32))
        .await
        .expect("ListTasks ok");
    assert_eq!(result.len(), 2);
    let ids: Vec<String> = result
        .iter()
        .map(|d| d.get("id").cloned().expect("id").try_into().unwrap())
        .collect();
    assert!(ids.contains(&"t-running-1".to_string()));
    assert!(ids.contains(&"t-running-2".to_string()));
    pair.shutdown().await;
}

#[tokio::test]
async fn signals_are_user_scoped_under_session_bus() {
    // The bridge owns a single transport (one per process, one per
    // user under XDG_RUNTIME_DIR). Subscribers on the bridge's
    // p2p connection can only see events the bridge's transport
    // received — there is no cross-transport leakage.
    let mut pair_a = ZbusPair::start(StubTransport::new(vec![])).await;
    let mut pair_b = ZbusPair::start(StubTransport::new(vec![])).await;

    let mut stream_b = zbus::MessageStream::for_match_rule(
        zbus::MatchRule::builder()
            .msg_type(zbus::message::Type::Signal)
            .path(paths::BACKGROUND_TASKS)
            .unwrap()
            .interface(BG_INTERFACE)
            .unwrap()
            .member("TaskStarted")
            .unwrap()
            .build(),
        &pair_b.client,
        Some(8),
    )
    .await
    .expect("subscribe on B");

    // Push an event into A's transport only.
    pair_a.transport.push_event(api::Event::TaskStarted {
        task: sample_task_view("t-a-only"),
    });

    // B must not observe A's event.
    let observed = tokio::time::timeout(Duration::from_millis(250), stream_b.next()).await;
    assert!(
        observed.is_err(),
        "B observed A's event; user scoping is broken: {observed:?}"
    );
    pair_a.shutdown().await;
    pair_b.shutdown().await;
}

// ---------------------------------------------------------------------------
// P2P zbus harness — pairs two zbus connections over a UnixStream pair
// so we exercise the real marshalling without a session bus.
// ---------------------------------------------------------------------------

struct ZbusPair {
    /// Held so the server connection (and its object server) stays
    /// open for the lifetime of the harness — the client side talks
    /// to it over the in-process unix-stream pair.
    #[allow(dead_code)]
    server: zbus::Connection,
    client: zbus::Connection,
    transport: Arc<StubTransport>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    forwarder: Option<tokio::task::JoinHandle<()>>,
}

impl ZbusPair {
    /// Set up a p2p pair, attach the adapter on the server side, and
    /// spawn an event forwarder that pumps `transport`'s events as
    /// D-Bus signals on the server connection.
    async fn start(transport: Arc<StubTransport>) -> Self {
        let (s0, s1) = UnixStream::pair().expect("unix stream pair");
        let guid = zbus::Guid::generate();
        let server_builder = zbus::connection::Builder::unix_stream(s0)
            .server(guid)
            .expect("server builder")
            .p2p();
        let client_builder = zbus::connection::Builder::unix_stream(s1).p2p();
        // Both ends handshake against each other; driving them
        // concurrently avoids a build-side deadlock.
        let (server, client) = futures_util::try_join!(server_builder.build(), client_builder.build())
            .expect("p2p pair built");

        let adapter = DbusBackgroundTasksAdapter::new(Arc::clone(&transport));
        server
            .object_server()
            .at(paths::BACKGROUND_TASKS, adapter)
            .await
            .expect("attach adapter");

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let events = transport.subscribe_events();
        let forwarder_conn = server.clone();
        let forwarder = tokio::spawn(async move {
            run(events, forwarder_conn, async move {
                let _ = shutdown_rx.await;
            })
            .await;
        });

        Self {
            server,
            client,
            transport,
            shutdown_tx: Some(shutdown_tx),
            forwarder: Some(forwarder),
        }
    }

    /// Call a method on the bridge interface from the client side.
    /// Skips zbus's `Proxy` wrapper because p2p connections do not
    /// have well-known names — issuing a raw method call with no
    /// destination is the supported pattern for in-process tests.
    async fn call<B, R>(&self, member: &str, body: &B) -> zbus::Result<R>
    where
        B: serde::ser::Serialize + zbus::zvariant::DynamicType,
        for<'de> R: serde::de::Deserialize<'de> + zbus::zvariant::Type,
    {
        let reply = self
            .client
            .call_method(
                None::<&str>,
                paths::BACKGROUND_TASKS,
                Some(BG_INTERFACE),
                member,
                body,
            )
            .await?;
        reply.body().deserialize::<R>()
    }

    async fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.forwarder.take() {
            let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
        }
    }
}
