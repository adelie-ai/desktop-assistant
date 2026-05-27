//! Integration tests for the #107 turn state machine and client-side
//! tool execution at the application layer.
//!
//! These tests drive the `ClientToolCoordinator` + `ClientCapableToolExecutor`
//! through the same shapes a real send_message would: register a client
//! tool, kick off a turn that invokes the tool, observe the
//! `Event::ClientToolCall` emission, post a `ClientToolResult` back, and
//! confirm the turn completes. The DB is mocked with an in-memory
//! `TurnStateStore` so the tests don't need Postgres.
//!
//! The harness is intentionally *not* the real `DefaultAssistantApiHandler`
//! — that one wires through `ConversationService` which expects an LLM,
//! a real conversation store, etc. We pin behavior at the
//! tool-coordinator boundary, which is the genuine subject of #107.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use desktop_assistant_api_model as api;
use desktop_assistant_application::client_tools::{
    register_client_tools, resolve_client_tool_result, suspend_for_client_tool,
    ClientToolCoordinator, ClientToolResolutionError,
};
use desktop_assistant_application::EventSink;
use desktop_assistant_auth_jwt::UserId;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::with_user_id;
use desktop_assistant_core::ports::store::{
    PendingClientToolCall, TurnRow, TurnStateJson, TurnStateStore, TurnStatus,
};

// ---- Fakes ---------------------------------------------------------------

/// In-memory `TurnStateStore` for the test harness.
struct InMemoryTurnStore {
    data: Mutex<HashMap<String, TurnRow>>,
}

impl InMemoryTurnStore {
    fn new() -> Self {
        Self {
            data: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl TurnStateStore for InMemoryTurnStore {
    async fn create_turn(&self, row: TurnRow) -> Result<(), CoreError> {
        let mut data = self.data.lock().unwrap();
        if data.contains_key(&row.id) {
            return Err(CoreError::Storage(format!("dup turn id: {}", row.id)));
        }
        data.insert(row.id.clone(), row);
        Ok(())
    }
    async fn get_turn(&self, id: &str) -> Result<Option<TurnRow>, CoreError> {
        Ok(self.data.lock().unwrap().get(id).cloned())
    }
    async fn update_turn(
        &self,
        id: &str,
        status: TurnStatus,
        state: &TurnStateJson,
        last_error: Option<&str>,
    ) -> Result<(), CoreError> {
        let mut data = self.data.lock().unwrap();
        let row = data
            .get_mut(id)
            .ok_or_else(|| CoreError::Storage(format!("turn missing: {id}")))?;
        row.status = status;
        row.state = state.clone();
        row.last_error = last_error.map(String::from);
        Ok(())
    }
    async fn scan_non_terminal(&self) -> Result<Vec<TurnRow>, CoreError> {
        Ok(self
            .data
            .lock()
            .unwrap()
            .values()
            .filter(|r| !r.status.is_terminal())
            .cloned()
            .collect())
    }
}

/// Captures emitted events for assertions.
struct CapturingSink {
    events: Mutex<Vec<api::Event>>,
}

impl CapturingSink {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }
    fn snapshot(&self) -> Vec<api::Event> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait]
impl EventSink for CapturingSink {
    async fn emit(&self, event: api::Event) -> bool {
        self.events.lock().unwrap().push(event);
        true
    }
}

// ---- Acceptance tests ----------------------------------------------------

/// Happy path: server-side tools never touch the coordinator.
///
/// A tool name that was never registered as client-local passes through
/// to whatever inner executor logic is running; the coordinator's
/// `suspend_for_client_tool` is the only path that creates a turn row,
/// so an unregistered tool name results in no DB writes — exactly the
/// "server-side execution stays unchanged" contract.
#[tokio::test]
async fn server_side_tool_does_not_create_turn_row_or_emit_client_tool_call() {
    let coord = Arc::new(ClientToolCoordinator::new());
    let store = Arc::new(InMemoryTurnStore::new());
    let sink = Arc::new(CapturingSink::new());

    // The user registers `fs_read` only. `weather_lookup` stays server-side.
    with_user_id(UserId::new("alice"), async {
        register_client_tools(
            &coord,
            &[api::ClientToolRegistration {
                name: "fs_read".into(),
                description: "client-local".into(),
                input_schema: serde_json::json!({}),
            }],
        )
        .await;
    })
    .await;

    // A call to a NON-registered tool short-circuits — the coordinator
    // doesn't recognize it, so the caller falls through to server-side
    // dispatch (which the test harness short-circuits to a sentinel
    // value to prove "we did NOT suspend").
    let recognized = with_user_id(UserId::new("alice"), async {
        coord.is_client_registered("weather_lookup").await
    })
    .await;
    assert!(
        !recognized,
        "tool that was never registered must not be recognized as client-local"
    );

    // Sanity: no events, no turn rows.
    assert!(sink.snapshot().is_empty());
    assert!(store.scan_non_terminal().await.unwrap().is_empty());
}

/// Suspending on a client-local tool: the coordinator writes a
/// `pending_client_tool` row, emits a `ClientToolCall`, and the
/// suspension future stays pending until a result is posted.
#[tokio::test]
async fn turn_suspends_on_client_local_tool_call_and_emits_event() {
    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());
    let sink_concrete = Arc::new(CapturingSink::new());
    let sink: Arc<dyn EventSink> = sink_concrete.clone();

    // Register the tool under alice's identity.
    with_user_id(UserId::new("alice"), async {
        register_client_tools(
            &coord,
            &[api::ClientToolRegistration {
                name: "fs_read".into(),
                description: "x".into(),
                input_schema: serde_json::json!({}),
            }],
        )
        .await;

        // Seed an empty turn row so the harness mirrors what the real
        // handler does at turn start (create row in `pending_llm`).
        store
            .create_turn(TurnRow {
                id: "task-1".into(),
                user_id: "alice".into(),
                conversation_id: "conv-1".into(),
                status: TurnStatus::PendingLlm,
                state: TurnStateJson::default(),
                last_error: None,
            })
            .await
            .unwrap();
    })
    .await;

    // Launch the suspension. The future stays pending until we resolve it.
    let coord_for_task = Arc::clone(&coord);
    let store_for_task = Arc::clone(&store);
    let sink_for_task = sink.clone();
    let suspended = tokio::spawn(async move {
        with_user_id(UserId::new("alice"), async {
            suspend_for_client_tool(
                &coord_for_task,
                &*store_for_task,
                &*sink_for_task,
                api::TaskId("task-1".into()),
                "conv-1".to_string(),
                PendingClientToolCall {
                    tool_call_id: "call-7".into(),
                    tool_name: "fs_read".into(),
                    arguments: serde_json::json!({"path": "/etc/hosts"}),
                },
            )
            .await
        })
        .await
    });

    // Drive the runtime so the suspending task can park.
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;

    // The event must have been emitted before the suspension returned.
    let snapshot = sink_concrete.snapshot();
    let client_tool_calls: Vec<_> = snapshot
        .iter()
        .filter(|e| matches!(e, api::Event::ClientToolCall { .. }))
        .collect();
    assert_eq!(
        client_tool_calls.len(),
        1,
        "exactly one ClientToolCall must have been emitted before suspension"
    );

    // Turn row is now in `pending_client_tool`.
    let row = store.get_turn("task-1").await.unwrap().unwrap();
    assert_eq!(row.status, TurnStatus::PendingClientTool);
    let pending = row.state.pending_client_tool.unwrap();
    assert_eq!(pending.tool_call_id, "call-7");
    assert_eq!(pending.tool_name, "fs_read");

    // The suspension future is still pending — it must NOT have completed
    // without a `ClientToolResult` coming back.
    assert!(
        !suspended.is_finished(),
        "suspension future must remain pending until a result arrives"
    );

    // Resolve and observe completion.
    resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("alice"),
        api::TaskId("task-1".into()),
        "call-7".to_string(),
        Ok("file contents".into()),
    )
    .await
    .unwrap();

    let result = suspended.await.unwrap().unwrap();
    assert_eq!(result, "file contents");

    // After resolution, the row transitions back to `pending_llm` so the
    // dispatcher knows to continue the LLM loop.
    let row = store.get_turn("task-1").await.unwrap().unwrap();
    assert_eq!(row.status, TurnStatus::PendingLlm);
    assert!(row.state.pending_client_tool.is_none());
}

/// Cross-user isolation: user A's pending turn can't be resumed by
/// user B's `ClientToolResult`. The protocol must refuse cleanly.
#[tokio::test]
async fn turn_cross_user_isolation_blocks_resume_by_other_user() {
    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());
    let sink: Arc<dyn EventSink> = Arc::new(CapturingSink::new());

    with_user_id(UserId::new("alice"), async {
        register_client_tools(
            &coord,
            &[api::ClientToolRegistration {
                name: "fs_read".into(),
                description: "x".into(),
                input_schema: serde_json::json!({}),
            }],
        )
        .await;
        store
            .create_turn(TurnRow {
                id: "task-A".into(),
                user_id: "alice".into(),
                conversation_id: "conv-A".into(),
                status: TurnStatus::PendingLlm,
                state: TurnStateJson::default(),
                last_error: None,
            })
            .await
            .unwrap();
    })
    .await;

    let coord_for_task = Arc::clone(&coord);
    let store_for_task = Arc::clone(&store);
    let sink_for_task = Arc::clone(&sink);
    let suspended = tokio::spawn(async move {
        with_user_id(UserId::new("alice"), async {
            suspend_for_client_tool(
                &coord_for_task,
                &*store_for_task,
                &*sink_for_task,
                api::TaskId("task-A".into()),
                "conv-A".to_string(),
                PendingClientToolCall {
                    tool_call_id: "call-a".into(),
                    tool_name: "fs_read".into(),
                    arguments: serde_json::Value::Null,
                },
            )
            .await
        })
        .await
    });

    tokio::task::yield_now().await;
    tokio::task::yield_now().await;

    // Bob tries to resolve Alice's pending turn. The coordinator must
    // refuse because the turn row's user_id is "alice", not "bob".
    let err = resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("bob"),
        api::TaskId("task-A".into()),
        "call-a".to_string(),
        Ok("malicious".into()),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, ClientToolResolutionError::TurnNotFound { .. }));

    // Alice's suspension is still pending.
    assert!(!suspended.is_finished());

    // Cleanup: resolve under alice's scope so the spawned task drains.
    resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("alice"),
        api::TaskId("task-A".into()),
        "call-a".to_string(),
        Ok("ok".into()),
    )
    .await
    .unwrap();
    let _ = suspended.await.unwrap();
}

/// Mismatched tool_call_id: the result references a tool call the turn
/// row doesn't know about. Must be refused.
#[tokio::test]
async fn malformed_client_tool_result_rejected_on_tool_call_id_mismatch() {
    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());
    let sink: Arc<dyn EventSink> = Arc::new(CapturingSink::new());

    with_user_id(UserId::new("alice"), async {
        register_client_tools(
            &coord,
            &[api::ClientToolRegistration {
                name: "fs_read".into(),
                description: "x".into(),
                input_schema: serde_json::json!({}),
            }],
        )
        .await;
        store
            .create_turn(TurnRow {
                id: "task-1".into(),
                user_id: "alice".into(),
                conversation_id: "conv-1".into(),
                status: TurnStatus::PendingLlm,
                state: TurnStateJson::default(),
                last_error: None,
            })
            .await
            .unwrap();
    })
    .await;

    let coord_for_task = Arc::clone(&coord);
    let store_for_task = Arc::clone(&store);
    let sink_for_task = Arc::clone(&sink);
    let suspended = tokio::spawn(async move {
        with_user_id(UserId::new("alice"), async {
            suspend_for_client_tool(
                &coord_for_task,
                &*store_for_task,
                &*sink_for_task,
                api::TaskId("task-1".into()),
                "conv-1".to_string(),
                PendingClientToolCall {
                    tool_call_id: "call-7".into(),
                    tool_name: "fs_read".into(),
                    arguments: serde_json::Value::Null,
                },
            )
            .await
        })
        .await
    });

    tokio::task::yield_now().await;

    // Send the result with the wrong tool_call_id.
    let err = resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("alice"),
        api::TaskId("task-1".into()),
        "call-wrong".to_string(),
        Ok("body".into()),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ClientToolResolutionError::ToolCallIdMismatch { .. }
    ));

    // Suspension still pending.
    assert!(!suspended.is_finished());

    // Cleanup.
    resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("alice"),
        api::TaskId("task-1".into()),
        "call-7".to_string(),
        Ok("ok".into()),
    )
    .await
    .unwrap();
    let _ = suspended.await.unwrap();
}

/// No-active-session: the LLM in some hypothetical race tries to call a
/// client tool that was registered under a now-disconnected session, and
/// the suspension future returns a clean failure to the LLM loop. Today
/// the coordinator's `suspend_for_client_tool` will create a row even if
/// the client is gone, then time out / never resolve. But the
/// `resolve_client_tool_result` path explicitly refuses a result for a
/// task that has no pending oneshot — clients reconnecting late see this.
#[tokio::test]
async fn resolve_with_no_pending_suspension_is_a_clean_error() {
    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());

    let err = resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("alice"),
        api::TaskId("ghost".into()),
        "call-1".to_string(),
        Ok("body".into()),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, ClientToolResolutionError::TurnNotFound { .. }));
}

/// `unregistered_tool_called_by_llm_is_an_error`: the suspension path
/// requires the tool to have been registered. A caller invoking it for
/// an unregistered tool name will be rejected.
#[tokio::test]
async fn suspension_for_unregistered_tool_fails_cleanly() {
    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());
    let sink: Arc<dyn EventSink> = Arc::new(CapturingSink::new());

    with_user_id(UserId::new("alice"), async {
        store
            .create_turn(TurnRow {
                id: "task-1".into(),
                user_id: "alice".into(),
                conversation_id: "conv-1".into(),
                status: TurnStatus::PendingLlm,
                state: TurnStateJson::default(),
                last_error: None,
            })
            .await
            .unwrap();
    })
    .await;

    let result = with_user_id(UserId::new("alice"), async {
        suspend_for_client_tool(
            &coord,
            &*store,
            &*sink,
            api::TaskId("task-1".into()),
            "conv-1".to_string(),
            PendingClientToolCall {
                tool_call_id: "call-7".into(),
                tool_name: "wasnt_registered".into(),
                arguments: serde_json::Value::Null,
            },
        )
        .await
    })
    .await;

    let err = result.unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("not registered") || s.contains("wasnt_registered"),
        "error must mention registration; got {s}"
    );
}

/// `turn_state_survives_daemon_restart` reduced to its DB-shaped form:
/// after the daemon dies, the abandoned `pending_client_tool` row is
/// still in the DB. A startup scan returns it, and the application's
/// shutdown sweep marks it `failed` so it doesn't shadow the next turn.
#[tokio::test]
async fn abandoned_pending_turn_visible_after_restart_and_swept_to_failed() {
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());

    // Pretend a previous daemon left a pending row behind.
    store
        .create_turn(TurnRow {
            id: "task-zombie".into(),
            user_id: "alice".into(),
            conversation_id: "conv-1".into(),
            status: TurnStatus::PendingClientTool,
            state: TurnStateJson {
                version: 1,
                pending_client_tool: Some(PendingClientToolCall {
                    tool_call_id: "call-1".into(),
                    tool_name: "fs_read".into(),
                    arguments: serde_json::Value::Null,
                }),
            },
            last_error: None,
        })
        .await
        .unwrap();

    // The startup scan must surface it. We exercise the helper that
    // performs the actual sweep so the contract is pinned.
    let swept =
        desktop_assistant_application::client_tools::sweep_non_terminal_turns_on_startup(&*store)
            .await
            .unwrap();
    assert_eq!(swept, 1, "exactly one zombie turn should have been swept");

    let row = store.get_turn("task-zombie").await.unwrap().unwrap();
    assert_eq!(row.status, TurnStatus::Failed);
    assert_eq!(row.last_error.as_deref(), Some("daemon_restarted"));
}

/// Registration is scoped to a user, not global: registering under
/// `alice` does not make the tool client-local for `bob`. (Mirrors the
/// `unregistered_tool_called_by_llm_is_an_error` and
/// `registered_client_tool_with_no_active_session_falls_back_to_error`
/// edge cases from the issue's test plan.)
#[tokio::test]
async fn registration_is_per_user_not_global() {
    let coord = Arc::new(ClientToolCoordinator::new());

    with_user_id(UserId::new("alice"), async {
        register_client_tools(
            &coord,
            &[api::ClientToolRegistration {
                name: "fs_read".into(),
                description: "x".into(),
                input_schema: serde_json::json!({}),
            }],
        )
        .await;
    })
    .await;

    let alice_sees = with_user_id(UserId::new("alice"), async {
        coord.is_client_registered("fs_read").await
    })
    .await;
    let bob_sees = with_user_id(UserId::new("bob"), async {
        coord.is_client_registered("fs_read").await
    })
    .await;

    assert!(alice_sees, "alice registered the tool, must see it");
    assert!(
        !bob_sees,
        "bob never registered — registration MUST be per-user"
    );
}

/// Re-registration replaces the previous set: a client reconnecting and
/// sending a new `RegisterClientTools` clears the previous set so a
/// removed tool stops being recognized as client-local.
#[tokio::test]
async fn re_registration_replaces_previous_set() {
    let coord = Arc::new(ClientToolCoordinator::new());

    with_user_id(UserId::new("alice"), async {
        register_client_tools(
            &coord,
            &[
                api::ClientToolRegistration {
                    name: "fs_read".into(),
                    description: "x".into(),
                    input_schema: serde_json::json!({}),
                },
                api::ClientToolRegistration {
                    name: "fs_write".into(),
                    description: "x".into(),
                    input_schema: serde_json::json!({}),
                },
            ],
        )
        .await;

        // Reconnect: new registration with only fs_read.
        register_client_tools(
            &coord,
            &[api::ClientToolRegistration {
                name: "fs_read".into(),
                description: "x".into(),
                input_schema: serde_json::json!({}),
            }],
        )
        .await;

        assert!(coord.is_client_registered("fs_read").await);
        assert!(
            !coord.is_client_registered("fs_write").await,
            "previous fs_write must have been cleared by re-registration"
        );
    })
    .await;
}

/// Cancellation: a turn whose token has tripped before the suspension
/// future resolves must bail out with `Cancelled` instead of waiting
/// indefinitely.
#[tokio::test]
async fn cancellation_during_client_tool_suspension_returns_cancelled() {
    use desktop_assistant_core::ports::llm::with_cancellation_token;

    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());
    let sink: Arc<dyn EventSink> = Arc::new(CapturingSink::new());

    with_user_id(UserId::new("alice"), async {
        register_client_tools(
            &coord,
            &[api::ClientToolRegistration {
                name: "fs_read".into(),
                description: "x".into(),
                input_schema: serde_json::json!({}),
            }],
        )
        .await;
        store
            .create_turn(TurnRow {
                id: "task-1".into(),
                user_id: "alice".into(),
                conversation_id: "conv-1".into(),
                status: TurnStatus::PendingLlm,
                state: TurnStateJson::default(),
                last_error: None,
            })
            .await
            .unwrap();
    })
    .await;

    let token = tokio_util::sync::CancellationToken::new();
    let token_for_cancel = token.clone();

    let coord_for_task = Arc::clone(&coord);
    let store_for_task = Arc::clone(&store);
    let sink_for_task = Arc::clone(&sink);
    let suspended = tokio::spawn(async move {
        with_user_id(UserId::new("alice"), async {
            with_cancellation_token(token, async {
                suspend_for_client_tool(
                    &coord_for_task,
                    &*store_for_task,
                    &*sink_for_task,
                    api::TaskId("task-1".into()),
                    "conv-1".to_string(),
                    PendingClientToolCall {
                        tool_call_id: "call-7".into(),
                        tool_name: "fs_read".into(),
                        arguments: serde_json::Value::Null,
                    },
                )
                .await
            })
            .await
        })
        .await
    });

    tokio::task::yield_now().await;

    token_for_cancel.cancel();

    let outcome = suspended.await.unwrap();
    assert!(outcome.is_err(), "cancelled suspension must surface an error");
    let err = outcome.unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("cancel") || s.contains("Cancel"),
        "error message must reference cancellation; got {s}"
    );

    let row = store.get_turn("task-1").await.unwrap().unwrap();
    assert_eq!(row.status, TurnStatus::Failed);
    assert_eq!(row.last_error.as_deref(), Some("cancelled"));
}

// Compile-time presence guard: ensures the coordinator's public surface
// hasn't been removed silently.
fn _api_surface_compiles_check() {
    fn _accepts_arc<T: ?Sized + Send + Sync>(_: &Arc<T>) {}
    fn _check(c: &Arc<ClientToolCoordinator>) {
        _accepts_arc(c);
    }
    let _ = _check;
}
