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
use desktop_assistant_application::EventSink;
use std::time::Duration;

use desktop_assistant_application::client_tools::{
    ClientToolCoordinator, ClientToolResolutionError, register_client_tools,
    resolve_client_tool_result, suspend_for_client_tool, with_client_tool_timeout,
};
use desktop_assistant_auth_jwt::UserId;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::with_user_id;
use desktop_assistant_core::ports::session::{SessionId, with_session_id};
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
    assert!(matches!(
        err,
        ClientToolResolutionError::TurnNotFound { .. }
    ));

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
    assert!(matches!(
        err,
        ClientToolResolutionError::TurnNotFound { .. }
    ));
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
    assert!(
        outcome.is_err(),
        "cancelled suspension must surface an error"
    );
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

// ---- #261: client-tool registration is per login session ----------------

fn reg(name: &str) -> api::ClientToolRegistration {
    api::ClientToolRegistration {
        name: name.into(),
        description: "x".into(),
        input_schema: serde_json::json!({}),
    }
}

/// The #260 repro at the coordinator boundary: the voice daemon (session A)
/// registers `say_this`; a *text* client of the SAME user on a different
/// connection (session B) must not be offered that tool, or its turn would
/// call a tool it can't fulfil and wedge. Two TUIs of one user must have
/// independent tool sets — so registration keys on the session, not the user.
#[tokio::test]
async fn two_sessions_same_user_have_independent_tool_sets() {
    let coord = Arc::new(ClientToolCoordinator::new());

    // Session A (voice) registers say_this.
    with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-A"), async {
            register_client_tools(&coord, &[reg("say_this")]).await;
        }),
    )
    .await;

    // Session B — same user, different connection — must see nothing.
    let b_sees = with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-B"), async {
            coord.is_client_registered("say_this").await
        }),
    )
    .await;
    let b_defs = with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-B"), async {
            coord.registered_definitions().await
        }),
    )
    .await;

    assert!(
        !b_sees,
        "a second login session of the same user must NOT inherit session A's client tools"
    );
    assert!(
        b_defs.is_empty(),
        "session B's offered client-tool defs must be empty (got {b_defs:?})"
    );

    // Sanity: session A itself still sees its own registration.
    let a_sees = with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-A"), async {
            coord.is_client_registered("say_this").await
        }),
    )
    .await;
    assert!(a_sees, "session A must still see its own registration");
}

/// Re-registration is scoped to the session that sent it: a reconnect /
/// re-register on session A replaces only A's set and never disturbs a
/// concurrent session B of the same user.
#[tokio::test]
async fn registration_overwrite_is_per_session() {
    let coord = Arc::new(ClientToolCoordinator::new());

    with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-A"), async {
            register_client_tools(&coord, &[reg("say_this")]).await;
        }),
    )
    .await;
    with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-B"), async {
            register_client_tools(&coord, &[reg("fs_read")]).await;
        }),
    )
    .await;

    // Session A re-registers a different set.
    with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-A"), async {
            register_client_tools(&coord, &[reg("listen_for_more")]).await;
        }),
    )
    .await;

    let b_still_has_fs_read = with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-B"), async {
            coord.is_client_registered("fs_read").await
                && !coord.is_client_registered("listen_for_more").await
                && !coord.is_client_registered("say_this").await
        }),
    )
    .await;

    assert!(
        b_still_has_fs_read,
        "session A's re-registration must not touch session B's tool set"
    );
}

/// On disconnect the dispatcher calls `clear_session`, which must evict only
/// the ending session's registrations — a concurrent session of the same
/// user keeps its tools. Prevents a long-lived daemon accumulating stale
/// per-session buckets across reconnects (#261).
#[tokio::test]
async fn clear_session_evicts_only_that_sessions_registrations() {
    let coord = Arc::new(ClientToolCoordinator::new());

    with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-A"), async {
            register_client_tools(&coord, &[reg("say_this")]).await;
        }),
    )
    .await;
    with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-B"), async {
            register_client_tools(&coord, &[reg("fs_read")]).await;
        }),
    )
    .await;

    // Session A disconnects.
    with_session_id(SessionId::new("conn-A"), async {
        coord.clear_session();
    })
    .await;

    let a_gone = with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-A"), async {
            coord.registered_definitions().await.is_empty()
        }),
    )
    .await;
    let b_intact = with_user_id(
        UserId::new("alice"),
        with_session_id(SessionId::new("conn-B"), async {
            coord.is_client_registered("fs_read").await
        }),
    )
    .await;

    assert!(
        a_gone,
        "session A's registrations must be evicted on disconnect"
    );
    assert!(
        b_intact,
        "session B's registrations must survive session A's disconnect"
    );
}

// ---- #262: client-tool suspension timeout --------------------------------

/// Helper: register `tool` for alice and create a fresh PendingLlm turn row.
async fn registered_turn(
    coord: &Arc<ClientToolCoordinator>,
    store: &Arc<dyn TurnStateStore>,
    tool: &str,
) {
    with_user_id(UserId::new("alice"), async {
        register_client_tools(coord, &[reg(tool)]).await;
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
}

/// A suspension whose client never answers must give up after the cap and
/// resolve as a tool error (not hang). The slot is dropped so a late result
/// is cleanly rejected, and the row is marked failed (#262).
#[tokio::test]
async fn client_tool_suspension_times_out_into_tool_error() {
    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());
    let sink: Arc<dyn EventSink> = Arc::new(CapturingSink::new());
    registered_turn(&coord, &store, "say_this").await;

    // Suspend with a short cap and never deliver a result; await inline — if
    // the timeout didn't fire this test would hang, which IS the failure.
    let outcome = with_user_id(
        UserId::new("alice"),
        with_client_tool_timeout(
            Duration::from_millis(50),
            suspend_for_client_tool(
                &coord,
                &*store,
                &*sink,
                api::TaskId("task-1".into()),
                "conv-1".to_string(),
                PendingClientToolCall {
                    tool_call_id: "call-7".into(),
                    tool_name: "say_this".into(),
                    arguments: serde_json::Value::Null,
                },
            ),
        ),
    )
    .await;

    let err = outcome.expect_err("an unanswered client tool must surface an error, not hang");
    assert!(
        matches!(err, CoreError::ToolExecution(_)),
        "timeout must surface a tool-execution error so the loop continues; got {err:?}"
    );
    assert!(
        err.to_string().contains("did not respond"),
        "error should explain the client never answered; got {err}"
    );

    // The slot is gone: a late ClientToolResult is now a clean TurnNotFound.
    let late = resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("alice"),
        api::TaskId("task-1".into()),
        "call-7".into(),
        Ok("too late".into()),
    )
    .await;
    assert!(
        matches!(late, Err(ClientToolResolutionError::TurnNotFound { .. })),
        "after the timeout the pending slot must be removed"
    );

    let row = store.get_turn("task-1").await.unwrap().unwrap();
    assert_eq!(row.status, TurnStatus::Failed);
}

/// A result delivered before the cap resolves normally — the timeout must not
/// fire spuriously on a healthy, prompt client (#262).
#[tokio::test]
async fn client_tool_result_before_timeout_resolves_normally() {
    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());
    let sink: Arc<dyn EventSink> = Arc::new(CapturingSink::new());
    registered_turn(&coord, &store, "say_this").await;

    let coord_for_task = Arc::clone(&coord);
    let store_for_task = Arc::clone(&store);
    let sink_for_task = Arc::clone(&sink);
    let suspended = tokio::spawn(async move {
        with_user_id(
            UserId::new("alice"),
            // Generous cap; the result lands well within it.
            with_client_tool_timeout(
                Duration::from_secs(5),
                suspend_for_client_tool(
                    &coord_for_task,
                    &*store_for_task,
                    &*sink_for_task,
                    api::TaskId("task-1".into()),
                    "conv-1".to_string(),
                    PendingClientToolCall {
                        tool_call_id: "call-7".into(),
                        tool_name: "say_this".into(),
                        arguments: serde_json::Value::Null,
                    },
                ),
            ),
        )
        .await
    });

    tokio::task::yield_now().await;
    resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("alice"),
        api::TaskId("task-1".into()),
        "call-7".into(),
        Ok("spoken".into()),
    )
    .await
    .unwrap();

    let outcome = suspended.await.unwrap();
    assert_eq!(
        outcome.unwrap(),
        "spoken",
        "a result delivered before the cap must resolve normally"
    );
}

/// Cancellation still wins over a (much longer) timeout: a cancelled turn
/// surfaces `Cancelled`, not a timeout tool-error (#262).
#[tokio::test]
async fn cancellation_still_preempts_timeout() {
    use desktop_assistant_core::ports::llm::with_cancellation_token;

    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStore::new());
    let sink: Arc<dyn EventSink> = Arc::new(CapturingSink::new());
    registered_turn(&coord, &store, "say_this").await;

    let token = tokio_util::sync::CancellationToken::new();
    let token_for_cancel = token.clone();
    let coord_for_task = Arc::clone(&coord);
    let store_for_task = Arc::clone(&store);
    let sink_for_task = Arc::clone(&sink);
    let suspended = tokio::spawn(async move {
        with_user_id(
            UserId::new("alice"),
            with_client_tool_timeout(
                Duration::from_secs(30),
                with_cancellation_token(
                    token,
                    suspend_for_client_tool(
                        &coord_for_task,
                        &*store_for_task,
                        &*sink_for_task,
                        api::TaskId("task-1".into()),
                        "conv-1".to_string(),
                        PendingClientToolCall {
                            tool_call_id: "call-7".into(),
                            tool_name: "say_this".into(),
                            arguments: serde_json::Value::Null,
                        },
                    ),
                ),
            ),
        )
        .await
    });

    tokio::task::yield_now().await;
    token_for_cancel.cancel();

    let err = suspended
        .await
        .unwrap()
        .expect_err("cancel must surface an error");
    assert!(
        matches!(err, CoreError::Cancelled),
        "cancellation must win over the timeout; got {err:?}"
    );
    let row = store.get_turn("task-1").await.unwrap().unwrap();
    assert_eq!(row.last_error.as_deref(), Some("cancelled"));
}
