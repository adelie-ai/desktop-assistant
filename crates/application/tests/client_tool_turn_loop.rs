//! End-to-end integration test for the activated client-tool turn loop
//! (#234).
//!
//! Unlike `client_tool_turn_state.rs` (which drives the coordinator's
//! suspend/resolve primitives directly), this test wires the *real* core
//! [`ConversationHandler`] turn loop to the *real*
//! [`ClientToolCoordinator`] via the [`CoordinatorClientToolPort`] adapter —
//! exactly the path a live `SendMessage` takes. A fake LLM calls a
//! registered client tool; the test asserts the turn:
//!
//! 1. offers + invokes the client tool, emitting `Event::ClientToolCall`,
//! 2. parks (the `send_prompt` future stays pending) until a result arrives,
//! 3. resumes when `resolve_client_tool_result` posts the client's output,
//!    feeding that output back to the LLM, which then finishes the turn.
//!
//! Cross-user isolation is re-verified at the turn-loop level: a result
//! posted under the wrong user must not wake the parked turn.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use desktop_assistant_api_model as api;
use desktop_assistant_application::EventSink;
use desktop_assistant_application::client_tools::{
    ClientToolCoordinator, CoordinatorClientToolPort, InMemoryTurnStateStore,
    register_client_tools, resolve_client_tool_result,
};
use desktop_assistant_auth_jwt::UserId;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, Message, ToolCall, ToolDefinition,
};
use desktop_assistant_core::ports::auth::with_user_id;
use desktop_assistant_core::ports::client_tools::{ClientToolPort, with_client_tools};
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::ports::llm::{ChunkCallback, LlmClient, LlmResponse, ReasoningConfig};
use desktop_assistant_core::ports::session::{SessionId, with_session_id};
use desktop_assistant_core::ports::store::{ConversationStore, TurnStateStore};
use desktop_assistant_core::service::ConversationHandler;

// ---- Fakes ---------------------------------------------------------------

/// In-memory conversation store (mirrors the core test `MockStore`).
#[derive(Default)]
struct MockStore {
    conversations: Mutex<HashMap<String, Conversation>>,
}

impl ConversationStore for MockStore {
    async fn create(&self, conv: Conversation) -> Result<(), CoreError> {
        self.conversations
            .lock()
            .unwrap()
            .insert(conv.id.0.clone(), conv);
        Ok(())
    }
    async fn get(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        self.conversations
            .lock()
            .unwrap()
            .get(&id.0)
            .cloned()
            .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
    }
    async fn list(
        &self,
    ) -> Result<Vec<desktop_assistant_core::domain::ConversationSummary>, CoreError> {
        Ok(self
            .conversations
            .lock()
            .unwrap()
            .values()
            .map(desktop_assistant_core::domain::ConversationSummary::from)
            .collect())
    }
    async fn update(&self, conv: Conversation) -> Result<(), CoreError> {
        self.conversations
            .lock()
            .unwrap()
            .insert(conv.id.0.clone(), conv);
        Ok(())
    }
    async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.conversations.lock().unwrap().remove(&id.0);
        Ok(())
    }
    async fn archive(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }
    async fn unarchive(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }
    async fn create_summary(
        &self,
        _conversation_id: &ConversationId,
        _summary: String,
        _start_ordinal: usize,
        _end_ordinal: usize,
    ) -> Result<String, CoreError> {
        Ok("summary-1".to_string())
    }
    async fn expand_summary(&self, _summary_id: &str) -> Result<(), CoreError> {
        Ok(())
    }
}

/// Fake LLM that returns a queued sequence of responses and records the tool
/// set it was offered on each call, so a test can assert exactly which tools
/// were advertised to the model.
struct ScriptedLlm {
    responses: Mutex<Vec<LlmResponse>>,
    offered_tools: Arc<Mutex<Vec<String>>>,
}

impl ScriptedLlm {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            offered_tools: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Shared handle to the accumulated set of tool names offered to the model
    /// across every `stream_completion` call.
    fn offered_recorder(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.offered_tools)
    }
}

#[async_trait]
impl LlmClient for ScriptedLlm {
    async fn stream_completion(
        &self,
        _messages: Vec<Message>,
        tools: &[ToolDefinition],
        _reasoning: ReasoningConfig,
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        {
            let mut offered = self.offered_tools.lock().unwrap();
            for t in tools {
                offered.push(t.name.clone());
            }
        }
        let response = {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Ok(LlmResponse::text("fallback"));
            }
            responses.remove(0)
        };
        if !response.text.is_empty() {
            on_chunk(response.text.clone());
        }
        Ok(response)
    }
}

/// Captures emitted events for assertions.
#[derive(Default)]
struct CapturingSink {
    events: Mutex<Vec<api::Event>>,
}

impl CapturingSink {
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

fn noop_chunk() -> ChunkCallback {
    Box::new(|_| true)
}
fn noop_status() -> desktop_assistant_core::ports::llm::StatusCallback {
    Box::new(|_| {})
}

fn make_handler(responses: Vec<LlmResponse>) -> ConversationHandler<MockStore, ScriptedLlm> {
    use std::sync::atomic::{AtomicU64, Ordering};
    let counter = Arc::new(AtomicU64::new(0));
    // `ConversationHandler::new` uses the `NoopToolExecutor`, so there are no
    // server-side tools at all — a `fs_read` call can only succeed via the
    // client-tool port, which is what this test pins.
    ConversationHandler::new(
        MockStore::default(),
        ScriptedLlm::new(responses),
        Box::new(move || {
            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
            format!("conv-{n}")
        }),
    )
}

/// Like [`make_handler`] but also returns a shared handle to the set of tool
/// names the model was offered, so a test can assert what was advertised.
fn make_handler_recording(
    responses: Vec<LlmResponse>,
) -> (
    ConversationHandler<MockStore, ScriptedLlm>,
    Arc<Mutex<Vec<String>>>,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    let counter = Arc::new(AtomicU64::new(0));
    let llm = ScriptedLlm::new(responses);
    let recorder = llm.offered_recorder();
    let handler = ConversationHandler::new(
        MockStore::default(),
        llm,
        Box::new(move || {
            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
            format!("conv-{n}")
        }),
    );
    (handler, recorder)
}

// ---- Test ----------------------------------------------------------------

/// The genuine #234 acceptance test: a fake LLM calls a registered client
/// tool, the turn emits `ClientToolCall` and parks, and posting a
/// `ClientToolResult` wakes it so the LLM finishes the turn.
#[tokio::test]
async fn turn_loop_suspends_on_client_tool_then_resumes_on_result() {
    let alice = UserId::new("alice");
    let task_id = api::TaskId("task-1".into());

    // The model first asks for `fs_read`, then answers using its output.
    let responses = vec![
        LlmResponse::with_tool_calls(
            "",
            vec![ToolCall::new(
                "call-1",
                "fs_read",
                r#"{"path":"/etc/hosts"}"#,
            )],
        ),
        LlmResponse::text("The hosts file maps 127.0.0.1 to localhost"),
    ];
    let handler = Arc::new(make_handler(responses));
    let conv = handler
        .create_conversation("Test".into(), vec![])
        .await
        .unwrap();
    let conv_id = conv.id.0.clone();

    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStateStore::new());
    let sink = Arc::new(CapturingSink::default());

    // Alice registers the client-local tool.
    with_user_id(alice.clone(), async {
        register_client_tools(
            &coord,
            &[api::ClientToolRegistration {
                name: "fs_read".into(),
                description: "Read a file on the client".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
        )
        .await;
    })
    .await;

    // The per-turn adapter, exactly as `run_send_turn` builds it.
    let port: Arc<dyn ClientToolPort> = Arc::new(CoordinatorClientToolPort::new(
        Arc::clone(&coord),
        Arc::clone(&store),
        sink.clone() as Arc<dyn EventSink>,
        task_id.clone(),
        conv_id.clone(),
    ));

    // Drive the real turn loop under alice's scope with the port installed.
    let handler_for_task = Arc::clone(&handler);
    let conv_id_for_task = conv_id.clone();
    let turn = tokio::spawn(async move {
        with_user_id(alice.clone(), async {
            with_client_tools(
                port,
                handler_for_task.send_prompt(
                    &ConversationId::from(conv_id_for_task.as_str()),
                    "Read /etc/hosts".into(),
                    noop_chunk(),
                    noop_status(),
                ),
            )
            .await
        })
        .await
    });

    // Let the turn run up to the suspension point.
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    // The turn emitted exactly one ClientToolCall and is parked.
    let calls: Vec<_> = sink
        .snapshot()
        .into_iter()
        .filter(|e| matches!(e, api::Event::ClientToolCall { .. }))
        .collect();
    assert_eq!(calls.len(), 1, "expected one ClientToolCall; got {calls:?}");
    match &calls[0] {
        api::Event::ClientToolCall {
            task_id: t,
            tool_call_id,
            tool_name,
            ..
        } => {
            assert_eq!(t, &task_id);
            assert_eq!(tool_call_id, "call-1");
            assert_eq!(tool_name, "fs_read");
        }
        other => panic!("unexpected event: {other:?}"),
    }
    assert!(
        !turn.is_finished(),
        "turn must remain parked until a ClientToolResult arrives"
    );

    // A cross-user result must NOT wake the parked turn (isolation).
    let cross_user_err = resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("mallory"),
        task_id.clone(),
        "call-1".to_string(),
        Ok("malicious".into()),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        cross_user_err,
        desktop_assistant_application::client_tools::ClientToolResolutionError::TurnNotFound { .. }
    ));
    assert!(
        !turn.is_finished(),
        "cross-user result must not wake the turn"
    );

    // The legitimate result wakes the turn; the LLM sees it and finishes.
    resolve_client_tool_result(
        &coord,
        &*store,
        UserId::new("alice"),
        task_id.clone(),
        "call-1".to_string(),
        Ok("127.0.0.1 localhost".into()),
    )
    .await
    .unwrap();

    let response = turn.await.unwrap().unwrap();
    assert_eq!(response, "The hosts file maps 127.0.0.1 to localhost");

    // The client's result was threaded into history as the tool result.
    let updated = handler
        .get_conversation(&ConversationId::from(conv_id.as_str()))
        .await
        .unwrap();
    let tool_msg = updated
        .messages
        .iter()
        .find(|m| m.role == desktop_assistant_core::domain::Role::Tool)
        .expect("a tool result message");
    assert_eq!(tool_msg.content, "127.0.0.1 localhost");
    assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call-1"));
}

/// #440 HIGH: cross-*session* isolation at the live turn loop. The existing
/// `#234` test proves cross-*user* isolation; this proves that two sessions of
/// the *same* user have independent client-tool sets (#261). A tool registered
/// only on session B must be neither advertised to the LLM nor callable when
/// the turn is driven under session A — even though both sessions belong to
/// the same user, so a `user_id`-only scope would wrongly conflate them.
#[tokio::test]
async fn session_b_tool_not_offered_or_callable_under_session_a() {
    let alice = UserId::new("alice");
    let session_a = SessionId::new("session-A");
    let session_b = SessionId::new("session-B");
    let task_id = api::TaskId("task-sess".into());

    // The model (ignoring what it was offered) tries to call session B's tool,
    // then answers. Under session A that call must not reach the client.
    let responses = vec![
        LlmResponse::with_tool_calls("", vec![ToolCall::new("call-b", "tool_b", r#"{}"#)]),
        LlmResponse::text("answered without tool_b"),
    ];
    let (handler, offered) = make_handler_recording(responses);
    let handler = Arc::new(handler);
    let conv = handler
        .create_conversation("Test".into(), vec![])
        .await
        .unwrap();
    let conv_id = conv.id.0.clone();

    let coord = Arc::new(ClientToolCoordinator::new());
    let store: Arc<dyn TurnStateStore> = Arc::new(InMemoryTurnStateStore::new());
    let sink = Arc::new(CapturingSink::default());

    // Two sessions of the SAME user register different tools.
    with_user_id(
        alice.clone(),
        with_session_id(session_a.clone(), async {
            register_client_tools(
                &coord,
                &[api::ClientToolRegistration {
                    name: "tool_a".into(),
                    description: "session A's tool".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }],
            )
            .await;
        }),
    )
    .await;
    with_user_id(
        alice.clone(),
        with_session_id(session_b.clone(), async {
            register_client_tools(
                &coord,
                &[api::ClientToolRegistration {
                    name: "tool_b".into(),
                    description: "session B's tool".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }],
            )
            .await;
        }),
    )
    .await;

    let port: Arc<dyn ClientToolPort> = Arc::new(CoordinatorClientToolPort::new(
        Arc::clone(&coord),
        Arc::clone(&store),
        sink.clone() as Arc<dyn EventSink>,
        task_id.clone(),
        conv_id.clone(),
    ));

    // The offered set is session-scoped: A sees only tool_a; B sees tool_b.
    let a_defs = with_user_id(
        alice.clone(),
        with_session_id(session_a.clone(), port.tool_definitions()),
    )
    .await;
    let a_names: Vec<String> = a_defs.iter().map(|d| d.name.clone()).collect();
    assert!(
        a_names.iter().any(|n| n == "tool_a"),
        "session A offers its own tool, got {a_names:?}"
    );
    assert!(
        !a_names.iter().any(|n| n == "tool_b"),
        "session A must NOT offer session B's tool, got {a_names:?}"
    );
    let b_defs = with_user_id(
        alice.clone(),
        with_session_id(session_b.clone(), port.tool_definitions()),
    )
    .await;
    assert!(
        b_defs.iter().any(|d| d.name == "tool_b"),
        "session B genuinely offers its own tool (registration is session-scoped, not globally lost)"
    );

    // Drive the real turn loop under session A. The scripted LLM calls tool_b,
    // which A can't reach — core routes it to the (empty) server-side executor
    // rather than suspending on the client, so the turn finishes on its own.
    let response = with_user_id(
        alice.clone(),
        with_session_id(
            session_a.clone(),
            with_client_tools(
                Arc::clone(&port),
                handler.send_prompt(
                    &ConversationId::from(conv_id.as_str()),
                    "please use tool_b".into(),
                    noop_chunk(),
                    noop_status(),
                ),
            ),
        ),
    )
    .await
    .expect("turn completes without hanging on an uncallable tool");
    assert_eq!(response, "answered without tool_b");

    // tool_b was never routed to the client port under session A, so no
    // ClientToolCall was ever emitted — it was not callable.
    let calls: Vec<_> = sink
        .snapshot()
        .into_iter()
        .filter(|e| matches!(e, api::Event::ClientToolCall { .. }))
        .collect();
    assert!(
        calls.is_empty(),
        "session A must not be able to call session B's tool, got {calls:?}"
    );

    // And tool_b was never advertised to the model under session A.
    let offered_names = offered.lock().unwrap().clone();
    assert!(
        offered_names.iter().any(|n| n == "tool_a"),
        "session A's own tool was advertised, got {offered_names:?}"
    );
    assert!(
        !offered_names.iter().any(|n| n == "tool_b"),
        "session B's tool must never be advertised under session A, got {offered_names:?}"
    );
}
