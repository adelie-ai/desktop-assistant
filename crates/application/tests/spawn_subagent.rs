//! Acceptance tests for the `spawn_subagent` / `get_subagent_status`
//! builtin tools (#112). Written TDD-first against the API described
//! in the issue: a parent task can spawn a child task that runs a
//! fresh conversation to completion, with `wait=true` returning the
//! child's final assistant text and `wait=false` returning a handle
//! the parent can poll.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_application::UserId;
use desktop_assistant_application::background_tasks::{BackgroundTaskRegistry, current_task_id};
use desktop_assistant_application::subagent_tools::{
    SubagentTools, TOOL_GET_SUBAGENT_STATUS, TOOL_SPAWN_SUBAGENT, tool_definitions,
};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, ConversationId, ConversationSummary, Message};
use desktop_assistant_core::ports::auth::with_user_id;
use desktop_assistant_core::ports::inbound::{
    ConversationService, PromptDispatchOutcome, PromptSelectionOverride,
};
use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};
use tokio::sync::Notify;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

// --------------------------------------------------------------------
// Fake ConversationService that lets tests drive child turns precisely.
// --------------------------------------------------------------------

/// A single recorded turn the LLM "ran" — captures the conversation id,
/// the prompt, and any override so tests can assert wire-through.
#[derive(Debug, Clone)]
struct RecordedTurn {
    conversation_id: String,
    #[allow(dead_code)]
    prompt: String,
    override_selection: Option<PromptSelectionOverride>,
}

/// Per-conversation behaviour the test wires up before driving the
/// subagent. The behaviour mutates as turns run so e.g. a "grandchild"
/// turn can spawn its own subagent before returning text.
type BoxedResult =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, CoreError>> + Send>>;
type TurnBehaviour = Arc<dyn Fn(String, String) -> BoxedResult + Send + Sync>;

#[derive(Default)]
struct FakeConvState {
    conversations: HashMap<String, Conversation>,
    turns: Vec<RecordedTurn>,
}

#[derive(Clone)]
struct FakeConversations {
    state: Arc<Mutex<FakeConvState>>,
    default_behaviour: TurnBehaviour,
    per_conv_behaviour: Arc<Mutex<HashMap<String, TurnBehaviour>>>,
    id_counter: Arc<AtomicUsize>,
}

impl FakeConversations {
    fn new(default_text: &str) -> Self {
        let text = default_text.to_string();
        let default_behaviour: TurnBehaviour = Arc::new(move |_cid, _prompt| {
            let t = text.clone();
            Box::pin(async move { Ok(t) })
        });
        Self {
            state: Arc::new(Mutex::new(FakeConvState::default())),
            default_behaviour,
            per_conv_behaviour: Arc::new(Mutex::new(HashMap::new())),
            id_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn with_behaviour<F, Fut>(self, conversation_id: &str, body: F) -> Self
    where
        F: Fn(String, String) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<String, CoreError>> + Send + 'static,
    {
        let wrapper: TurnBehaviour = Arc::new(move |cid, p| Box::pin(body(cid, p)));
        self.per_conv_behaviour
            .lock()
            .unwrap()
            .insert(conversation_id.to_string(), wrapper);
        self
    }

    fn turns(&self) -> Vec<RecordedTurn> {
        self.state.lock().unwrap().turns.clone()
    }

    fn next_id(&self) -> String {
        let n = self.id_counter.fetch_add(1, Ordering::SeqCst);
        format!("conv-{n}")
    }
}

#[async_trait::async_trait]
impl ConversationService for FakeConversations {
    async fn create_conversation(
        &self,
        title: String,
        _tags: Vec<String>,
    ) -> Result<Conversation, CoreError> {
        let id = self.next_id();
        let conv = Conversation::new(id.clone(), title);
        self.state
            .lock()
            .unwrap()
            .conversations
            .insert(id, conv.clone());
        Ok(conv)
    }

    async fn list_conversations(
        &self,
        _max_age_days: Option<u32>,
        _include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        Ok(vec![])
    }

    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        self.state
            .lock()
            .unwrap()
            .conversations
            .get(id.as_str())
            .cloned()
            .ok_or_else(|| {
                CoreError::ConversationNotFound(format!("conversation {} not found", id.as_str()))
            })
    }

    async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }
    async fn rename_conversation(
        &self,
        _id: &ConversationId,
        _title: String,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }
    async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }
    async fn clear_all_history(&self) -> Result<u32, CoreError> {
        Ok(0)
    }

    async fn send_prompt(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        _on_chunk: ChunkCallback,
        _on_status: StatusCallback,
    ) -> Result<String, CoreError> {
        // Delegate via send_prompt_with_override with no override so
        // tests configure one behaviour set.
        let outcome = self
            .send_prompt_with_override(
                conversation_id,
                prompt,
                None,
                String::new(),
                Box::new(|_: String| true),
                Box::new(|_| {}),
                CancellationToken::new(),
            )
            .await?;
        Ok(outcome.response)
    }

    async fn send_prompt_with_override(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        override_selection: Option<PromptSelectionOverride>,
        _system_refinement: String,
        _on_chunk: ChunkCallback,
        _on_status: StatusCallback,
        cancellation: CancellationToken,
    ) -> Result<PromptDispatchOutcome, CoreError> {
        let cid = conversation_id.as_str().to_string();
        let turn = RecordedTurn {
            conversation_id: cid.clone(),
            prompt: prompt.clone(),
            override_selection: override_selection.clone(),
        };
        {
            let mut state = self.state.lock().unwrap();
            state.turns.push(turn);
            // Track the prompt as a user message for inspection.
            if let Some(conv) = state.conversations.get_mut(&cid) {
                conv.messages.push(Message::new(
                    desktop_assistant_core::domain::Role::User,
                    &prompt,
                ));
            }
        }

        // Pick behaviour: per-conversation override, else default.
        let behaviour = {
            let per = self.per_conv_behaviour.lock().unwrap();
            per.get(&cid)
                .cloned()
                .unwrap_or_else(|| self.default_behaviour.clone())
        };

        // Install the cancellation token for the duration of the call,
        // mirroring how the real `send_prompt_with_override` works.
        let inner = behaviour(cid.clone(), prompt);
        let result =
            desktop_assistant_core::ports::llm::with_cancellation_token(cancellation, inner)
                .await?;

        // Record assistant message in the conversation history.
        {
            let mut state = self.state.lock().unwrap();
            if let Some(conv) = state.conversations.get_mut(&cid) {
                conv.messages.push(Message::new(
                    desktop_assistant_core::domain::Role::Assistant,
                    &result,
                ));
            }
        }

        Ok(PromptDispatchOutcome {
            response: result,
            warnings: Vec::new(),
        })
    }
}

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

fn unique_user(label: &str) -> UserId {
    UserId::new(format!("user-{label}-{}", uuid::Uuid::new_v4()))
}

/// Run `body` as if it were the body of a parent `BackgroundTask`. The
/// registry installs `CURRENT_TASK_ID` only inside spawned bodies — this
/// helper exists so a test can call the tool from "inside" a synthetic
/// parent without spinning up a real LLM stack.
async fn under_parent_task<F, Fut, T>(
    registry: &BackgroundTaskRegistry,
    user: UserId,
    parent_conv: &str,
    body: F,
) -> (api::TaskId, T)
where
    F: FnOnce(api::TaskId) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let result_slot: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
    let result_slot_inner = Arc::clone(&result_slot);
    let parent_id = registry.spawn(
        user,
        api::TaskKind::Conversation {
            conversation_id: parent_conv.into(),
        },
        "parent".into(),
        move |ctx| async move {
            let parent_id = ctx.task_id.clone();
            let token = ctx.token.clone();
            // Install the per-turn cancellation token the same way
            // `send_prompt_with_override` would. Tool bodies read this
            // to propagate cancellation into child registry tasks.
            let value =
                desktop_assistant_core::ports::llm::with_cancellation_token(token, body(parent_id))
                    .await;
            *result_slot_inner.lock().unwrap() = Some(value);
            Ok(())
        },
    );
    timeout(Duration::from_secs(5), registry.wait(&parent_id))
        .await
        .expect("parent task must finish, not hang");
    let value = result_slot
        .lock()
        .unwrap()
        .take()
        .expect("body produced a value");
    (parent_id, value)
}

/// Like [`under_parent_task`] but the parent body does NOT return until
/// the caller fires the returned release `Notify`. The body runs `body`,
/// publishes its produced value, then parks — so the parent (and any
/// `wait=false` child it left running) stay live in the registry while
/// the test inspects them. Terminal entries are evicted immediately on
/// finalize (#158), so live inspection requires the producing task to
/// still be running. Returns the parent id, the body's value, and the
/// release handle; the test must fire it (and drain) to avoid leaks.
async fn under_live_parent_task<F, Fut, T>(
    registry: &BackgroundTaskRegistry,
    user: UserId,
    parent_conv: &str,
    body: F,
) -> (api::TaskId, T, Arc<Notify>)
where
    F: FnOnce(api::TaskId) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let release = Arc::new(Notify::new());
    let release_for_task = Arc::clone(&release);
    let (value_tx, value_rx) = tokio::sync::oneshot::channel::<T>();
    let parent_id = registry.spawn(
        user,
        api::TaskKind::Conversation {
            conversation_id: parent_conv.into(),
        },
        "parent".into(),
        move |ctx| async move {
            let parent_id = ctx.task_id.clone();
            let token = ctx.token.clone();
            // Install the per-turn cancellation token the same way
            // `send_prompt_with_override` would.
            let value =
                desktop_assistant_core::ports::llm::with_cancellation_token(token, body(parent_id))
                    .await;
            // Publish the value, then park so the parent (and any live
            // child) remain registered for the test to inspect.
            let _ = value_tx.send(value);
            release_for_task.notified().await;
            Ok(())
        },
    );
    let value = timeout(Duration::from_secs(5), value_rx)
        .await
        .expect("parent body must publish its value, not hang")
        .expect("parent body produced a value");
    (parent_id, value, release)
}

/// Drain `events` until a terminal `TaskCompleted` for `task_id` arrives,
/// returning its `(status, last_error)`. Used by tests that inspect
/// terminal status now that finalize evicts the entry from `list`/`get`
/// (#158) — the broadcast event is the surviving record.
async fn wait_for_completion(
    events: &mut tokio::sync::broadcast::Receiver<api::Event>,
    task_id: &api::TaskId,
) -> (api::TaskStatus, Option<String>) {
    let want = task_id.0.clone();
    loop {
        match timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Ok(api::Event::TaskCompleted {
                id,
                status,
                last_error,
            })) if id == want => return (status, last_error),
            Ok(Ok(_)) => continue,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(e)) => panic!("event channel closed before TaskCompleted: {e:?}"),
            Err(_) => panic!("timed out waiting for TaskCompleted({want})"),
        }
    }
}

// --------------------------------------------------------------------
// 1. spawn_subagent with wait=true returns the child's final text
// --------------------------------------------------------------------

#[tokio::test]
async fn spawn_subagent_with_wait_true_returns_child_final_message() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let conversations = Arc::new(FakeConversations::new("hello"));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let (_parent_id, result) =
        under_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
            let tools = tools_for_body;
            let user = user_for_body;
            async move {
                with_user_id(user, async move {
                    tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "researcher",
                                "prompt": "say hello",
                                "wait": true,
                            }),
                        )
                        .await
                })
                .await
            }
        })
        .await;

    let response = result.expect("tool returned Ok");
    // The tool returns the child's final assistant text directly (per
    // the issue's "tool result is 'hello'" acceptance phrasing).
    assert_eq!(response, "hello");
}

// --------------------------------------------------------------------
// 2. spawn_subagent with wait=false returns id immediately
// --------------------------------------------------------------------

#[tokio::test]
async fn spawn_subagent_with_wait_false_returns_task_id_immediately() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let release = Arc::new(Notify::new());
    let release_for_conv = Arc::clone(&release);
    let conversations = Arc::new(FakeConversations::new("slow-default").with_behaviour(
        "conv-0",
        move |_cid, _p| {
            let r = Arc::clone(&release_for_conv);
            async move {
                r.notified().await;
                Ok("eventual".to_string())
            }
        },
    ));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    // Subscribe before anything spawns so we never miss the child's
    // terminal event — finalize evicts the entry (#158), so the broadcast
    // is the authoritative post-completion status.
    let mut events = registry.subscribe(&user);

    // Keep the parent live (it returns wait=false immediately, but we need
    // the child still in-flight so we can observe its Running status).
    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let registry_for_body = Arc::clone(&registry);
    let (_parent_id, child_task_id, parent_release) =
        under_live_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
            let tools = tools_for_body;
            let user = user_for_body;
            let registry = registry_for_body;
            async move {
                let r = timeout(
                    Duration::from_millis(500),
                    with_user_id(user.clone(), async move {
                        tools
                            .execute_tool(
                                TOOL_SPAWN_SUBAGENT,
                                serde_json::json!({
                                    "name": "researcher",
                                    "prompt": "do something slow",
                                    "wait": false,
                                }),
                            )
                            .await
                    }),
                )
                .await
                .expect("wait=false must return within timeout")
                .expect("tool succeeded");

                // The result is a JSON object with child_task_id.
                let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
                let child_task_id = parsed["child_task_id"].as_str().unwrap().to_string();
                assert!(parsed["child_conversation_id"].as_str().is_some());

                // The child is still Running until we release it — it's
                // visible because it hasn't reached a terminal state.
                let view = registry
                    .get(&user, &api::TaskId(child_task_id.clone()))
                    .expect("child registered");
                assert_eq!(view.status, api::TaskStatus::Running);

                child_task_id
            }
        })
        .await;

    // Now release the child and observe it complete via the broadcast
    // stream (the entry is evicted on finalize, so we can't `get` it).
    release.notify_one();
    let child_id = api::TaskId(child_task_id);
    let (status, _) = wait_for_completion(&mut events, &child_id).await;
    assert_eq!(status, api::TaskStatus::Completed);
    assert!(
        registry.get(&user, &child_id).is_none(),
        "completed child must be evicted"
    );

    // Drain the parent so the test doesn't leak it.
    parent_release.notify_one();
}

// --------------------------------------------------------------------
// 3. subagent appears in registry with parent link
// --------------------------------------------------------------------

#[tokio::test]
async fn subagent_appears_in_registry_with_parent_link() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    // The child conversation (conv-0) blocks so the child stays Running
    // and remains visible in the registry while we inspect it (terminal
    // tasks are evicted on finalize, #158).
    let release = Arc::new(Notify::new());
    let release_for_conv = Arc::clone(&release);
    let conversations = Arc::new(FakeConversations::new("ok").with_behaviour(
        "conv-0",
        move |_cid, _p| {
            let r = Arc::clone(&release_for_conv);
            async move {
                r.notified().await;
                Ok("done".to_string())
            }
        },
    ));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    // Spawn the child wait=false and keep the parent live so both stay in
    // the registry for inspection.
    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let (parent_id, _child_id, parent_release) =
        under_live_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
            let tools = tools_for_body;
            let user = user_for_body;
            async move {
                let r = with_user_id(user, async move {
                    tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "researcher",
                                "prompt": "go",
                                "wait": false,
                            }),
                        )
                        .await
                        .expect("spawn ok")
                })
                .await;
                let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
                parsed["child_task_id"].as_str().unwrap().to_string()
            }
        })
        .await;

    // List, find the subagent with kind=Subagent and parent_task_id=parent_id.
    let tasks = registry.list(&user, true, None);
    let subagent = tasks
        .iter()
        .find(|t| matches!(t.kind, api::TaskKind::Subagent { .. }))
        .expect("subagent registered");
    let api::TaskKind::Subagent {
        parent_task_id,
        name,
        ..
    } = &subagent.kind
    else {
        unreachable!()
    };
    assert_eq!(parent_task_id, &parent_id);
    assert_eq!(name, "researcher");
    assert_eq!(subagent.parent.as_ref(), Some(&parent_id));

    // Release the child and parent so the test doesn't leak.
    release.notify_one();
    parent_release.notify_one();
}

// --------------------------------------------------------------------
// 4. parent log records the spawn_subagent tool call with child ids
// --------------------------------------------------------------------

#[tokio::test]
async fn parent_log_records_subagent_tool_call_with_child_ids() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let conversations = Arc::new(FakeConversations::new("ok"));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    // Keep the parent live so its log buffer survives for inspection —
    // finalize evicts the entry (and its logs) immediately (#158). The
    // spawn_subagent tool appends the ToolCall log to the parent
    // regardless of wait, so wait=false (child completes on its own) is
    // sufficient and keeps the test simple.
    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let (parent_id, _child_id, parent_release) =
        under_live_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
            let tools = tools_for_body;
            let user = user_for_body;
            async move {
                let r = with_user_id(user, async move {
                    tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "researcher",
                                "prompt": "go",
                                "wait": false,
                            }),
                        )
                        .await
                        .expect("spawn ok")
                })
                .await;
                let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
                parsed["child_task_id"].as_str().unwrap().to_string()
            }
        })
        .await;

    // Parent's log must contain a ToolCall entry with child_task_id +
    // child_conversation_id in `data`.
    let (entries, _) = registry.logs(&user, &parent_id, 0, 1000).unwrap();
    let tool_call_entry = entries
        .iter()
        .find(|e| e.category == api::LogCategory::ToolCall)
        .expect("parent log has tool-call entry");
    let data = tool_call_entry
        .data
        .as_ref()
        .expect("tool-call entry has data payload");
    assert!(data["child_task_id"].is_string(), "data has child_task_id");
    assert!(
        data["child_conversation_id"].is_string(),
        "data has child_conversation_id"
    );

    // Drain the parent so the test doesn't leak.
    parent_release.notify_one();
}

// --------------------------------------------------------------------
// 5. cancelling parent cancels subagents recursively
// --------------------------------------------------------------------

#[tokio::test]
async fn cancelling_parent_cancels_subagents_recursively() {
    let registry = Arc::new(BackgroundTaskRegistry::new());

    // Grandchild blocks forever (until cancellation propagates).
    let grandchild_started = Arc::new(Notify::new());
    let grandchild_started_for_conv = Arc::clone(&grandchild_started);

    // The child conversation, when prompted, calls `spawn_subagent`
    // again (spawning a grandchild) and waits for it.
    let registry_for_child = Arc::clone(&registry);
    let conv2_arc: Arc<Mutex<Option<Arc<FakeConversations>>>> = Arc::new(Mutex::new(None));
    let conv2_arc_for_child = Arc::clone(&conv2_arc);
    let grandchild_started_for_child = Arc::clone(&grandchild_started);
    let conv = Arc::new(
        FakeConversations::new("never-default")
            .with_behaviour("conv-0", move |_cid, _p| {
                // This is the child conversation. Spawn a grandchild.
                let registry = Arc::clone(&registry_for_child);
                let conv2_arc = Arc::clone(&conv2_arc_for_child);
                let _started = Arc::clone(&grandchild_started_for_child);
                async move {
                    let conv2 = conv2_arc
                        .lock()
                        .unwrap()
                        .clone()
                        .expect("convs handle stashed");
                    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conv2));
                    // Call spawn_subagent for the grandchild.
                    tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "grandchild",
                                "prompt": "block forever",
                                "wait": true,
                            }),
                        )
                        .await
                        .map_err(|e| CoreError::ToolExecution(e.to_string()))
                }
            })
            .with_behaviour("conv-1", move |_cid, _p| {
                // This is the grandchild conversation. Block until
                // cancellation.
                let started = Arc::clone(&grandchild_started_for_conv);
                async move {
                    started.notify_one();
                    let token = desktop_assistant_core::ports::llm::current_cancellation_token()
                        .unwrap_or_default();
                    token.cancelled().await;
                    Err(CoreError::Cancelled)
                }
            }),
    );
    *conv2_arc.lock().unwrap() = Some(Arc::clone(&conv));

    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conv));
    let user = unique_user("alice");

    // Subscribe before spawning so we observe every TaskCompleted —
    // terminal entries are evicted on finalize (#158), so `list` can no
    // longer enumerate the cancelled generations; the broadcast stream is
    // the surviving record of all three reaching Cancelled.
    let mut events = registry.subscribe(&user);

    // Spawn the parent and have it spawn a (waiting) child.
    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let parent_id = registry.spawn(
        user.clone(),
        api::TaskKind::Conversation {
            conversation_id: "parent-conv".into(),
        },
        "parent".into(),
        move |ctx| {
            let tools = tools_for_body;
            let user = user_for_body;
            let token = ctx.token.clone();
            async move {
                desktop_assistant_core::ports::llm::with_cancellation_token(
                    token,
                    with_user_id(user, async move {
                        let _ = tools
                            .execute_tool(
                                TOOL_SPAWN_SUBAGENT,
                                serde_json::json!({
                                    "name": "child",
                                    "prompt": "spawn grandchild",
                                    "wait": true,
                                }),
                            )
                            .await;
                    }),
                )
                .await;
                Ok(())
            }
        },
    );

    // Wait until the grandchild has registered itself and started.
    timeout(Duration::from_secs(5), grandchild_started.notified())
        .await
        .expect("grandchild started");

    // Cancel the parent.
    registry
        .cancel(&user, &parent_id)
        .expect("parent cancellable");

    // Wait for the parent to wind down (its `wait` resolving implies the
    // child it blocked on, and the grandchild that child blocked on, have
    // all reached a terminal state).
    timeout(Duration::from_secs(5), registry.wait(&parent_id))
        .await
        .expect("parent terminates");

    // All three generations (parent + child + grandchild) must reach
    // Cancelled. We collect the three TaskCompleted events from the
    // broadcast stream rather than `list()`, because finalize evicts each
    // terminal entry immediately (#158).
    let mut cancelled_count = 0;
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while cancelled_count < 3 {
        let remaining = deadline
            .checked_duration_since(std::time::Instant::now())
            .expect("timed out collecting 3 TaskCompleted events");
        match timeout(remaining, events.recv()).await {
            Ok(Ok(api::Event::TaskCompleted { id, status, .. })) => {
                assert_eq!(
                    status,
                    api::TaskStatus::Cancelled,
                    "task {id} completed with {status:?}, expected Cancelled"
                );
                cancelled_count += 1;
            }
            Ok(Ok(_)) => continue,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(e)) => panic!("event channel closed before 3 TaskCompleted: {e:?}"),
            Err(_) => panic!("timed out collecting 3 TaskCompleted events"),
        }
    }
    assert_eq!(
        cancelled_count, 3,
        "parent + child + grandchild must each reach Cancelled"
    );
}

// --------------------------------------------------------------------
// 6. subagent uses its own connection/model override
// --------------------------------------------------------------------

#[tokio::test]
async fn subagent_uses_own_connection_model_override() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let conversations = Arc::new(FakeConversations::new("ok"));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let conv_for_body = Arc::clone(&conversations);
    under_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
        let tools = tools_for_body;
        let user = user_for_body;
        let conv = conv_for_body;
        async move {
            with_user_id(user, async move {
                let _ = tools
                    .execute_tool(
                        TOOL_SPAWN_SUBAGENT,
                        serde_json::json!({
                            "name": "researcher",
                            "prompt": "go",
                            "connection": "ollama",
                            "model": "llama3",
                            "wait": true,
                        }),
                    )
                    .await;
            })
            .await;

            // Inspect recorded turns: the child turn must carry the
            // override the tool was asked to apply.
            let turns = conv.turns();
            let child_turn = turns
                .iter()
                .find(|t| t.conversation_id != "parent-conv")
                .expect("child turn recorded");
            let ov = child_turn
                .override_selection
                .as_ref()
                .expect("override forwarded");
            assert_eq!(ov.connection_id, "ollama");
            assert_eq!(ov.model_id, "llama3");
        }
    })
    .await;
}

// --------------------------------------------------------------------
// 7. get_subagent_status for unknown id returns structured not_found
// --------------------------------------------------------------------

#[tokio::test]
async fn get_subagent_status_for_unknown_id_returns_not_found() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let conversations = Arc::new(FakeConversations::new("ok"));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    let result = with_user_id(user, async {
        tools
            .execute_tool(
                TOOL_GET_SUBAGENT_STATUS,
                serde_json::json!({"task_id": "does-not-exist"}),
            )
            .await
            .unwrap()
    })
    .await;
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["error"], "not_found");
}

// --------------------------------------------------------------------
// 8. get_subagent_status for other user's task is not_found (no leak)
// --------------------------------------------------------------------

#[tokio::test]
async fn get_subagent_status_for_other_users_task_returns_not_found() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    // Alice's subagent blocks (conv-0) so it stays RUNNING while Bob
    // probes — this proves the `not_found` Bob sees is genuine cross-user
    // opacity (#105), not just the post-completion eviction (#158).
    let release = Arc::new(Notify::new());
    let release_for_conv = Arc::clone(&release);
    let conversations = Arc::new(FakeConversations::new("ok").with_behaviour(
        "conv-0",
        move |_cid, _p| {
            let r = Arc::clone(&release_for_conv);
            async move {
                r.notified().await;
                Ok("done".to_string())
            }
        },
    ));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let alice = unique_user("alice");
    let bob = unique_user("bob");

    // Alice spawns a subagent that stays in-flight; keep her parent live
    // so we can capture the (still-Running) child id.
    let alice_for_body = alice.clone();
    let tools_for_body = tools.clone();
    let registry_for_body = Arc::clone(&registry);
    let alice_for_capture = alice.clone();
    let (_pid, child_id, parent_release) =
        under_live_parent_task(&registry, alice.clone(), "alice-conv", move |_pid| {
            let tools = tools_for_body;
            let user = alice_for_body;
            let registry = registry_for_body;
            let alice = alice_for_capture;
            async move {
                with_user_id(user.clone(), async move {
                    let _ = tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "child",
                                "prompt": "go",
                                "wait": false,
                            }),
                        )
                        .await
                        .unwrap();
                })
                .await;
                // Find the still-Running subagent id.
                let tasks = registry.list(&alice, true, None);
                tasks
                    .iter()
                    .find(|t| matches!(t.kind, api::TaskKind::Subagent { .. }))
                    .map(|t| t.id.clone())
                    .expect("subagent registered")
            }
        })
        .await;

    // Sanity: the child genuinely exists for Alice right now.
    assert!(
        registry.get(&alice, &child_id).is_some(),
        "child must be live so the probe tests opacity, not eviction"
    );

    // Bob asks about Alice's (live) child task: must come back as
    // not_found — existence must not leak.
    let child_id_for_bob = child_id.0.clone();
    let result = with_user_id(bob, async {
        tools
            .execute_tool(
                TOOL_GET_SUBAGENT_STATUS,
                serde_json::json!({"task_id": child_id_for_bob}),
            )
            .await
            .unwrap()
    })
    .await;
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["error"], "not_found");

    // Clean up: release the child and parent so the test doesn't leak.
    release.notify_one();
    timeout(Duration::from_secs(5), registry.wait(&child_id))
        .await
        .expect("child completes");
    parent_release.notify_one();
}

// --------------------------------------------------------------------
// 9. spawn_subagent inherits parent's user_id
// --------------------------------------------------------------------

#[tokio::test]
async fn spawn_subagent_inherits_parent_user_id() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    // The child blocks (conv-0) so it stays Running and visible while we
    // assert ownership — terminal tasks are evicted on finalize (#158).
    let release = Arc::new(Notify::new());
    let release_for_conv = Arc::clone(&release);
    let conversations = Arc::new(FakeConversations::new("ok").with_behaviour(
        "conv-0",
        move |_cid, _p| {
            let r = Arc::clone(&release_for_conv);
            async move {
                r.notified().await;
                Ok("done".to_string())
            }
        },
    ));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let alice = unique_user("alice");

    let alice_for_body = alice.clone();
    let tools_for_body = tools.clone();
    let (_pid, _child_id, parent_release) =
        under_live_parent_task(&registry, alice.clone(), "parent-conv", move |_pid| {
            let tools = tools_for_body;
            let user = alice_for_body;
            async move {
                let r = with_user_id(user, async move {
                    tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "child",
                                "prompt": "go",
                                "wait": false,
                            }),
                        )
                        .await
                        .unwrap()
                })
                .await;
                let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
                parsed["child_task_id"].as_str().unwrap().to_string()
            }
        })
        .await;

    // The child task must be owned by Alice, not by the default sentinel.
    let alice_tasks = registry.list(&alice, true, None);
    let subagent = alice_tasks
        .iter()
        .find(|t| matches!(t.kind, api::TaskKind::Subagent { .. }))
        .expect("subagent visible to alice");
    // A different user must not see it.
    let bob = unique_user("bob");
    let bob_view = registry.get(&bob, &subagent.id);
    assert!(bob_view.is_none(), "bob must not see alice's child");

    // Release the child and parent so the test doesn't leak.
    release.notify_one();
    parent_release.notify_one();
}

// --------------------------------------------------------------------
// 10. spawn_subagent records TaskKind::Subagent with correct link
// --------------------------------------------------------------------

#[tokio::test]
async fn spawn_subagent_records_task_kind_subagent_with_correct_link() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    // The child blocks (conv-0) so it stays Running and visible while we
    // inspect its kind — terminal tasks are evicted on finalize (#158).
    let release = Arc::new(Notify::new());
    let release_for_conv = Arc::clone(&release);
    let conversations = Arc::new(FakeConversations::new("ok").with_behaviour(
        "conv-0",
        move |_cid, _p| {
            let r = Arc::clone(&release_for_conv);
            async move {
                r.notified().await;
                Ok("done".to_string())
            }
        },
    ));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let (parent_id, _child_id, parent_release) =
        under_live_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
            let tools = tools_for_body;
            let user = user_for_body;
            async move {
                let r = with_user_id(user, async move {
                    tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "fred",
                                "prompt": "go",
                                "wait": false,
                            }),
                        )
                        .await
                        .unwrap()
                })
                .await;
                let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
                parsed["child_task_id"].as_str().unwrap().to_string()
            }
        })
        .await;

    let tasks = registry.list(&user, true, None);
    let child = tasks
        .iter()
        .find(|t| matches!(t.kind, api::TaskKind::Subagent { .. }))
        .expect("subagent recorded");
    let api::TaskKind::Subagent {
        parent_task_id,
        conversation_id,
        name,
    } = &child.kind
    else {
        unreachable!();
    };
    assert_eq!(parent_task_id, &parent_id);
    assert_eq!(name, "fred");
    // The conversation_id must match a freshly-created conversation.
    assert!(!conversation_id.is_empty());
    assert_ne!(conversation_id, "parent-conv");

    // Release the child and parent so the test doesn't leak.
    release.notify_one();
    parent_release.notify_one();
}

// --------------------------------------------------------------------
// 11. business outcome: returned text is the actual child assistant text
// --------------------------------------------------------------------

#[tokio::test]
async fn business_outcome_subagent_result_contains_actual_assistant_text() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let conversations = Arc::new(FakeConversations::new("the price of tea in china is $42"));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let (_pid, result) = under_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
        let tools = tools_for_body;
        let user = user_for_body;
        async move {
            with_user_id(user, async move {
                tools
                    .execute_tool(
                        TOOL_SPAWN_SUBAGENT,
                        serde_json::json!({
                            "name": "tea-researcher",
                            "prompt": "look up tea prices",
                            "wait": true,
                        }),
                    )
                    .await
            })
            .await
        }
    })
    .await;

    let text = result.expect("ok");
    // Business outcome: not a placeholder, but the actual child text.
    assert!(
        text.contains("$42"),
        "expected the child's actual assistant text, got: {text}"
    );
}

// --------------------------------------------------------------------
// 12. tool definitions advertise the expected JSON schema fields
// --------------------------------------------------------------------

#[tokio::test]
async fn tool_definitions_publish_expected_schema() {
    let defs = tool_definitions();
    let names: Vec<String> = defs.iter().map(|t| t.name.clone()).collect();
    assert!(names.contains(&TOOL_SPAWN_SUBAGENT.to_string()));
    assert!(names.contains(&TOOL_GET_SUBAGENT_STATUS.to_string()));

    let spawn = defs.iter().find(|t| t.name == TOOL_SPAWN_SUBAGENT).unwrap();
    let required = spawn.parameters["required"].as_array().unwrap();
    let required_names: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
    assert!(required_names.contains(&"name"));
    assert!(required_names.contains(&"prompt"));
    // Optional fields must be advertised as properties.
    let properties = spawn.parameters["properties"].as_object().unwrap();
    for opt in ["system_prompt", "connection", "model", "tools", "wait"] {
        assert!(
            properties.contains_key(opt),
            "missing optional property: {opt}"
        );
    }
}

// --------------------------------------------------------------------
// 13. current_task_id is None outside a registry body and Some inside
// --------------------------------------------------------------------

#[tokio::test]
async fn current_task_id_outside_registry_is_none_inside_is_some() {
    // The new task-local must default to None for code that doesn't
    // run inside `BackgroundTaskRegistry::spawn` — same shape as
    // `current_user_id`'s sentinel behaviour but here we surface the
    // absence explicitly so subagent tooling can fail cleanly when
    // misused.
    assert!(current_task_id().is_none());

    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("alice");
    let seen = Arc::new(AtomicBool::new(false));
    let seen_for_body = Arc::clone(&seen);
    let id = registry.spawn(
        user.clone(),
        api::TaskKind::Conversation {
            conversation_id: "c".into(),
        },
        "t".into(),
        move |_ctx| async move {
            if current_task_id().is_some() {
                seen_for_body.store(true, Ordering::SeqCst);
            }
            Ok(())
        },
    );
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");
    assert!(seen.load(Ordering::SeqCst));
}

// --------------------------------------------------------------------
// 14. recursion depth limit (issue #291)
// --------------------------------------------------------------------

/// Register a chain of nested `Subagent` tasks `depth` levels deep
/// (level 0's parent is a `Conversation` task) and run `body` inside the
/// deepest one, where `current_task_id()` resolves to that level's id and
/// the registry holds the full parent chain. Returns the body's value.
///
/// Used by the depth-limit tests: they drive `tools.spawn(..)` from deep
/// inside a real Subagent chain so the dispatch-time depth check has a
/// genuine ancestry to walk.
async fn under_subagent_chain<F, Fut, T>(
    registry: &Arc<BackgroundTaskRegistry>,
    user: UserId,
    depth: usize,
    body: F,
) -> T
where
    F: FnOnce(api::TaskId) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    // First register the root Conversation parent so the chain hangs off
    // a real ancestor.
    let result_slot: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
    let result_for_body = Arc::clone(&result_slot);
    let registry_outer = Arc::clone(registry);
    let user_outer = user.clone();

    let root_id = registry.spawn(
        user.clone(),
        api::TaskKind::Conversation {
            conversation_id: "root-conv".into(),
        },
        "root".into(),
        move |ctx| {
            let registry = registry_outer;
            let user = user_outer;
            let result_for_body = result_for_body;
            async move {
                with_user_id(user.clone(), async move {
                    let token = ctx.token.clone();
                    desktop_assistant_core::ports::llm::with_cancellation_token(
                        token,
                        async move {
                            let parent = ctx.task_id.clone();
                            let value = spawn_nested(&registry, user, parent, depth, body).await;
                            *result_for_body.lock().unwrap() = Some(value);
                        },
                    )
                    .await;
                    Ok(())
                })
                .await
            }
        },
    );
    timeout(Duration::from_secs(10), registry.wait(&root_id))
        .await
        .expect("subagent chain must finish, not hang");
    result_slot
        .lock()
        .unwrap()
        .take()
        .expect("chain body produced a value")
}

/// Spawn one more `Subagent` level under `parent`; recurse until
/// `remaining == 0`, then run `body` inside the deepest body.
fn spawn_nested<F, Fut, T>(
    registry: &Arc<BackgroundTaskRegistry>,
    user: UserId,
    parent: api::TaskId,
    remaining: usize,
    body: F,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>
where
    F: FnOnce(api::TaskId) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let registry = Arc::clone(registry);
    Box::pin(async move {
        let result_slot: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
        let result_for_body = Arc::clone(&result_slot);
        let registry_inner = Arc::clone(&registry);
        let user_inner = user.clone();
        let parent_for_kind = parent.clone();
        let child_id = registry.spawn(
            user.clone(),
            api::TaskKind::Subagent {
                parent_task_id: parent_for_kind,
                conversation_id: format!("sub-conv-{remaining}"),
                name: format!("level-{remaining}"),
            },
            format!("level-{remaining}"),
            move |ctx| {
                let registry = registry_inner;
                let user = user_inner;
                let result_for_body = result_for_body;
                async move {
                    with_user_id(user.clone(), async move {
                        let token = ctx.token.clone();
                        desktop_assistant_core::ports::llm::with_cancellation_token(
                            token,
                            async move {
                                let me = ctx.task_id.clone();
                                let value = if remaining <= 1 {
                                    body(me).await
                                } else {
                                    spawn_nested(&registry, user, me, remaining - 1, body).await
                                };
                                *result_for_body.lock().unwrap() = Some(value);
                            },
                        )
                        .await;
                        Ok(())
                    })
                    .await
                }
            },
        );
        timeout(Duration::from_secs(10), registry.wait(&child_id))
            .await
            .expect("nested subagent level must finish, not hang");
        result_slot
            .lock()
            .unwrap()
            .take()
            .expect("nested body produced a value")
    })
}

#[tokio::test]
async fn spawn_subagent_rejected_past_recursion_depth_limit() {
    use desktop_assistant_application::subagent_tools::MAX_SUBAGENT_DEPTH;

    let registry = Arc::new(BackgroundTaskRegistry::new());
    let conversations = Arc::new(FakeConversations::new("child text"));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("deep");

    // Run from inside a Subagent chain already at MAX_SUBAGENT_DEPTH
    // Subagent ancestors. Spawning one more would exceed the cap, so the
    // tool must refuse with a recoverable error and create no child task.
    let tools_for_body = tools.clone();
    let result: Result<String, CoreError> = under_subagent_chain(
        &registry,
        user.clone(),
        MAX_SUBAGENT_DEPTH,
        move |_deepest| async move {
            tools_for_body
                .execute_tool(
                    TOOL_SPAWN_SUBAGENT,
                    serde_json::json!({
                        "name": "too-deep",
                        "prompt": "go deeper",
                        "wait": true,
                    }),
                )
                .await
        },
    )
    .await;

    let err = result.expect_err("spawn past the depth cap must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("depth"),
        "rejection should mention the depth limit, got: {msg}"
    );
}

#[tokio::test]
async fn spawn_subagent_allowed_within_recursion_depth_limit() {
    use desktop_assistant_application::subagent_tools::MAX_SUBAGENT_DEPTH;

    let registry = Arc::new(BackgroundTaskRegistry::new());
    let conversations = Arc::new(FakeConversations::new("child text"));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("shallow");

    // One level below the cap: a spawn here is still permitted, proving
    // the limit doesn't fire prematurely.
    let tools_for_body = tools.clone();
    let result: Result<String, CoreError> = under_subagent_chain(
        &registry,
        user.clone(),
        MAX_SUBAGENT_DEPTH - 1,
        move |_deepest| async move {
            tools_for_body
                .execute_tool(
                    TOOL_SPAWN_SUBAGENT,
                    serde_json::json!({
                        "name": "still-ok",
                        "prompt": "one more",
                        "wait": true,
                    }),
                )
                .await
        },
    )
    .await;

    assert_eq!(
        result.expect("a spawn within the depth limit must succeed"),
        "child text"
    );
}

// --------------------------------------------------------------------
// 15. #440: a wait=true child that fails surfaces the error to the parent
// --------------------------------------------------------------------

#[tokio::test]
async fn subagent_wait_true_child_error_surfaces_to_parent() {
    // A wait=true subagent whose turn returns `Err` must make the parent's
    // `execute_tool` return `ToolExecution("subagent failed: ...")` — NOT an
    // empty-text success. Otherwise a failed delegation looks like a silent
    // no-op to the parent LLM (subagent_tools.rs:366-374).
    let registry = Arc::new(BackgroundTaskRegistry::new());
    // The child conversation (conv-0) fails its turn.
    let conversations = Arc::new(
        FakeConversations::new("unused-default")
            .with_behaviour("conv-0", move |_cid, _p| async move {
                Err(CoreError::ToolExecution("upstream boom".to_string()))
            }),
    );
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    let (_parent_id, result) =
        under_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
            let tools = tools_for_body;
            let user = user_for_body;
            async move {
                with_user_id(user, async move {
                    tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "doomed",
                                "prompt": "do the thing",
                                "wait": true,
                            }),
                        )
                        .await
                })
                .await
            }
        })
        .await;

    let err = result.expect_err("a failed wait=true child must surface as Err");
    let msg = format!("{err}");
    assert!(
        msg.contains("subagent failed"),
        "error must name the subagent failure, got: {msg}"
    );
    assert!(
        msg.contains("upstream boom"),
        "error must carry the child's failure reason, got: {msg}"
    );
    assert!(
        matches!(err, CoreError::ToolExecution(_)),
        "the surfaced error must be a recoverable ToolExecution, got: {err:?}"
    );
}

// --------------------------------------------------------------------
// 16. #440: one failed sibling does not cancel the others
// --------------------------------------------------------------------

#[tokio::test]
async fn one_failed_subagent_does_not_cancel_siblings() {
    // A parent spawns three wait=false children; the middle one fails. Sibling
    // isolation: the failure of one must NOT cancel or fail the others — they
    // each reach Completed on their own.
    let registry = Arc::new(BackgroundTaskRegistry::new());
    // Children are created in spawn order: conv-0, conv-1, conv-2. The middle
    // one fails; the outer two succeed.
    let conversations = Arc::new(
        FakeConversations::new("unused-default")
            .with_behaviour(
                "conv-0",
                move |_c, _p| async move { Ok("done-0".to_string()) },
            )
            .with_behaviour("conv-1", move |_c, _p| async move {
                Err(CoreError::ToolExecution("sibling boom".to_string()))
            })
            .with_behaviour(
                "conv-2",
                move |_c, _p| async move { Ok("done-2".to_string()) },
            ),
    );
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    // Subscribe before spawning so we capture every child's terminal event —
    // finalize evicts the entry (#158), so the broadcast is the record.
    let mut events = registry.subscribe(&user);

    let user_for_body = user.clone();
    let tools_for_body = tools.clone();
    // The body spawns three wait=false children and returns their
    // (task_id, conversation_id) pairs.
    let (_parent_id, children): (api::TaskId, Vec<(String, String)>) =
        under_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
            let tools = tools_for_body;
            let user = user_for_body;
            async move {
                with_user_id(user, async move {
                    let mut out = Vec::new();
                    for name in ["first", "second", "third"] {
                        let r = tools
                            .execute_tool(
                                TOOL_SPAWN_SUBAGENT,
                                serde_json::json!({
                                    "name": name,
                                    "prompt": "go",
                                    "wait": false,
                                }),
                            )
                            .await
                            .expect("spawn ok");
                        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
                        out.push((
                            parsed["child_task_id"].as_str().unwrap().to_string(),
                            parsed["child_conversation_id"]
                                .as_str()
                                .unwrap()
                                .to_string(),
                        ));
                    }
                    out
                })
                .await
            }
        })
        .await;

    assert_eq!(children.len(), 3, "three children spawned");

    // Collect each child's terminal status from the broadcast stream.
    let want: std::collections::HashSet<String> =
        children.iter().map(|(tid, _)| tid.clone()).collect();
    let mut statuses: HashMap<String, api::TaskStatus> = HashMap::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while statuses.len() < want.len() {
        let remaining = deadline
            .checked_duration_since(std::time::Instant::now())
            .expect("timed out collecting child terminal events");
        match timeout(remaining, events.recv()).await {
            Ok(Ok(api::Event::TaskCompleted { id, status, .. })) if want.contains(&id) => {
                statuses.insert(id, status);
            }
            Ok(Ok(_)) => continue,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(e)) => panic!("event channel closed early: {e:?}"),
            Err(_) => panic!("timed out collecting child terminal events, got {statuses:?}"),
        }
    }

    // Map task_id -> conversation_id so we can name the expected outcome.
    let by_conv: HashMap<String, String> = children
        .iter()
        .map(|(tid, cid)| (cid.clone(), tid.clone()))
        .collect();
    let failing = &by_conv["conv-1"];
    let ok_a = &by_conv["conv-0"];
    let ok_c = &by_conv["conv-2"];

    assert_eq!(
        statuses.get(failing),
        Some(&api::TaskStatus::Failed),
        "the middle child failed"
    );
    assert_eq!(
        statuses.get(ok_a),
        Some(&api::TaskStatus::Completed),
        "the first sibling completed despite the middle one failing"
    );
    assert_eq!(
        statuses.get(ok_c),
        Some(&api::TaskStatus::Completed),
        "the last sibling completed despite the middle one failing"
    );
}

// --------------------------------------------------------------------
// #287 slice 5: a subagent adopts the child scope the dispatch loop
// installs (session pad + owner_todo + snapshot marker), carried across
// the registry's tokio::spawn as data and re-installed in the child body.
// --------------------------------------------------------------------

#[tokio::test]
async fn subagent_body_runs_under_installed_child_scope() {
    use desktop_assistant_core::ports::scratchpad_scope::{
        SubagentScope, current_owner_todo, current_scratchpad_scope, with_pending_child_scope,
    };

    let registry = Arc::new(BackgroundTaskRegistry::new());
    // The child body records the scope it observes; the first conversation the
    // spawn creates is "conv-0".
    let observed: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
    let obs = Arc::clone(&observed);
    let conversations = Arc::new(FakeConversations::new("hi").with_behaviour(
        "conv-0",
        move |_cid, _prompt| {
            let obs = Arc::clone(&obs);
            async move {
                let owner = current_owner_todo().unwrap_or_default();
                let scope_conv = current_scratchpad_scope()
                    .map(|c| c.as_str().to_string())
                    .unwrap_or_default();
                *obs.lock().unwrap() = Some((owner, scope_conv));
                Ok("hi".to_string())
            }
        },
    ));
    let tools = SubagentTools::new(Arc::clone(&registry), Arc::clone(&conversations));
    let user = unique_user("alice");

    let scope = SubagentScope {
        session_conversation_id: ConversationId::from("session-1"),
        owner_todo: "1.1".to_string(),
        visible_before: "marker".to_string(),
        ancestors: vec![String::new()],
    };

    let tools_for_body = tools.clone();
    let user_for_body = user.clone();
    let scope_for_body = scope.clone();
    let (_pid, result) = under_parent_task(&registry, user.clone(), "parent-conv", move |_pid| {
        let tools = tools_for_body;
        let user = user_for_body;
        let scope = scope_for_body;
        async move {
            // Mirror the dispatch loop: install the pending child scope
            // around the spawn tool's execution.
            with_user_id(
                user,
                with_pending_child_scope(scope, async move {
                    tools
                        .execute_tool(
                            TOOL_SPAWN_SUBAGENT,
                            serde_json::json!({
                                "name": "researcher",
                                "prompt": "go",
                                "wait": true,
                            }),
                        )
                        .await
                }),
            )
            .await
        }
    })
    .await;

    result.expect("spawn returned Ok");
    let (owner, scope_conv) = observed
        .lock()
        .unwrap()
        .clone()
        .expect("child body ran and recorded its scope");
    assert_eq!(
        owner, "1.1",
        "child body runs under the installed owner_todo"
    );
    assert_eq!(
        scope_conv, "session-1",
        "child scratchpad ops target the session pad, not the child conversation"
    );
}
