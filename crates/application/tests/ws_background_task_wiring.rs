//! Acceptance tests for the background-task command arms wired into
//! [`DefaultAssistantApiHandler`] (#114).
//!
//! These exercise the *handler-level* contract — given a registry and a
//! per-user request context, each `Command::*BackgroundTask*` arm must
//! consult the registry and return the documented `CommandResult` (or a
//! structured error). The transport-level wiring (Subscribe forwarder,
//! SendMessageAck plumbing) lives in `transport-dispatch` and is exercised
//! by tests in that crate; the arms below are the building blocks that
//! the dispatcher composes against.
//!
//! TDD: written before the stubs in `application/src/lib.rs` are replaced
//! with real registry calls. Every test asserts a documented business
//! outcome from the #114 issue body.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_application::{
    AssistantApiHandler, DefaultAssistantApiHandler, RequestContext, UserId,
    background_tasks::BackgroundTaskRegistry,
};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    Conversation, ConversationId, ConversationSummary, KnowledgeEntry, Message, Role,
};
use desktop_assistant_core::ports::auth::{current_user_id, with_user_id};
use desktop_assistant_core::ports::inbound::{
    AssistantService, BackendTasksSettingsView, ConnectionConfigPayload, ConnectionsService,
    ConnectorDefaultsView, ConversationService, DatabaseSettingsView, EmbeddingsSettingsView,
    KnowledgeService, LlmSettingsView, ModelListing as CoreModelListing, PersistenceSettingsView,
    PromptDispatchOutcome, PromptSelectionOverride, PurposeConfigPayload, PurposeKind,
    PurposesView as CorePurposesView, SettingsService, WsAuthSettingsView,
};
use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

// ---------- Minimal fakes ----------

struct FakeAssistant;
impl AssistantService for FakeAssistant {
    fn version(&self) -> &str {
        "0.0.0-test"
    }
    fn ping(&self) -> &str {
        "pong"
    }
}

struct FakeKnowledge;
impl KnowledgeService for FakeKnowledge {
    async fn list_entries(
        &self,
        _l: usize,
        _o: usize,
        _t: Option<Vec<String>>,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        Ok(vec![])
    }
    async fn get_entry(&self, _id: String) -> Result<Option<KnowledgeEntry>, CoreError> {
        Ok(None)
    }
    async fn search_entries(
        &self,
        _q: String,
        _t: Option<Vec<String>>,
        _l: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        Ok(vec![])
    }
    async fn create_entry(
        &self,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        let mut e = KnowledgeEntry::new("kb", content, tags);
        e.metadata = metadata;
        Ok(e)
    }
    async fn update_entry(
        &self,
        id: String,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        let mut e = KnowledgeEntry::new(id, content, tags);
        e.metadata = metadata;
        Ok(e)
    }
    async fn delete_entry(&self, _id: String) -> Result<(), CoreError> {
        Ok(())
    }
}

struct FakeConnections;
impl ConnectionsService for FakeConnections {
    async fn list_connections(
        &self,
    ) -> Result<Vec<desktop_assistant_core::ports::inbound::ConnectionView>, CoreError> {
        Ok(vec![])
    }
    async fn create_connection(
        &self,
        _id: String,
        _c: ConnectionConfigPayload,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn update_connection(
        &self,
        _id: String,
        _c: ConnectionConfigPayload,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn delete_connection(&self, _id: String, _f: bool) -> Result<(), CoreError> {
        Ok(())
    }
    async fn list_available_models(
        &self,
        _c: Option<String>,
        _r: bool,
    ) -> Result<Vec<CoreModelListing>, CoreError> {
        Ok(vec![])
    }
    async fn get_purposes(&self) -> Result<CorePurposesView, CoreError> {
        Ok(CorePurposesView::default())
    }
    async fn set_purpose(
        &self,
        _p: PurposeKind,
        _c: PurposeConfigPayload,
    ) -> Result<(), CoreError> {
        Ok(())
    }
}

struct FakeSettings;
impl SettingsService for FakeSettings {
    async fn get_llm_settings(&self) -> Result<LlmSettingsView, CoreError> {
        Ok(LlmSettingsView {
            connector: "x".into(),
            model: "y".into(),
            base_url: "z".into(),
            has_api_key: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            hosted_tool_search: None,
        })
    }
    async fn set_llm_settings(
        &self,
        _c: String,
        _m: Option<String>,
        _b: Option<String>,
        _t: Option<f64>,
        _p: Option<f64>,
        _x: Option<u32>,
        _h: Option<bool>,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn set_api_key(&self, _k: String) -> Result<(), CoreError> {
        Ok(())
    }
    async fn generate_ws_jwt(&self, _s: Option<String>) -> Result<String, CoreError> {
        Ok("t".into())
    }
    async fn validate_ws_jwt(&self, _t: String) -> Result<bool, CoreError> {
        Ok(true)
    }
    async fn get_embeddings_settings(&self) -> Result<EmbeddingsSettingsView, CoreError> {
        Ok(EmbeddingsSettingsView {
            connector: "x".into(),
            model: "y".into(),
            base_url: "z".into(),
            has_api_key: false,
            available: false,
            is_default: true,
        })
    }
    async fn set_embeddings_settings(
        &self,
        _c: Option<String>,
        _m: Option<String>,
        _b: Option<String>,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn get_connector_defaults(&self, _c: String) -> Result<ConnectorDefaultsView, CoreError> {
        Ok(ConnectorDefaultsView {
            llm_model: "m".into(),
            llm_base_url: "u".into(),
            backend_llm_model: "bm".into(),
            embeddings_model: "em".into(),
            embeddings_base_url: "eu".into(),
            embeddings_available: false,
            hosted_tool_search_available: false,
        })
    }
    async fn get_persistence_settings(&self) -> Result<PersistenceSettingsView, CoreError> {
        Ok(PersistenceSettingsView {
            enabled: false,
            remote_url: "".into(),
            remote_name: "origin".into(),
            push_on_update: false,
        })
    }
    async fn set_persistence_settings(
        &self,
        _e: bool,
        _u: Option<String>,
        _n: Option<String>,
        _p: bool,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
        Ok(DatabaseSettingsView {
            url: String::new(),
            max_connections: 5,
        })
    }
    async fn set_database_settings(&self, _u: Option<String>, _m: u32) -> Result<(), CoreError> {
        Ok(())
    }
    async fn get_backend_tasks_settings(&self) -> Result<BackendTasksSettingsView, CoreError> {
        Ok(BackendTasksSettingsView {
            has_separate_llm: false,
            llm_connector: "x".into(),
            llm_model: "y".into(),
            llm_base_url: "z".into(),
            dreaming_enabled: false,
            dreaming_interval_secs: 0,
            archive_after_days: 0,
        })
    }
    async fn set_backend_tasks_settings(
        &self,
        _c: Option<String>,
        _m: Option<String>,
        _b: Option<String>,
        _e: bool,
        _i: u64,
        _a: u32,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn list_mcp_servers(
        &self,
    ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
        Ok(vec![])
    }
    async fn add_mcp_server(
        &self,
        _n: String,
        _c: String,
        _a: Vec<String>,
        _ns: Option<String>,
        _e: bool,
    ) -> Result<(), CoreError> {
        Ok(())
    }
    async fn remove_mcp_server(&self, _n: String) -> Result<(), CoreError> {
        Ok(())
    }
    async fn set_mcp_server_enabled(&self, _n: String, _e: bool) -> Result<(), CoreError> {
        Ok(())
    }
    async fn mcp_server_action(
        &self,
        _a: String,
        _s: Option<String>,
    ) -> Result<Vec<desktop_assistant_core::ports::inbound::McpServerView>, CoreError> {
        Ok(vec![])
    }
    async fn get_ws_auth_settings(&self) -> Result<WsAuthSettingsView, CoreError> {
        Ok(WsAuthSettingsView {
            methods: vec![],
            oidc_issuer: String::new(),
            oidc_auth_endpoint: String::new(),
            oidc_token_endpoint: String::new(),
            oidc_client_id: String::new(),
            oidc_scopes: String::new(),
        })
    }
    async fn set_ws_auth_settings(
        &self,
        _m: Vec<String>,
        _i: String,
        _a: String,
        _t: String,
        _c: String,
        _s: String,
    ) -> Result<(), CoreError> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct SendCall {
    #[allow(dead_code)]
    conversation_id: String,
    prompt: String,
    /// `current_user_id()` observed inside `send_prompt_with_override`.
    /// Used by #154 to assert that the registry-spawned send body
    /// inherits the caller's user_id scope.
    seen_user_id: String,
}

struct RecordingConversations {
    sends: Mutex<Vec<SendCall>>,
    next_conv_id: AtomicU32,
    cancelled: Arc<AtomicBool>,
    /// When `Some`, send_prompt_with_override blocks on this notify before
    /// returning. Used to assert that ListBackgroundTasks sees the task as
    /// `Running` while the LLM is still working.
    block_until: Mutex<Option<Arc<Notify>>>,
}

impl RecordingConversations {
    fn new() -> Self {
        Self {
            sends: Mutex::new(vec![]),
            next_conv_id: AtomicU32::new(0),
            cancelled: Arc::new(AtomicBool::new(false)),
            block_until: Mutex::new(None),
        }
    }

    fn seen_user_ids(&self) -> Vec<String> {
        self.sends
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.seen_user_id.clone())
            .collect()
    }

    fn block_on(&self, n: Arc<Notify>) {
        *self.block_until.lock().unwrap() = Some(n);
    }
}

impl ConversationService for RecordingConversations {
    async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
        let id = format!("conv-{}", self.next_conv_id.fetch_add(1, Ordering::SeqCst));
        Ok(Conversation::new(id, title))
    }
    async fn list_conversations(
        &self,
        _m: Option<u32>,
        _i: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        Ok(vec![])
    }
    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        let mut c = Conversation::new(id.as_str(), "t");
        if let Some(last) = self.sends.lock().unwrap().last().cloned() {
            c.messages.push(Message::new(Role::User, last.prompt));
            c.messages
                .push(Message::new(Role::Assistant, "ok".to_string()));
        }
        Ok(c)
    }
    async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
        Ok(())
    }
    async fn rename_conversation(&self, _id: &ConversationId, _t: String) -> Result<(), CoreError> {
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
        _conversation_id: &ConversationId,
        _prompt: String,
        _on_chunk: ChunkCallback,
        _on_status: StatusCallback,
    ) -> Result<String, CoreError> {
        Ok(String::new())
    }
    async fn send_prompt_with_override(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        _override_selection: Option<PromptSelectionOverride>,
        mut on_chunk: ChunkCallback,
        _on_status: StatusCallback,
        cancellation: CancellationToken,
    ) -> Result<PromptDispatchOutcome, CoreError> {
        self.sends.lock().unwrap().push(SendCall {
            conversation_id: conversation_id.as_str().to_string(),
            prompt: prompt.clone(),
            seen_user_id: current_user_id().as_str().to_string(),
        });
        let block = self.block_until.lock().unwrap().clone();
        if let Some(block) = block {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    self.cancelled.store(true, Ordering::SeqCst);
                    return Err(CoreError::Cancelled);
                }
                _ = block.notified() => {}
            }
        }
        on_chunk("ok".into());
        Ok(PromptDispatchOutcome {
            response: "ok".into(),
            warnings: Vec::new(),
        })
    }
}

// ---------- Helpers ----------

fn unique_user(label: &str) -> UserId {
    use std::sync::atomic::AtomicU64;
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    UserId::new(format!("user-{label}-{n}"))
}

fn make_handler(
    convs: Arc<RecordingConversations>,
    registry: Arc<BackgroundTaskRegistry>,
) -> DefaultAssistantApiHandler<
    FakeAssistant,
    RecordingConversations,
    FakeSettings,
    FakeConnections,
    FakeKnowledge,
> {
    DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        convs,
        Arc::new(FakeSettings),
        Arc::new(FakeConnections),
        Arc::new(FakeKnowledge),
    )
    .with_registry(registry)
}

fn spawn_dummy_task(registry: &BackgroundTaskRegistry, user: &UserId, title: &str) -> api::TaskId {
    let release = Arc::new(Notify::new());
    let r2 = Arc::clone(&release);
    let id = registry.spawn(
        user.clone(),
        api::TaskKind::Standalone {
            name: title.into(),
            conversation_id: "c-x".into(),
        },
        title.into(),
        move |_ctx| async move {
            r2.notified().await;
            Ok(())
        },
    );
    // Leak the release sender — these test tasks stay running, kept alive
    // only by the spawn() body holding the notified future. Tests that
    // need the task in a terminal state should call `registry.cancel`
    // or use `spawn_completing_task` below.
    std::mem::forget(release);
    id
}

fn spawn_completing_task(
    registry: &BackgroundTaskRegistry,
    user: &UserId,
    title: &str,
) -> api::TaskId {
    registry.spawn(
        user.clone(),
        api::TaskKind::Standalone {
            name: title.into(),
            conversation_id: "c-done".into(),
        },
        title.into(),
        |_ctx| async move { Ok(()) },
    )
}

// ---------- Tests ----------

/// Acceptance: ListBackgroundTasks returns the user's registered tasks.
#[tokio::test]
async fn list_background_tasks_returns_registered_tasks() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));

    let user = unique_user("alice");
    let id1 = spawn_dummy_task(&registry, &user, "first");
    let id2 = spawn_dummy_task(&registry, &user, "second");

    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::ListBackgroundTasks {
                include_finished: false,
                limit: None,
            },
        )
        .await
        .expect("list ok");

    let tasks = match result {
        api::CommandResult::BackgroundTasks(t) => t,
        other => panic!("unexpected result: {other:?}"),
    };
    let ids: std::collections::HashSet<_> = tasks.iter().map(|t| t.id.clone()).collect();
    assert!(ids.contains(&id1), "first task present");
    assert!(ids.contains(&id2), "second task present");
}

/// Acceptance: GetBackgroundTask returns the requested task view.
#[tokio::test]
async fn get_background_task_returns_specific_task() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));

    let user = unique_user("alice");
    let id = spawn_dummy_task(&registry, &user, "my-task");

    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::GetBackgroundTask { id: id.0.clone() },
        )
        .await
        .expect("get ok");

    match result {
        api::CommandResult::BackgroundTask(t) => {
            assert_eq!(t.id, id);
            assert_eq!(t.title, "my-task");
        }
        other => panic!("unexpected result: {other:?}"),
    }
}

/// Unhappy: GetBackgroundTask on an unknown id returns a structured error,
/// not a panic and not a silent Ok. The transport adapter surfaces this
/// as a `WsFrame::Error` so the client can render the failure.
#[tokio::test]
async fn get_background_task_unknown_id_returns_error() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));

    let user = unique_user("alice");
    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::GetBackgroundTask {
                id: "does-not-exist".into(),
            },
        )
        .await;
    assert!(
        result.is_err(),
        "unknown id must surface as an error, not a silent Ok"
    );
}

/// Acceptance: CancelBackgroundTask trips the registry's cancellation
/// token and the targeted task reaches the `Cancelled` terminal state.
#[tokio::test]
async fn cancel_background_task_propagates_to_registry() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));

    let user = unique_user("alice");

    // Spawn a cooperatively-cancellable task that watches its token.
    let started = Arc::new(Notify::new());
    let started_clone = Arc::clone(&started);
    let id = registry.spawn(
        user.clone(),
        api::TaskKind::Standalone {
            name: "cancel-me".into(),
            conversation_id: "c".into(),
        },
        "cancel-me".into(),
        move |ctx| async move {
            started_clone.notify_one();
            ctx.token.cancelled().await;
            Ok(())
        },
    );
    started.notified().await;

    // Subscribe BEFORE cancel so we don't race the TaskCompleted event —
    // terminal entries are evicted from the registry, so a missed event
    // would leave us with no way to observe the terminal status.
    let mut events = registry.subscribe(&user);

    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::CancelBackgroundTask { id: id.0.clone() },
        )
        .await
        .expect("cancel ok");
    assert!(matches!(result, api::CommandResult::Ack));

    registry.wait(&id).await;
    loop {
        match tokio::time::timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Ok(api::Event::TaskCompleted { id: ev_id, status, .. })) if ev_id == id.0 => {
                assert_eq!(status, api::TaskStatus::Cancelled);
                break;
            }
            Ok(Ok(_)) => continue,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(e)) => panic!("event channel closed: {e:?}"),
            Err(_) => panic!("timed out waiting for TaskCompleted"),
        }
    }
    assert!(registry.get(&user, &id).is_none());
}

/// Unhappy: cancelling a task that has already completed surfaces an
/// error frame instead of pretending the cancel succeeded — without it,
/// "cancel" on a finished row would look like a silent no-op. Since
/// terminal entries are evicted from the registry on finalize (#158),
/// the underlying error is now `NotFound` (existence-hiding contract
/// from #105) rather than `AlreadyTerminal`. The user-visible outcome —
/// an error rather than a silent success — is unchanged.
#[tokio::test]
async fn cancel_completed_task_returns_structured_error() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));

    let user = unique_user("alice");
    let id = spawn_completing_task(&registry, &user, "done");
    registry.wait(&id).await;

    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::CancelBackgroundTask { id: id.0.clone() },
        )
        .await;
    assert!(
        result.is_err(),
        "cancelling a completed task must be an error, got {result:?}"
    );
}

/// Unhappy: user B cannot cancel user A's task. The error reads as
/// NotFound (existence-hiding contract from #105), never as Forbidden,
/// so the surface doesn't leak that task A exists.
#[tokio::test]
async fn cross_user_cancel_returns_not_found() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));

    let alice = unique_user("alice");
    let bob = unique_user("bob");
    let id = spawn_dummy_task(&registry, &alice, "alice-task");

    let result = handler
        .handle_command_for(
            RequestContext::for_user(bob.clone()),
            api::Command::CancelBackgroundTask { id: id.0.clone() },
        )
        .await;
    assert!(
        result.is_err(),
        "bob must not be able to cancel alice's task; got {result:?}"
    );
    // The error message must NOT reveal existence — it must read as
    // "not found", not "forbidden" or similar.
    let err = result.unwrap_err().to_string();
    assert!(
        err.to_lowercase().contains("not found"),
        "cross-user cancel must surface as 'not found' to avoid leaking existence; got {err:?}",
    );
}

/// Acceptance: GetBackgroundTaskLogs paginates by `after_seq`.
#[tokio::test]
async fn task_log_pagination_works_via_after_seq() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));

    let user = unique_user("alice");

    // Spawn a task that emits 50 status log lines then waits.
    let release = Arc::new(Notify::new());
    let release_clone = Arc::clone(&release);
    let id = registry.spawn(
        user.clone(),
        api::TaskKind::Standalone {
            name: "logger".into(),
            conversation_id: "c".into(),
        },
        "logger".into(),
        move |ctx| async move {
            for i in 0..50 {
                ctx.logs.append(
                    api::LogLevel::Info,
                    api::LogCategory::Status,
                    format!("line {i}"),
                    None,
                );
            }
            release_clone.notified().await;
            Ok(())
        },
    );
    // Give the task time to emit its log lines.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let first = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::GetBackgroundTaskLogs {
                id: id.0.clone(),
                after_seq: Some(0),
                limit: Some(20),
            },
        )
        .await
        .expect("logs ok");
    let (entries1, next_seq1) = match first {
        api::CommandResult::BackgroundTaskLogs { entries, next_seq } => (entries, next_seq),
        other => panic!("unexpected result: {other:?}"),
    };
    // The first lifecycle "task started" entry uses seq=1; the 50 status
    // lines occupy seq=2..=51. A page with after_seq=0 limit=20 returns
    // seq=1..=20 inclusive — so 20 entries, next_seq=21.
    assert_eq!(entries1.len(), 20);
    assert_eq!(next_seq1, 21);

    let second = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::GetBackgroundTaskLogs {
                id: id.0.clone(),
                after_seq: Some(next_seq1 - 1),
                limit: Some(20),
            },
        )
        .await
        .expect("logs ok");
    let (entries2, _) = match second {
        api::CommandResult::BackgroundTaskLogs { entries, next_seq } => (entries, next_seq),
        other => panic!("unexpected result: {other:?}"),
    };
    // No log entry should be returned twice across pages.
    let seqs1: std::collections::HashSet<_> = entries1.iter().map(|e| e.seq).collect();
    let seqs2: std::collections::HashSet<_> = entries2.iter().map(|e| e.seq).collect();
    assert!(seqs1.is_disjoint(&seqs2), "pages must not overlap");
    assert_eq!(entries2.len(), 20);

    release.notify_one();
    registry.wait(&id).await;
}

/// Default limits: when `after_seq` and `limit` are omitted the daemon
/// returns the recent slice (up to the documented default of 200) without
/// failing. Pinning this protects clients that send the minimal payload.
#[tokio::test]
async fn task_log_defaults_apply_when_optional_fields_omitted() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));

    let user = unique_user("alice");
    let id = spawn_completing_task(&registry, &user, "tiny");
    registry.wait(&id).await;

    let result = handler
        .handle_command_for(
            RequestContext::for_user(user),
            api::Command::GetBackgroundTaskLogs {
                id: id.0.clone(),
                after_seq: None,
                limit: None,
            },
        )
        .await
        .expect("logs ok");
    match result {
        api::CommandResult::BackgroundTaskLogs { entries, .. } => {
            // start + completion lifecycle markers.
            assert!(!entries.is_empty(), "expected lifecycle entries");
        }
        other => panic!("unexpected result: {other:?}"),
    }
}

/// Subscribe and Unsubscribe Ack immediately at the handler level (the
/// dispatcher attaches the real broadcast forwarder separately, but the
/// arms themselves must return `Ack`).
#[tokio::test]
async fn subscribe_and_unsubscribe_ack_immediately() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));
    let user = unique_user("alice");

    let sub = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::SubscribeBackgroundTasks,
        )
        .await
        .expect("subscribe ok");
    assert!(matches!(sub, api::CommandResult::Ack));

    let unsub = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::UnsubscribeBackgroundTasks,
        )
        .await
        .expect("unsubscribe ok");
    assert!(matches!(unsub, api::CommandResult::Ack));
}

/// The default handler exposes the registry's per-user broadcast receiver
/// so the dispatcher (or any other transport-level forwarder) can fan
/// `Event::Task*` out to a connection. Without this hook, Subscribe is
/// a no-op at the wire level.
#[tokio::test]
async fn handler_exposes_user_broadcast_receiver_when_registry_attached() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(convs, Arc::clone(&registry));
    let user = unique_user("alice");

    let receiver = with_user_id(user.clone(), async {
        handler.subscribe_user_events().await
    })
    .await;
    let mut rx = receiver.expect("handler returns receiver when registry attached");

    // Trigger an event by spawning a task — the receiver should observe
    // TaskStarted.
    let _id = spawn_completing_task(&registry, &user, "trigger");

    let ev = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("event arrived")
        .expect("channel open");
    assert!(matches!(ev, api::Event::TaskStarted { .. }));
}

/// When no registry is attached, the handler must return None from
/// `subscribe_user_events` so the dispatcher knows to error out the
/// Subscribe command rather than spawn a forwarder that would never
/// receive anything.
#[tokio::test]
async fn handler_returns_none_receiver_when_registry_not_attached() {
    let handler = DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(RecordingConversations::new()),
        Arc::new(FakeSettings),
        Arc::new(FakeConnections),
        Arc::new(FakeKnowledge),
    );
    let user = unique_user("alice");
    let receiver = with_user_id(user, async { handler.subscribe_user_events().await }).await;
    assert!(receiver.is_none());
}

/// Acceptance: `SendMessage` is registered as a background task and the
/// new ack carries the task id. Without it, clients can't correlate the
/// streaming events back to a `ListBackgroundTasks` row.
#[tokio::test]
async fn start_send_message_returns_task_id_that_appears_in_listing() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    // Block the underlying send so the task stays Running long enough to
    // appear in the list. Terminal entries are evicted from the registry,
    // so without the block the task can complete before we list it.
    let release = Arc::new(Notify::new());
    convs.block_on(Arc::clone(&release));
    let handler = make_handler(convs, Arc::clone(&registry));

    let user = unique_user("alice");
    let sink: Arc<dyn desktop_assistant_application::EventSink> = Arc::new(NoopSink);

    let task_id = with_user_id(user.clone(), async {
        handler
            .start_send_message(
                "conv-x".into(),
                "hello".into(),
                None,
                "req-1".into(),
                Arc::clone(&sink),
            )
            .await
    })
    .await
    .expect("start_send_message ok");
    let task_id = task_id.expect("registry attached, expected Some(task_id)");

    // The new id is visible to the same user via List while it's running.
    let list = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::ListBackgroundTasks {
                include_finished: false,
                limit: None,
            },
        )
        .await
        .expect("list ok");
    let tasks = match list {
        api::CommandResult::BackgroundTasks(t) => t,
        other => panic!("unexpected: {other:?}"),
    };
    assert!(tasks.iter().any(|t| t.id == task_id));

    release.notify_one();
    registry.wait(&task_id).await;
}

/// Issue #154: the registry-spawned `SendMessage` body must observe the
/// caller's user_id scope. `tokio::spawn` doesn't propagate task-locals,
/// so the body has to re-install `with_user_id`. Without that, storage
/// queries inside the body see the `"default"` sentinel and a
/// per-user-scoped `WHERE user_id = $1` lookup misses the row, surfacing
/// as `ConversationNotFound` to the user.
#[tokio::test]
async fn start_send_message_propagates_user_id_to_spawned_body() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let sink: Arc<dyn desktop_assistant_application::EventSink> = Arc::new(NoopSink);

    let task_id = with_user_id(user.clone(), async {
        handler
            .start_send_message(
                "conv-x".into(),
                "hello".into(),
                None,
                "req-1".into(),
                Arc::clone(&sink),
            )
            .await
    })
    .await
    .expect("start_send_message ok")
    .expect("registry attached, expected Some(task_id)");

    registry.wait(&task_id).await;

    let seen = convs.seen_user_ids();
    assert_eq!(
        seen,
        vec![user.as_str().to_string()],
        "spawned body must observe the caller's user_id, not the default sentinel"
    );
}

/// Issue #154 sibling: the same scope requirement applies to
/// `handle_send_message_with_override`, which routes through
/// `send_message_via_registry` when a registry is attached. This path
/// is exercised by the legacy bare-Ack transport adapters and must stay
/// in sync with the `start_send_message` path.
#[tokio::test]
async fn handle_send_message_with_override_propagates_user_id_to_spawned_body() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let sink: Arc<dyn desktop_assistant_application::EventSink> = Arc::new(NoopSink);

    with_user_id(user.clone(), async {
        handler
            .handle_send_message_with_override(
                "conv-y".into(),
                "hello".into(),
                None,
                "req-2".into(),
                sink,
            )
            .await
    })
    .await
    .expect("handle_send_message_with_override ok");

    let seen = convs.seen_user_ids();
    assert_eq!(
        seen,
        vec![user.as_str().to_string()],
        "spawned body must observe the caller's user_id, not the default sentinel"
    );
}

/// Without a registry the handler's `start_send_message` returns `None`
/// so the dispatcher knows to fall back to the pre-#114 streaming path.
#[tokio::test]
async fn start_send_message_returns_none_without_registry() {
    let handler = DefaultAssistantApiHandler::new(
        Arc::new(FakeAssistant),
        Arc::new(RecordingConversations::new()),
        Arc::new(FakeSettings),
        Arc::new(FakeConnections),
        Arc::new(FakeKnowledge),
    );
    let user = unique_user("alice");
    let sink: Arc<dyn desktop_assistant_application::EventSink> = Arc::new(NoopSink);

    let result = with_user_id(user, async {
        handler
            .start_send_message("conv-x".into(), "hello".into(), None, "req-1".into(), sink)
            .await
    })
    .await
    .expect("start ok");
    assert!(result.is_none(), "no registry → no task id");
}

/// Business outcome: a user can list their running standalone agent.
/// This is the end-user-facing acceptance test from #114.
#[tokio::test]
async fn business_outcome_user_can_list_their_running_standalone_agent() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new());
    let release = Arc::new(Notify::new());
    convs.block_on(Arc::clone(&release));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");

    // Spawn a standalone agent.
    let spawn_result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::SpawnStandaloneAgent {
                name: "researcher".into(),
                initial_prompt: "go".into(),
                override_selection: None,
                tools: None,
            },
        )
        .await
        .expect("spawn ok");
    let task_id = match spawn_result {
        api::CommandResult::BackgroundTaskSpawned { id } => api::TaskId(id),
        other => panic!("unexpected: {other:?}"),
    };

    // Briefly wait for the registered row to settle into Running.
    tokio::time::sleep(Duration::from_millis(20)).await;

    let list = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            api::Command::ListBackgroundTasks {
                include_finished: false,
                limit: None,
            },
        )
        .await
        .expect("list ok");
    let tasks = match list {
        api::CommandResult::BackgroundTasks(t) => t,
        other => panic!("unexpected: {other:?}"),
    };
    let me = tasks
        .iter()
        .find(|t| t.id == task_id)
        .expect("standalone visible in list");
    assert_eq!(me.status, api::TaskStatus::Running);
    assert!(me.title.contains("researcher"), "title: {}", me.title);

    release.notify_one();
    registry.wait(&task_id).await;
}

struct NoopSink;
#[async_trait::async_trait]
impl desktop_assistant_application::EventSink for NoopSink {
    async fn emit(&self, _event: api::Event) -> bool {
        true
    }
}

#[allow(dead_code)]
fn _silence_unused(c: &RecordingConversations) -> &Mutex<Vec<SendCall>> {
    let _ = current_user_id();
    &c.sends
}
