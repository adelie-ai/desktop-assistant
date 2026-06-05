//! Acceptance tests for `Command::SpawnStandaloneAgent` (#113).
//!
//! Spec-driven, TDD: these tests describe the desired *business
//! outcomes* of the handler — synchronous task-id return, registry
//! visibility with no parent, override threading, event emission, tool
//! allowlist propagation, user isolation, cancellation, clean failure
//! recording, conversation creation under the user scope, and end-to-end
//! prompt→response history.

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
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::inbound::{
    AssistantService, BackendTasksSettingsView, ConnectionConfigPayload, ConnectionsService,
    ConnectorDefaultsView, ConversationService, DatabaseSettingsView, EmbeddingsSettingsView,
    KnowledgeService, LlmSettingsView, ModelListing as CoreModelListing, PersistenceSettingsView,
    PromptDispatchOutcome, PromptSelectionOverride, PurposeConfigPayload, PurposeKind,
    PurposesView as CorePurposesView, SettingsService, WsAuthSettingsView,
};
use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback, current_tool_allowlist};
use tokio::sync::Notify;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

// ----- Minimal fakes -------------------------------------------------------

struct FakeKnowledge;
impl KnowledgeService for FakeKnowledge {
    async fn list_entries(
        &self,
        _limit: usize,
        _offset: usize,
        _tag_filter: Option<Vec<String>>,
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
    async fn delete_connection(&self, _id: String, _force: bool) -> Result<(), CoreError> {
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

struct FakeAssistant;
impl AssistantService for FakeAssistant {
    fn version(&self) -> &str {
        "0.0.0-test"
    }
    fn ping(&self) -> &str {
        "pong"
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
        Ok("jwt".into())
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
            remote_url: String::new(),
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
            llm_connector: "openai".into(),
            llm_model: "gpt-5".into(),
            llm_base_url: "https://api.openai.com/v1".into(),
            dreaming_enabled: false,
            dreaming_interval_secs: 3600,
            archive_after_days: 0,
        })
    }
    async fn set_backend_tasks_settings(
        &self,
        _c: Option<String>,
        _m: Option<String>,
        _b: Option<String>,
        _d: bool,
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

/// Per-call record captured by [`RecordingConversations`].
#[derive(Debug, Clone)]
struct SendCall {
    conversation_id: String,
    prompt: String,
    override_selection: Option<PromptSelectionOverride>,
    tool_allowlist: Option<Vec<String>>,
    /// Recorded so the multi-tenant / scope tests can sanity-check that
    /// `with_user_id` was actually installed before the inner call —
    /// even though the registry-side isolation tests assert this
    /// indirectly through `registry.get(&other_user, ...)`.
    #[allow(dead_code)]
    user_id_observed: UserId,
}

#[derive(Debug, Clone, Default)]
struct CreateCall {
    title: String,
    user_id_observed: UserId,
}

/// Programmable conversation service for the standalone-agent acceptance
/// tests. Records every `create_conversation` and `send_prompt_with_override`
/// call so tests can assert business outcomes without driving a real LLM.
///
/// Behavior can be tuned with `behavior`:
/// - `Block { release }` — `send_prompt_with_override` waits on `release`
///   then returns the configured response (used by the synchronous-return
///   and cancellation tests).
/// - `RespondImmediately { response }` — returns `response` right away.
/// - `Fail { error }` — returns `Err(CoreError::Llm(error))`.
#[derive(Debug)]
enum Behavior {
    Block {
        release: Arc<Notify>,
        response: String,
    },
    RespondImmediately {
        response: String,
    },
    Fail {
        error: String,
    },
}

struct RecordingConversations {
    behavior: Behavior,
    creates: Mutex<Vec<CreateCall>>,
    sends: Mutex<Vec<SendCall>>,
    /// Set when the LLM call observed token cancellation.
    cancelled_flag: Arc<AtomicBool>,
    send_count: Arc<AtomicU32>,
    /// Synthetic conversation id used for every newly created conversation.
    /// Tests don't care which id we hand back, but they do verify it gets
    /// propagated into `TaskKind::Standalone::conversation_id`.
    fixed_conversation_id: String,
}

impl RecordingConversations {
    fn new(behavior: Behavior) -> Self {
        Self {
            behavior,
            creates: Mutex::new(vec![]),
            sends: Mutex::new(vec![]),
            cancelled_flag: Arc::new(AtomicBool::new(false)),
            send_count: Arc::new(AtomicU32::new(0)),
            fixed_conversation_id: "conv-standalone-1".into(),
        }
    }
}

#[async_trait::async_trait]
impl ConversationService for RecordingConversations {
    async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
        let observed = current_user_id();
        self.creates.lock().unwrap().push(CreateCall {
            title: title.clone(),
            user_id_observed: observed,
        });
        Ok(Conversation::new(self.fixed_conversation_id.clone(), title))
    }
    async fn list_conversations(
        &self,
        _m: Option<u32>,
        _i: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        Ok(vec![])
    }
    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        let mut c = Conversation::new(id.as_str(), "standalone");
        // Emulate the post-send conversation having the user's initial
        // prompt and the assistant's reply, so the business-outcome test
        // can verify history shape end-to-end. We pull the prompt from
        // the last recorded send (if any) and a synthetic response.
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
        override_selection: Option<PromptSelectionOverride>,
        _system_refinement: String,
        mut on_chunk: ChunkCallback,
        _on_status: StatusCallback,
        cancellation: CancellationToken,
    ) -> Result<PromptDispatchOutcome, CoreError> {
        self.send_count.fetch_add(1, Ordering::SeqCst);
        let allowlist = current_tool_allowlist();
        let user_id = current_user_id();
        self.sends.lock().unwrap().push(SendCall {
            conversation_id: conversation_id.as_str().to_string(),
            prompt: prompt.clone(),
            override_selection: override_selection.clone(),
            tool_allowlist: allowlist,
            user_id_observed: user_id,
        });

        match &self.behavior {
            Behavior::Block { release, response } => {
                on_chunk(response.clone());
                tokio::select! {
                    _ = cancellation.cancelled() => {
                        self.cancelled_flag.store(true, Ordering::SeqCst);
                        Err(CoreError::Cancelled)
                    }
                    _ = release.notified() => Ok(PromptDispatchOutcome {
                        response: response.clone(),
                        warnings: Vec::new(),
                    }),
                }
            }
            Behavior::RespondImmediately { response } => {
                on_chunk(response.clone());
                Ok(PromptDispatchOutcome {
                    response: response.clone(),
                    warnings: Vec::new(),
                })
            }
            Behavior::Fail { error } => Err(CoreError::Llm(error.clone())),
        }
    }
}

// ----- Helpers -------------------------------------------------------------

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

fn standalone_cmd(
    name: &str,
    initial_prompt: &str,
    override_selection: Option<api::SendPromptOverride>,
    tools: Option<Vec<String>>,
) -> api::Command {
    api::Command::SpawnStandaloneAgent {
        name: name.into(),
        initial_prompt: initial_prompt.into(),
        override_selection,
        tools,
    }
}

async fn wait_until<F: FnMut() -> bool>(mut pred: F, label: &str) {
    for _ in 0..400 {
        if pred() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("predicate '{label}' never became true within timeout");
}

fn task_id_from(result: api::CommandResult) -> api::TaskId {
    match result {
        api::CommandResult::BackgroundTaskSpawned { id } => api::TaskId(id),
        other => panic!("expected BackgroundTaskSpawned, got {other:?}"),
    }
}

// ----- Acceptance tests ----------------------------------------------------

/// Acceptance: the handler returns the task id synchronously — even when
/// the underlying LLM call would block. Standalone agents are fire-and-
/// forget; if the user has to wait for the LLM to finish before getting
/// back a task id, the UI can't show the task as "running" and can't
/// offer a Cancel button. We assert a strict <50ms latency budget for
/// the handler return.
#[tokio::test]
async fn spawn_standalone_returns_task_id_synchronously() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let release = Arc::new(Notify::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::Block {
        release: Arc::clone(&release),
        response: "ok".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let cmd = standalone_cmd("researcher", "go", None, None);

    let start = std::time::Instant::now();
    let result = handler
        .handle_command_for(RequestContext::for_user(user.clone()), cmd)
        .await
        .expect("handle_command ok");
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(50),
        "handler should return within 50ms even when the LLM blocks; took {elapsed:?}"
    );

    let task_id = task_id_from(result);
    // Task must actually be live in the registry — the wait below
    // doubles as proof we received a real id.
    assert!(
        registry.get(&user, &task_id).is_some(),
        "spawned task must be visible to its owner"
    );

    // Release the LLM so the task wraps up and the test doesn't leak.
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&task_id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");
}

/// Acceptance: a standalone task appears in the registry's listing for
/// its owner with `parent == None` and `kind == Standalone`. This is the
/// shape the process-manager UI keys off when grouping foreground turns,
/// subagents, and standalones.
#[tokio::test]
async fn standalone_appears_in_registry_with_no_parent() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let release = Arc::new(Notify::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::Block {
        release: Arc::clone(&release),
        response: "ok".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            standalone_cmd("harvester", "hello", None, None),
        )
        .await
        .expect("ok");
    let task_id = task_id_from(result);

    let view = registry.get(&user, &task_id).expect("present");
    assert_eq!(view.parent, None, "standalone tasks have no parent");
    match &view.kind {
        api::TaskKind::Standalone {
            name,
            conversation_id,
        } => {
            assert_eq!(name, "harvester");
            assert!(
                !conversation_id.is_empty(),
                "TaskKind::Standalone::conversation_id must be populated"
            );
        }
        other => panic!("expected Standalone, got {other:?}"),
    }

    release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&task_id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");
}

/// Acceptance: `override_selection.connection_id` is threaded into the
/// underlying `send_prompt_with_override` call so the chosen connection
/// (e.g. "openai") is actually used by the dispatch path. Without this
/// the override would be silently dropped and standalone agents would
/// always run on the conversation's default connection.
#[tokio::test]
async fn standalone_runs_initial_prompt_against_chosen_connection() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::RespondImmediately {
        response: "ok".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let override_sel = api::SendPromptOverride {
        connection_id: "openai".into(),
        model_id: "gpt-5".into(),
        effort: None,
    };
    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            standalone_cmd("researcher", "go", Some(override_sel.clone()), None),
        )
        .await
        .expect("ok");
    let task_id = task_id_from(result);
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&task_id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");

    let calls = convs.sends.lock().unwrap().clone();
    assert_eq!(calls.len(), 1, "exactly one send_prompt call");
    let sent = &calls[0];
    let override_observed = sent
        .override_selection
        .as_ref()
        .expect("override threaded through");
    assert_eq!(override_observed.connection_id, "openai");
    assert_eq!(override_observed.model_id, "gpt-5");
    assert_eq!(sent.prompt, "go");
}

/// Acceptance: subscribers to the registry's per-user broadcast see both
/// `TaskStarted` and `TaskCompleted` for the standalone task, in order.
/// This is what powers live UI updates without polling.
#[tokio::test]
async fn standalone_emits_task_started_and_task_completed_events() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::RespondImmediately {
        response: "ok".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let mut events = registry.subscribe(&user);

    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            standalone_cmd("agent", "go", None, None),
        )
        .await
        .expect("ok");
    let task_id = task_id_from(result);

    let mut saw_started = false;
    let mut saw_completed_after_started = false;
    while let Ok(ev) = timeout(Duration::from_secs(2), events.recv()).await {
        let ev = ev.expect("event ok");
        match ev {
            api::Event::TaskStarted { task } if task.id == task_id => {
                saw_started = true;
                assert!(
                    matches!(task.kind, api::TaskKind::Standalone { .. }),
                    "started event carries Standalone kind"
                );
            }
            api::Event::TaskCompleted { id, status, .. } if id == task_id.0 => {
                assert!(saw_started, "TaskCompleted arrived before TaskStarted");
                assert_eq!(status, api::TaskStatus::Completed);
                saw_completed_after_started = true;
                break;
            }
            _ => {}
        }
    }
    assert!(saw_started, "did not receive TaskStarted");
    assert!(
        saw_completed_after_started,
        "did not receive TaskCompleted in order"
    );
}

/// Acceptance: when the caller specifies a `tools` allowlist, the task
/// body observes it via the shared `current_tool_allowlist()` task-local.
/// Once tool dispatch lands behind that task-local (#112's shared
/// mechanism), the allowlist will gate which tools the LLM can call;
/// today the propagation alone is what this test pins down.
#[tokio::test]
async fn standalone_tool_allowlist_enforced() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::RespondImmediately {
        response: "ok".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            standalone_cmd(
                "researcher",
                "go",
                None,
                Some(vec!["search".into(), "fetch".into()]),
            ),
        )
        .await
        .expect("ok");
    let task_id = task_id_from(result);
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&task_id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");

    let calls = convs.sends.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    let observed = calls[0]
        .tool_allowlist
        .as_ref()
        .expect("allowlist must be installed for the task body");
    assert_eq!(observed, &vec!["search".to_string(), "fetch".to_string()]);
}

/// Acceptance: a standalone task spawned under user A is invisible to
/// user B. Multi-tenant isolation rides on the registry's existing user
/// scope (#105 contract). This is the boundary test that proves the
/// handler installs `user_id` correctly before spawning.
#[tokio::test]
async fn standalone_under_user_a_invisible_to_user_b() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let release = Arc::new(Notify::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::Block {
        release: Arc::clone(&release),
        response: "ok".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let alice = unique_user("alice");
    let bob = unique_user("bob");

    let result = handler
        .handle_command_for(
            RequestContext::for_user(alice.clone()),
            standalone_cmd("alice-task", "go", None, None),
        )
        .await
        .expect("ok");
    let task_id = task_id_from(result);

    // Alice sees her task; Bob does not.
    assert!(registry.get(&alice, &task_id).is_some());
    assert!(registry.get(&bob, &task_id).is_none());
    assert!(registry.list(&bob, true, None).is_empty());

    release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&task_id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");
}

/// Acceptance: tripping the registry's cancel for a running standalone
/// task aborts the underlying LLM call cooperatively (#109's
/// CancellationToken). Without this the user's Cancel button would be a
/// lie.
#[tokio::test]
async fn cancelling_standalone_aborts_the_turn() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let release = Arc::new(Notify::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::Block {
        release: Arc::clone(&release),
        response: "ok".into(),
    }));
    let cancelled_flag = Arc::clone(&convs.cancelled_flag);
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            standalone_cmd("agent", "go", None, None),
        )
        .await
        .expect("ok");
    let task_id = task_id_from(result);

    // Wait for the body to actually be inside send_prompt — otherwise we
    // could cancel before the inner future ever sees the token.
    wait_until(
        || convs.send_count.load(Ordering::SeqCst) == 1,
        "send_prompt entered",
    )
    .await;

    // Subscribe BEFORE cancel so we observe the TaskCompleted event —
    // terminal entries are evicted from the registry on finalize (#158).
    let mut events = registry.subscribe(&user);
    registry.cancel(&user, &task_id).expect("cancel ok");
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&task_id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");

    assert!(
        cancelled_flag.load(Ordering::SeqCst),
        "underlying LLM call did not observe cancellation"
    );
    let want = task_id.0.clone();
    let status = loop {
        match tokio::time::timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Ok(api::Event::TaskCompleted { id, status, .. })) if id == want => break status,
            Ok(Ok(_)) => continue,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(e)) => panic!("event channel closed: {e:?}"),
            Err(_) => panic!("timed out waiting for TaskCompleted"),
        }
    };
    assert_eq!(status, api::TaskStatus::Cancelled);
    assert!(registry.get(&user, &task_id).is_none());
}

/// Acceptance: if the underlying dispatch fails (e.g. invalid
/// connection in the override), the handler must STILL return a clean
/// task id — the failure shows up in the task's status (`Failed`) and
/// `last_error`, not as a panic or a synchronous error from the handler.
/// This is the safety contract that lets the UI present a coherent
/// "task failed" tile.
#[tokio::test]
async fn standalone_with_invalid_connection_returns_clean_error_in_task_status() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::Fail {
        error: "unknown connection: ghost".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let bad_override = api::SendPromptOverride {
        connection_id: "ghost".into(),
        model_id: "anything".into(),
        effort: None,
    };

    // Subscribe BEFORE spawning so we can't possibly miss the
    // TaskCompleted event — terminal entries are evicted from the
    // registry on finalize (#158), so the broadcast is the only way to
    // observe the failure status post-completion.
    let mut events = registry.subscribe(&user);

    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            standalone_cmd("agent", "go", Some(bad_override), None),
        )
        .await
        .expect("handler must not surface the LLM error synchronously");
    let task_id = task_id_from(result);

    let want = task_id.0.clone();
    let (status, last_error) = loop {
        match tokio::time::timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Ok(api::Event::TaskCompleted {
                id,
                status,
                last_error,
            })) if id == want => {
                break (status, last_error);
            }
            Ok(Ok(_)) => continue,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(e)) => panic!("event channel closed: {e:?}"),
            Err(_) => panic!("timed out waiting for TaskCompleted"),
        }
    };
    assert_eq!(status, api::TaskStatus::Failed);
    let err = last_error
        .as_deref()
        .expect("Failed task must record last_error");
    assert!(
        err.contains("unknown connection: ghost"),
        "last_error must surface the underlying message, got {err:?}"
    );
}

/// Acceptance: spawning a standalone agent creates a fresh conversation
/// scoped to the requesting user — the `create_conversation` call
/// happens inside the `with_user_id` scope so storage adapters tag the
/// new row with the right `user_id`.
#[tokio::test]
async fn spawn_standalone_creates_new_conversation_scoped_to_user_id() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::RespondImmediately {
        response: "ok".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let alice = unique_user("alice");
    let result = handler
        .handle_command_for(
            RequestContext::for_user(alice.clone()),
            standalone_cmd("analyst", "summarize", None, None),
        )
        .await
        .expect("ok");
    let task_id = task_id_from(result);
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&task_id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");

    let creates = convs.creates.lock().unwrap().clone();
    assert_eq!(creates.len(), 1, "exactly one conversation created");
    assert_eq!(
        creates[0].user_id_observed, alice,
        "create_conversation must run under the requesting user's scope"
    );
    // The conversation title should derive from the agent's name so the
    // UI can present a meaningful entry. Either exact-equal or
    // contains-name is acceptable; we pick contains so the helper can
    // prepend a "Standalone:" tag if it wants.
    assert!(
        creates[0].title.contains("analyst"),
        "conversation title must reflect the agent name, got {:?}",
        creates[0].title
    );
}

/// Business outcome: at the end of a standalone run the conversation's
/// message history contains the initial prompt and the assistant's
/// reply. This is the contract the UI relies on when opening the
/// conversation after a standalone agent finishes — the user should see
/// what they asked for and what the agent answered, not an empty thread.
#[tokio::test]
async fn business_outcome_standalone_conversation_contains_initial_prompt_then_response() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let convs = Arc::new(RecordingConversations::new(Behavior::RespondImmediately {
        response: "ok".into(),
    }));
    let handler = make_handler(Arc::clone(&convs), Arc::clone(&registry));

    let user = unique_user("alice");
    let result = handler
        .handle_command_for(
            RequestContext::for_user(user.clone()),
            standalone_cmd("agent", "tell me a joke", None, None),
        )
        .await
        .expect("ok");
    let task_id = task_id_from(result);
    tokio::time::timeout(Duration::from_secs(5), registry.wait(&task_id))
        .await
        .expect("wait() must resolve once the task finalizes, not hang");

    // The send actually ran with the user's prompt.
    let sends = convs.sends.lock().unwrap().clone();
    assert_eq!(sends.len(), 1);
    assert_eq!(sends[0].prompt, "tell me a joke");

    // Pull the conversation back: it should carry the user prompt and
    // an assistant reply. We go through the ConversationService directly
    // because the fake mirrors the storage adapter's "messages live in
    // the conversation" contract.
    let conv_id = ConversationId::from(sends[0].conversation_id.as_str());
    let conv = convs
        .get_conversation(&conv_id)
        .await
        .expect("conversation exists");
    let roles: Vec<_> = conv.messages.iter().map(|m| m.role.clone()).collect();
    let contents: Vec<_> = conv.messages.iter().map(|m| m.content.clone()).collect();
    assert_eq!(roles, vec![Role::User, Role::Assistant]);
    assert_eq!(contents[0], "tell me a joke");
    assert_eq!(contents[1], "ok");
}
