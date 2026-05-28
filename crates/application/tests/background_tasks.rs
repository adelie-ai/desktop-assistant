//! Acceptance tests for the in-memory `BackgroundTaskRegistry` (#111).
//!
//! These tests are intentionally written before the implementation
//! (TDD) — they describe the desired *business outcomes* of the
//! registry: unique ids, user scoping, cooperative cancellation,
//! bounded log ring buffer, broadcast event fan-out, and the wrapper
//! around the existing foreground send path.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use desktop_assistant_api_model as api;
use desktop_assistant_application::background_tasks::{
    BackgroundTaskRegistry, RegistryConfig, TaskError,
};
use desktop_assistant_application::UserId;
use tokio::sync::Notify;
use tokio::time::timeout;

fn unique_user(label: &str) -> UserId {
    UserId::new(format!("user-{label}-{}", uuid_like_label()))
}

fn uuid_like_label() -> String {
    use std::sync::atomic::AtomicU64;
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    format!("{n}")
}

fn conv_kind(id: &str) -> api::TaskKind {
    api::TaskKind::Conversation {
        conversation_id: id.into(),
    }
}

/// Wait until `pred()` returns true, polling at most `tries` times with a
/// short sleep. Useful for tests that watch lifecycle transitions driven
/// by the background runtime without coupling to exact scheduling.
async fn wait_until<F: FnMut() -> bool>(mut pred: F, label: &str) {
    for _ in 0..200 {
        if pred() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("predicate '{label}' never became true within timeout");
}

/// Drain events for `task_id` from a broadcast receiver until a terminal
/// `TaskCompleted` arrives, then return its `(status, last_error)`. Used
/// by tests now that terminal entries are evicted from the registry —
/// callers can no longer inspect post-completion state via `get`/`list`.
async fn wait_for_completion(
    events: &mut tokio::sync::broadcast::Receiver<api::Event>,
    task_id: &api::TaskId,
) -> (api::TaskStatus, Option<String>) {
    let want = task_id.0.clone();
    loop {
        match timeout(Duration::from_secs(5), events.recv()).await {
            Ok(Ok(api::Event::TaskCompleted { id, status, last_error })) if id == want => {
                return (status, last_error);
            }
            Ok(Ok(_)) => continue,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(e)) => panic!("event channel closed before TaskCompleted: {e:?}"),
            Err(_) => panic!("timed out waiting for TaskCompleted({})", want),
        }
    }
}

// ----------------------------------------------------------------------
// 1. Unique task ids
// ----------------------------------------------------------------------

#[tokio::test]
async fn registry_spawn_returns_unique_task_id() {
    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("alice");

    let mut ids: HashSet<api::TaskId> = HashSet::new();
    for i in 0..100 {
        let id = registry.spawn(
            user.clone(),
            conv_kind(&format!("c-{i}")),
            format!("t-{i}"),
            |_ctx| async move { Ok(()) },
        );
        assert!(ids.insert(id.clone()), "duplicate task id: {:?}", id);
    }
    assert_eq!(ids.len(), 100);
}

// ----------------------------------------------------------------------
// 2. Terminal tasks are evicted from the registry
// ----------------------------------------------------------------------

#[tokio::test]
async fn registry_evicts_terminal_tasks_so_list_returns_only_in_flight() {
    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("alice");
    let mut events = registry.subscribe(&user);

    // One task that completes immediately.
    let done = registry.spawn(
        user.clone(),
        conv_kind("c-done"),
        "done".into(),
        |_ctx| async move { Ok(()) },
    );

    // One task that blocks until we let it go.
    let release = Arc::new(Notify::new());
    let release_for_task = Arc::clone(&release);
    let running = registry.spawn(
        user.clone(),
        conv_kind("c-running"),
        "running".into(),
        move |_ctx| async move {
            release_for_task.notified().await;
            Ok(())
        },
    );

    // Observe the completion of `done` via the broadcast channel; the
    // registry evicts terminal entries, so it's no longer queryable.
    let (status, _) = wait_for_completion(&mut events, &done).await;
    assert_eq!(status, api::TaskStatus::Completed);

    // Both flag values now return only the in-flight task. The
    // include_finished flag is retained as a public API but only ever
    // surfaces sweep-resurrected rows after a daemon restart.
    for include_finished in [false, true] {
        let listing = registry.list(&user, include_finished, None);
        assert_eq!(listing.len(), 1, "include_finished={include_finished} {listing:?}");
        assert_eq!(listing[0].id, running);
    }
    assert!(registry.get(&user, &done).is_none(), "terminal task must be evicted");

    // Let the running task complete so the test doesn't leak it.
    release.notify_one();
    let _ = wait_for_completion(&mut events, &running).await;
}

// ----------------------------------------------------------------------
// 3. User scoping isolates listings
// ----------------------------------------------------------------------

#[tokio::test]
async fn registry_user_scope_isolates_listings() {
    let registry = BackgroundTaskRegistry::new();
    let alice = unique_user("alice");
    let bob = unique_user("bob");

    // Per-task Notify so we can release each independently — using one
    // shared Notify here is racy because `notify_one` only stores a
    // single permit regardless of how many times it's called.
    let r_a1 = Arc::new(Notify::new());
    let r_a2 = Arc::new(Notify::new());
    let r_b1 = Arc::new(Notify::new());

    let r_a1_t = Arc::clone(&r_a1);
    let a1 = registry.spawn(alice.clone(), conv_kind("c1"), "a1".into(), move |_| async move {
        r_a1_t.notified().await;
        Ok(())
    });
    let r_a2_t = Arc::clone(&r_a2);
    let a2 = registry.spawn(alice.clone(), conv_kind("c2"), "a2".into(), move |_| async move {
        r_a2_t.notified().await;
        Ok(())
    });
    let r_b1_t = Arc::clone(&r_b1);
    let b1 = registry.spawn(bob.clone(), conv_kind("c3"), "b1".into(), move |_| async move {
        r_b1_t.notified().await;
        Ok(())
    });

    let alice_view = registry.list(&alice, true, None);
    let alice_ids: HashSet<_> = alice_view.iter().map(|v| v.id.clone()).collect();
    assert_eq!(alice_ids, HashSet::from([a1.clone(), a2.clone()]));

    let bob_view = registry.list(&bob, true, None);
    let bob_ids: HashSet<_> = bob_view.iter().map(|v| v.id.clone()).collect();
    assert_eq!(bob_ids, HashSet::from([b1.clone()]));

    // Cross-user get returns None — must not leak existence.
    assert!(registry.get(&bob, &a1).is_none());
    assert!(registry.get(&alice, &b1).is_none());

    // Same-user get returns Some.
    assert!(registry.get(&alice, &a1).is_some());
    assert!(registry.get(&bob, &b1).is_some());

    // Drain so the runtime tasks exit cleanly.
    r_a1.notify_one();
    r_a2.notify_one();
    r_b1.notify_one();
    registry.wait(&a1).await;
    registry.wait(&a2).await;
    registry.wait(&b1).await;
}

// ----------------------------------------------------------------------
// 4. Cancellation fires the token and transitions to Cancelled
// ----------------------------------------------------------------------

#[tokio::test]
async fn registry_cancel_fires_token_and_transitions_status() {
    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("alice");

    let observed_cancel = Arc::new(AtomicBool::new(false));
    let observed_for_task = Arc::clone(&observed_cancel);

    let mut events = registry.subscribe(&user);

    let task_id = registry.spawn(
        user.clone(),
        conv_kind("c"),
        "cancellable".into(),
        move |ctx| async move {
            ctx.token.cancelled().await;
            observed_for_task.store(true, Ordering::SeqCst);
            // The task body is responsible for reporting cancellation.
            Err(anyhow::anyhow!("cancelled by token"))
        },
    );

    // The first event should be TaskStarted.
    let first = timeout(Duration::from_secs(2), events.recv())
        .await
        .expect("event")
        .expect("event ok");
    match first {
        api::Event::TaskStarted { task } => {
            assert_eq!(task.id, task_id);
        }
        other => panic!("expected TaskStarted, got {other:?}"),
    }

    // Cancel — and watch for the terminal event.
    registry.cancel(&user, &task_id).expect("cancel succeeds");

    let mut saw_cancel_complete = false;
    while let Ok(ev) = timeout(Duration::from_secs(2), events.recv()).await {
        let ev = ev.expect("event ok");
        if let api::Event::TaskCompleted { id, status, .. } = ev {
            assert_eq!(id, task_id.0);
            assert_eq!(status, api::TaskStatus::Cancelled);
            saw_cancel_complete = true;
            break;
        }
    }
    assert!(saw_cancel_complete, "did not see TaskCompleted{{Cancelled}}");
    assert!(observed_cancel.load(Ordering::SeqCst));

    // Terminal entries are evicted; the broadcast event above is the
    // authoritative resolution of the cancel.
    assert!(registry.get(&user, &task_id).is_none());
}

// ----------------------------------------------------------------------
// 5. Cross-user cancel returns NotFound
// ----------------------------------------------------------------------

#[tokio::test]
async fn registry_cancel_cross_user_returns_not_found() {
    let registry = BackgroundTaskRegistry::new();
    let alice = unique_user("alice");
    let bob = unique_user("bob");

    let mut alice_events = registry.subscribe(&alice);

    let observed_cancel = Arc::new(AtomicBool::new(false));
    let observed_for_task = Arc::clone(&observed_cancel);
    let release = Arc::new(Notify::new());
    let release_for_task = Arc::clone(&release);

    let task_id = registry.spawn(
        alice.clone(),
        conv_kind("c"),
        "alice's".into(),
        move |ctx| async move {
            tokio::select! {
                _ = ctx.token.cancelled() => {
                    observed_for_task.store(true, Ordering::SeqCst);
                    Err(anyhow::anyhow!("cancelled"))
                }
                _ = release_for_task.notified() => Ok(()),
            }
        },
    );

    // Bob tries to cancel Alice's task — must be NotFound, must not
    // leak existence, must not actually fire the token.
    let err = registry
        .cancel(&bob, &task_id)
        .expect_err("cross-user cancel must error");
    assert!(matches!(err, TaskError::NotFound));

    // Alice's task continued — let it complete normally.
    release.notify_one();
    let (status, _) = wait_for_completion(&mut alice_events, &task_id).await;
    assert_eq!(status, api::TaskStatus::Completed);
    assert!(
        !observed_cancel.load(Ordering::SeqCst),
        "token was tripped by cross-user cancel"
    );
}

// ----------------------------------------------------------------------
// 6. Log ring drops oldest entries when full
// ----------------------------------------------------------------------

#[tokio::test]
async fn registry_log_ring_drops_oldest_when_bounded() {
    let registry = BackgroundTaskRegistry::with_config(RegistryConfig {
        log_ring_capacity: 1000,
        ..RegistryConfig::default()
    });
    let user = unique_user("alice");

    let release = Arc::new(Notify::new());
    let release_for_task = Arc::clone(&release);

    let task_id = registry.spawn(
        user.clone(),
        conv_kind("c"),
        "logger".into(),
        move |ctx| async move {
            for i in 0..1500 {
                ctx.logs.append(
                    api::LogLevel::Info,
                    api::LogCategory::Status,
                    format!("entry-{i}"),
                    None,
                );
            }
            release_for_task.notified().await;
            Ok(())
        },
    );

    // Wait until the ring is full (the loop ran past 1000 entries).
    wait_until(
        || {
            let (entries, _) = registry
                .logs(&user, &task_id, 0, 5000)
                .expect("task exists");
            entries.len() >= 1000
        },
        "ring saturated",
    )
    .await;

    let (entries, next_seq) = registry.logs(&user, &task_id, 0, 5000).expect("task");
    assert_eq!(
        entries.len(),
        1000,
        "ring buffer must cap at config size, got {}",
        entries.len()
    );
    // Sequence numbers are monotonic per task; the most recent 1000 are
    // retained and `next_seq` points one past the last retained seq.
    let last_seq = entries.last().unwrap().seq;
    let first_seq = entries.first().unwrap().seq;
    assert_eq!(
        last_seq - first_seq + 1,
        1000,
        "retained range must be exactly the ring capacity"
    );
    assert_eq!(next_seq, last_seq + 1);

    // Resuming from a known seq must skip entries already seen.
    let resume_from = last_seq - 9;
    let (resumed, next_resumed) = registry
        .logs(&user, &task_id, resume_from, 100)
        .expect("task");
    assert_eq!(resumed.len(), 9);
    assert_eq!(resumed.first().unwrap().seq, resume_from + 1);
    assert_eq!(next_resumed, last_seq + 1);

    release.notify_one();
    registry.wait(&task_id).await;
}

// ----------------------------------------------------------------------
// 7. subscribe receives TaskStarted + TaskCompleted
// ----------------------------------------------------------------------

#[tokio::test]
async fn registry_subscribe_receives_task_events() {
    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("alice");

    let mut events = registry.subscribe(&user);

    let task_id = registry.spawn(
        user.clone(),
        conv_kind("c"),
        "subscribe".into(),
        |_ctx| async move { Ok(()) },
    );

    let mut saw_started = false;
    let mut saw_completed = false;
    while let Ok(ev) = timeout(Duration::from_secs(2), events.recv()).await {
        let ev = ev.expect("event ok");
        match ev {
            api::Event::TaskStarted { task } if task.id == task_id => saw_started = true,
            api::Event::TaskCompleted { id, status, .. } if id == task_id.0 => {
                assert_eq!(status, api::TaskStatus::Completed);
                saw_completed = true;
            }
            _ => {}
        }
        if saw_started && saw_completed {
            break;
        }
    }
    assert!(saw_started, "did not receive TaskStarted");
    assert!(saw_completed, "did not receive TaskCompleted");
}

// ----------------------------------------------------------------------
// 8. Foreground send-message registers a Conversation task
// ----------------------------------------------------------------------

mod foreground_send {
    use super::*;
    use desktop_assistant_application::{
        AssistantApiHandler, DefaultAssistantApiHandler, EventSink, RequestContext,
    };
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::domain::{
        Conversation, ConversationId, ConversationSummary, Message, Role,
    };
    use desktop_assistant_core::ports::inbound::{
        AssistantService, BackendTasksSettingsView, ConnectionConfigPayload, ConnectionsService,
        ConnectorDefaultsView, ConversationService, DatabaseSettingsView, EmbeddingsSettingsView,
        KnowledgeService, LlmSettingsView, ModelListing as CoreModelListing,
        PersistenceSettingsView, PromptDispatchOutcome, PromptSelectionOverride, PurposeConfigPayload,
        PurposeKind, PurposesView as CorePurposesView, SettingsService, WsAuthSettingsView,
    };
    use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};

    struct FakeKnowledge;
    impl KnowledgeService for FakeKnowledge {
        async fn list_entries(
            &self,
            _limit: usize,
            _offset: usize,
            _tag_filter: Option<Vec<String>>,
        ) -> Result<Vec<desktop_assistant_core::domain::KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }
        async fn get_entry(
            &self,
            _id: String,
        ) -> Result<Option<desktop_assistant_core::domain::KnowledgeEntry>, CoreError> {
            Ok(None)
        }
        async fn search_entries(
            &self,
            _query: String,
            _tag_filter: Option<Vec<String>>,
            _limit: usize,
        ) -> Result<Vec<desktop_assistant_core::domain::KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }
        async fn create_entry(
            &self,
            content: String,
            tags: Vec<String>,
            metadata: serde_json::Value,
        ) -> Result<desktop_assistant_core::domain::KnowledgeEntry, CoreError> {
            let mut e = desktop_assistant_core::domain::KnowledgeEntry::new("kb", content, tags);
            e.metadata = metadata;
            Ok(e)
        }
        async fn update_entry(
            &self,
            id: String,
            content: String,
            tags: Vec<String>,
            metadata: serde_json::Value,
        ) -> Result<desktop_assistant_core::domain::KnowledgeEntry, CoreError> {
            let mut e = desktop_assistant_core::domain::KnowledgeEntry::new(id, content, tags);
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
            _config: ConnectionConfigPayload,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn update_connection(
            &self,
            _id: String,
            _config: ConnectionConfigPayload,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn delete_connection(&self, _id: String, _force: bool) -> Result<(), CoreError> {
            Ok(())
        }
        async fn list_available_models(
            &self,
            _connection_id: Option<String>,
            _refresh: bool,
        ) -> Result<Vec<CoreModelListing>, CoreError> {
            Ok(vec![])
        }
        async fn get_purposes(&self) -> Result<CorePurposesView, CoreError> {
            Ok(CorePurposesView::default())
        }
        async fn set_purpose(
            &self,
            _purpose: PurposeKind,
            _config: PurposeConfigPayload,
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
            _connector: String,
            _model: Option<String>,
            _base_url: Option<String>,
            _temperature: Option<f64>,
            _top_p: Option<f64>,
            _max_tokens: Option<u32>,
            _hosted_tool_search: Option<bool>,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_api_key(&self, _api_key: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn generate_ws_jwt(&self, _subject: Option<String>) -> Result<String, CoreError> {
            Ok("jwt".into())
        }
        async fn validate_ws_jwt(&self, _token: String) -> Result<bool, CoreError> {
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
            _connector: Option<String>,
            _model: Option<String>,
            _base_url: Option<String>,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_connector_defaults(
            &self,
            _connector: String,
        ) -> Result<ConnectorDefaultsView, CoreError> {
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
            _enabled: bool,
            _remote_url: Option<String>,
            _remote_name: Option<String>,
            _push_on_update: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn get_database_settings(&self) -> Result<DatabaseSettingsView, CoreError> {
            Ok(DatabaseSettingsView {
                url: String::new(),
                max_connections: 5,
            })
        }
        async fn set_database_settings(
            &self,
            _url: Option<String>,
            _max_connections: u32,
        ) -> Result<(), CoreError> {
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
            _llm_connector: Option<String>,
            _llm_model: Option<String>,
            _llm_base_url: Option<String>,
            _dreaming_enabled: bool,
            _dreaming_interval_secs: u64,
            _archive_after_days: u32,
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
            _name: String,
            _command: String,
            _args: Vec<String>,
            _namespace: Option<String>,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn remove_mcp_server(&self, _name: String) -> Result<(), CoreError> {
            Ok(())
        }
        async fn set_mcp_server_enabled(
            &self,
            _name: String,
            _enabled: bool,
        ) -> Result<(), CoreError> {
            Ok(())
        }
        async fn mcp_server_action(
            &self,
            _action: String,
            _server: Option<String>,
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
            _methods: Vec<String>,
            _oidc_issuer: String,
            _oidc_auth_endpoint: String,
            _oidc_token_endpoint: String,
            _oidc_client_id: String,
            _oidc_scopes: String,
        ) -> Result<(), CoreError> {
            Ok(())
        }
    }

    /// A conversation service whose `send_prompt_with_override` blocks
    /// on a `Notify` so the test can observe the task being `Running`
    /// before letting it complete. Also accepts a cancellation token so
    /// the cancellation-via-registry test can prove the underlying LLM
    /// call is interrupted.
    pub struct ControllableConversations {
        pub release: Arc<Notify>,
        pub cancelled_flag: Arc<AtomicBool>,
        pub send_count: Arc<AtomicU32>,
    }

    impl ControllableConversations {
        pub fn new(release: Arc<Notify>) -> Self {
            Self {
                release,
                cancelled_flag: Arc::new(AtomicBool::new(false)),
                send_count: Arc::new(AtomicU32::new(0)),
            }
        }
    }

    impl ConversationService for ControllableConversations {
        async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c1", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            let mut c = Conversation::new(id.as_str(), "t");
            c.messages.push(Message::new(Role::User, "hi"));
            Ok(c)
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
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            self.send_count.fetch_add(1, Ordering::SeqCst);
            on_chunk("hi".into());
            self.release.notified().await;
            Ok("hi".into())
        }
        async fn send_prompt_with_override(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            _override_selection: Option<PromptSelectionOverride>,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
            cancellation: tokio_util::sync::CancellationToken,
        ) -> Result<PromptDispatchOutcome, CoreError> {
            self.send_count.fetch_add(1, Ordering::SeqCst);
            on_chunk("hi".into());
            tokio::select! {
                _ = cancellation.cancelled() => {
                    self.cancelled_flag.store(true, Ordering::SeqCst);
                    Err(CoreError::Cancelled)
                }
                _ = self.release.notified() => Ok(PromptDispatchOutcome {
                    response: "hi".into(),
                    warnings: Vec::new(),
                }),
            }
        }
    }

    struct CollectSink(tokio::sync::Mutex<Vec<api::Event>>);
    #[async_trait::async_trait]
    impl EventSink for CollectSink {
        async fn emit(&self, event: api::Event) -> bool {
            self.0.lock().await.push(event);
            true
        }
    }

    fn make_handler_with_registry(
        convs: Arc<ControllableConversations>,
        registry: Arc<BackgroundTaskRegistry>,
    ) -> DefaultAssistantApiHandler<
        FakeAssistant,
        ControllableConversations,
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

    #[tokio::test]
    async fn foreground_send_message_now_registers_a_conversation_task() {
        let registry = Arc::new(BackgroundTaskRegistry::new());
        let release = Arc::new(Notify::new());
        let convs = Arc::new(ControllableConversations::new(Arc::clone(&release)));
        let handler = Arc::new(make_handler_with_registry(
            Arc::clone(&convs),
            Arc::clone(&registry),
        ));

        let alice = unique_user("alice");
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        let handler_for_task = Arc::clone(&handler);
        let alice_for_task = alice.clone();
        let sink_for_task = Arc::clone(&sink);
        let join = tokio::spawn(async move {
            handler_for_task
                .handle_send_message_with_override_for(
                    RequestContext::for_user(alice_for_task),
                    "conv-9".into(),
                    "hi".into(),
                    None,
                    "req-1".into(),
                    sink_for_task,
                )
                .await
                .expect("send ok");
        });

        // Eventually a Conversation task exists for alice.
        let task_id = {
            let registry = Arc::clone(&registry);
            let alice = alice.clone();
            let mut found = None;
            for _ in 0..200 {
                let listing = registry.list(&alice, true, None);
                if let Some(view) = listing.into_iter().find(|v| {
                    matches!(&v.kind, api::TaskKind::Conversation { conversation_id }
                        if conversation_id == "conv-9")
                }) {
                    found = Some(view);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let view = found.expect("conversation task registered");
            assert_eq!(view.status, api::TaskStatus::Running);
            view.id
        };

        // Bob doesn't see Alice's task.
        let bob = unique_user("bob");
        assert!(registry.list(&bob, true, None).is_empty());
        assert!(registry.get(&bob, &task_id).is_none());

        // Let the underlying call finish. After completion the task is
        // evicted from the registry — `list` (either flag) returns it
        // no longer, and `get` is None.
        let mut alice_events = registry.subscribe(&alice);
        release.notify_one();
        join.await.expect("join");
        let (status, _) = wait_for_completion(&mut alice_events, &task_id).await;
        assert_eq!(status, api::TaskStatus::Completed);
        for include_finished in [false, true] {
            let listing = registry.list(&alice, include_finished, None);
            assert!(
                listing.iter().all(|v| v.id != task_id),
                "evicted task surfaced with include_finished={include_finished}: {listing:?}"
            );
        }
        assert!(registry.get(&alice, &task_id).is_none());
    }

    #[tokio::test]
    async fn cancellation_via_registry_aborts_underlying_llm_call() {
        let registry = Arc::new(BackgroundTaskRegistry::new());
        let release = Arc::new(Notify::new());
        let convs = Arc::new(ControllableConversations::new(Arc::clone(&release)));
        let handler = Arc::new(make_handler_with_registry(
            Arc::clone(&convs),
            Arc::clone(&registry),
        ));

        let alice = unique_user("alice");
        let sink = Arc::new(CollectSink(tokio::sync::Mutex::new(vec![])));

        let handler_for_task = Arc::clone(&handler);
        let alice_for_task = alice.clone();
        let sink_for_task = Arc::clone(&sink);
        let join = tokio::spawn(async move {
            let _ = handler_for_task
                .handle_send_message_with_override_for(
                    RequestContext::for_user(alice_for_task),
                    "conv-x".into(),
                    "hi".into(),
                    None,
                    "req-1".into(),
                    sink_for_task,
                )
                .await;
        });

        // Wait for the task to appear and be running.
        let task_id = {
            let registry = Arc::clone(&registry);
            let alice = alice.clone();
            let mut found = None;
            for _ in 0..200 {
                let listing = registry.list(&alice, true, None);
                if let Some(view) = listing.into_iter().next()
                    && view.status == api::TaskStatus::Running
                {
                    found = Some(view.id);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            found.expect("running task")
        };

        // Cancel via the registry — must trip the token and end the call.
        let mut alice_events = registry.subscribe(&alice);
        registry.cancel(&alice, &task_id).expect("cancel ok");
        join.await.expect("join");

        assert!(
            convs.cancelled_flag.load(Ordering::SeqCst),
            "the underlying LLM call did not observe cancellation"
        );

        // Terminal entries are evicted; the broadcast event is the
        // authoritative status indicator.
        let (status, _) = wait_for_completion(&mut alice_events, &task_id).await;
        assert_eq!(status, api::TaskStatus::Cancelled);
        assert!(registry.get(&alice, &task_id).is_none());
    }
}

// ----------------------------------------------------------------------
// 9. Slow subscriber does not block other subscribers
// ----------------------------------------------------------------------

#[tokio::test]
async fn slow_subscriber_does_not_block_other_subscribers() {
    // Configure a small broadcast capacity so we can deliberately overflow.
    let registry = BackgroundTaskRegistry::with_config(RegistryConfig {
        broadcast_capacity: 4,
        ..RegistryConfig::default()
    });
    let user = unique_user("alice");

    // First subscriber: never reads. Should not block the second.
    let _slow = registry.subscribe(&user);
    let mut fast = registry.subscribe(&user);

    // Emit a burst of events that exceeds the broadcast capacity.
    let mut ids = Vec::new();
    for _ in 0..20 {
        let id = registry.spawn(
            user.clone(),
            conv_kind("c"),
            "burst".into(),
            |_ctx| async move { Ok(()) },
        );
        ids.push(id);
    }
    // Let all of them complete so events flush through.
    for id in &ids {
        registry.wait(id).await;
    }

    // The fast subscriber must still receive at least one event despite
    // the slow subscriber sitting on its receiver. We accept either a
    // direct event or a lagged error followed by a real event — both are
    // valid broadcast outcomes per `tokio::sync::broadcast::Receiver`.
    let mut saw_any = false;
    for _ in 0..32 {
        match timeout(Duration::from_millis(200), fast.recv()).await {
            Ok(Ok(_)) => {
                saw_any = true;
                break;
            }
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(_)) => break,
            Err(_) => break,
        }
    }
    assert!(
        saw_any,
        "fast subscriber starved by slow subscriber — broadcast semantics broken"
    );
}

// ----------------------------------------------------------------------
// 10. Log sink emits TaskLogAppended events
// ----------------------------------------------------------------------

#[tokio::test]
async fn log_sink_emits_log_appended_event() {
    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("alice");

    let mut events = registry.subscribe(&user);
    let release = Arc::new(Notify::new());
    let release_for_task = Arc::clone(&release);

    let task_id = registry.spawn(
        user.clone(),
        conv_kind("c"),
        "logger".into(),
        move |ctx| async move {
            ctx.logs.append(
                api::LogLevel::Warn,
                api::LogCategory::ToolResult,
                "tool finished".into(),
                Some(serde_json::json!({"ok": true})),
            );
            release_for_task.notified().await;
            Ok(())
        },
    );

    // We're hunting specifically for the user-emitted Warn/ToolResult
    // entry — the registry also emits Lifecycle markers on every spawn
    // and the test should ignore those.
    let mut saw = false;
    for _ in 0..50 {
        match timeout(Duration::from_millis(200), events.recv()).await {
            Ok(Ok(api::Event::TaskLogAppended { id, entry }))
                if id == task_id.0
                    && entry.category == api::LogCategory::ToolResult =>
            {
                assert_eq!(entry.level, api::LogLevel::Warn);
                assert_eq!(entry.message, "tool finished");
                assert_eq!(entry.data, Some(serde_json::json!({"ok": true})));
                saw = true;
                break;
            }
            Ok(Ok(_)) => continue,
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(_)) => break,
            Err(_) => break,
        }
    }
    assert!(saw, "no TaskLogAppended event received");

    release.notify_one();
    registry.wait(&task_id).await;
}

// ----------------------------------------------------------------------
// 11. Concurrent spawns under same user are thread-safe
// ----------------------------------------------------------------------

#[tokio::test]
async fn concurrent_spawns_under_same_user_are_thread_safe() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let user = unique_user("alice");

    // Each task gets its own Notify so we can keep them all in-flight
    // simultaneously for the visibility check (terminal entries are now
    // evicted, so a task that returns Ok immediately disappears before
    // we can list it).
    let mut releases: Vec<Arc<Notify>> = Vec::with_capacity(100);
    let mut handles = Vec::new();
    for _ in 0..100 {
        let release = Arc::new(Notify::new());
        releases.push(Arc::clone(&release));
        let registry = Arc::clone(&registry);
        let user = user.clone();
        handles.push(tokio::spawn(async move {
            registry.spawn(user, conv_kind("c"), "concurrent".into(), move |_| async move {
                release.notified().await;
                Ok(())
            })
        }));
    }

    let mut ids: HashSet<api::TaskId> = HashSet::new();
    for h in handles {
        let id = h.await.expect("join");
        assert!(ids.insert(id), "duplicate id under concurrent spawn");
    }

    // While all 100 are still running, every id is visible from `list`.
    let listing = registry.list(&user, true, None);
    let listed: HashSet<_> = listing.iter().map(|v| v.id.clone()).collect();
    assert_eq!(listed, ids);

    // Release everything so the runtime tasks exit cleanly.
    for release in &releases {
        release.notify_one();
    }
    for id in &ids {
        registry.wait(id).await;
    }
}

// ----------------------------------------------------------------------
// 12. TaskContext can update progress_hint
// ----------------------------------------------------------------------

#[tokio::test]
async fn task_view_progress_hint_updates_via_context() {
    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("alice");

    let release = Arc::new(Notify::new());
    let release_for_task = Arc::clone(&release);

    let task_id = registry.spawn(
        user.clone(),
        conv_kind("c"),
        "progresser".into(),
        move |ctx| async move {
            ctx.set_progress_hint(Some("step 2/4".into()));
            release_for_task.notified().await;
            Ok(())
        },
    );

    // Wait for the hint to land.
    wait_until(
        || {
            registry
                .get(&user, &task_id)
                .map(|v| v.progress_hint.as_deref() == Some("step 2/4"))
                .unwrap_or(false)
        },
        "progress hint visible",
    )
    .await;

    release.notify_one();
    registry.wait(&task_id).await;
}

// ----------------------------------------------------------------------
// 13. Subagent / Standalone kinds compile and are handled like any task
// ----------------------------------------------------------------------

#[tokio::test]
async fn business_outcome_subagent_or_standalone_kinds_compile_and_are_listable_while_in_flight() {
    let registry = BackgroundTaskRegistry::new();
    let user = unique_user("alice");

    // Subagent needs a parent task that's still in-flight so its parent
    // id resolves. Block all three on per-task Notifies; inspect kinds
    // while running, then release so they evict cleanly.
    let parent_release = Arc::new(Notify::new());
    let sub_release = Arc::new(Notify::new());
    let standalone_release = Arc::new(Notify::new());

    let parent_release_for_task = Arc::clone(&parent_release);
    let parent = registry.spawn(
        user.clone(),
        conv_kind("conv-parent"),
        "parent".into(),
        move |_ctx| async move {
            parent_release_for_task.notified().await;
            Ok(())
        },
    );

    let sub_release_for_task = Arc::clone(&sub_release);
    let sub = registry.spawn(
        user.clone(),
        api::TaskKind::Subagent {
            parent_task_id: parent.clone(),
            conversation_id: "conv-child".into(),
            name: "researcher".into(),
        },
        "subagent".into(),
        move |_ctx| async move {
            sub_release_for_task.notified().await;
            Ok(())
        },
    );
    let standalone_release_for_task = Arc::clone(&standalone_release);
    let standalone = registry.spawn(
        user.clone(),
        api::TaskKind::Standalone {
            name: "harvester".into(),
            conversation_id: "conv-standalone".into(),
        },
        "standalone".into(),
        move |_ctx| async move {
            standalone_release_for_task.notified().await;
            Ok(())
        },
    );

    let everything = registry.list(&user, true, None);
    let kinds: Vec<_> = everything.iter().map(|v| v.kind.clone()).collect();
    assert!(kinds
        .iter()
        .any(|k| matches!(k, api::TaskKind::Subagent { name, .. } if name == "researcher")));
    assert!(kinds
        .iter()
        .any(|k| matches!(k, api::TaskKind::Standalone { name, .. } if name == "harvester")));

    let sub_view = registry.get(&user, &sub).expect("present");
    assert_eq!(sub_view.parent, Some(parent.clone()));

    // Drain: release subs/standalone first (parent still keeps them
    // parented in `children` until they evict), then parent.
    sub_release.notify_one();
    standalone_release.notify_one();
    registry.wait(&sub).await;
    registry.wait(&standalone).await;
    parent_release.notify_one();
    registry.wait(&parent).await;
}
