use std::sync::Arc;

use desktop_assistant_core::domain::ConversationId;
use desktop_assistant_core::ports::auth::{current_user_id, with_user_id};
use desktop_assistant_core::ports::inbound::ConversationService;
use tokio::sync::mpsc;
use zbus::object_server::SignalEmitter;
use zbus::{fdo, interface};

use crate::resolve_dbus_user_id;

/// D-Bus adapter for the ConversationService.
///
/// Exposes conversation management and streaming prompt/response
/// over D-Bus signals.
pub struct DbusConversationAdapter<S: ConversationService + 'static> {
    service: Arc<S>,
}

impl<S: ConversationService + 'static> DbusConversationAdapter<S> {
    pub fn new(service: Arc<S>) -> Self {
        Self { service }
    }
}

/// Messages sent from the streaming task to the signal emitter.
pub(crate) enum StreamEvent {
    Chunk(String),
    Complete(String),
    Error(String),
}

/// LLM-call body extracted from `send_prompt` so the spawn-body fix
/// (#156) can be unit-tested without standing up a real D-Bus
/// connection. Production code wraps this future in
/// `with_user_id(user_id, ...)` before handing it to `tokio::spawn` so
/// storage queries inside the LLM turn scope to the caller's user_id
/// instead of the `"default"` sentinel.
///
/// The function is generic over `ConversationService` rather than
/// boxed because the trait uses RPITIT (`impl Future`) which isn't
/// dyn-compatible.
pub(crate) async fn run_send_prompt_llm_task<S>(
    service: Arc<S>,
    conversation_id: String,
    prompt: String,
    tx: mpsc::UnboundedSender<StreamEvent>,
) where
    S: ConversationService + 'static,
{
    let tx_chunk = tx.clone();
    let callback: desktop_assistant_core::ports::llm::ChunkCallback =
        Box::new(move |chunk| tx_chunk.send(StreamEvent::Chunk(chunk)).is_ok());

    let on_status: desktop_assistant_core::ports::llm::StatusCallback = Box::new(|_| {});

    match service
        .send_prompt(
            &ConversationId::from(conversation_id.as_str()),
            prompt,
            callback,
            on_status,
        )
        .await
    {
        Ok(full_response) => {
            let _ = tx.send(StreamEvent::Complete(full_response));
        }
        Err(e) => {
            tracing::error!(
                conversation_id = %conversation_id,
                "I hit an LLM backend error and could not complete this request. Details: {e}"
            );
            let _ = tx.send(StreamEvent::Error(e.to_string()));
        }
    }
}

/// Schedule the LLM-call body on a fresh tokio task.
///
/// `tokio::spawn` does not propagate task-locals, so this helper is
/// the single fix surface for #156: it captures `current_user_id()`
/// at the call site (which the outer D-Bus method has installed via
/// [`with_user_id`]) and re-installs that scope inside the spawned
/// future. Without that wrap, storage queries inside the LLM turn
/// fall through to the `"default"` sentinel and miss rows owned by
/// the real user — the exact failure mode the analogous WS fix in
/// #155 addressed.
pub(crate) fn spawn_send_prompt_llm_task<S>(
    service: Arc<S>,
    conversation_id: String,
    prompt: String,
    tx: mpsc::UnboundedSender<StreamEvent>,
) -> tokio::task::JoinHandle<()>
where
    S: ConversationService + 'static,
{
    let user_id_for_body = current_user_id();
    tokio::spawn(with_user_id(
        user_id_for_body,
        run_send_prompt_llm_task(service, conversation_id, prompt, tx),
    ))
}

#[interface(name = "org.desktopAssistant.Conversations")]
impl<S: ConversationService + 'static> DbusConversationAdapter<S> {
    /// Create a new conversation and return its ID.
    async fn create_conversation(&self, title: &str) -> fdo::Result<String> {
        with_user_id(resolve_dbus_user_id(), async {
            let conv = self
                .service
                .create_conversation(title.to_string())
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))?;
            Ok(conv.id.0)
        })
        .await
    }

    /// List conversations as an array of (id, title, message_count, updated_at, archived),
    /// optionally filtered by max age in days (0 means no filtering).
    async fn list_conversations(
        &self,
        max_age_days: i32,
        include_archived: bool,
    ) -> fdo::Result<Vec<(String, String, u32, String, bool)>> {
        with_user_id(resolve_dbus_user_id(), async {
            let max_age = u32::try_from(max_age_days).ok().filter(|days| *days > 0);
            let summaries = self
                .service
                .list_conversations(max_age, include_archived)
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))?;
            Ok(summaries
                .into_iter()
                .map(|s| {
                    (
                        s.id.0,
                        s.title,
                        s.message_count as u32,
                        s.updated_at,
                        s.archived,
                    )
                })
                .collect())
        })
        .await
    }

    /// Archive a conversation by ID.
    async fn archive_conversation(&self, id: &str) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .archive_conversation(&ConversationId::from(id))
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))
        })
        .await
    }

    /// Unarchive a conversation by ID.
    async fn unarchive_conversation(&self, id: &str) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .unarchive_conversation(&ConversationId::from(id))
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))
        })
        .await
    }

    /// Get a conversation by ID, returns (id, title, messages) where
    /// messages is an array of (role, content).
    async fn get_conversation(
        &self,
        id: &str,
    ) -> fdo::Result<(String, String, Vec<(String, String)>)> {
        with_user_id(resolve_dbus_user_id(), async {
            let conv = self
                .service
                .get_conversation(&ConversationId::from(id))
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))?;
            let messages: Vec<(String, String)> = conv
                .messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        desktop_assistant_core::domain::Role::User => "user",
                        desktop_assistant_core::domain::Role::Assistant => "assistant",
                        desktop_assistant_core::domain::Role::System => "system",
                        desktop_assistant_core::domain::Role::Tool => "tool",
                    };
                    (role.to_string(), m.content.clone())
                })
                .collect();
            Ok((conv.id.0, conv.title, messages))
        })
        .await
    }

    /// Get messages from a conversation with optional pagination and role filtering.
    ///
    /// - `tail`: max messages to return from the *filtered* set (0 = unlimited).
    ///   Ignored when `after_count` >= 0.
    /// - `after_count`: skip the first N raw (pre-filter) messages; -1 means unused.
    /// - `include_roles`: allowlist of roles to return, e.g. `["user", "assistant"]`.
    ///   An empty list disables filtering and returns all roles.
    ///
    /// Returns `(total_raw_count, truncated, messages)`.
    /// `total_raw_count` always reflects the unfiltered length so callers can
    /// use it as the next `after_count` for incremental fetches.
    async fn get_messages(
        &self,
        id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> fdo::Result<(u32, bool, Vec<(String, String)>)> {
        with_user_id(resolve_dbus_user_id(), async {
            let conv = self
                .service
                .get_conversation(&ConversationId::from(id))
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))?;

            let total = conv.messages.len() as u32;

            let all: Vec<(String, String)> = conv
                .messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        desktop_assistant_core::domain::Role::User => "user",
                        desktop_assistant_core::domain::Role::Assistant => "assistant",
                        desktop_assistant_core::domain::Role::System => "system",
                        desktop_assistant_core::domain::Role::Tool => "tool",
                    };
                    (role.to_string(), m.content.clone())
                })
                .collect();

            // Slice by raw position first so after_count is always a stable index.
            let use_after = after_count >= 0;
            let sliced: Vec<(String, String)> = if use_after {
                let start = (after_count as usize).min(all.len());
                all[start..].to_vec()
            } else {
                all
            };

            // Apply role allowlist (empty = no filtering).
            let filtered: Vec<(String, String)> = sliced
                .into_iter()
                .filter(|(role, _)| include_roles.is_empty() || include_roles.contains(role))
                .collect();

            // Apply tail limit to the filtered set (tail mode only).
            let (truncated, messages) = if !use_after && tail > 0 && filtered.len() > tail as usize
            {
                let start = filtered.len() - tail as usize;
                (true, filtered[start..].to_vec())
            } else {
                (false, filtered)
            };

            Ok((total, truncated, messages))
        })
        .await
    }

    /// Delete a conversation by ID.
    async fn delete_conversation(&self, id: &str) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .delete_conversation(&ConversationId::from(id))
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))
        })
        .await
    }

    /// Rename a conversation.
    async fn rename_conversation(&self, id: &str, title: &str) -> fdo::Result<()> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .rename_conversation(&ConversationId::from(id), title.to_string())
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))
        })
        .await
    }

    /// Delete every conversation and return how many were removed.
    async fn clear_all_history(&self) -> fdo::Result<u32> {
        with_user_id(resolve_dbus_user_id(), async {
            self.service
                .clear_all_history()
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))
        })
        .await
    }

    /// Send a prompt and stream the response via signals.
    /// Returns a request_id that correlates the signals.
    async fn send_prompt(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        conversation_id: &str,
        prompt: &str,
    ) -> fdo::Result<String> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let conv_id = conversation_id.to_string();
        let prompt = prompt.to_string();
        let service = Arc::clone(&self.service);
        let req_id = request_id.clone();

        let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();

        // Install the D-Bus user_id scope before spawning the LLM
        // task so `spawn_send_prompt_llm_task` captures the right id
        // for the in-spawn `with_user_id` wrap (#156).
        let llm_conv_id = conv_id.clone();
        with_user_id(resolve_dbus_user_id(), async {
            drop(spawn_send_prompt_llm_task(service, llm_conv_id, prompt, tx));
        })
        .await;

        // Spawn the signal emitter task. Signal emission does not
        // touch per-user storage, so it doesn't need the scope.
        let emitter = emitter.to_owned();
        let signal_conv_id = conv_id.clone();
        let signal_req_id = req_id.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Chunk(chunk) => {
                        if let Err(e) =
                            Self::response_chunk(&emitter, &signal_conv_id, &signal_req_id, &chunk)
                                .await
                        {
                            tracing::error!("failed to emit ResponseChunk signal: {e}");
                        }
                    }
                    StreamEvent::Complete(full) => {
                        if let Err(e) = Self::response_complete(
                            &emitter,
                            &signal_conv_id,
                            &signal_req_id,
                            &full,
                        )
                        .await
                        {
                            tracing::error!("failed to emit ResponseComplete signal: {e}");
                        }
                        break;
                    }
                    StreamEvent::Error(err) => {
                        if let Err(e) =
                            Self::response_error(&emitter, &signal_conv_id, &signal_req_id, &err)
                                .await
                        {
                            tracing::error!("failed to emit ResponseError signal: {e}");
                        }
                        break;
                    }
                }
            }
        });

        Ok(request_id)
    }

    /// Signal emitted for each chunk of a streaming response.
    #[zbus(signal)]
    async fn response_chunk(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        chunk: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted when a streaming response is complete.
    #[zbus(signal)]
    async fn response_complete(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        full_response: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted when a streaming response encounters an error.
    #[zbus(signal)]
    async fn response_error(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        error: &str,
    ) -> zbus::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::CoreError;
    use desktop_assistant_core::domain::{Conversation, ConversationSummary, Message, Role};
    use desktop_assistant_core::ports::auth::{UserId, current_user_id, with_user_id};
    use desktop_assistant_core::ports::llm::{ChunkCallback, StatusCallback};
    use std::sync::Mutex;

    /// Recording fake that captures `current_user_id()` at every inbound
    /// service call so #156 acceptance tests can assert the D-Bus
    /// boundary installed a `with_user_id` scope before invoking the
    /// service. Mirrors the `RecordingConversations` fake added in #155.
    struct RecordingConversationService {
        seen: Mutex<Vec<String>>,
    }

    impl RecordingConversationService {
        fn new() -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
            }
        }

        fn record(&self) {
            self.seen
                .lock()
                .unwrap()
                .push(current_user_id().as_str().to_string());
        }

        fn observed(&self) -> Vec<String> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl ConversationService for RecordingConversationService {
        async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
            self.record();
            Ok(Conversation::new("rec-id", title))
        }
        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            self.record();
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            self.record();
            Ok(Conversation::new(id.as_str(), "rec"))
        }
        async fn delete_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            self.record();
            Ok(())
        }
        async fn rename_conversation(
            &self,
            _id: &ConversationId,
            _title: String,
        ) -> Result<(), CoreError> {
            self.record();
            Ok(())
        }
        async fn archive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            self.record();
            Ok(())
        }
        async fn unarchive_conversation(&self, _id: &ConversationId) -> Result<(), CoreError> {
            self.record();
            Ok(())
        }
        async fn clear_all_history(&self) -> Result<u32, CoreError> {
            self.record();
            Ok(0)
        }
        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            self.record();
            on_chunk("ok".to_string());
            Ok("ok".to_string())
        }
    }

    /// `resolve_dbus_user_id` reads the local OS user from `$USER`
    /// because the D-Bus session bus is local-only (no JWT to extract
    /// from). When `$USER` is unset, the helper falls through to
    /// `UserId::default()` — the schema sentinel `"default"` — which
    /// matches the single-tenant fallback used everywhere else in the
    /// codebase. Documenting this trust boundary in a test pins the
    /// policy so a reviewer can challenge it.
    #[test]
    fn resolve_dbus_user_id_uses_user_env_var() {
        let _guard = crate::testing::UserEnvGuard::set("alice-from-env");
        let resolved = crate::resolve_dbus_user_id();
        assert_eq!(resolved, UserId::new("alice-from-env"));
    }

    #[test]
    fn resolve_dbus_user_id_falls_back_to_default_when_user_env_missing() {
        let _guard = crate::testing::UserEnvGuard::unset();
        let resolved = crate::resolve_dbus_user_id();
        assert_eq!(resolved, UserId::default());
        assert_eq!(resolved.as_str(), "default");
    }

    /// Issue #156: every D-Bus method must install `with_user_id` at
    /// the dispatch boundary so per-user-scoped storage queries see
    /// the local OS user instead of the `"default"` sentinel. Without
    /// the wrap, `current_user_id()` returns `"default"` and rows owned
    /// by the local user (after the multi-tenant migration) are
    /// invisible to adele-kde. This test fixes `$USER` to a known
    /// value, calls each non-streaming D-Bus method, and asserts the
    /// inbound service observed the same value at SQL composition
    /// time.
    #[tokio::test]
    async fn dbus_methods_install_user_id_scope_at_method_entry() {
        let service = Arc::new(RecordingConversationService::new());
        let adapter = DbusConversationAdapter::new(Arc::clone(&service));

        let _guard = crate::testing::UserEnvGuard::set("alice-dbus");

        // Exercise every non-streaming D-Bus interface method. These
        // are the ones called by adele-kde for conversation listing,
        // titling, archival, and deletion.
        let _id = adapter.create_conversation("Test").await.unwrap();
        let _list = adapter.list_conversations(0, false).await.unwrap();
        let _conv = adapter.get_conversation("c").await.unwrap();
        let _msgs = adapter.get_messages("c", 0, -1, vec![]).await.unwrap();
        adapter.archive_conversation("c").await.unwrap();
        adapter.unarchive_conversation("c").await.unwrap();
        adapter.rename_conversation("c", "x").await.unwrap();
        adapter.delete_conversation("c").await.unwrap();
        let _count = adapter.clear_all_history().await.unwrap();

        let observed = service.observed();
        assert!(
            !observed.is_empty(),
            "expected the inbound service to be called at least once"
        );
        for seen in observed {
            assert_eq!(
                seen, "alice-dbus",
                "every D-Bus method must scope storage to the resolved local user, not the default sentinel"
            );
        }
    }

    /// Issue #156 spawn-body fix (mirrors the WS #155 fix). The
    /// `send_prompt` body spawns a `tokio::spawn` for the LLM call.
    /// `tokio::spawn` does not propagate task-locals, so the spawn
    /// body must re-install `with_user_id(user_id, ...)` before
    /// invoking the service. Without that wrap, even if the outer
    /// D-Bus method installed the scope, the LLM-call body would see
    /// `"default"` and storage queries inside the turn would scope
    /// wrong.
    ///
    /// We test the extracted helper directly because constructing a
    /// `SignalEmitter` requires a real D-Bus connection. The test
    /// installs `with_user_id` around the *outer* call (simulating the
    /// method-entry wrap) and then dispatches the helper through
    /// `tokio::spawn`. The fix wraps the inner future in
    /// `with_user_id`; without that wrap the spawned body sees
    /// `"default"` and this test fails the assertion.
    #[tokio::test]
    async fn send_prompt_spawn_body_propagates_user_id() {
        let service = Arc::new(RecordingConversationService::new());
        let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();
        let svc_for_task = Arc::clone(&service);

        // Simulate the production call site: the outer D-Bus method
        // entry installs `with_user_id`, then calls a function that
        // internally does `tokio::spawn(...)`. The spawned body must
        // observe the same user_id — which only happens if the body
        // itself is wrapped in `with_user_id`. The outer
        // `with_user_id` returns the spawned task's `JoinHandle` so
        // the test can drive it to completion. The lint about an
        // async block yielding an awaitable is intentional here: the
        // outer scope yields, the inner JoinHandle is driven
        // separately.
        let user = UserId::new("alice-spawn");
        #[allow(clippy::async_yields_async)]
        let join = with_user_id(user.clone(), async move {
            crate::conversation::spawn_send_prompt_llm_task(
                svc_for_task,
                "conv-x".to_string(),
                "hello".to_string(),
                tx,
            )
        })
        .await;
        join.await.expect("spawn join");

        // Drain at least the Complete event so we know the body ran.
        let mut saw_complete = false;
        while let Some(ev) = rx.recv().await {
            if matches!(ev, StreamEvent::Complete(_)) {
                saw_complete = true;
            }
        }
        assert!(saw_complete, "spawn body must reach completion");

        let observed = service.observed();
        assert_eq!(
            observed,
            vec!["alice-spawn".to_string()],
            "spawned LLM-call body must observe the caller's user_id, not the default sentinel"
        );
    }

    struct FakeConversationService;

    impl ConversationService for FakeConversationService {
        async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("test-id", title))
        }

        async fn list_conversations(
            &self,
            _max_age_days: Option<u32>,
            _include_archived: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![ConversationSummary {
                id: ConversationId::from("test-id"),
                title: "Test".to_string(),
                created_at: "2026-02-16 00:00:00".to_string(),
                updated_at: "2026-02-16 00:00:00".to_string(),
                message_count: 0,
                archived: false,
            }])
        }

        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            let mut conv = Conversation::new(id.as_str(), "Test");
            conv.messages.push(Message::new(Role::User, "hi"));
            Ok(conv)
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
            Ok(1)
        }

        async fn send_prompt(
            &self,
            _conversation_id: &ConversationId,
            _prompt: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
        ) -> Result<String, CoreError> {
            on_chunk("hello ".to_string());
            on_chunk("world".to_string());
            Ok("hello world".to_string())
        }
    }

    #[test]
    fn adapter_construction() {
        let service = Arc::new(FakeConversationService);
        let _adapter = DbusConversationAdapter::new(service);
    }

    #[tokio::test]
    async fn adapter_create_conversation() {
        let service = Arc::new(FakeConversationService);
        let adapter = DbusConversationAdapter::new(service);
        // We can't test D-Bus methods directly without a bus connection,
        // but we can verify the service is accessible.
        let conv = adapter
            .service
            .create_conversation("Test".into())
            .await
            .unwrap();
        assert_eq!(conv.id.as_str(), "test-id");
    }

    #[tokio::test]
    async fn adapter_list_conversations() {
        let service = Arc::new(FakeConversationService);
        let adapter = DbusConversationAdapter::new(service);
        let summaries = adapter
            .service
            .list_conversations(None, false)
            .await
            .unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].title, "Test");
    }

    #[tokio::test]
    async fn adapter_get_conversation() {
        let service = Arc::new(FakeConversationService);
        let adapter = DbusConversationAdapter::new(service);
        let conv = adapter
            .service
            .get_conversation(&ConversationId::from("test-id"))
            .await
            .unwrap();
        assert_eq!(conv.messages.len(), 1);
    }

    #[tokio::test]
    async fn get_messages_empty_include_returns_all() {
        let service = Arc::new(FakeConversationService);
        let adapter = DbusConversationAdapter::new(Arc::clone(&service));
        // Use the service directly since we can't call D-Bus methods in unit tests.
        let conv = adapter
            .service
            .get_conversation(&ConversationId::from("test-id"))
            .await
            .unwrap();
        // Empty include_roles → no filtering, all roles returned.
        assert_eq!(conv.messages.len(), 1);
        assert_eq!(conv.messages[0].role, Role::User);
    }

    #[tokio::test]
    async fn get_messages_include_filters_to_allowlist() {
        use desktop_assistant_core::domain::Message;
        struct MultiRoleService;
        impl ConversationService for MultiRoleService {
            async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
                Ok(Conversation::new("id", title))
            }
            async fn list_conversations(
                &self,
                _: Option<u32>,
                _: bool,
            ) -> Result<Vec<ConversationSummary>, CoreError> {
                Ok(vec![])
            }
            async fn get_conversation(
                &self,
                id: &ConversationId,
            ) -> Result<Conversation, CoreError> {
                let mut conv = Conversation::new(id.as_str(), "T");
                conv.messages.push(Message::new(Role::User, "hello"));
                conv.messages
                    .push(Message::new(Role::Assistant, "response"));
                conv.messages
                    .push(Message::tool_result("c1", "tool output"));
                conv.messages.push(Message::new(Role::User, "follow-up"));
                Ok(conv)
            }
            async fn delete_conversation(&self, _: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn rename_conversation(
                &self,
                _: &ConversationId,
                _: String,
            ) -> Result<(), CoreError> {
                Ok(())
            }
            async fn archive_conversation(&self, _: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn unarchive_conversation(&self, _: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn clear_all_history(&self) -> Result<u32, CoreError> {
                Ok(0)
            }
            async fn send_prompt(
                &self,
                _: &ConversationId,
                _: String,
                _: ChunkCallback,
                _: StatusCallback,
            ) -> Result<String, CoreError> {
                Ok(String::new())
            }
        }

        let adapter = DbusConversationAdapter::new(Arc::new(MultiRoleService));
        let conv = adapter
            .service
            .get_conversation(&ConversationId::from("id"))
            .await
            .unwrap();

        // Raw: 4 messages (user, assistant, tool, user)
        assert_eq!(conv.messages.len(), 4);

        // Simulate GetMessages with include_roles=["user", "assistant"].
        let total = conv.messages.len() as u32;
        let include = vec!["user".to_string(), "assistant".to_string()];
        let all: Vec<(String, String)> = conv
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                };
                (role.to_string(), m.content.clone())
            })
            .collect();
        let filtered: Vec<_> = all
            .iter()
            .filter(|(r, _)| include.is_empty() || include.contains(r))
            .collect();
        assert_eq!(total, 4);
        assert_eq!(filtered.len(), 3); // user, assistant, user — tool excluded
        assert!(filtered.iter().all(|(r, _)| r != "tool"));
    }

    #[tokio::test]
    async fn get_messages_after_count_slices_raw() {
        use desktop_assistant_core::domain::Message;
        struct SeqService;
        impl ConversationService for SeqService {
            async fn create_conversation(&self, t: String) -> Result<Conversation, CoreError> {
                Ok(Conversation::new("id", t))
            }
            async fn list_conversations(
                &self,
                _: Option<u32>,
                _: bool,
            ) -> Result<Vec<ConversationSummary>, CoreError> {
                Ok(vec![])
            }
            async fn get_conversation(
                &self,
                id: &ConversationId,
            ) -> Result<Conversation, CoreError> {
                let mut conv = Conversation::new(id.as_str(), "T");
                conv.messages.push(Message::new(Role::User, "u1"));
                conv.messages.push(Message::tool_result("c1", "t1"));
                conv.messages.push(Message::new(Role::Assistant, "a1"));
                conv.messages.push(Message::new(Role::User, "u2"));
                Ok(conv)
            }
            async fn delete_conversation(&self, _: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn rename_conversation(
                &self,
                _: &ConversationId,
                _: String,
            ) -> Result<(), CoreError> {
                Ok(())
            }
            async fn archive_conversation(&self, _: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn unarchive_conversation(&self, _: &ConversationId) -> Result<(), CoreError> {
                Ok(())
            }
            async fn clear_all_history(&self) -> Result<u32, CoreError> {
                Ok(0)
            }
            async fn send_prompt(
                &self,
                _: &ConversationId,
                _: String,
                _: ChunkCallback,
                _: StatusCallback,
            ) -> Result<String, CoreError> {
                Ok(String::new())
            }
        }

        let svc = Arc::new(SeqService);
        let conv = svc
            .get_conversation(&ConversationId::from("id"))
            .await
            .unwrap();
        let total = conv.messages.len() as u32; // 4 raw
        let include = vec!["user".to_string(), "assistant".to_string()];
        let all: Vec<(String, String)> = conv
            .messages
            .iter()
            .map(|m| {
                let r = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                };
                (r.to_string(), m.content.clone())
            })
            .collect();

        // after_count=2 -> skip first 2 raw messages (user, tool)
        let sliced: Vec<_> = all[2..].to_vec();
        let filtered: Vec<_> = sliced
            .into_iter()
            .filter(|(r, _)| include.is_empty() || include.contains(r))
            .collect();
        assert_eq!(total, 4);
        // sliced: [assistant, user]; include=[user,assistant] → both pass
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, "assistant");
        assert_eq!(filtered[1].0, "user");
    }
}
