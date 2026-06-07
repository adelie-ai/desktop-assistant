use std::sync::Arc;

use desktop_assistant_core::domain::ConversationId;
use desktop_assistant_core::ports::auth::{current_user_id, with_user_id};
use desktop_assistant_core::ports::inbound::ConversationService;
use desktop_assistant_core::prompts::{PersonalityLevel, PersonalityOverride};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
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

    /// Shared body for `SendPrompt` / `SendPromptWithSystemRefinement`:
    /// spawn the LLM-call task (carrying `system_refinement`; empty =
    /// none) under the resolved user_id scope, spawn the signal-emitter
    /// task, and return the correlation `request_id`. Keeping this in one
    /// place guarantees the two D-Bus methods stay identical apart from
    /// the refinement.
    async fn start_streaming_send(
        &self,
        emitter: SignalEmitter<'_>,
        conversation_id: &str,
        prompt: &str,
        system_refinement: String,
    ) -> String {
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
            drop(spawn_send_prompt_llm_task(
                service,
                llm_conv_id,
                prompt,
                system_refinement,
                tx,
            ));
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

        request_id
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
    system_refinement: String,
    tx: mpsc::UnboundedSender<StreamEvent>,
) where
    S: ConversationService + 'static,
{
    let tx_chunk = tx.clone();
    let callback: desktop_assistant_core::ports::llm::ChunkCallback =
        Box::new(move |chunk| tx_chunk.send(StreamEvent::Chunk(chunk)).is_ok());

    let on_status: desktop_assistant_core::ports::llm::StatusCallback = Box::new(|_| {});

    // `system_refinement` (empty = none) is a per-request addition to the
    // system prompt for THIS turn only. We route through
    // `send_prompt_with_override` with no model override and install the
    // refinement as a task-local; the core service appends it to the system
    // message for the LLM call but never stores it, so the visible transcript
    // records only the clean `prompt`. A fresh `CancellationToken` preserves
    // pre-existing behaviour (this in-process adapter never cancels).
    match service
        .send_prompt_with_override(
            &ConversationId::from(conversation_id.as_str()),
            prompt,
            None,
            system_refinement,
            callback,
            on_status,
            CancellationToken::new(),
        )
        .await
        .map(|outcome| outcome.response)
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
    system_refinement: String,
    tx: mpsc::UnboundedSender<StreamEvent>,
) -> tokio::task::JoinHandle<()>
where
    S: ConversationService + 'static,
{
    let user_id_for_body = current_user_id();
    tokio::spawn(with_user_id(
        user_id_for_body,
        run_send_prompt_llm_task(service, conversation_id, prompt, system_refinement, tx),
    ))
}

/// D-Bus ordinal contract for one personality-override trait (#227): `-1`
/// means "unset" (fall back to the global config), `0..=4` pins the level
/// (Never=0 … Always=4). Out-of-range positives are rejected. Mirrors the
/// signed-int "unset" convention the settings interface uses (e.g.
/// `llm_hosted_tool_search = -1`), while reusing the 0..=4 personality ordinal
/// contract the KCM already binds to.
fn trait_from_dbus_ordinal(name: &str, n: i32) -> fdo::Result<Option<PersonalityLevel>> {
    if n < 0 {
        return Ok(None);
    }
    let ordinal = u8::try_from(n).ok().filter(|v| *v <= 4).ok_or_else(|| {
        fdo::Error::InvalidArgs(format!(
            "personality trait {name}: ordinal {n} out of range 0..=4 (or -1 to leave unset)"
        ))
    })?;
    Ok(PersonalityLevel::from_ordinal(ordinal))
}

/// Inverse of [`trait_from_dbus_ordinal`]: `None` → `-1`, `Some(level)` → its
/// 0..=4 ordinal.
fn trait_to_dbus_ordinal(level: Option<PersonalityLevel>) -> i32 {
    match level {
        None => -1,
        Some(l) => l.as_ordinal() as i32,
    }
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

    /// Set (or clear) a conversation's personality override (#227, Phase 2).
    ///
    /// Each trait is a signed ordinal: `-1` leaves it unset (falls back to the
    /// global config on every send), `0..=4` pins the level (Never=0 …
    /// Always=4). When every trait is `-1` the override is cleared
    /// (global-only). The override sets only the *initial disposition* — the
    /// assistant stays soft/adaptive. Arguments are in the fixed trait order:
    /// professionalism, warmth, directness, enthusiasm, humor, sarcasm,
    /// pretentiousness. Returns the stored override echoed back as the same
    /// 7-ordinal tuple (cleared → all `-1`).
    #[allow(clippy::too_many_arguments)]
    async fn set_conversation_personality(
        &self,
        conversation_id: &str,
        professionalism: i32,
        warmth: i32,
        directness: i32,
        enthusiasm: i32,
        humor: i32,
        sarcasm: i32,
        pretentiousness: i32,
    ) -> fdo::Result<(i32, i32, i32, i32, i32, i32, i32)> {
        let ovr = PersonalityOverride {
            professionalism: trait_from_dbus_ordinal("professionalism", professionalism)?,
            warmth: trait_from_dbus_ordinal("warmth", warmth)?,
            directness: trait_from_dbus_ordinal("directness", directness)?,
            enthusiasm: trait_from_dbus_ordinal("enthusiasm", enthusiasm)?,
            humor: trait_from_dbus_ordinal("humor", humor)?,
            sarcasm: trait_from_dbus_ordinal("sarcasm", sarcasm)?,
            pretentiousness: trait_from_dbus_ordinal("pretentiousness", pretentiousness)?,
        };
        with_user_id(resolve_dbus_user_id(), async {
            let id = ConversationId::from(conversation_id);
            self.service
                .set_conversation_personality(&id, ovr)
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))?;
            // Echo the stored value (cleared → all-None → all -1).
            let stored = self
                .service
                .get_conversation_personality(&id)
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))?
                .unwrap_or_default();
            Ok((
                trait_to_dbus_ordinal(stored.professionalism),
                trait_to_dbus_ordinal(stored.warmth),
                trait_to_dbus_ordinal(stored.directness),
                trait_to_dbus_ordinal(stored.enthusiasm),
                trait_to_dbus_ordinal(stored.humor),
                trait_to_dbus_ordinal(stored.sarcasm),
                trait_to_dbus_ordinal(stored.pretentiousness),
            ))
        })
        .await
    }

    /// Read a conversation's personality override (#227) as the 7-ordinal tuple
    /// (`-1` = unset / global fallback; `0..=4` = pinned level). When no
    /// override is stored every value is `-1`.
    async fn get_conversation_personality(
        &self,
        conversation_id: &str,
    ) -> fdo::Result<(i32, i32, i32, i32, i32, i32, i32)> {
        with_user_id(resolve_dbus_user_id(), async {
            let stored = self
                .service
                .get_conversation_personality(&ConversationId::from(conversation_id))
                .await
                .map_err(|e| fdo::Error::Failed(e.to_string()))?
                .unwrap_or_default();
            Ok((
                trait_to_dbus_ordinal(stored.professionalism),
                trait_to_dbus_ordinal(stored.warmth),
                trait_to_dbus_ordinal(stored.directness),
                trait_to_dbus_ordinal(stored.enthusiasm),
                trait_to_dbus_ordinal(stored.humor),
                trait_to_dbus_ordinal(stored.sarcasm),
                trait_to_dbus_ordinal(stored.pretentiousness),
            ))
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
        Ok(self
            .start_streaming_send(emitter, conversation_id, prompt, String::new())
            .await)
    }

    /// Like [`send_prompt`](Self::send_prompt) but attaches a
    /// per-request `system_refinement` that the daemon appends to the
    /// system prompt for THIS turn only (empty = none). The refinement is
    /// never stored as a message and never affects later turns, so the
    /// visible transcript records only the clean `prompt`. Added
    /// additively (issue #200 follow-up) for the voice daemon; the chat
    /// clients' `send_prompt` is left byte-identical.
    async fn send_prompt_with_system_refinement(
        &self,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> fdo::Result<String> {
        Ok(self
            .start_streaming_send(
                emitter,
                conversation_id,
                prompt,
                system_refinement.to_string(),
            )
            .await)
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

    #[async_trait::async_trait]
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
                String::new(),
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

    /// #207 regression guard. The future `tokio::spawn` receives at the
    /// send-prompt site must be a thin state machine that keeps the
    /// handler's work behind a boxed `Pin<Box<dyn Future>>`, not the
    /// deeply nested generic future inlined by value. Before #207
    /// `ConversationService` used RPITIT (`-> impl Future`), so
    /// `send_prompt_with_override` monomorphized the whole
    /// handler/LLM/tool stack into one multi-MB async state machine —
    /// constructing/moving it onto the 2 MB tokio worker stack at
    /// `tokio::spawn` overflowed the guard page (#205/#206).
    /// `#[async_trait]` boxes that future, so the spawned task frame is
    /// now a few hundred bytes. If the trait ever reverts to RPITIT the
    /// future inlines the stack again and balloons far past the
    /// threshold below.
    #[test]
    fn spawned_send_prompt_future_stays_small() {
        let service = Arc::new(FakeConversationService);
        let (tx, _rx) = mpsc::unbounded_channel::<StreamEvent>();
        let fut = crate::conversation::run_send_prompt_llm_task(
            service,
            "conv-1".to_string(),
            "hello".to_string(),
            String::new(),
            tx,
        );
        let size = std::mem::size_of_val(&fut);
        assert!(
            size < 8 * 1024,
            "spawned send-prompt future is {size} bytes; expected < 8 KiB. \
             A multi-KB/MB size means `ConversationService` lost its \
             `#[async_trait]` boxing and the handler future is being inlined \
             again — the #205/#206 worker-stack-overflow regression."
        );
    }

    /// Service double that records the `prompt` and `system_refinement`
    /// it was dispatched with via `send_prompt_with_override`, so the
    /// `SendPromptWithSystemRefinement` plumbing can be asserted without a
    /// real D-Bus connection.
    struct RefinementCapturingService {
        prompt: Mutex<Option<String>>,
        refinement: Mutex<Option<String>>,
    }

    impl RefinementCapturingService {
        fn new() -> Self {
            Self {
                prompt: Mutex::new(None),
                refinement: Mutex::new(None),
            }
        }
    }

    #[async_trait::async_trait]
    impl ConversationService for RefinementCapturingService {
        async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("rec-id", title))
        }
        async fn list_conversations(
            &self,
            _: Option<u32>,
            _: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            Ok(Conversation::new(id.as_str(), "rec"))
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
            // Should not be hit: the adapter routes through
            // `send_prompt_with_override` so the refinement is carried.
            panic!("plain send_prompt must not be called by the streaming send path");
        }
        async fn send_prompt_with_override(
            &self,
            _conversation_id: &ConversationId,
            prompt: String,
            _override_selection: Option<
                desktop_assistant_core::ports::inbound::PromptSelectionOverride,
            >,
            system_refinement: String,
            mut on_chunk: ChunkCallback,
            _on_status: StatusCallback,
            _cancellation: CancellationToken,
        ) -> Result<desktop_assistant_core::ports::inbound::PromptDispatchOutcome, CoreError>
        {
            *self.prompt.lock().unwrap() = Some(prompt);
            *self.refinement.lock().unwrap() = Some(system_refinement);
            on_chunk("ok".to_string());
            Ok(
                desktop_assistant_core::ports::inbound::PromptDispatchOutcome {
                    response: "ok".to_string(),
                    warnings: Vec::new(),
                },
            )
        }
    }

    /// The `SendPromptWithSystemRefinement` path must pass the CLEAN
    /// prompt as `content` and the caller-supplied text as
    /// `system_refinement` through to the core service — mirroring the
    /// `SendPrompt` handler but with the refinement populated. (The core
    /// service is what guarantees the refinement is appended to the
    /// system prompt for this turn only and never stored; see
    /// `service::tests::system_refinement_is_appended_to_system_prompt_for_the_request`.)
    #[tokio::test]
    async fn send_prompt_with_system_refinement_routes_clean_prompt_and_refinement() {
        let service = Arc::new(RefinementCapturingService::new());
        let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();

        run_send_prompt_llm_task(
            Arc::clone(&service),
            "conv-x".to_string(),
            "what's the weather?".to_string(),
            "Respond briefly, by voice.".to_string(),
            tx,
        )
        .await;

        // Drain so we know the body ran to completion.
        while rx.recv().await.is_some() {}

        assert_eq!(
            service.prompt.lock().unwrap().as_deref(),
            Some("what's the weather?"),
            "the clean prompt (no blurb) must be sent as content"
        );
        assert_eq!(
            service.refinement.lock().unwrap().as_deref(),
            Some("Respond briefly, by voice."),
            "the configured hint must be carried as the per-request system_refinement"
        );
    }

    // --- #227: per-conversation personality ordinal contract ---------------

    #[test]
    fn personality_trait_ordinal_round_trips_through_dbus() {
        // -1 means "unset / fall back to global".
        assert_eq!(trait_from_dbus_ordinal("humor", -1).unwrap(), None);
        assert_eq!(trait_to_dbus_ordinal(None), -1);
        // 0..=4 round-trips to the level and back.
        for (n, level) in [
            (0, PersonalityLevel::Never),
            (1, PersonalityLevel::Rarely),
            (2, PersonalityLevel::Sometimes),
            (3, PersonalityLevel::Often),
            (4, PersonalityLevel::Always),
        ] {
            assert_eq!(
                trait_from_dbus_ordinal("humor", n).unwrap(),
                Some(level),
                "ordinal {n} must map to {level:?}"
            );
            assert_eq!(trait_to_dbus_ordinal(Some(level)), n);
        }
        // Out-of-range positive is rejected, not clamped.
        assert!(trait_from_dbus_ordinal("humor", 5).is_err());
    }

    /// Conversation service double that stores a per-conversation personality
    /// override in memory so the D-Bus method's persist + echo can be asserted.
    struct PersonalityStoringService {
        stored: Mutex<Option<PersonalityOverride>>,
    }

    impl PersonalityStoringService {
        fn new() -> Self {
            Self {
                stored: Mutex::new(None),
            }
        }
    }

    #[async_trait::async_trait]
    impl ConversationService for PersonalityStoringService {
        async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
            Ok(Conversation::new("c", title))
        }
        async fn list_conversations(
            &self,
            _: Option<u32>,
            _: bool,
        ) -> Result<Vec<ConversationSummary>, CoreError> {
            Ok(vec![])
        }
        async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            Ok(Conversation::new(id.as_str(), "t"))
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
        async fn get_conversation_personality(
            &self,
            _: &ConversationId,
        ) -> Result<Option<PersonalityOverride>, CoreError> {
            Ok(*self.stored.lock().unwrap())
        }
        async fn set_conversation_personality(
            &self,
            _: &ConversationId,
            personality: PersonalityOverride,
        ) -> Result<(), CoreError> {
            // Mirror the routing wrapper: an empty override clears the store.
            *self.stored.lock().unwrap() = if personality.is_empty() {
                None
            } else {
                Some(personality)
            };
            Ok(())
        }
    }

    #[tokio::test]
    async fn dbus_set_get_conversation_personality_round_trips_and_clears() {
        let service = Arc::new(PersonalityStoringService::new());
        let adapter = DbusConversationAdapter::new(Arc::clone(&service));
        let _guard = crate::testing::UserEnvGuard::set("alice");

        // Pin humor=Never(0) and directness=Always(4); leave the rest unset(-1).
        let echoed = adapter
            .set_conversation_personality("c1", -1, -1, 4, -1, 0, -1, -1)
            .await
            .unwrap();
        // Echo: directness=4, humor=0, rest -1.
        assert_eq!(echoed, (-1, -1, 4, -1, 0, -1, -1));

        // GET reflects the stored value.
        let got = adapter.get_conversation_personality("c1").await.unwrap();
        assert_eq!(got, (-1, -1, 4, -1, 0, -1, -1));

        // All -1 clears the override → GET returns all -1.
        let cleared = adapter
            .set_conversation_personality("c1", -1, -1, -1, -1, -1, -1, -1)
            .await
            .unwrap();
        assert_eq!(cleared, (-1, -1, -1, -1, -1, -1, -1));
        assert!(service.stored.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn dbus_set_conversation_personality_rejects_out_of_range() {
        let service = Arc::new(PersonalityStoringService::new());
        let adapter = DbusConversationAdapter::new(service);
        let _guard = crate::testing::UserEnvGuard::set("alice");
        // ordinal 5 is out of range for the humor slot.
        let err = adapter
            .set_conversation_personality("c1", -1, -1, -1, -1, 5, -1, -1)
            .await;
        assert!(err.is_err(), "out-of-range ordinal must be rejected");
    }

    struct FakeConversationService;

    #[async_trait::async_trait]
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
        #[async_trait::async_trait]
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
        let include = ["user".to_string(), "assistant".to_string()];
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
        #[async_trait::async_trait]
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
        let include = ["user".to_string(), "assistant".to_string()];
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
