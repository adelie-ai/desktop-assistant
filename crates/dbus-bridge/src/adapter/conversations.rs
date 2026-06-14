//! D-Bus adapter for `/org/desktopAssistant/Conversations`.
//!
//! Mirrors `crates/dbus-interface/src/conversation.rs` method-for-method.
//! Translations of method args → `api::Command`, and of
//! `api::CommandResult` → method return values, match the WS adapter
//! semantics so existing TUI/GTK/KDE clients keep working unchanged.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use zbus::object_server::SignalEmitter;
use zbus::{fdo, interface};

use crate::transport::{BridgeTransport, BridgeTransportError};

fn to_fdo<E: std::fmt::Display>(error: E) -> fdo::Error {
    fdo::Error::Failed(error.to_string())
}

/// D-Bus ordinal contract for one personality-override trait (#227): `-1` =
/// unset (fall back to global), `0..=4` pins the level (Never=0 … Always=4).
/// Mirrors `dbus-interface`'s `trait_from_dbus_ordinal`.
fn trait_from_dbus_ordinal(name: &str, n: i32) -> fdo::Result<Option<api::PersonalityLevel>> {
    if n < 0 {
        return Ok(None);
    }
    let ordinal = u8::try_from(n).ok().filter(|v| *v <= 4).ok_or_else(|| {
        fdo::Error::InvalidArgs(format!(
            "personality trait {name}: ordinal {n} out of range 0..=4 (or -1 to leave unset)"
        ))
    })?;
    Ok(api::PersonalityLevel::from_ordinal(ordinal))
}

/// Inverse of [`trait_from_dbus_ordinal`]: `None` → `-1`, `Some` → 0..=4.
fn trait_to_dbus_ordinal(level: Option<api::PersonalityLevel>) -> i32 {
    match level {
        None => -1,
        Some(l) => l.as_ordinal() as i32,
    }
}

/// Build a [`api::PersonalityOverride`] from the 7-ordinal tuple in fixed trait
/// order (professionalism, warmth, directness, enthusiasm, humor, sarcasm,
/// pretentiousness).
#[allow(clippy::too_many_arguments)]
fn personality_override_from_ordinals(
    professionalism: i32,
    warmth: i32,
    directness: i32,
    enthusiasm: i32,
    humor: i32,
    sarcasm: i32,
    pretentiousness: i32,
) -> fdo::Result<api::PersonalityOverride> {
    Ok(api::PersonalityOverride {
        professionalism: trait_from_dbus_ordinal("professionalism", professionalism)?,
        warmth: trait_from_dbus_ordinal("warmth", warmth)?,
        directness: trait_from_dbus_ordinal("directness", directness)?,
        enthusiasm: trait_from_dbus_ordinal("enthusiasm", enthusiasm)?,
        humor: trait_from_dbus_ordinal("humor", humor)?,
        sarcasm: trait_from_dbus_ordinal("sarcasm", sarcasm)?,
        pretentiousness: trait_from_dbus_ordinal("pretentiousness", pretentiousness)?,
    })
}

/// Inverse of [`personality_override_from_ordinals`].
fn personality_override_to_ordinals(
    ovr: &api::PersonalityOverride,
) -> (i32, i32, i32, i32, i32, i32, i32) {
    (
        trait_to_dbus_ordinal(ovr.professionalism),
        trait_to_dbus_ordinal(ovr.warmth),
        trait_to_dbus_ordinal(ovr.directness),
        trait_to_dbus_ordinal(ovr.enthusiasm),
        trait_to_dbus_ordinal(ovr.humor),
        trait_to_dbus_ordinal(ovr.sarcasm),
        trait_to_dbus_ordinal(ovr.pretentiousness),
    )
}

/// Translate a transport-level error to a D-Bus error. Daemon-level
/// errors propagate verbatim; everything else gets a `Failed` with a
/// descriptive prefix.
fn map_transport_err(error: BridgeTransportError) -> fdo::Error {
    match error {
        BridgeTransportError::Daemon(msg) => fdo::Error::Failed(msg),
        other => fdo::Error::Failed(other.to_string()),
    }
}

/// D-Bus adapter for conversation management. The non-streaming
/// methods translate into `api::Command` dispatches; `SendPrompt`
/// triggers a `SendMessage` command which the daemon streams back as
/// `AssistantDelta` / `AssistantCompleted` / `AssistantError` events —
/// those are translated to `ResponseChunk` / `ResponseComplete` /
/// `ResponseError` signals by [`super::event_forwarder`].
pub struct DbusConversationsAdapter<T: BridgeTransport + 'static> {
    transport: Arc<T>,
}

impl<T: BridgeTransport + 'static> DbusConversationsAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.transport.request(cmd).await.map_err(map_transport_err)
    }

    /// Shared body for `SendPrompt` / `SendPromptWithSystemRefinement`:
    /// builds the `SendMessage` command (with the given
    /// `system_refinement`; empty = none), dispatches it, and maps the
    /// immediate result to a correlation id for the caller. Keeping this
    /// in one place guarantees the two D-Bus methods stay
    /// byte-identical apart from the refinement.
    async fn dispatch_send_message(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: String,
    ) -> fdo::Result<String> {
        let result = self
            .dispatch(api::Command::SendMessage {
                conversation_id: conversation_id.to_string(),
                content: prompt.to_string(),
                override_selection: None,
                system_refinement,
                // The D-Bus bridge does not originate idempotency keys (#204).
                idempotency_key: None,
            })
            .await
            .map_err(|e| {
                // SendMessage returns an immediate Ack on success; if
                // the daemon refused the request before streaming, we
                // surface the error directly.
                to_fdo(e)
            })?;

        match result {
            api::CommandResult::Ack => {
                // Legacy bare `Ack`: the daemon told us no correlation id, so
                // use a placeholder for the caller to log against. (The events
                // flowing through the forwarder carry the daemon's own
                // correlation id, which a bare-Ack daemon never surfaced here.)
                Ok(uuid::Uuid::new_v4().to_string())
            }
            // Return the turn `request_id` — the id every streamed `Assistant*`
            // event carries — so the D-Bus caller can correlate the response
            // events the bridge forwards (voice#49). The `task_id` keys `Task*`
            // events, not the response stream, so it is not the value to hand
            // back here.
            api::CommandResult::SendMessageAck { request_id, .. } => Ok(request_id),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SendMessage result: {other:?}"
            ))),
        }
    }
}

#[interface(name = "org.desktopAssistant.Conversations")]
impl<T: BridgeTransport + 'static> DbusConversationsAdapter<T> {
    /// Create a new conversation and return its ID.
    async fn create_conversation(&self, title: &str) -> fdo::Result<String> {
        let result = self
            .dispatch(api::Command::CreateConversation {
                title: title.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::ConversationId { id } => Ok(id),
            other => Err(fdo::Error::Failed(format!(
                "unexpected CreateConversation result: {other:?}"
            ))),
        }
    }

    /// List conversations. Wire shape matches the in-process adapter:
    /// `(id, title, message_count, updated_at, archived)` tuples.
    async fn list_conversations(
        &self,
        max_age_days: i32,
        include_archived: bool,
    ) -> fdo::Result<Vec<(String, String, u32, String, bool)>> {
        let max_age_days = u32::try_from(max_age_days).ok().filter(|d| *d > 0);
        let result = self
            .dispatch(api::Command::ListConversations {
                max_age_days,
                include_archived,
            })
            .await?;
        match result {
            api::CommandResult::Conversations(summaries) => Ok(summaries
                .into_iter()
                .map(|s| (s.id, s.title, s.message_count, s.updated_at, s.archived))
                .collect()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListConversations result: {other:?}"
            ))),
        }
    }

    /// Archive a conversation by ID.
    async fn archive_conversation(&self, id: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::ArchiveConversation { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ArchiveConversation result: {other:?}"
            ))),
        }
    }

    /// Unarchive a conversation by ID.
    async fn unarchive_conversation(&self, id: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::UnarchiveConversation { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected UnarchiveConversation result: {other:?}"
            ))),
        }
    }

    /// Get a conversation by ID: `(id, title, messages)` where
    /// `messages` is `(role, content)` tuples.
    async fn get_conversation(
        &self,
        id: &str,
    ) -> fdo::Result<(String, String, Vec<(String, String)>)> {
        let result = self
            .dispatch(api::Command::GetConversation { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Conversation(conv) => {
                let messages = conv
                    .messages
                    .into_iter()
                    .map(|m| (m.role, m.content))
                    .collect();
                Ok((conv.id, conv.title, messages))
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetConversation result: {other:?}"
            ))),
        }
    }

    /// Get messages with pagination + role filter. Returns
    /// `(total_raw_count, truncated, messages)`. Slicing is performed
    /// on the bridge side using the daemon's full conversation view
    /// because the daemon's command set does not (yet) expose this
    /// exact pagination shape on the wire.
    async fn get_messages(
        &self,
        id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> fdo::Result<(u32, bool, Vec<(String, String)>)> {
        let result = self
            .dispatch(api::Command::GetConversation { id: id.to_string() })
            .await?;
        let conv = match result {
            api::CommandResult::Conversation(c) => c,
            other => {
                return Err(fdo::Error::Failed(format!(
                    "unexpected GetConversation result for get_messages: {other:?}"
                )));
            }
        };

        let total = conv.messages.len() as u32;
        let all: Vec<(String, String)> = conv
            .messages
            .into_iter()
            .map(|m| (m.role, m.content))
            .collect();

        let use_after = after_count >= 0;
        let sliced: Vec<(String, String)> = if use_after {
            let start = (after_count as usize).min(all.len());
            all[start..].to_vec()
        } else {
            all
        };

        let filtered: Vec<(String, String)> = sliced
            .into_iter()
            .filter(|(role, _)| include_roles.is_empty() || include_roles.contains(role))
            .collect();

        let (truncated, messages) = if !use_after && tail > 0 && filtered.len() > tail as usize {
            let start = filtered.len() - tail as usize;
            (true, filtered[start..].to_vec())
        } else {
            (false, filtered)
        };

        Ok((total, truncated, messages))
    }

    /// Delete a conversation by ID.
    async fn delete_conversation(&self, id: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::DeleteConversation { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected DeleteConversation result: {other:?}"
            ))),
        }
    }

    /// Rename a conversation.
    async fn rename_conversation(&self, id: &str, title: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::RenameConversation {
                id: id.to_string(),
                title: title.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected RenameConversation result: {other:?}"
            ))),
        }
    }

    /// Delete every conversation; returns the count.
    async fn clear_all_history(&self) -> fdo::Result<u32> {
        let result = self.dispatch(api::Command::ClearAllHistory).await?;
        match result {
            api::CommandResult::Cleared { deleted_count } => Ok(deleted_count),
            other => Err(fdo::Error::Failed(format!(
                "unexpected ClearAllHistory result: {other:?}"
            ))),
        }
    }

    /// Set (or clear) a conversation's personality override (#227, Phase 2).
    /// Each trait is a signed ordinal: `-1` = unset (fall back to global),
    /// `0..=4` pins the level (Never=0 … Always=4); all `-1` clears the
    /// override. Args in fixed trait order: professionalism, warmth,
    /// directness, enthusiasm, humor, sarcasm, pretentiousness. Returns the
    /// stored override echoed as the same 7-ordinal tuple. Mirrors the
    /// in-process `dbus-interface` method.
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
        let personality = personality_override_from_ordinals(
            professionalism,
            warmth,
            directness,
            enthusiasm,
            humor,
            sarcasm,
            pretentiousness,
        )?;
        let result = self
            .dispatch(api::Command::SetConversationPersonality {
                conversation_id: conversation_id.to_string(),
                personality,
            })
            .await?;
        match result {
            api::CommandResult::ConversationPersonality(stored) => {
                Ok(personality_override_to_ordinals(&stored))
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected SetConversationPersonality result: {other:?}"
            ))),
        }
    }

    /// Read a conversation's personality override (#227) as the 7-ordinal
    /// tuple (`-1` = unset; `0..=4` = pinned). All `-1` when no override is
    /// stored. Reads `conversation_personality` from `GetConversation`.
    async fn get_conversation_personality(
        &self,
        conversation_id: &str,
    ) -> fdo::Result<(i32, i32, i32, i32, i32, i32, i32)> {
        let result = self
            .dispatch(api::Command::GetConversation {
                id: conversation_id.to_string(),
            })
            .await?;
        match result {
            api::CommandResult::Conversation(view) => Ok(personality_override_to_ordinals(
                &view.conversation_personality.unwrap_or_default(),
            )),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetConversation result: {other:?}"
            ))),
        }
    }

    /// Send a prompt; daemon streams back via `AssistantDelta` events
    /// which the event forwarder turns into `ResponseChunk` /
    /// `ResponseComplete` / `ResponseError` signals.
    ///
    /// Returns the `request_id` the daemon will use for event
    /// correlation. The daemon currently echoes the request id via
    /// the streamed event payloads, so the bridge picks its own UUID
    /// and the daemon's id is what shows up on the signal — same as
    /// the in-process adapter where the dbus-interface created its
    /// own request id.
    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> fdo::Result<String> {
        self.dispatch_send_message(conversation_id, prompt, String::new())
            .await
    }

    /// Like [`send_prompt`](Self::send_prompt) but attaches a
    /// per-request `system_refinement` that the daemon appends to the
    /// system prompt for THIS turn only. The refinement is never stored
    /// as a message and never affects later turns (see
    /// `api::Command::SendMessage`), so the visible transcript records
    /// only the clean `prompt`. An empty `system_refinement` is
    /// equivalent to [`send_prompt`](Self::send_prompt).
    ///
    /// Added additively (issue #200 follow-up) so the voice daemon can
    /// dictate "respond briefly, by voice" into an existing chat without
    /// polluting history. `send_prompt` is intentionally left
    /// byte-identical for existing chat clients.
    async fn send_prompt_with_system_refinement(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> fdo::Result<String> {
        self.dispatch_send_message(conversation_id, prompt, system_refinement.to_string())
            .await
    }

    /// Signal emitted for each chunk of a streaming response.
    /// Body is forwarded by [`super::event_forwarder`] from
    /// `Event::AssistantDelta`.
    #[zbus(signal)]
    async fn response_chunk(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        chunk: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted when a streaming response is complete.
    /// Forwarded from `Event::AssistantCompleted`.
    #[zbus(signal)]
    async fn response_complete(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        full_response: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted on streaming failure. Forwarded from
    /// `Event::AssistantError`.
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
    use tokio::sync::Mutex;

    /// Recording [`BridgeTransport`] that captures every dispatched
    /// command so a test can assert the exact `api::Command` the adapter
    /// builds. Returns a fixed `SendMessageAck` so the adapter's
    /// result-mapping path is exercised end to end.
    struct RecordingTransport {
        commands: Mutex<Vec<api::Command>>,
    }

    impl RecordingTransport {
        fn new() -> Self {
            Self {
                commands: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl BridgeTransport for RecordingTransport {
        async fn request(
            &self,
            command: api::Command,
        ) -> Result<api::CommandResult, BridgeTransportError> {
            self.commands.lock().await.push(command);
            Ok(api::CommandResult::SendMessageAck {
                request_id: "req-1".to_string(),
                task_id: "task-1".to_string(),
            })
        }
    }

    /// A non-empty refinement must produce a `SendMessage` whose
    /// `system_refinement` carries that text and whose `content` is the
    /// CLEAN prompt (no blurb prepended). This is the wire half of the
    /// voice "system-prompt refinement" feature: the daemon appends the
    /// refinement to the system prompt for this turn only, while the
    /// visible transcript records just `content`.
    #[tokio::test]
    async fn send_prompt_with_system_refinement_sets_refinement_and_clean_content() {
        let transport = Arc::new(RecordingTransport::new());
        let adapter = DbusConversationsAdapter::new(Arc::clone(&transport));

        let request_id = adapter
            .send_prompt_with_system_refinement(
                "conv-1",
                "what's the weather?",
                "Respond briefly, by voice.",
            )
            .await
            .expect("send returns the daemon correlation id");
        // The adapter returns the turn `request_id` (what streamed events carry,
        // so the D-Bus caller can correlate the response), not the `task_id`
        // (voice#49).
        assert_eq!(request_id, "req-1");

        let commands = transport.commands.lock().await;
        assert_eq!(commands.len(), 1, "exactly one command dispatched");
        match &commands[0] {
            api::Command::SendMessage {
                conversation_id,
                content,
                override_selection,
                system_refinement,
                ..
            } => {
                assert_eq!(conversation_id, "conv-1");
                // The prompt is stored CLEAN — no "You are Adele…" blurb.
                assert_eq!(content, "what's the weather?");
                assert_eq!(system_refinement, "Respond briefly, by voice.");
                assert!(override_selection.is_none());
            }
            other => panic!("expected SendMessage, got {other:?}"),
        }
    }

    /// `SendPrompt` must stay byte-identical: it builds a `SendMessage`
    /// with an EMPTY `system_refinement` so existing chat clients are
    /// unaffected by the additive method.
    #[tokio::test]
    async fn send_prompt_leaves_system_refinement_empty() {
        let transport = Arc::new(RecordingTransport::new());
        let adapter = DbusConversationsAdapter::new(Arc::clone(&transport));

        adapter.send_prompt("conv-2", "hello").await.unwrap();

        let commands = transport.commands.lock().await;
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            api::Command::SendMessage {
                conversation_id,
                content,
                override_selection,
                system_refinement,
                ..
            } => {
                assert_eq!(conversation_id, "conv-2");
                assert_eq!(content, "hello");
                assert!(
                    system_refinement.is_empty(),
                    "send_prompt must not set a refinement"
                );
                assert!(override_selection.is_none());
            }
            other => panic!("expected SendMessage, got {other:?}"),
        }
    }

    /// Transport that records commands and returns a caller-supplied result,
    /// so methods whose result is not `SendMessageAck` (e.g. personality) can
    /// exercise their result-mapping path.
    struct CannedTransport {
        commands: Mutex<Vec<api::Command>>,
        result: api::CommandResult,
    }

    impl CannedTransport {
        fn new(result: api::CommandResult) -> Self {
            Self {
                commands: Mutex::new(Vec::new()),
                result,
            }
        }
    }

    #[async_trait::async_trait]
    impl BridgeTransport for CannedTransport {
        async fn request(
            &self,
            command: api::Command,
        ) -> Result<api::CommandResult, BridgeTransportError> {
            self.commands.lock().await.push(command);
            Ok(self.result.clone())
        }
    }

    #[tokio::test]
    async fn set_conversation_personality_builds_command_and_maps_ordinals() {
        // The bridge must translate the 7-ordinal tuple into a partial
        // `PersonalityOverride` (only pinned traits present) and map the
        // `ConversationPersonality` result back to ordinals.
        let stored = api::PersonalityOverride {
            humor: Some(api::PersonalityLevel::Never),
            directness: Some(api::PersonalityLevel::Always),
            ..api::PersonalityOverride::default()
        };
        let transport = Arc::new(CannedTransport::new(
            api::CommandResult::ConversationPersonality(stored),
        ));
        let adapter = DbusConversationsAdapter::new(Arc::clone(&transport));

        // Pin directness=Always(4), humor=Never(0); rest unset(-1).
        let echoed = adapter
            .set_conversation_personality("conv-1", -1, -1, 4, -1, 0, -1, -1)
            .await
            .unwrap();
        assert_eq!(echoed, (-1, -1, 4, -1, 0, -1, -1));

        let commands = transport.commands.lock().await;
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            api::Command::SetConversationPersonality {
                conversation_id,
                personality,
            } => {
                assert_eq!(conversation_id, "conv-1");
                assert_eq!(personality.directness, Some(api::PersonalityLevel::Always));
                assert_eq!(personality.humor, Some(api::PersonalityLevel::Never));
                // Unset traits are `None` (fall back to global).
                assert_eq!(personality.warmth, None);
            }
            other => panic!("expected SetConversationPersonality, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn set_conversation_personality_rejects_out_of_range_before_dispatch() {
        let transport = Arc::new(CannedTransport::new(
            api::CommandResult::ConversationPersonality(api::PersonalityOverride::default()),
        ));
        let adapter = DbusConversationsAdapter::new(Arc::clone(&transport));
        let err = adapter
            .set_conversation_personality("conv-1", -1, -1, -1, -1, 9, -1, -1)
            .await;
        assert!(err.is_err(), "out-of-range ordinal must be rejected");
        // Validation happens before dispatch — no command sent.
        assert!(transport.commands.lock().await.is_empty());
    }
}
