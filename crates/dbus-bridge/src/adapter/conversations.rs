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

use crate::session::SessionRegistry;
use crate::transport::{BridgeTransport, BridgeTransportError};

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
    /// Per-sender daemon sessions (#367/#320). Wired in production via
    /// [`with_sessions`](Self::with_sessions); `None` in unit tests, where
    /// turn-driving falls back to the shared `transport`.
    sessions: Option<Arc<SessionRegistry>>,
}

impl<T: BridgeTransport + 'static> DbusConversationsAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self {
            transport,
            sessions: None,
        }
    }

    /// Wire the per-sender session registry (production). Turn-driving methods
    /// then dispatch through the *caller's* own daemon session, so a turn's
    /// streamed events come back on that session and unicast to only that caller
    /// (#367/#320). Without it (unit tests), they use the shared transport.
    pub fn with_sessions(mut self, sessions: Arc<SessionRegistry>) -> Self {
        self.sessions = Some(sessions);
        self
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.transport.request(cmd).await.map_err(map_transport_err)
    }

    /// Dispatch a command that may be **session-scoped**, routing it through the
    /// per-sender [`SessionRegistry`]: a turn runs on the caller's own session (so
    /// its streamed events come back there, to be unicast to only this caller);
    /// `SubscribeConversations` is pinned to that session too (#367); everything
    /// else uses the shared connection. The registry's
    /// [`route`](SessionRegistry::route) makes that decision in one place, so the
    /// typed methods and the generic Commands channel route identically. Without a
    /// registry (unit tests) the command falls back to the shared transport.
    /// `caller` is the D-Bus sender's unique bus name from the message header.
    async fn dispatch_routed(
        &self,
        caller: Option<&str>,
        cmd: api::Command,
    ) -> fdo::Result<api::CommandResult> {
        match self.sessions.as_ref() {
            Some(registry) => registry
                .route(caller, cmd, self.transport.as_ref())
                .await
                .map_err(map_transport_err),
            None => self.transport.request(cmd).await.map_err(map_transport_err),
        }
    }

    /// Shared body for `SendPrompt` / `SendPromptWithSystemRefinement`:
    /// builds the `SendMessage` command (with the given
    /// `system_refinement`; empty = none), dispatches it **through the caller's
    /// session**, and maps the immediate result to a correlation id for the
    /// caller. Keeping this in one place guarantees the two D-Bus methods stay
    /// byte-identical apart from the refinement.
    async fn dispatch_send_message(
        &self,
        caller: Option<&str>,
        conversation_id: &str,
        prompt: &str,
        system_refinement: String,
    ) -> fdo::Result<String> {
        // SendMessage returns an immediate Ack/SendMessageAck; a daemon refusal
        // before streaming surfaces here as the mapped transport error.
        let result = self
            .dispatch_routed(
                caller,
                api::Command::SendMessage {
                    conversation_id: conversation_id.to_string(),
                    content: prompt.to_string(),
                    override_selection: None,
                    system_refinement,
                    // The D-Bus bridge does not originate idempotency keys (#204).
                    idempotency_key: None,
                },
            )
            .await?;

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

    /// Shared body for `SubscribeConversations` (#367): set-replace the set of
    /// conversations this caller is viewing, **pinned to the caller's own
    /// session**, so the daemon fans those conversations' turn events
    /// (`UserMessageAdded` + the response stream, including turns this caller did
    /// not initiate) back on that session — which the per-sender unicast forwarder
    /// delivers to only this caller. An empty list unsubscribes from all.
    /// Factored out (no header) so it is unit-testable; the `#[interface]` method
    /// extracts `caller` and calls it.
    async fn dispatch_subscribe_conversations(
        &self,
        caller: Option<&str>,
        conversation_ids: Vec<String>,
    ) -> fdo::Result<()> {
        let result = self
            .dispatch_routed(
                caller,
                api::Command::SubscribeConversations { conversation_ids },
            )
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected SubscribeConversations result: {other:?}"
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
    /// `(total_raw_count, truncated, messages)` where `messages` is
    /// `(role, content)` tuples.
    ///
    /// The windowing is the daemon's: `Command::GetMessages` runs the single
    /// `window_messages` slicer (#363) that every UDS/WS client also uses, so
    /// this adapter is a thin translator — no slicing logic of its own to drift
    /// from the others. The D-Bus signature is unchanged.
    async fn get_messages(
        &self,
        id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> fdo::Result<(u32, bool, Vec<(String, String)>)> {
        let result = self
            .dispatch(api::Command::GetMessages {
                conversation_id: id.to_string(),
                tail,
                after_count,
                include_roles,
            })
            .await?;
        match result {
            api::CommandResult::Messages(view) => Ok((
                view.total_raw_count,
                view.truncated,
                view.messages
                    .into_iter()
                    .map(|m| (m.role, m.content))
                    .collect(),
            )),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetMessages result: {other:?}"
            ))),
        }
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
    async fn send_prompt(
        &self,
        #[zbus(header)] hdr: zbus::message::Header<'_>,
        conversation_id: &str,
        prompt: &str,
    ) -> fdo::Result<String> {
        let caller = hdr.sender().map(|s| s.as_str());
        self.dispatch_send_message(caller, conversation_id, prompt, String::new())
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
        #[zbus(header)] hdr: zbus::message::Header<'_>,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> fdo::Result<String> {
        let caller = hdr.sender().map(|s| s.as_str());
        self.dispatch_send_message(
            caller,
            conversation_id,
            prompt,
            system_refinement.to_string(),
        )
        .await
    }

    /// Set-replace the conversations this caller is viewing, for live
    /// multi-client sync (#367). The daemon fans these conversations' turn events
    /// back on the caller's per-sender session — including turns it did NOT
    /// initiate (a voice turn, or another client) — delivered as
    /// `UserMessageAdded` + `ResponseChunk`/`ResponseComplete`/`ResponseError`
    /// signals unicast to this caller. Send the WHOLE set each time it changes
    /// (open/switch/close); an empty list unsubscribes from all. A caller still
    /// receives turns it drives itself regardless of this set.
    async fn subscribe_conversations(
        &self,
        #[zbus(header)] hdr: zbus::message::Header<'_>,
        conversation_ids: Vec<String>,
    ) -> fdo::Result<()> {
        let caller = hdr.sender().map(|s| s.as_str());
        self.dispatch_subscribe_conversations(caller, conversation_ids)
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

    /// Signal emitted when a user message is committed and a turn starts in a
    /// conversation this caller is viewing (via `SubscribeConversations`) —
    /// including turns this caller did NOT initiate. Forwarded from
    /// `Event::UserMessageAdded`; the initiator dedupes on `request_id` (#367).
    #[zbus(signal)]
    async fn user_message_added(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
        request_id: &str,
        content: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted when the user's conversation list changed
    /// (created/renamed/deleted/(un)archived) by any client or the voice daemon.
    /// Broadcast to every D-Bus client so sidebars refresh; carries only the
    /// affected `conversation_id` (clients re-fetch the list). Forwarded from
    /// `Event::ConversationListChanged` (#367).
    #[zbus(signal)]
    async fn conversation_list_changed(
        emitter: &SignalEmitter<'_>,
        conversation_id: &str,
    ) -> zbus::Result<()>;

    /// Signal emitted when a turn suspends on a client-side tool call (#320),
    /// **unicast** to the session that registered the tool. The client runs
    /// `tool_name` with `arguments_json` (the tool input as a JSON string) and
    /// posts the outcome back via a `ClientToolResult` command on the generic
    /// `Commands` channel, carrying the same `task_id` + `tool_call_id`. Forwarded
    /// from `Event::ClientToolCall`.
    #[zbus(signal)]
    async fn client_tool_call(
        emitter: &SignalEmitter<'_>,
        task_id: &str,
        conversation_id: &str,
        tool_call_id: &str,
        tool_name: &str,
        arguments_json: &str,
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

        // Drives the shared body directly (the `#[interface]` method only adds
        // header→caller extraction, which zbus fills at dispatch). `None` caller
        // ⇒ shared transport, exactly what a registry-less adapter does.
        let request_id = adapter
            .dispatch_send_message(
                None,
                "conv-1",
                "what's the weather?",
                "Respond briefly, by voice.".to_string(),
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

        adapter
            .dispatch_send_message(None, "conv-2", "hello", String::new())
            .await
            .unwrap();

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

    // -------------------------------------------------------------------------
    // Contract coverage for the remaining methods: each builds the canonical
    // `api::Command` and maps its result; the rich logic (get_messages slicing,
    // list_conversations normalization) gets edge-case tests.
    // -------------------------------------------------------------------------

    fn conv(messages: Vec<(&str, &str)>) -> api::CommandResult {
        api::CommandResult::Conversation(api::ConversationView {
            id: "c1".to_string(),
            title: "Chat".to_string(),
            messages: messages
                .into_iter()
                .map(|(role, content)| api::MessageView {
                    id: String::new(),
                    role: role.to_string(),
                    content: content.to_string(),
                })
                .collect(),
            warnings: Vec::new(),
            model_selection: None,
            conversation_personality: None,
        })
    }

    async fn only_command(t: &CannedTransport) -> api::Command {
        let cmds = t.commands.lock().await;
        assert_eq!(cmds.len(), 1, "expected exactly one dispatched command");
        cmds[0].clone()
    }

    // --- create_conversation --------------------------------------------------

    #[tokio::test]
    async fn create_conversation_builds_command_and_returns_id() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::ConversationId {
            id: "new-id".to_string(),
        }));
        let id = DbusConversationsAdapter::new(Arc::clone(&t))
            .create_conversation("Trip planning")
            .await
            .unwrap();
        assert_eq!(id, "new-id");
        assert!(matches!(
            only_command(&t).await,
            api::Command::CreateConversation { title } if title == "Trip planning"
        ));
    }

    // --- list_conversations ---------------------------------------------------

    #[tokio::test]
    async fn list_conversations_positive_max_age_passes_through() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::Conversations(
            Vec::new(),
        )));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .list_conversations(7, true)
            .await
            .unwrap();
        assert!(matches!(
            only_command(&t).await,
            api::Command::ListConversations {
                max_age_days: Some(7),
                include_archived: true
            }
        ));
    }

    #[tokio::test]
    async fn list_conversations_nonpositive_max_age_means_no_limit() {
        // The contract: 0 or negative max_age_days means "no age cutoff" → None,
        // never Some(0) (which the daemon would read as "younger than 0 days").
        for days in [0, -1, -365] {
            let t = Arc::new(CannedTransport::new(api::CommandResult::Conversations(
                Vec::new(),
            )));
            DbusConversationsAdapter::new(Arc::clone(&t))
                .list_conversations(days, false)
                .await
                .unwrap();
            assert!(
                matches!(
                    only_command(&t).await,
                    api::Command::ListConversations {
                        max_age_days: None,
                        ..
                    }
                ),
                "days={days} must normalize to None"
            );
        }
    }

    #[tokio::test]
    async fn list_conversations_maps_summaries_to_tuples() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::Conversations(
            vec![api::ConversationSummary {
                id: "a".into(),
                title: "Alpha".into(),
                message_count: 3,
                updated_at: "2026-06-14".into(),
                archived: false,
            }],
        )));
        let rows = DbusConversationsAdapter::new(Arc::clone(&t))
            .list_conversations(0, false)
            .await
            .unwrap();
        assert_eq!(
            rows,
            vec![(
                "a".to_string(),
                "Alpha".to_string(),
                3,
                "2026-06-14".to_string(),
                false
            )]
        );
    }

    // --- archive / unarchive / delete / rename (Ack) --------------------------

    #[tokio::test]
    async fn archive_conversation_builds_command_and_acks() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::Ack));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .archive_conversation("x")
            .await
            .unwrap();
        assert!(
            matches!(only_command(&t).await, api::Command::ArchiveConversation { id } if id == "x")
        );
    }

    #[tokio::test]
    async fn unarchive_conversation_builds_command_and_acks() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::Ack));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .unarchive_conversation("x")
            .await
            .unwrap();
        assert!(
            matches!(only_command(&t).await, api::Command::UnarchiveConversation { id } if id == "x")
        );
    }

    #[tokio::test]
    async fn delete_conversation_builds_command_and_acks() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::Ack));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .delete_conversation("x")
            .await
            .unwrap();
        assert!(
            matches!(only_command(&t).await, api::Command::DeleteConversation { id } if id == "x")
        );
    }

    #[tokio::test]
    async fn rename_conversation_builds_command_with_id_and_title() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::Ack));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .rename_conversation("c1", "Renamed")
            .await
            .unwrap();
        assert!(matches!(
            only_command(&t).await,
            api::Command::RenameConversation { id, title } if id == "c1" && title == "Renamed"
        ));
    }

    // --- get_conversation -----------------------------------------------------

    #[tokio::test]
    async fn get_conversation_maps_view_to_id_title_and_role_content_tuples() {
        let t = Arc::new(CannedTransport::new(conv(vec![
            ("user", "hi"),
            ("assistant", "hello"),
        ])));
        let (id, title, messages) = DbusConversationsAdapter::new(Arc::clone(&t))
            .get_conversation("c1")
            .await
            .unwrap();
        assert_eq!(id, "c1");
        assert_eq!(title, "Chat");
        assert_eq!(
            messages,
            vec![
                ("user".to_string(), "hi".to_string()),
                ("assistant".to_string(), "hello".to_string())
            ]
        );
    }

    // --- get_messages: a thin translator over the wire windowing -------------
    // The slicing lives in — and is tested by — the daemon's `window_messages`
    // (#363); the bridge only forwards the window args and maps the result, so
    // these tests pin the *translation*, not the slicing. (This is what keeps
    // the D-Bus path identical to every UDS/WS client: one slicer, shared.)

    fn messages_view(
        total_raw_count: u32,
        truncated: bool,
        msgs: Vec<(&str, &str)>,
    ) -> api::CommandResult {
        api::CommandResult::Messages(api::MessagesView {
            total_raw_count,
            truncated,
            messages: msgs
                .into_iter()
                .map(|(role, content)| api::MessageView {
                    id: String::new(),
                    role: role.to_string(),
                    content: content.to_string(),
                })
                .collect(),
        })
    }

    #[tokio::test]
    async fn get_messages_forwards_window_args_to_the_daemon_verbatim() {
        // The bridge must NOT interpret tail/after_count/roles itself — it
        // forwards them so the daemon's single `window_messages` does the work.
        let t = Arc::new(CannedTransport::new(messages_view(0, false, vec![])));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .get_messages("c1", 50, 10, vec!["user".to_string()])
            .await
            .unwrap();
        match only_command(&t).await {
            api::Command::GetMessages {
                conversation_id,
                tail,
                after_count,
                include_roles,
            } => {
                assert_eq!(conversation_id, "c1");
                assert_eq!(tail, 50);
                assert_eq!(after_count, 10);
                assert_eq!(include_roles, vec!["user".to_string()]);
            }
            other => panic!("expected GetMessages, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn get_messages_maps_view_to_total_truncated_and_tuples() {
        let t = Arc::new(CannedTransport::new(messages_view(
            42,
            true,
            vec![("user", "hi"), ("assistant", "yo")],
        )));
        let (total, truncated, messages) = DbusConversationsAdapter::new(Arc::clone(&t))
            .get_messages("c1", 2, -1, vec![])
            .await
            .unwrap();
        assert_eq!(total, 42, "total_raw_count passes through");
        assert!(truncated, "truncated passes through");
        assert_eq!(
            messages,
            vec![
                ("user".to_string(), "hi".to_string()),
                ("assistant".to_string(), "yo".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn get_messages_errors_on_unexpected_result_variant() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::Ack));
        let err = DbusConversationsAdapter::new(Arc::clone(&t))
            .get_messages("c1", -1, -1, vec![])
            .await
            .expect_err("a non-Messages result must error");
        assert!(matches!(err, fdo::Error::Failed(_)));
    }

    // --- clear_all_history ----------------------------------------------------

    #[tokio::test]
    async fn clear_all_history_returns_the_deleted_count() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::Cleared {
            deleted_count: 9,
        }));
        let n = DbusConversationsAdapter::new(Arc::clone(&t))
            .clear_all_history()
            .await
            .unwrap();
        assert_eq!(n, 9);
        assert!(matches!(
            only_command(&t).await,
            api::Command::ClearAllHistory
        ));
    }

    // --- get_conversation_personality -----------------------------------------

    #[tokio::test]
    async fn get_conversation_personality_is_all_unset_when_none_stored() {
        // No stored override → every trait reads back as -1 (fall back to global).
        let t = Arc::new(CannedTransport::new(conv(vec![])));
        let ordinals = DbusConversationsAdapter::new(Arc::clone(&t))
            .get_conversation_personality("c1")
            .await
            .unwrap();
        assert_eq!(ordinals, (-1, -1, -1, -1, -1, -1, -1));
    }

    // --- send_prompt ----------------------------------------------------------

    #[tokio::test]
    async fn send_prompt_builds_send_message_with_empty_refinement() {
        let t = Arc::new(CannedTransport::new(api::CommandResult::SendMessageAck {
            request_id: "r".into(),
            task_id: "t".into(),
        }));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .dispatch_send_message(None, "c1", "hello there", String::new())
            .await
            .unwrap();
        match only_command(&t).await {
            api::Command::SendMessage {
                conversation_id,
                content,
                system_refinement,
                ..
            } => {
                assert_eq!(conversation_id, "c1");
                assert_eq!(content, "hello there");
                assert!(
                    system_refinement.is_empty(),
                    "plain send_prompt carries no refinement"
                );
            }
            other => panic!("expected SendMessage, got {other:?}"),
        }
    }

    // --- subscribe_conversations (#367) ---------------------------------------

    #[tokio::test]
    async fn subscribe_conversations_builds_set_replace_command() {
        // The bridge forwards the WHOLE viewed-set verbatim — the daemon does the
        // set-replace. (Routing to the caller's session is covered in session.rs;
        // with no registry wired here it falls back to the shared transport, which
        // is enough to assert the command the adapter builds.)
        let t = Arc::new(CannedTransport::new(api::CommandResult::Ack));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .dispatch_subscribe_conversations(Some(":1.10"), vec!["c1".into(), "c2".into()])
            .await
            .unwrap();
        match only_command(&t).await {
            api::Command::SubscribeConversations { conversation_ids } => {
                assert_eq!(conversation_ids, vec!["c1".to_string(), "c2".to_string()]);
            }
            other => panic!("expected SubscribeConversations, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn subscribe_conversations_empty_list_is_forwarded_as_unsubscribe() {
        // An empty set is the documented "unsubscribe from all"; it must still be
        // forwarded (not dropped) so the daemon clears the viewed-set.
        let t = Arc::new(CannedTransport::new(api::CommandResult::Ack));
        DbusConversationsAdapter::new(Arc::clone(&t))
            .dispatch_subscribe_conversations(Some(":1.10"), vec![])
            .await
            .unwrap();
        assert!(matches!(
            only_command(&t).await,
            api::Command::SubscribeConversations { conversation_ids } if conversation_ids.is_empty()
        ));
    }

    #[tokio::test]
    async fn subscribe_conversations_errors_on_unexpected_result_variant() {
        // A non-Ack result is a contract violation and must surface as an error,
        // not be swallowed.
        let t = Arc::new(CannedTransport::new(api::CommandResult::ConversationId {
            id: "x".into(),
        }));
        let err = DbusConversationsAdapter::new(Arc::clone(&t))
            .dispatch_subscribe_conversations(Some(":1.10"), vec!["c1".into()])
            .await
            .expect_err("a non-Ack SubscribeConversations result must error");
        assert!(matches!(err, fdo::Error::Failed(_)));
    }

    // -------------------------------------------------------------------------
    // Cross-transport parity guard.
    //
    // The bridge adapters and client-common's `AssistantCommands` (tui/gtk over
    // UDS/WS) are two independent layers that translate the same operation into
    // an `api::Command`. They must build the SAME command — past that point the
    // daemon's single handler makes behavior identical, so command-equality IS
    // the parity contract, asserted where divergence can happen. (This is the
    // regression guard for the get_messages divergence: pre-fix the bridge built
    // GetConversation + sliced client-side while client-common built GetMessages.)
    // -------------------------------------------------------------------------
    use desktop_assistant_client_common::AssistantCommands;

    /// Records the command client-common builds — every typed method funnels
    /// through `send_command`.
    struct RecordingClient {
        seen: std::sync::Mutex<Vec<api::Command>>,
        reply: api::CommandResult,
    }
    impl RecordingClient {
        fn new(reply: api::CommandResult) -> Self {
            Self {
                seen: std::sync::Mutex::new(Vec::new()),
                reply,
            }
        }
        fn command(&self) -> api::Command {
            self.seen
                .lock()
                .unwrap()
                .first()
                .cloned()
                .expect("a command was sent")
        }
    }
    #[async_trait::async_trait]
    impl AssistantCommands for RecordingClient {
        async fn send_command(&self, command: api::Command) -> anyhow::Result<api::CommandResult> {
            self.seen.lock().unwrap().push(command);
            Ok(self.reply.clone())
        }
    }

    #[tokio::test]
    async fn get_messages_builds_the_same_command_on_dbus_and_client_common() {
        let reply = api::CommandResult::Messages(api::MessagesView {
            total_raw_count: 0,
            truncated: false,
            messages: Vec::new(),
        });
        let bridge = Arc::new(CannedTransport::new(reply.clone()));
        DbusConversationsAdapter::new(Arc::clone(&bridge))
            .get_messages("c1", 50, 10, vec!["user".to_string()])
            .await
            .unwrap();
        let client = RecordingClient::new(reply);
        client
            .get_messages("c1", 50, 10, vec!["user".to_string()])
            .await
            .unwrap();
        assert_eq!(
            only_command(&bridge).await,
            client.command(),
            "get_messages must build the same api::Command over D-Bus and UDS/WS"
        );
    }

    #[tokio::test]
    async fn create_conversation_builds_the_same_command_on_dbus_and_client_common() {
        let reply = api::CommandResult::ConversationId {
            id: "x".to_string(),
        };
        let bridge = Arc::new(CannedTransport::new(reply.clone()));
        DbusConversationsAdapter::new(Arc::clone(&bridge))
            .create_conversation("Trip planning")
            .await
            .unwrap();
        let client = RecordingClient::new(reply);
        client.create_conversation("Trip planning").await.unwrap();
        assert_eq!(only_command(&bridge).await, client.command());
    }

    #[tokio::test]
    async fn delete_conversation_builds_the_same_command_on_dbus_and_client_common() {
        let bridge = Arc::new(CannedTransport::new(api::CommandResult::Ack));
        DbusConversationsAdapter::new(Arc::clone(&bridge))
            .delete_conversation("c1")
            .await
            .unwrap();
        let client = RecordingClient::new(api::CommandResult::Ack);
        client.delete_conversation("c1").await.unwrap();
        assert_eq!(only_command(&bridge).await, client.command());
    }
}
