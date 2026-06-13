//! Shared, transport-agnostic command surface.
//!
//! Both the WebSocket and Unix-domain-socket clients speak the same
//! `WsRequest`/`WsFrame` JSON protocol — only the connect step and the
//! on-the-wire framing differ. [`AssistantCommands`] captures that command
//! surface once: an implementor provides [`AssistantCommands::send_command`]
//! (its transport-specific request/response correlation) and inherits every
//! typed command method as a provided default.
//!
//! This is the impl-sharing counterpart to the public
//! [`crate::transport::AssistantClient`] facade, which dispatches across the
//! `TransportClient` enum. `DbusClient` talks a different wire protocol and so
//! implements its typed methods independently rather than via this trait.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use desktop_assistant_api_model as api;

use crate::types::{ConversationDetail, ConversationSummary};

/// Outcome of a single in-flight request, stored by request id in each
/// client's pending map: `Ok` on a `WsFrame::Result`, `Err(message)` on a
/// `WsFrame::Error` or a connection-level failure.
pub type PendingResult = Result<api::CommandResult, String>;

#[async_trait]
pub trait AssistantCommands: Send + Sync {
    /// Serialize `command` as a `WsRequest`, send it over the transport, and
    /// await the correlated `WsFrame` response.
    async fn send_command(&self, command: api::Command) -> Result<api::CommandResult>;

    async fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        let result = self
            .send_command(api::Command::ListConversations {
                max_age_days: None,
                include_archived: false,
            })
            .await?;
        let api::CommandResult::Conversations(items) = result else {
            return Err(anyhow!("unexpected response for list_conversations"));
        };
        Ok(items.into_iter().map(ConversationSummary::from).collect())
    }

    async fn list_conversations_with_archived(&self) -> Result<Vec<ConversationSummary>> {
        let result = self
            .send_command(api::Command::ListConversations {
                max_age_days: None,
                include_archived: true,
            })
            .await?;
        let api::CommandResult::Conversations(items) = result else {
            return Err(anyhow!("unexpected response for list_conversations"));
        };
        Ok(items.into_iter().map(ConversationSummary::from).collect())
    }

    async fn get_conversation(&self, id: &str) -> Result<ConversationDetail> {
        let result = self
            .send_command(api::Command::GetConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Conversation(conversation) = result else {
            return Err(anyhow!("unexpected response for get_conversation"));
        };
        Ok(ConversationDetail::from(conversation))
    }

    /// Windowed message fetch (CC-5 / #361): a slice of a conversation's
    /// messages instead of the whole transcript, with the UUIDv7 id on each so
    /// the client can dedupe/order/back-page. `after_count >= 0` = from that raw
    /// index; else `tail > 0` = the last `tail`; `include_roles` empty = all.
    async fn get_messages(
        &self,
        conversation_id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> Result<api::MessagesView> {
        let result = self
            .send_command(api::Command::GetMessages {
                conversation_id: conversation_id.to_string(),
                tail,
                after_count,
                include_roles,
            })
            .await?;
        let api::CommandResult::Messages(messages) = result else {
            return Err(anyhow!("unexpected response for get_messages"));
        };
        Ok(messages)
    }

    async fn create_conversation(&self, title: &str) -> Result<String> {
        let result = self
            .send_command(api::Command::CreateConversation {
                title: title.to_string(),
            })
            .await?;
        let api::CommandResult::ConversationId { id } = result else {
            return Err(anyhow!("unexpected response for create_conversation"));
        };
        Ok(id)
    }

    async fn delete_conversation(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::DeleteConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for delete_conversation"));
        };
        Ok(())
    }

    async fn rename_conversation(&self, id: &str, title: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::RenameConversation {
                id: id.to_string(),
                title: title.to_string(),
            })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for rename_conversation"));
        };
        Ok(())
    }

    async fn archive_conversation(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::ArchiveConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for archive_conversation"));
        };
        Ok(())
    }

    async fn unarchive_conversation(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::UnarchiveConversation { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for unarchive_conversation"));
        };
        Ok(())
    }

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String> {
        self.send_prompt_full(conversation_id, prompt, None, String::new())
            .await
    }

    /// Send a prompt with an optional per-message model/connection override
    /// (issue #34). Mirrors [`send_prompt`](AssistantCommands::send_prompt) but
    /// threads `override_selection` into the `SendMessage` command, so callers
    /// can pin a single message to a specific model without mutating stored
    /// conversation settings.
    async fn send_prompt_with_override(
        &self,
        conversation_id: &str,
        prompt: &str,
        override_selection: Option<api::SendPromptOverride>,
    ) -> Result<String> {
        self.send_prompt_full(conversation_id, prompt, override_selection, String::new())
            .await
    }

    /// Send a prompt with a per-request **system-prompt refinement**: an
    /// addition to the system prompt that applies to THIS turn only.
    ///
    /// `system_refinement` is appended after the conversation's normal system
    /// prompt for the LLM call, but is never stored as a message and never
    /// attached to the conversation — so it does not appear in chat history
    /// and does not affect later turns. This is how a voice client attaches
    /// instructions like "respond briefly, by voice" to a turn dictated into
    /// an existing chat without polluting the visible transcript. An empty
    /// `system_refinement` is equivalent to [`send_prompt`].
    async fn send_prompt_with_system_refinement(
        &self,
        conversation_id: &str,
        prompt: &str,
        system_refinement: &str,
    ) -> Result<String> {
        self.send_prompt_full(conversation_id, prompt, None, system_refinement.to_string())
            .await
    }

    /// Full `SendMessage` send: optional per-message model override plus an
    /// optional per-request system-prompt refinement. The three convenience
    /// methods above delegate here so the ack-handling and wire shape live in
    /// one place. `system_refinement` is omitted on the wire when empty
    /// (`#[serde(skip_serializing_if = "String::is_empty")]`), so existing
    /// callers produce a byte-identical `SendMessage`.
    async fn send_prompt_full(
        &self,
        conversation_id: &str,
        prompt: &str,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
    ) -> Result<String> {
        self.send_prompt_idempotent(
            conversation_id,
            prompt,
            override_selection,
            system_refinement,
            None,
        )
        .await
    }

    /// Like [`send_prompt_full`](AssistantCommands::send_prompt_full) but with a
    /// client-supplied **idempotency key** scoped to the conversation (#204).
    ///
    /// A retry carrying the same key after a dropped connection is
    /// de-duplicated by the daemon — re-attached to the still-running turn, or
    /// (if it already finished) the committed reply replayed — instead of
    /// re-running the turn and re-processing an action. `None` is identical to
    /// [`send_prompt_full`]. Every `send_prompt*` method routes through here so
    /// the ack-handling and wire shape live in one place.
    async fn send_prompt_idempotent(
        &self,
        conversation_id: &str,
        prompt: &str,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        idempotency_key: Option<String>,
    ) -> Result<String> {
        let result = self
            .send_command(api::Command::SendMessage {
                conversation_id: conversation_id.to_string(),
                content: prompt.to_string(),
                override_selection,
                system_refinement,
                idempotency_key,
            })
            .await?;
        // The daemon replies with `SendMessageAck { request_id, task_id }`. We
        // return the `request_id` — the id every streamed `AssistantDelta` /
        // `AssistantCompleted` / `AssistantError` event for this turn is
        // stamped with — so a streaming client can correlate the response back
        // to its send (voice#49). The `task_id` (background-task id, used to
        // correlate `Task*` events / drive Cancel) is not what a response
        // stream is keyed on, so it is intentionally not the return value here.
        //
        // A legacy / pre-#114 daemon that still replies with a bare `Ack`
        // carries no correlation id; we surface an empty string as before
        // (such a client falls back to matching events loosely).
        match result {
            api::CommandResult::SendMessageAck { request_id, .. } => Ok(request_id),
            api::CommandResult::Ack => Ok(String::new()),
            other => Err(anyhow!("unexpected response for send_prompt: {other:?}")),
        }
    }

    /// List models across every healthy connection. Pass `connection_id =
    /// Some(_)` to scope to a single connection. `refresh = true` bypasses
    /// connector caches (e.g. Bedrock).
    async fn list_available_models(
        &self,
        connection_id: Option<&str>,
        refresh: bool,
    ) -> Result<Vec<api::ModelListing>> {
        let result = self
            .send_command(api::Command::ListAvailableModels {
                connection_id: connection_id.map(str::to_string),
                refresh,
            })
            .await?;
        let api::CommandResult::Models(items) = result else {
            return Err(anyhow!("unexpected response for list_available_models"));
        };
        Ok(items)
    }

    // --- Knowledge management (issue #73) -------------------------------

    async fn list_knowledge_entries(
        &self,
        limit: u32,
        offset: u32,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<api::KnowledgeEntryView>> {
        let result = self
            .send_command(api::Command::ListKnowledgeEntries {
                limit,
                offset,
                tag_filter,
            })
            .await?;
        let api::CommandResult::KnowledgeEntries(items) = result else {
            return Err(anyhow!("unexpected response for list_knowledge_entries"));
        };
        Ok(items)
    }

    async fn get_knowledge_entry(&self, id: &str) -> Result<Option<api::KnowledgeEntryView>> {
        let result = self
            .send_command(api::Command::GetKnowledgeEntry { id: id.to_string() })
            .await?;
        let api::CommandResult::KnowledgeEntry(entry) = result else {
            return Err(anyhow!("unexpected response for get_knowledge_entry"));
        };
        Ok(entry)
    }

    async fn search_knowledge_entries(
        &self,
        query: &str,
        tag_filter: Option<Vec<String>>,
        limit: u32,
    ) -> Result<Vec<api::KnowledgeEntryView>> {
        let result = self
            .send_command(api::Command::SearchKnowledgeEntries {
                query: query.to_string(),
                tag_filter,
                limit,
            })
            .await?;
        let api::CommandResult::KnowledgeEntries(items) = result else {
            return Err(anyhow!("unexpected response for search_knowledge_entries"));
        };
        Ok(items)
    }

    async fn create_knowledge_entry(
        &self,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView> {
        let result = self
            .send_command(api::Command::CreateKnowledgeEntry {
                content: content.to_string(),
                tags,
                metadata,
            })
            .await?;
        let api::CommandResult::KnowledgeEntryWritten(entry) = result else {
            return Err(anyhow!("unexpected response for create_knowledge_entry"));
        };
        Ok(entry)
    }

    async fn update_knowledge_entry(
        &self,
        id: &str,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView> {
        let result = self
            .send_command(api::Command::UpdateKnowledgeEntry {
                id: id.to_string(),
                content: content.to_string(),
                tags,
                metadata,
            })
            .await?;
        let api::CommandResult::KnowledgeEntryWritten(entry) = result else {
            return Err(anyhow!("unexpected response for update_knowledge_entry"));
        };
        Ok(entry)
    }

    async fn delete_knowledge_entry(&self, id: &str) -> Result<()> {
        let result = self
            .send_command(api::Command::DeleteKnowledgeEntry { id: id.to_string() })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for delete_knowledge_entry"));
        };
        Ok(())
    }

    // --- Conversation scratchpad (issue #190) -----------------------------

    /// Read a conversation's scratchpad notes (ordered by type then sequence).
    async fn get_conversation_scratchpad(
        &self,
        conversation_id: &str,
        max_results: Option<u32>,
    ) -> Result<Vec<api::ScratchpadNoteView>> {
        let result = self
            .send_command(api::Command::GetConversationScratchpad {
                conversation_id: conversation_id.to_string(),
                max_results,
            })
            .await?;
        let api::CommandResult::Scratchpad(items) = result else {
            return Err(anyhow!(
                "unexpected response for get_conversation_scratchpad"
            ));
        };
        Ok(items)
    }

    /// Upsert a single scratchpad note (re-writing a key replaces its fields —
    /// e.g. set `done` to check a todo off). Returns the saved note(s).
    #[allow(clippy::too_many_arguments)]
    async fn set_scratchpad_note(
        &self,
        conversation_id: &str,
        key: &str,
        content: &str,
        note_type: &str,
        sequence: Option<i32>,
        done: bool,
    ) -> Result<Vec<api::ScratchpadNoteView>> {
        let result = self
            .send_command(api::Command::SetScratchpadNote {
                conversation_id: conversation_id.to_string(),
                key: key.to_string(),
                content: content.to_string(),
                note_type: note_type.to_string(),
                sequence,
                done,
            })
            .await?;
        let api::CommandResult::Scratchpad(items) = result else {
            return Err(anyhow!("unexpected response for set_scratchpad_note"));
        };
        Ok(items)
    }

    /// Delete scratchpad notes by key, or clear the whole pad with `all: true`.
    async fn delete_scratchpad_notes(
        &self,
        conversation_id: &str,
        keys: Vec<String>,
        all: bool,
    ) -> Result<()> {
        let result = self
            .send_command(api::Command::DeleteScratchpadNotes {
                conversation_id: conversation_id.to_string(),
                keys,
                all,
            })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for delete_scratchpad_notes"));
        };
        Ok(())
    }

    // --- Per-conversation personality override (issue #227) ----------------

    /// Set (or clear) a conversation's personality override (#227, Phase 2).
    ///
    /// `personality` is a *partial* [`api::ConversationPersonalityView`] (a
    /// [`api::PersonalityOverride`]): each `Some` trait pins that trait for the
    /// conversation, each `None` falls back to the global config on every send.
    /// An empty/all-`None` override clears it (back to global-only). Returns the
    /// stored override after the write (cleared → empty). The current value is
    /// also surfaced on [`ConversationDetail::conversation_personality`] from
    /// `get_conversation`, which a picker pre-fills from. Used by the tui/gtk
    /// personality pickers.
    async fn set_conversation_personality(
        &self,
        conversation_id: &str,
        personality: api::ConversationPersonalityView,
    ) -> Result<api::ConversationPersonalityView> {
        let result = self
            .send_command(api::Command::SetConversationPersonality {
                conversation_id: conversation_id.to_string(),
                personality,
            })
            .await?;
        let api::CommandResult::ConversationPersonality(stored) = result else {
            return Err(anyhow!(
                "unexpected response for set_conversation_personality"
            ));
        };
        Ok(stored)
    }

    // --- Client-side tool execution (issue #107 / #231) -------------------

    /// Advertise the set of client-local MCP tools this connection can run
    /// (#107). The daemon replaces any previously-registered set on each call —
    /// send the full list, not deltas — so re-register on every connect.
    /// Returns the count of tools the daemon accepted (from
    /// `CommandResult::ClientToolsRegistered`).
    async fn register_client_tools(
        &self,
        tools: Vec<api::ClientToolRegistration>,
    ) -> Result<usize> {
        let result = self
            .send_command(api::Command::RegisterClientTools { tools })
            .await?;
        let api::CommandResult::ClientToolsRegistered { count } = result else {
            return Err(anyhow!("unexpected response for register_client_tools"));
        };
        Ok(count as usize)
    }

    /// Deliver the outcome of a `ClientToolCall` back to the daemon so the
    /// suspended turn can resume (#107). Pass the `task_id` and `tool_call_id`
    /// the [`SignalEvent::ClientToolCall`](crate::SignalEvent::ClientToolCall)
    /// carried, and exactly one of `result` / `error` — the daemon treats both
    /// `None` as an error.
    async fn submit_client_tool_result(
        &self,
        task_id: &str,
        tool_call_id: &str,
        result: Result<String, String>,
    ) -> Result<()> {
        let (ok, err) = match result {
            Ok(value) => (Some(value), None),
            Err(message) => (None, Some(message)),
        };
        let outcome = self
            .send_command(api::Command::ClientToolResult {
                task_id: api::TaskId(task_id.to_string()),
                tool_call_id: tool_call_id.to_string(),
                result: ok,
                error: err,
            })
            .await?;
        let api::CommandResult::Ack = outcome else {
            return Err(anyhow!("unexpected response for submit_client_tool_result"));
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Records the last `Command` passed to `send_command` and replies with a
    /// canned `CommandResult`, so we can assert the wire command a provided
    /// default method emits without standing up a real transport.
    struct RecordingClient {
        last: Mutex<Option<api::Command>>,
        reply: api::CommandResult,
    }

    impl RecordingClient {
        fn new(reply: api::CommandResult) -> Self {
            Self {
                last: Mutex::new(None),
                reply,
            }
        }

        fn last(&self) -> api::Command {
            self.last.lock().unwrap().clone().expect("no command sent")
        }
    }

    #[async_trait]
    impl AssistantCommands for RecordingClient {
        async fn send_command(&self, command: api::Command) -> Result<api::CommandResult> {
            *self.last.lock().unwrap() = Some(command);
            Ok(self.reply.clone())
        }
    }

    #[tokio::test]
    async fn send_prompt_with_override_emits_send_message_with_override() {
        let client = RecordingClient::new(api::CommandResult::SendMessageAck {
            request_id: "req-1".to_string(),
            task_id: "task-1".to_string(),
        });
        let override_selection = Some(api::SendPromptOverride {
            connection_id: "conn-1".to_string(),
            model_id: "model-1".to_string(),
            effort: None,
        });

        let returned = client
            .send_prompt_with_override("conv-1", "hello", override_selection.clone())
            .await
            .unwrap();

        // The send returns the turn `request_id` (what streamed events carry),
        // not the `task_id` (voice#49).
        assert_eq!(returned, "req-1");
        match client.last() {
            api::Command::SendMessage {
                conversation_id,
                content,
                override_selection: emitted,
                system_refinement,
                ..
            } => {
                assert_eq!(conversation_id, "conv-1");
                assert_eq!(content, "hello");
                assert_eq!(emitted, override_selection);
                assert!(emitted.is_some());
                // The override path carries no system refinement.
                assert!(system_refinement.is_empty());
            }
            other => panic!("expected Command::SendMessage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn send_prompt_idempotent_emits_send_message_with_key() {
        let client = RecordingClient::new(api::CommandResult::SendMessageAck {
            request_id: "req-1".to_string(),
            task_id: "task-1".to_string(),
        });

        let returned = client
            .send_prompt_idempotent(
                "conv-1",
                "hello",
                None,
                String::new(),
                Some("turn-key-1".to_string()),
            )
            .await
            .unwrap();

        // Returns the correlation `request_id`, not the `task_id` (voice#49).
        assert_eq!(returned, "req-1");
        match client.last() {
            api::Command::SendMessage {
                conversation_id,
                content,
                idempotency_key,
                ..
            } => {
                assert_eq!(conversation_id, "conv-1");
                assert_eq!(content, "hello");
                assert_eq!(idempotency_key.as_deref(), Some("turn-key-1"));
            }
            other => panic!("expected Command::SendMessage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn send_prompt_full_stays_idempotency_key_free() {
        // Non-breaking: the existing entry point must keep emitting a key-less
        // SendMessage so callers that don't opt into idempotency are unchanged.
        let client = RecordingClient::new(api::CommandResult::SendMessageAck {
            request_id: "r".to_string(),
            task_id: "t".to_string(),
        });
        client
            .send_prompt_full("c", "hi", None, String::new())
            .await
            .unwrap();
        match client.last() {
            api::Command::SendMessage {
                idempotency_key, ..
            } => assert!(
                idempotency_key.is_none(),
                "send_prompt_full must not attach an idempotency key"
            ),
            other => panic!("expected Command::SendMessage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn send_prompt_with_system_refinement_emits_send_message_with_refinement() {
        let client = RecordingClient::new(api::CommandResult::SendMessageAck {
            request_id: "req-3".to_string(),
            task_id: "task-3".to_string(),
        });

        let returned = client
            .send_prompt_with_system_refinement(
                "conv-1",
                "what's the weather?",
                "You are Adele, responding by voice. Keep it brief.",
            )
            .await
            .unwrap();

        // Returns the correlation `request_id`, not the `task_id` (voice#49).
        assert_eq!(returned, "req-3");
        match client.last() {
            api::Command::SendMessage {
                conversation_id,
                content,
                override_selection,
                system_refinement,
                ..
            } => {
                assert_eq!(conversation_id, "conv-1");
                // The visible user message is the clean prompt — the
                // refinement rides a separate field, not the content.
                assert_eq!(content, "what's the weather?");
                assert!(override_selection.is_none());
                assert_eq!(
                    system_refinement,
                    "You are Adele, responding by voice. Keep it brief."
                );
            }
            other => panic!("expected Command::SendMessage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn send_prompt_with_override_accepts_legacy_bare_ack() {
        let client = RecordingClient::new(api::CommandResult::Ack);
        let task_id = client
            .send_prompt_with_override("conv-1", "hello", None)
            .await
            .unwrap();
        // Legacy daemons reply with a bare `Ack`; the task id is then empty
        // and surfaced via streaming events instead.
        assert_eq!(task_id, String::new());
    }

    #[tokio::test]
    async fn list_available_models_emits_list_available_models_command() {
        let client = RecordingClient::new(api::CommandResult::Models(vec![]));
        let models = client
            .list_available_models(Some("conn-1"), true)
            .await
            .unwrap();

        assert!(models.is_empty());
        match client.last() {
            api::Command::ListAvailableModels {
                connection_id,
                refresh,
            } => {
                assert_eq!(connection_id.as_deref(), Some("conn-1"));
                assert!(refresh);
            }
            other => panic!("expected Command::ListAvailableModels, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn promoted_methods_are_reachable_through_dyn_trait_object() {
        // The whole point of adele-gtk#49: these commands must be issuable
        // through a `&dyn AssistantCommands` (which is what `UdsClient` is
        // reached as via `TransportClient::as_commands`), not only on a
        // concrete `WsClient`.
        let client = RecordingClient::new(api::CommandResult::SendMessageAck {
            request_id: "req-2".to_string(),
            task_id: "task-2".to_string(),
        });
        let commands: &dyn AssistantCommands = &client;

        let returned = commands
            .send_prompt_with_override(
                "conv-2",
                "hi",
                Some(api::SendPromptOverride {
                    connection_id: "conn-2".to_string(),
                    model_id: "model-2".to_string(),
                    effort: None,
                }),
            )
            .await
            .unwrap();
        // Returns the correlation `request_id`, not the `task_id` (voice#49).
        assert_eq!(returned, "req-2");
        assert!(matches!(
            client.last(),
            api::Command::SendMessage {
                override_selection: Some(_),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn get_conversation_scratchpad_emits_command_and_unwraps_notes() {
        let client = RecordingClient::new(api::CommandResult::Scratchpad(vec![
            api::ScratchpadNoteView {
                id: "sp-1".into(),
                key: "goal".into(),
                content: "ship it".into(),
                note_type: "note".into(),
                sequence: None,
                done: false,
                updated_at: "t".into(),
            },
        ]));
        let notes = client
            .get_conversation_scratchpad("conv-1", Some(20))
            .await
            .unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].key, "goal");
        match client.last() {
            api::Command::GetConversationScratchpad {
                conversation_id,
                max_results,
            } => {
                assert_eq!(conversation_id, "conv-1");
                assert_eq!(max_results, Some(20));
            }
            other => panic!("expected GetConversationScratchpad, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn set_scratchpad_note_emits_command() {
        let client = RecordingClient::new(api::CommandResult::Scratchpad(vec![]));
        client
            .set_scratchpad_note("conv-1", "t1", "wire it", "todo", Some(2), true)
            .await
            .unwrap();
        match client.last() {
            api::Command::SetScratchpadNote {
                conversation_id,
                key,
                content,
                note_type,
                sequence,
                done,
            } => {
                assert_eq!(conversation_id, "conv-1");
                assert_eq!(key, "t1");
                assert_eq!(content, "wire it");
                assert_eq!(note_type, "todo");
                assert_eq!(sequence, Some(2));
                assert!(done);
            }
            other => panic!("expected SetScratchpadNote, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn delete_scratchpad_notes_emits_command() {
        let client = RecordingClient::new(api::CommandResult::Ack);
        client
            .delete_scratchpad_notes("conv-1", vec!["t1".into()], false)
            .await
            .unwrap();
        match client.last() {
            api::Command::DeleteScratchpadNotes {
                conversation_id,
                keys,
                all,
            } => {
                assert_eq!(conversation_id, "conv-1");
                assert_eq!(keys, vec!["t1".to_string()]);
                assert!(!all);
            }
            other => panic!("expected DeleteScratchpadNotes, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn set_conversation_personality_emits_command_and_returns_stored() {
        // The picker-facing client method must emit the
        // `SetConversationPersonality` command with the partial override and
        // unwrap the stored value from `ConversationPersonality`.
        let stored = api::ConversationPersonalityView {
            humor: Some(api::PersonalityLevel::Never),
            ..api::ConversationPersonalityView::default()
        };
        let client = RecordingClient::new(api::CommandResult::ConversationPersonality(stored));
        let sent = api::ConversationPersonalityView {
            humor: Some(api::PersonalityLevel::Never),
            directness: Some(api::PersonalityLevel::Always),
            ..api::ConversationPersonalityView::default()
        };
        let got = client
            .set_conversation_personality("conv-1", sent)
            .await
            .unwrap();
        assert_eq!(got, stored);
        match client.last() {
            api::Command::SetConversationPersonality {
                conversation_id,
                personality,
            } => {
                assert_eq!(conversation_id, "conv-1");
                assert_eq!(personality, sent);
            }
            other => panic!("expected SetConversationPersonality, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn register_client_tools_emits_command_and_returns_count() {
        let client = RecordingClient::new(api::CommandResult::ClientToolsRegistered { count: 2 });
        let tools = vec![
            api::ClientToolRegistration {
                name: "weather".into(),
                description: "look up the weather".into(),
                input_schema: serde_json::json!({ "type": "object" }),
            },
            api::ClientToolRegistration {
                name: "calendar".into(),
                description: String::new(),
                input_schema: serde_json::Value::Null,
            },
        ];
        let count = client.register_client_tools(tools.clone()).await.unwrap();
        assert_eq!(count, 2);
        match client.last() {
            api::Command::RegisterClientTools { tools: emitted } => {
                assert_eq!(emitted, tools);
            }
            other => panic!("expected RegisterClientTools, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn submit_client_tool_result_ok_emits_result_field() {
        let client = RecordingClient::new(api::CommandResult::Ack);
        client
            .submit_client_tool_result("task-1", "call-1", Ok("sunny".into()))
            .await
            .unwrap();
        match client.last() {
            api::Command::ClientToolResult {
                task_id,
                tool_call_id,
                result,
                error,
            } => {
                assert_eq!(task_id, api::TaskId("task-1".into()));
                assert_eq!(tool_call_id, "call-1");
                assert_eq!(result.as_deref(), Some("sunny"));
                assert!(error.is_none());
            }
            other => panic!("expected ClientToolResult, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn submit_client_tool_result_err_emits_error_field() {
        let client = RecordingClient::new(api::CommandResult::Ack);
        client
            .submit_client_tool_result("task-2", "call-2", Err("tool blew up".into()))
            .await
            .unwrap();
        match client.last() {
            api::Command::ClientToolResult { result, error, .. } => {
                // Exactly one of result/error is populated — an Err maps to the
                // error field with result left None (the daemon treats both
                // None as an error).
                assert!(result.is_none());
                assert_eq!(error.as_deref(), Some("tool blew up"));
            }
            other => panic!("expected ClientToolResult, got {other:?}"),
        }
    }
}
