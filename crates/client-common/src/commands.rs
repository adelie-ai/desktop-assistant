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

/// Why a command did not succeed, as the daemon reported it.
///
/// `Display` is the daemon's message verbatim, so every existing caller that
/// renders the error string sees exactly what it saw before. `detail` carries
/// the daemon's machine-readable classification when there is one (#728), so a
/// client can tell an authorization refusal from a technical failure without
/// matching English text - the whole point of the structured shape.
///
/// `detail` is `None` when the daemon did not classify the failure, or when the
/// failure never reached the daemon at all (a transport error, a timeout, a
/// closed connection). Absent means "unclassified", never "not an
/// authorization problem".
///
/// [`AssistantCommands::send_command`] returns `anyhow::Result`, which keeps
/// this as the error's source, so a caller recovers the classification with
/// `err.downcast_ref::<CommandFailure>()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandFailure {
    /// The daemon's human-readable message.
    pub message: String,
    /// The daemon's classification, when it supplied one.
    pub detail: Option<api::ErrorDetail>,
}

impl CommandFailure {
    /// A failure the daemon did not classify - a transport error, a timeout, or
    /// a message from a daemon that predates the classification.
    pub fn unclassified(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            detail: None,
        }
    }

    /// The stable code, when the daemon supplied one. Branch on this, never on
    /// [`Self::message`].
    pub fn code(&self) -> Option<&api::ErrorCode> {
        self.detail.as_ref().map(|d| &d.code)
    }

    /// Whether the daemon refused this command for lack of capability (#728).
    /// A client uses it to prompt for a different account, or to mark the
    /// surface read-only, rather than showing a bare failure.
    pub fn is_not_authorized(&self) -> bool {
        self.code() == Some(&api::ErrorCode::NotAuthorized)
    }

    /// Text fit to show a person: the daemon's user-facing message when it
    /// classified the failure, otherwise the raw message.
    pub fn user_message(&self) -> &str {
        self.detail
            .as_ref()
            .map(|d| d.message.as_str())
            .unwrap_or(&self.message)
    }
}

impl std::fmt::Display for CommandFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CommandFailure {}

/// Outcome of a single in-flight request, stored by request id in each
/// client's pending map: `Ok` on a `WsFrame::Result`, `Err` on a
/// `WsFrame::Error` or a connection-level failure.
pub type PendingResult = Result<api::CommandResult, CommandFailure>;

/// Both correlation ids the daemon returns for an accepted `SendMessage`
/// (issue #138).
///
/// A streaming client usually needs only `request_id` (see
/// [`AssistantCommands::send_prompt_idempotent`], which returns just that). A
/// client that wants to offer **Cancel** for the in-flight turn also needs
/// `task_id` — the background-task handle `CancelBackgroundTask { id: task_id }`
/// acts on — so [`AssistantCommands::send_prompt_idempotent_ack`] surfaces both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SendAck {
    /// The turn correlation id every streamed `AssistantDelta` /
    /// `AssistantCompleted` / `AssistantError` event is stamped with (voice#49).
    /// Empty when a legacy daemon replied with a bare `Ack`.
    pub request_id: String,
    /// The registered background-task id, used to drive Cancel
    /// (`CancelBackgroundTask { id: task_id }`) and correlate `Task*` events.
    /// Empty when a legacy daemon replied with a bare `Ack` (no Cancel possible).
    pub task_id: String,
}

/// What a client says about the trace a turn belongs to.
///
/// A turn starts when a person commits an input, and that happens in the
/// client, so the client is the top of the trace and mints the id. The daemon
/// adopts it as the `request_id` it stamps every streamed event with, so one
/// value is greppable in the client's own log, in the daemon's log, and in a
/// trace backend.
///
/// None of this needs an exporter. A desktop client installed with
/// `cargo install` runs a default-feature build that exports no span at all,
/// and it still mints and sends the id, so the daemon produces a trace whose id
/// matches what the client printed. Correlation survives where export does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnTrace {
    /// The turn's correlation id, as a uuid. The daemon adopts a well-formed,
    /// non-nil value and mints its own otherwise.
    pub turn_id: String,
    /// A W3C `traceparent` for a caller that is already inside a trace, so the
    /// daemon continues that trace rather than starting one. `None` where the
    /// client is the top of the trace, which is the ordinary case.
    pub traceparent: Option<String>,
}

impl TurnTrace {
    /// A fresh turn id, and no trace to continue.
    ///
    /// The client is the root of its own trace. This is what every
    /// `send_prompt*` method uses.
    #[must_use]
    pub fn mint() -> Self {
        Self {
            turn_id: uuid::Uuid::new_v4().to_string(),
            traceparent: None,
        }
    }

    /// A turn id, plus the trace a caller is already inside.
    #[must_use]
    pub fn continuing(turn_id: String, traceparent: String) -> Self {
        Self {
            turn_id,
            traceparent: Some(traceparent),
        }
    }
}

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
        self.create_conversation_with_tags(title, vec![]).await
    }

    async fn create_conversation_with_tags(
        &self,
        title: &str,
        tags: Vec<String>,
    ) -> Result<String> {
        let result = self
            .send_command(api::Command::CreateConversation {
                title: title.to_string(),
                tags,
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
    /// `system_refinement` is equivalent to [`Self::send_prompt`].
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
    /// [`Self::send_prompt_full`]. Every `send_prompt*` method routes through here so
    /// the ack-handling and wire shape live in one place.
    async fn send_prompt_idempotent(
        &self,
        conversation_id: &str,
        prompt: &str,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        idempotency_key: Option<String>,
    ) -> Result<String> {
        // Delegate to the ack variant and keep only the correlation
        // `request_id` — the id every streamed `AssistantDelta` /
        // `AssistantCompleted` / `AssistantError` event for this turn is
        // stamped with, so a streaming client can correlate the response back to
        // its send (voice#49). The `task_id` (the background-task handle used to
        // drive Cancel) is dropped here on purpose; a caller that needs it uses
        // [`send_prompt_idempotent_ack`](Self::send_prompt_idempotent_ack).
        self.send_prompt_idempotent_ack(
            conversation_id,
            prompt,
            override_selection,
            system_refinement,
            idempotency_key,
        )
        .await
        .map(|ack| ack.request_id)
    }

    /// Like [`send_prompt_idempotent`](Self::send_prompt_idempotent) but returns
    /// **both** correlation ids as a [`SendAck`] — including the `task_id` a
    /// client needs to offer **Cancel** for the in-flight turn
    /// (`CancelBackgroundTask { id: task_id }`, issue #138), which the
    /// `request_id`-only variant intentionally drops.
    ///
    /// This is the single chokepoint every `send_prompt*` method routes through,
    /// so the ack-handling and wire shape live in one place.
    ///
    /// The turn id it sends is minted here. See
    /// [`send_prompt_traced`](Self::send_prompt_traced) for a caller that has a
    /// trace of its own to continue.
    async fn send_prompt_idempotent_ack(
        &self,
        conversation_id: &str,
        prompt: &str,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        idempotency_key: Option<String>,
    ) -> Result<SendAck> {
        self.send_prompt_traced(
            conversation_id,
            prompt,
            override_selection,
            system_refinement,
            idempotency_key,
            TurnTrace::mint(),
        )
        .await
    }

    /// Like [`send_prompt_idempotent_ack`](Self::send_prompt_idempotent_ack)
    /// but with the turn's trace context stated rather than minted.
    ///
    /// A caller that already sits inside a trace - the web BFF forwarding a
    /// browser's turn, one daemon calling another - passes what it has, and the
    /// daemon continues that trace instead of starting a second one. A caller
    /// that only wants the correlation id passes [`TurnTrace::mint`], which is
    /// what every other `send_prompt*` method does.
    async fn send_prompt_traced(
        &self,
        conversation_id: &str,
        prompt: &str,
        override_selection: Option<api::SendPromptOverride>,
        system_refinement: String,
        idempotency_key: Option<String>,
        trace: TurnTrace,
    ) -> Result<SendAck> {
        let result = self
            .send_command(api::Command::SendMessage {
                conversation_id: conversation_id.to_string(),
                content: prompt.to_string(),
                override_selection,
                system_refinement,
                // These shared clients carry their client context on the
                // per-connection handshake, not per turn (#557 is for the
                // browser-multiplexed web BFF); so no per-turn override here.
                client_context: None,
                idempotency_key,
                turn_id: Some(trace.turn_id),
                traceparent: trace.traceparent,
            })
            .await?;
        // The daemon replies with `SendMessageAck { request_id, task_id }`:
        // `request_id` correlates the streamed reply (voice#49); `task_id` is the
        // background-task handle Cancel acts on (#138).
        //
        // A legacy / pre-#114 daemon that still replies with a bare `Ack` carries
        // no correlation ids; we surface empty strings (such a client falls back
        // to matching events loosely and simply cannot offer Cancel).
        match result {
            api::CommandResult::SendMessageAck {
                request_id,
                task_id,
            } => Ok(SendAck {
                request_id,
                task_id,
            }),
            api::CommandResult::Ack => Ok(SendAck {
                request_id: String::new(),
                task_id: String::new(),
            }),
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

    /// Trigger an on-demand knowledge-maintenance pass (the "dream cycle"
    /// controls). Returns immediately with the background task's id; progress
    /// and completion arrive as `Task*` signals, and the pass broadcasts
    /// `KnowledgeChanged` as entries land. Cancel it via the task id.
    async fn start_knowledge_maintenance(&self, op: api::MaintenanceOp) -> Result<String> {
        let result = self
            .send_command(api::Command::StartKnowledgeMaintenance { op })
            .await?;
        let api::CommandResult::MaintenanceTaskStarted { task_id } = result else {
            return Err(anyhow!(
                "unexpected response for start_knowledge_maintenance"
            ));
        };
        Ok(task_id)
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
    /// e.g. set `done` to check a todo off). Returns the saved note(s): a
    /// one-element vec on a normal write.
    ///
    /// Returns an **empty** vec — `Ok(vec![])`, not an error — when the
    /// write was silently refused. Today the only cause is `conversation_id`
    /// naming a conversation this caller does not own: the daemon's
    /// cross-tenant guard fails closed there rather than touching another
    /// user's note, so nothing is written. Check the returned vec's length;
    /// `Ok` alone does not mean the note was saved.
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

    // --- Per-conversation tool-provenance-gate override (issue #1007) ------

    /// Set the conversation's tool-provenance-gate override (#1007).
    ///
    /// `true` disables the gate (`crates/core/src/tool_provenance.rs`) for
    /// every turn in this conversation, so a turn that reads outside content
    /// may still call an acting tool afterward. `false` leaves it enforced.
    /// Returns the stored value after the write. The current value is also
    /// surfaced on [`ConversationDetail::tool_gate_disabled`] from
    /// `get_conversation`.
    async fn set_conversation_tool_gate(
        &self,
        conversation_id: &str,
        disabled: bool,
    ) -> Result<bool> {
        let result = self
            .send_command(api::Command::SetConversationToolGate {
                conversation_id: conversation_id.to_string(),
                disabled,
            })
            .await?;
        let api::CommandResult::ConversationToolGate { disabled: stored } = result else {
            return Err(anyhow!(
                "unexpected response for set_conversation_tool_gate"
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

    // --- Database / backend-tasks / WS-auth settings (#314) ----------------
    //
    // Socket-transport equivalents of the in-process D-Bus settings methods
    // (bridge cutover 2/7). Each is a thin default over `send_command`, so the
    // UDS, WS, and D-Bus clients all inherit it.

    /// Read database settings (#314). The returned `url` is redacted: its
    /// password reads `api::secret_url::REDACTED_PASSWORD` and the rest of the
    /// connection string is intact, so it is safe to render but is not a
    /// usable credential.
    async fn get_database_settings(&self) -> Result<api::DatabaseSettingsView> {
        let result = self.send_command(api::Command::GetDatabaseSettings).await?;
        let api::CommandResult::DatabaseSettings(view) = result else {
            return Err(anyhow!("unexpected response for get_database_settings"));
        };
        Ok(view)
    }

    /// Update database settings (#314). An empty `url` clears it.
    ///
    /// A settings form may post back the redacted `url` it was given: that
    /// keeps the stored password rather than overwriting it, so long as
    /// nothing else in the URL was edited. Editing any other part requires
    /// re-entering the password — the daemon refuses to splice its stored
    /// credential into a URL the client has changed.
    async fn set_database_settings(&self, url: &str, max_connections: u32) -> Result<()> {
        let result = self
            .send_command(api::Command::SetDatabaseSettings {
                url: url.into(),
                max_connections,
            })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for set_database_settings"));
        };
        Ok(())
    }

    /// Read backend-tasks settings (#314). No secret is returned.
    async fn get_backend_tasks_settings(&self) -> Result<api::BackendTasksSettingsView> {
        let result = self
            .send_command(api::Command::GetBackendTasksSettings)
            .await?;
        let api::CommandResult::BackendTasksSettings(view) = result else {
            return Err(anyhow!(
                "unexpected response for get_backend_tasks_settings"
            ));
        };
        Ok(view)
    }

    /// Update backend-tasks settings (#314). An empty `llm_connector` clears the
    /// separate backend-tasks LLM override.
    #[allow(clippy::too_many_arguments)]
    async fn set_backend_tasks_settings(
        &self,
        llm_connector: &str,
        llm_model: &str,
        llm_base_url: &str,
        dreaming_enabled: bool,
        dreaming_interval_secs: u64,
        archive_after_days: u32,
    ) -> Result<()> {
        let result = self
            .send_command(api::Command::SetBackendTasksSettings {
                llm_connector: llm_connector.to_string(),
                llm_model: llm_model.to_string(),
                llm_base_url: llm_base_url.to_string(),
                dreaming_enabled,
                dreaming_interval_secs,
                archive_after_days,
            })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!(
                "unexpected response for set_backend_tasks_settings"
            ));
        };
        Ok(())
    }

    /// Read WebSocket auth settings (#314). No signing secret is returned — only
    /// the method list and the non-sensitive OIDC discovery fields.
    async fn get_ws_auth_settings(&self) -> Result<api::WsAuthSettingsView> {
        let result = self.send_command(api::Command::GetWsAuthSettings).await?;
        let api::CommandResult::WsAuthSettings(view) = result else {
            return Err(anyhow!("unexpected response for get_ws_auth_settings"));
        };
        Ok(view)
    }

    /// Update WebSocket auth settings (#314).
    async fn set_ws_auth_settings(
        &self,
        methods: Vec<String>,
        oidc_issuer: &str,
        oidc_auth_endpoint: &str,
        oidc_token_endpoint: &str,
        oidc_client_id: &str,
        oidc_scopes: &str,
    ) -> Result<()> {
        let result = self
            .send_command(api::Command::SetWsAuthSettings {
                methods,
                oidc_issuer: oidc_issuer.to_string(),
                oidc_auth_endpoint: oidc_auth_endpoint.to_string(),
                oidc_token_endpoint: oidc_token_endpoint.to_string(),
                oidc_client_id: oidc_client_id.to_string(),
                oidc_scopes: oidc_scopes.to_string(),
            })
            .await?;
        let api::CommandResult::Ack = result else {
            return Err(anyhow!("unexpected response for set_ws_auth_settings"));
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

    // -----------------------------------------------------------------
    // The client is the top of the trace (epic D12).
    // -----------------------------------------------------------------

    /// The turn id the last recorded send carried.
    fn sent_turn_id(client: &RecordingClient) -> String {
        match client.last() {
            api::Command::SendMessage { turn_id, .. } => {
                turn_id.expect("every send from this client carries a turn id")
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    fn ack_client() -> RecordingClient {
        RecordingClient::new(api::CommandResult::SendMessageAck {
            request_id: "req-1".to_string(),
            task_id: "task-1".to_string(),
        })
    }

    #[tokio::test]
    async fn client_mints_and_propagates_with_otel_off() {
        // This crate has no `otel` feature and resolves no opentelemetry crate.
        // A desktop install from `cargo install` runs exactly this build and
        // exports no span at all, and it still mints the id the daemon adopts,
        // so the daemon produces a trace whose id matches what the client
        // printed. Correlation survives where export does not.
        let client = ack_client();
        client.send_prompt("conv-1", "hello").await.unwrap();

        let turn_id = sent_turn_id(&client);
        let parsed = uuid::Uuid::parse_str(&turn_id)
            .unwrap_or_else(|e| panic!("the client must mint a uuid, got {turn_id:?}: {e}"));
        assert!(
            !parsed.is_nil(),
            "the nil uuid spells the trace id the W3C spec reserves as invalid,              so the daemon would discard it and mint its own"
        );
    }

    #[tokio::test]
    async fn every_send_path_mints_a_turn_id() {
        // Four public entry points reach one chokepoint. A path that skipped
        // the mint would leave that client silently rooted at the daemon.
        let client = ack_client();

        client.send_prompt("conv-1", "hello").await.unwrap();
        let plain = sent_turn_id(&client);

        client
            .send_prompt_with_override("conv-1", "hello", None)
            .await
            .unwrap();
        let with_override = sent_turn_id(&client);

        client
            .send_prompt_with_system_refinement("conv-1", "hello", "briefly")
            .await
            .unwrap();
        let with_refinement = sent_turn_id(&client);

        client
            .send_prompt_idempotent("conv-1", "hello", None, String::new(), Some("k".into()))
            .await
            .unwrap();
        let idempotent = sent_turn_id(&client);

        for id in [&plain, &with_override, &with_refinement, &idempotent] {
            assert!(
                uuid::Uuid::parse_str(id).is_ok_and(|u| !u.is_nil()),
                "{id:?}"
            );
        }
    }

    #[tokio::test]
    async fn two_turns_get_different_turn_ids() {
        // One id per turn, not one per client. Two turns sharing an id would
        // merge two traces and make each turn's duration meaningless.
        let client = ack_client();
        client.send_prompt("conv-1", "first").await.unwrap();
        let first = sent_turn_id(&client);
        client.send_prompt("conv-1", "second").await.unwrap();
        let second = sent_turn_id(&client);
        assert_ne!(first, second);
    }

    #[tokio::test]
    async fn a_minted_turn_sends_no_traceparent() {
        // The client is the root, so there is no trace to continue. Sending an
        // invented one would make the daemon join a trace nobody started.
        let client = ack_client();
        client.send_prompt("conv-1", "hello").await.unwrap();
        match client.last() {
            api::Command::SendMessage { traceparent, .. } => assert!(traceparent.is_none()),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_caller_inside_a_trace_states_it_rather_than_minting() {
        // The web BFF's case: a browser started the turn, so the BFF forwards
        // the browser's id and the trace it belongs to rather than replacing
        // either.
        let client = ack_client();
        let header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        client
            .send_prompt_traced(
                "conv-1",
                "hello",
                None,
                String::new(),
                None,
                TurnTrace::continuing("11111111-2222-4333-8444-555555555555".into(), header.into()),
            )
            .await
            .unwrap();

        match client.last() {
            api::Command::SendMessage {
                turn_id,
                traceparent,
                ..
            } => {
                assert_eq!(
                    turn_id.as_deref(),
                    Some("11111111-2222-4333-8444-555555555555")
                );
                assert_eq!(traceparent.as_deref(), Some(header));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_turn_id_never_becomes_the_idempotency_key() {
        // Two fields, two purposes. A send with no idempotency key must still
        // carry a turn id, and the key slot must stay empty.
        let client = ack_client();
        client.send_prompt("conv-1", "hello").await.unwrap();
        match client.last() {
            api::Command::SendMessage {
                idempotency_key,
                turn_id,
                ..
            } => {
                assert!(turn_id.is_some());
                assert!(
                    idempotency_key.is_none(),
                    "minting a turn id must not enrol the send in the retry path"
                );
            }
            other => panic!("unexpected {other:?}"),
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
    async fn send_prompt_idempotent_ack_returns_both_correlation_ids() {
        // #138: the ack variant surfaces the `task_id` (the background-task
        // handle Cancel acts on) alongside the `request_id`, which the
        // `request_id`-only `send_prompt_idempotent` intentionally drops.
        let client = RecordingClient::new(api::CommandResult::SendMessageAck {
            request_id: "req-1".to_string(),
            task_id: "task-1".to_string(),
        });

        let ack = client
            .send_prompt_idempotent_ack(
                "conv-1",
                "hello",
                None,
                String::new(),
                Some("turn-key-1".to_string()),
            )
            .await
            .unwrap();

        assert_eq!(ack.request_id, "req-1");
        assert_eq!(ack.task_id, "task-1");
        // Same wire shape as the request_id-only variant: a keyed SendMessage.
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
    async fn send_prompt_idempotent_ack_surfaces_empty_ids_on_legacy_ack() {
        // A legacy / pre-#114 daemon replies with a bare `Ack` carrying no
        // correlation ids. The ack variant surfaces empty strings for both, so
        // such a client can't offer Cancel (no task id) but doesn't error.
        let client = RecordingClient::new(api::CommandResult::Ack);

        let ack = client
            .send_prompt_idempotent_ack("conv-1", "hello", None, String::new(), None)
            .await
            .unwrap();

        assert_eq!(ack.request_id, "");
        assert_eq!(ack.task_id, "");
    }

    #[tokio::test]
    async fn send_prompt_idempotent_still_returns_only_request_id() {
        // The request_id-only variant is unchanged: it delegates to the ack
        // variant and keeps just the `request_id` (what streamed events carry),
        // so its existing callers are byte-for-byte unaffected (#138).
        let client = RecordingClient::new(api::CommandResult::SendMessageAck {
            request_id: "req-9".to_string(),
            task_id: "task-9".to_string(),
        });

        let returned = client
            .send_prompt_idempotent("conv-1", "hello", None, String::new(), None)
            .await
            .unwrap();

        assert_eq!(returned, "req-9");
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
    async fn set_conversation_tool_gate_emits_command_and_returns_stored() {
        // Mirrors `set_conversation_personality_emits_command_and_returns_stored`:
        // the client-facing method must emit `SetConversationToolGate` with
        // the requested value and unwrap the stored bool from
        // `ConversationToolGate`.
        let client =
            RecordingClient::new(api::CommandResult::ConversationToolGate { disabled: true });
        let got = client
            .set_conversation_tool_gate("conv-1", true)
            .await
            .unwrap();
        assert!(got);
        match client.last() {
            api::Command::SetConversationToolGate {
                conversation_id,
                disabled,
            } => {
                assert_eq!(conversation_id, "conv-1");
                assert!(disabled);
            }
            other => panic!("expected SetConversationToolGate, got {other:?}"),
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

    // --- #314 settings client methods --------------------------------------

    #[tokio::test]
    async fn get_database_settings_emits_command_and_decodes_view() {
        let client = RecordingClient::new(api::CommandResult::DatabaseSettings(
            api::DatabaseSettingsView {
                url: "postgres://u:p@host/db".into(),
                max_connections: 8,
            },
        ));
        let view = client.get_database_settings().await.unwrap();
        assert_eq!(view.url, "postgres://u:p@host/db");
        assert_eq!(view.max_connections, 8);
        assert!(matches!(client.last(), api::Command::GetDatabaseSettings));
    }

    #[tokio::test]
    async fn set_database_settings_emits_command_with_args() {
        let client = RecordingClient::new(api::CommandResult::Ack);
        client
            .set_database_settings("postgres://u:p@host/db", 8)
            .await
            .unwrap();
        match client.last() {
            api::Command::SetDatabaseSettings {
                url,
                max_connections,
            } => {
                assert_eq!(url.as_str(), "postgres://u:p@host/db");
                assert_eq!(max_connections, 8);
            }
            other => panic!("expected SetDatabaseSettings, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn get_backend_tasks_settings_emits_command_and_decodes_view() {
        let client = RecordingClient::new(api::CommandResult::BackendTasksSettings(
            api::BackendTasksSettingsView {
                has_separate_llm: true,
                llm_connector: "ollama".into(),
                llm_model: "qwen3".into(),
                llm_base_url: "http://localhost:11434".into(),
                dreaming_enabled: true,
                dreaming_interval_secs: 1800,
                archive_after_days: 7,
            },
        ));
        let view = client.get_backend_tasks_settings().await.unwrap();
        assert!(view.has_separate_llm);
        assert_eq!(view.llm_connector, "ollama");
        assert!(matches!(
            client.last(),
            api::Command::GetBackendTasksSettings
        ));
    }

    #[tokio::test]
    async fn set_backend_tasks_settings_emits_command_with_args() {
        let client = RecordingClient::new(api::CommandResult::Ack);
        client
            .set_backend_tasks_settings("ollama", "qwen3", "http://localhost:11434", true, 1800, 7)
            .await
            .unwrap();
        match client.last() {
            api::Command::SetBackendTasksSettings {
                llm_connector,
                llm_model,
                llm_base_url,
                dreaming_enabled,
                dreaming_interval_secs,
                archive_after_days,
            } => {
                assert_eq!(llm_connector, "ollama");
                assert_eq!(llm_model, "qwen3");
                assert_eq!(llm_base_url, "http://localhost:11434");
                assert!(dreaming_enabled);
                assert_eq!(dreaming_interval_secs, 1800);
                assert_eq!(archive_after_days, 7);
            }
            other => panic!("expected SetBackendTasksSettings, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn get_ws_auth_settings_emits_command_and_decodes_view() {
        let client = RecordingClient::new(api::CommandResult::WsAuthSettings(
            api::WsAuthSettingsView {
                methods: vec!["password".into()],
                oidc_issuer: "https://issuer.example".into(),
                oidc_auth_endpoint: String::new(),
                oidc_token_endpoint: String::new(),
                oidc_client_id: String::new(),
                oidc_scopes: String::new(),
            },
        ));
        let view = client.get_ws_auth_settings().await.unwrap();
        assert_eq!(view.methods, vec!["password".to_string()]);
        assert_eq!(view.oidc_issuer, "https://issuer.example");
        assert!(matches!(client.last(), api::Command::GetWsAuthSettings));
    }

    #[tokio::test]
    async fn set_ws_auth_settings_emits_command_with_args() {
        let client = RecordingClient::new(api::CommandResult::Ack);
        client
            .set_ws_auth_settings(
                vec!["password".into(), "oidc".into()],
                "https://issuer.example",
                "https://issuer.example/authorize",
                "https://issuer.example/token",
                "client-123",
                "openid profile",
            )
            .await
            .unwrap();
        match client.last() {
            api::Command::SetWsAuthSettings {
                methods,
                oidc_issuer,
                oidc_client_id,
                ..
            } => {
                assert_eq!(methods, vec!["password".to_string(), "oidc".to_string()]);
                assert_eq!(oidc_issuer, "https://issuer.example");
                assert_eq!(oidc_client_id, "client-123");
            }
            other => panic!("expected SetWsAuthSettings, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn settings_getters_reject_unexpected_result() {
        // A mismatched result variant must be a clean error, not a panic.
        let client = RecordingClient::new(api::CommandResult::Ack);
        assert!(client.get_database_settings().await.is_err());
        assert!(client.get_backend_tasks_settings().await.is_err());
        assert!(client.get_ws_auth_settings().await.is_err());
    }
}

#[cfg(test)]
mod failure_tests {
    use super::*;

    fn refusal() -> CommandFailure {
        CommandFailure {
            message: "not authorized: 'set_api_key' requires the administrator capability"
                .to_string(),
            detail: Some(api::ErrorDetail {
                code: api::ErrorCode::NotAuthorized,
                description: "not authorized: 'set_api_key' requires the administrator capability"
                    .to_string(),
                message: "Only a daemon administrator can do that.".to_string(),
                retryable: false,
            }),
        }
    }

    /// A client can tell an authorization refusal from a technical failure
    /// without matching the message text (#728).
    #[test]
    fn a_refusal_is_recognisable_without_reading_the_message() {
        let failure = refusal();
        assert!(failure.is_not_authorized());
        assert_eq!(failure.code(), Some(&api::ErrorCode::NotAuthorized));
        assert_eq!(
            failure.user_message(),
            "Only a daemon administrator can do that."
        );
    }

    /// An unclassified failure is not mistaken for an authorized outcome, and
    /// still renders something useful.
    #[test]
    fn an_unclassified_failure_is_not_read_as_authorized() {
        let failure = CommandFailure::unclassified("uds response channel closed");
        assert!(!failure.is_not_authorized());
        assert_eq!(failure.code(), None);
        assert_eq!(failure.user_message(), "uds response channel closed");
    }

    /// `Display` is unchanged, so every existing caller that renders the error
    /// string sees exactly what it saw before.
    #[test]
    fn display_is_the_daemons_message_verbatim() {
        assert_eq!(
            refusal().to_string(),
            "not authorized: 'set_api_key' requires the administrator capability"
        );
    }

    /// `send_command` returns `anyhow::Result`, so the classification must
    /// survive the conversion for a caller to recover it.
    #[test]
    fn the_classification_survives_the_anyhow_boundary() {
        let err = anyhow::Error::new(refusal());
        let recovered = err
            .downcast_ref::<CommandFailure>()
            .expect("the failure must remain downcastable");
        assert!(recovered.is_not_authorized());
    }
}
