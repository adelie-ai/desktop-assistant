//! The client-facing signal stream (`SignalEvent`) and the `api::Event` →
//! `SignalEvent` projection.
//!
//! This lives in `api-model` (not `client-common`) so the shared, wasm-targeting
//! client cores (`client-ui-common`, the web SPA) can consume the signal stream
//! and convert wire events **without** pulling `client-common`'s native
//! transport tail (tokio, tungstenite, rustls). `client-common` re-exports both
//! items so existing `client_common::SignalEvent` / `ws_client::map_event_to_signal`
//! paths are unchanged (#377).

use crate as api;

// `Clone` lets the `Connector` fan one signal stream out to many subscribers.
#[derive(Debug, Clone)]
pub enum SignalEvent {
    // The streaming events carry `conversation_id` (already present on the wire
    // frames) so a client can route a turn it did NOT initiate — e.g. a voice
    // turn streaming into a conversation the GUI is merely viewing — to the
    // right chat view, instead of only rendering streams it started itself.
    /// A user message was committed and a turn started (api `UserMessageAdded`).
    /// Emitted for every send turn, including ones this client did not initiate
    /// (a voice turn, or another client on the same account). The client renders
    /// the user bubble live in the matching `conversation_id`; the initiator
    /// dedupes on `request_id` (it already rendered the bubble optimistically).
    UserMessageAdded {
        conversation_id: String,
        request_id: String,
        content: String,
        /// Echoes the initiating `SendMessage.idempotency_key` (when present)
        /// so the initiator can correlate its optimistic user bubble by exact
        /// key match (#570). `None` for keyless send paths.
        idempotency_key: Option<String>,
    },
    Chunk {
        conversation_id: String,
        request_id: String,
        chunk: String,
    },
    Complete {
        conversation_id: String,
        request_id: String,
        full_response: String,
    },
    Error {
        conversation_id: String,
        request_id: String,
        error: String,
    },
    Status {
        conversation_id: String,
        request_id: String,
        message: String,
    },
    /// Per-turn context-window fill report (issue #341): `used_tokens` of
    /// `budget_tokens` consumed this turn, plus whether proactive compaction
    /// ran. Carries token COUNTS only — clients render a "used / budget (%)"
    /// indicator and shift colour toward the 0.85 compaction line. Delivered
    /// on the same stream as `Status`.
    ContextUsage {
        conversation_id: String,
        request_id: String,
        used_tokens: u64,
        budget_tokens: u64,
        compaction_active: bool,
    },
    TitleChanged {
        conversation_id: String,
        title: String,
    },
    /// The user's conversation list changed elsewhere — a conversation was
    /// created, renamed, deleted, or (un)archived by another client or the
    /// voice daemon (#1). The client re-fetches its conversation list so its
    /// sidebar stays in sync. Carries only the affected `conversation_id`.
    ConversationListChanged {
        conversation_id: String,
    },
    /// One-time advisory emitted by the daemon (e.g. the conversation's
    /// stored model selection no longer resolves and was cleared).
    ConversationWarning {
        conversation_id: String,
        warning: api::ConversationWarning,
    },
    /// A background task transitioned to `Pending`/`Running`. Carries the
    /// full `TaskView` so process-manager UIs can populate a list row
    /// without an extra round-trip. Sent in response to
    /// `Command::SubscribeBackgroundTasks` (issue #110).
    TaskStarted {
        task: api::TaskView,
    },
    /// Lightweight progress signal — typically used to update a per-row
    /// "progress hint" string without writing a log entry.
    TaskProgress {
        id: String,
        progress_hint: Option<String>,
    },
    /// A new log entry was appended to a task's bounded log buffer.
    TaskLogAppended {
        id: String,
        entry: api::TaskLogEntry,
    },
    /// Terminal event: the task reached `Completed`, `Failed`, or
    /// `Cancelled`. `last_error` is set for `Failed` and may be set for
    /// `Cancelled` when cancellation was driven by an upstream error.
    TaskCompleted {
        id: String,
        status: api::TaskStatus,
        last_error: Option<String>,
    },
    /// A conversation's scratchpad changed (note written or deleted), by the
    /// LLM's tools or a client command. Delivered on connections subscribed via
    /// `Command::SubscribeBackgroundTasks`; carries only the `conversation_id`
    /// so the client re-reads via `get_conversation_scratchpad` (issue #190).
    ScratchpadChanged {
        conversation_id: String,
    },
    /// The calling user's knowledge base changed (an entry was created, updated,
    /// deleted, or rewritten by a maintenance pass). Delivered on connections
    /// subscribed via `Command::SubscribeBackgroundTasks`; carries no payload so
    /// the client debounce-refetches its knowledge list (dream-cycle controls).
    KnowledgeChanged,
    /// The daemon's turn has suspended on a client-local MCP tool call (#107).
    /// The client is expected to execute `tool_name` with `arguments` against
    /// its local environment and post the outcome back via
    /// `Connector::submit_client_tool_result` with the same `task_id` and
    /// `tool_call_id`; until then the turn parks. `task_id` is the `api::TaskId`
    /// unwrapped to its inner `String`, matching the rest of this stream's id
    /// fields.
    ClientToolCall {
        task_id: String,
        conversation_id: String,
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    Disconnected {
        reason: String,
    },
}

/// Project a wire [`api::Event`] onto a client-facing [`SignalEvent`].
///
/// Returns `None` for wire events that carry no client signal (e.g.
/// `ConfigChanged`). Pure and dependency-light so it runs in both the native
/// transports and the wasm web client.
pub fn map_event_to_signal(event: api::Event) -> Option<SignalEvent> {
    match event {
        api::Event::UserMessageAdded {
            conversation_id,
            request_id,
            content,
            idempotency_key,
        } => Some(SignalEvent::UserMessageAdded {
            conversation_id,
            request_id,
            content,
            idempotency_key,
        }),
        api::Event::AssistantDelta {
            conversation_id,
            request_id,
            chunk,
        } => Some(SignalEvent::Chunk {
            conversation_id,
            request_id,
            chunk,
        }),
        api::Event::AssistantCompleted {
            conversation_id,
            request_id,
            full_response,
        } => Some(SignalEvent::Complete {
            conversation_id,
            request_id,
            full_response,
        }),
        api::Event::AssistantError {
            conversation_id,
            request_id,
            error,
        } => Some(SignalEvent::Error {
            conversation_id,
            request_id,
            error,
        }),
        api::Event::ConversationTitleChanged {
            conversation_id,
            title,
        } => Some(SignalEvent::TitleChanged {
            conversation_id,
            title,
        }),
        api::Event::ConversationListChanged { conversation_id } => {
            Some(SignalEvent::ConversationListChanged { conversation_id })
        }
        api::Event::AssistantStatus {
            conversation_id,
            request_id,
            message,
        } => Some(SignalEvent::Status {
            conversation_id,
            request_id,
            message,
        }),
        api::Event::ContextUsage {
            conversation_id,
            request_id,
            used_tokens,
            budget_tokens,
            compaction_active,
        } => Some(SignalEvent::ContextUsage {
            conversation_id,
            request_id,
            used_tokens,
            budget_tokens,
            compaction_active,
        }),
        api::Event::ConfigChanged { .. } => None,
        api::Event::ConversationWarningEmitted {
            conversation_id,
            warning,
        } => Some(SignalEvent::ConversationWarning {
            conversation_id,
            warning,
        }),
        // Background-task events (issue #110) — surfaced verbatim on the
        // signal channel so process-manager UIs (adele-tui#45, adele-gtk
        // follow-up) can react. The TaskView/TaskLogEntry types are
        // re-exported from `api-model`; clients consume them directly.
        api::Event::TaskStarted { task } => Some(SignalEvent::TaskStarted { task }),
        api::Event::TaskProgress { id, progress_hint } => {
            Some(SignalEvent::TaskProgress { id, progress_hint })
        }
        api::Event::TaskLogAppended { id, entry } => {
            Some(SignalEvent::TaskLogAppended { id, entry })
        }
        api::Event::TaskCompleted {
            id,
            status,
            last_error,
        } => Some(SignalEvent::TaskCompleted {
            id,
            status,
            last_error,
        }),
        api::Event::ScratchpadChanged { conversation_id } => {
            Some(SignalEvent::ScratchpadChanged { conversation_id })
        }
        api::Event::KnowledgeChanged => Some(SignalEvent::KnowledgeChanged),
        // Client-side tool execution (#107/#231): surfaced on the signal
        // stream so a client that advertised client-local tools (voice first)
        // can execute the requested tool and post the result back via
        // `Connector::submit_client_tool_result`. The `TaskId` newtype is
        // unwrapped to its inner `String` to match the rest of this stream.
        api::Event::ClientToolCall {
            task_id,
            conversation_id,
            tool_call_id,
            tool_name,
            arguments,
        } => Some(SignalEvent::ClientToolCall {
            task_id: task_id.0,
            conversation_id,
            tool_call_id,
            tool_name,
            arguments,
        }),
    }
}
