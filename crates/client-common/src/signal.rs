use desktop_assistant_api_model as api;

// `Clone` lets the `Connector` fan one signal stream out to many subscribers.
#[derive(Debug, Clone)]
pub enum SignalEvent {
    // The streaming events carry `conversation_id` (already present on the wire
    // frames) so a client can route a turn it did NOT initiate — e.g. a voice
    // turn streaming into a conversation the GUI is merely viewing — to the
    // right chat view, instead of only rendering streams it started itself.
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
    /// The daemon's turn has suspended on a client-local MCP tool call (#107).
    /// The client is expected to execute `tool_name` with `arguments` against
    /// its local environment and post the outcome back via
    /// [`Connector::submit_client_tool_result`](crate::Connector::submit_client_tool_result)
    /// with the same `task_id` and `tool_call_id`; until then the turn parks.
    /// `task_id` is the `api::TaskId` unwrapped to its inner `String`, matching
    /// the rest of this stream's id fields.
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
