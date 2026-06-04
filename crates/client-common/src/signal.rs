use desktop_assistant_api_model as api;

#[derive(Debug)]
pub enum SignalEvent {
    Chunk {
        request_id: String,
        chunk: String,
    },
    Complete {
        request_id: String,
        full_response: String,
    },
    Error {
        request_id: String,
        error: String,
    },
    Status {
        request_id: String,
        message: String,
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
    Disconnected {
        reason: String,
    },
}
