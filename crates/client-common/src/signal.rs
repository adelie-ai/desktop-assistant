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
    Disconnected {
        reason: String,
    },
}
