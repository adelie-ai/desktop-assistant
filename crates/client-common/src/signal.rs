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
    TitleChanged {
        conversation_id: String,
        title: String,
    },
}
