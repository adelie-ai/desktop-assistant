use crate::CoreError;
use crate::domain::{Conversation, ConversationId, ConversationSummary};
use crate::ports::llm::ChunkCallback;

/// Inbound port for health/status queries.
///
/// Any adapter that wants to expose assistant status (D-Bus, HTTP, etc.)
/// implements a handler that calls through this trait.
pub trait AssistantService: Send + Sync {
    /// Returns a version string for the running assistant.
    fn version(&self) -> &str;

    /// Simple liveness check.
    fn ping(&self) -> &str;
}

/// Inbound port for conversation management.
pub trait ConversationService: Send + Sync {
    fn create_conversation(
        &self,
        title: String,
    ) -> impl std::future::Future<Output = Result<Conversation, CoreError>> + Send;

    fn list_conversations(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<ConversationSummary>, CoreError>> + Send;

    fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<Conversation, CoreError>> + Send;

    fn delete_conversation(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn send_prompt(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        on_chunk: ChunkCallback,
    ) -> impl std::future::Future<Output = Result<String, CoreError>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockAssistant;

    impl AssistantService for MockAssistant {
        fn version(&self) -> &str {
            env!("CARGO_PKG_VERSION")
        }

        fn ping(&self) -> &str {
            "pong"
        }
    }

    #[test]
    fn mock_assistant_returns_version() {
        let assistant = MockAssistant;
        assert!(!assistant.version().is_empty(), "version must not be empty");
    }

    #[test]
    fn mock_assistant_responds_to_ping() {
        let assistant = MockAssistant;
        assert_eq!(assistant.ping(), "pong");
    }

    // ConversationService uses impl Future so not dyn-compatible,
    // but we verify it's implementable via the service tests in service.rs.
    fn _assert_conversation_service<T: ConversationService>() {}
}
