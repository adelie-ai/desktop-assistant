use crate::CoreError;
use crate::domain::{Conversation, ConversationId, ConversationSummary};
use crate::ports::llm::ChunkCallback;

#[derive(Debug, Clone)]
pub struct LlmSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
}

#[derive(Debug, Clone)]
pub struct EmbeddingsSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
    pub available: bool,
    pub is_default: bool,
}

#[derive(Debug, Clone)]
pub struct ConnectorDefaultsView {
    pub llm_model: String,
    pub llm_base_url: String,
    pub embeddings_model: String,
    pub embeddings_base_url: String,
    pub embeddings_available: bool,
}

#[derive(Debug, Clone)]
pub struct PersistenceSettingsView {
    pub enabled: bool,
    /// Empty string means no remote is configured.
    pub remote_url: String,
    pub remote_name: String,
    pub push_on_update: bool,
}

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
        max_age_days: Option<u32>,
    ) -> impl std::future::Future<Output = Result<Vec<ConversationSummary>, CoreError>> + Send;

    fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<Conversation, CoreError>> + Send;

    fn delete_conversation(
        &self,
        id: &ConversationId,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn clear_all_history(&self)
    -> impl std::future::Future<Output = Result<u32, CoreError>> + Send;

    fn send_prompt(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        on_chunk: ChunkCallback,
    ) -> impl std::future::Future<Output = Result<String, CoreError>> + Send;
}

/// Inbound port for assistant settings.
///
/// Secret values are write-only through this interface and never returned.
pub trait SettingsService: Send + Sync {
    fn get_llm_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<LlmSettingsView, CoreError>> + Send;

    fn set_llm_settings(
        &self,
        connector: String,
        model: Option<String>,
        base_url: Option<String>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn set_api_key(
        &self,
        api_key: String,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn get_embeddings_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<EmbeddingsSettingsView, CoreError>> + Send;

    fn set_embeddings_settings(
        &self,
        connector: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn get_connector_defaults(
        &self,
        connector: String,
    ) -> impl std::future::Future<Output = Result<ConnectorDefaultsView, CoreError>> + Send;

    fn get_persistence_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<PersistenceSettingsView, CoreError>> + Send;

    fn set_persistence_settings(
        &self,
        enabled: bool,
        remote_url: Option<String>,
        remote_name: Option<String>,
        push_on_update: bool,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;
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
    fn _assert_settings_service<T: SettingsService>() {}
}
