use crate::CoreError;
use crate::domain::{Conversation, ConversationId, ConversationSummary};
use crate::ports::llm::ChunkCallback;

#[derive(Debug, Clone)]
pub struct LlmSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
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

#[derive(Debug, Clone)]
pub struct DatabaseSettingsView {
    /// Empty string means no URL is configured.
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone)]
pub struct BackendTasksSettingsView {
    /// Whether `[backend_tasks.llm]` is explicitly configured (vs. falling back to primary LLM).
    pub has_separate_llm: bool,
    /// Resolved connector (from backend_tasks.llm or fallback).
    pub llm_connector: String,
    /// Resolved model (from backend_tasks.llm or fallback).
    pub llm_model: String,
    /// Resolved base URL (from backend_tasks.llm or fallback).
    pub llm_base_url: String,
    /// Whether periodic fact extraction ("dreaming") is enabled.
    pub dreaming_enabled: bool,
    /// Interval in seconds between dreaming cycles.
    pub dreaming_interval_secs: u64,
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

    fn rename_conversation(
        &self,
        id: &ConversationId,
        title: String,
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
        temperature: Option<f64>,
        top_p: Option<f64>,
        max_tokens: Option<u32>,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn set_api_key(
        &self,
        api_key: String,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn generate_ws_jwt(
        &self,
        subject: Option<String>,
    ) -> impl std::future::Future<Output = Result<String, CoreError>> + Send;

    fn validate_ws_jwt(
        &self,
        token: String,
    ) -> impl std::future::Future<Output = Result<bool, CoreError>> + Send;

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

    fn get_database_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<DatabaseSettingsView, CoreError>> + Send;

    fn set_database_settings(
        &self,
        url: Option<String>,
        max_connections: u32,
    ) -> impl std::future::Future<Output = Result<(), CoreError>> + Send;

    fn get_backend_tasks_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<BackendTasksSettingsView, CoreError>> + Send;

    /// Update backend-tasks settings (LLM override + dreaming config).
    ///
    /// Pass `llm_connector = None` to clear the separate LLM override (revert to primary).
    fn set_backend_tasks_settings(
        &self,
        llm_connector: Option<String>,
        llm_model: Option<String>,
        llm_base_url: Option<String>,
        dreaming_enabled: bool,
        dreaming_interval_secs: u64,
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
