/// Inbound ports — trait interfaces that adapters (e.g. D-Bus) call into.
pub mod inbound;

/// Outbound ports — trait interfaces the core uses to reach external services.
pub mod outbound;

/// LLM client port — outbound trait for LLM completion.
pub mod llm;

/// LLM profiling decorator — captures request/response context to JSONL.
pub mod llm_profiling;

/// Embedding client port — outbound trait for generating vector embeddings.
pub mod embedding;

/// Conversation store port — outbound trait for persistence.
pub mod store;

/// Tool executor port — outbound trait for executing tools via MCP or other providers.
pub mod tools;

/// Knowledge base store port — outbound trait for unified knowledge persistence.
pub mod knowledge;

/// Tool registry store port — outbound trait for tool definition persistence and search.
pub mod tool_registry;

/// Database query port — closure type for read-only SQL queries.
pub mod database;

/// Conversation search port — outbound trait for full-text search over past messages.
pub mod conversation_search;

#[cfg(test)]
mod tests {
    #[test]
    fn ports_modules_are_accessible() {
        // Validates that the port sub-modules compile and are reachable.
        let _ = std::any::type_name::<dyn super::inbound::AssistantService>();
        // These use impl Future, so they're not dyn-compatible.
        fn _assert_llm_exists<T: super::llm::LlmClient>() {}
        fn _assert_store_exists<T: super::store::ConversationStore>() {}
        fn _assert_system_exists<T: super::outbound::SystemServiceClient>() {}
        fn _assert_tools_exists<T: super::tools::ToolExecutor>() {}
        fn _assert_embedding_exists<T: super::embedding::EmbeddingClient>() {}
        fn _assert_knowledge_exists<T: super::knowledge::KnowledgeBaseStore>() {}
        fn _assert_tool_registry_exists<T: super::tool_registry::ToolRegistryStore>() {}
    }
}
