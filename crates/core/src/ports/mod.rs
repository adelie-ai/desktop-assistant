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

/// Scratchpad store port — outbound trait for ephemeral per-conversation notes.
pub mod scratchpad;

/// Tool registry store port — outbound trait for tool definition persistence and search.
pub mod tool_registry;

/// Database query port — closure type for read-only SQL queries.
pub mod database;

/// Conversation search port — outbound trait for full-text search over past messages.
pub mod conversation_search;

/// Request-scoped auth context — task-local `UserId` for SQL scoping (#105).
pub mod auth;

/// Request-scoped login-session identity — task-local [`session::SessionId`],
/// unique per client connection, so per-connection state (client-local tool
/// registration) doesn't bleed between two windows of the same user (#261).
pub mod session;

/// Request-scoped transport context — task-local [`crate::domain::TransportKind`]
/// so the turn loop can infer tool co-location (UDS/D-Bus ⇒ same machine,
/// WebSocket ⇒ possibly remote) when tagging tools with locality (#243).
pub mod transport;

/// Request-scoped conversation context — task-local `ConversationId` so tool
/// executors can scope per-conversation side state (e.g. the scratchpad).
pub mod conversation_ctx;

/// Bundle of the request-scoped task-locals that must cross a `tokio::spawn`
/// boundary ([`request_scope::RequestScope`]) — captured before the spawn and
/// re-installed in one call inside the spawned turn body, so a new
/// spawn-crossing local can never be silently dropped at a re-install site
/// (the #261 leak class, issue #305 item 4).
pub mod request_scope;

/// Client-side tool execution port — outbound trait the turn loop uses to
/// consult the current user's registered client-local tools and suspend the
/// turn on a client-tool call (#107 / #234).
pub mod client_tools;

/// Request-scoped tool-activity observer — task-local sink the turn loop
/// notifies of each tool/MCP call and its outcome, so a caller can surface a
/// live activity feed (e.g. the background-task panel) without threading a
/// sink through the `send_prompt` trait surface.
pub mod tool_observer;

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
        fn _assert_scratchpad_exists<T: super::scratchpad::ScratchpadStore>() {}
        fn _assert_tool_registry_exists<T: super::tool_registry::ToolRegistryStore>() {}
    }
}
