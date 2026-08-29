/// Inbound ports — trait interfaces that adapters (e.g. D-Bus) call into.
pub mod inbound;

/// Outbound ports — trait interfaces the core uses to reach external services.
pub mod outbound;

/// LLM client port — outbound trait for LLM completion.
pub mod llm;

/// Embedding client port — outbound trait for generating vector embeddings.
pub mod embedding;

/// Conversation store port — outbound trait for persistence.
pub mod store;

/// Tool executor port — outbound trait for executing tools via MCP or other providers.
pub mod tools;

/// Knowledge base store port — outbound trait for unified knowledge persistence.
pub mod knowledge;

pub mod knowledge_use;

/// Scratchpad store port — outbound trait for ephemeral per-conversation notes.
pub mod scratchpad;

/// Tool registry store port — outbound trait for tool definition persistence and search.
pub mod tool_registry;

/// Database query port — closure type for read-only SQL queries.
pub mod database;

/// Conversation search port — outbound trait for full-text search over past messages.
pub mod conversation_search;

// The modules below document themselves with a `//!` header. A summary here as
// well would merge with it and make every unqualified intra-doc link in that
// header resolve against THIS module instead of its own, which silently breaks
// the links.
pub mod auth;
pub mod client_tools;
pub mod conversation_ctx;
pub mod knowledge_delete;
pub mod negative_memory;
pub mod notify;
pub mod recall;
pub mod request_scope;
pub mod scratchpad_scope;
pub mod session;
pub mod skill_index;
pub mod skill_use;
pub mod tool_observer;
pub mod tool_usage;

// One durable record per turn of what filled its prompt, and what the turn was
// allowed to spend (#588). The measurement already happens on every turn; this
// is where it stops being discarded. The module carries its own `//!` header,
// so this stays a plain comment: a `///` here would merge with that header and
// resolve its links against THIS module.
pub mod context_breakdown;
// One in-memory record per turn of what the `[Recall]` lookup considered, how
// each candidate scored by activation term, and which it offered (#1327). The
// module carries its own `//!` header, for the same reason `context_breakdown`
// does.
pub mod context_plan;
pub mod transcript;
pub mod transport;
pub mod turn_capability;
pub mod turn_digest;
pub mod turn_interactivity;
pub mod turn_record;
pub mod turn_telemetry;

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
        fn _assert_negative_memory_exists<T: super::negative_memory::NegativeMemoryStore>() {}
        fn _assert_scratchpad_exists<T: super::scratchpad::ScratchpadStore>() {}
        fn _assert_tool_registry_exists<T: super::tool_registry::ToolRegistryStore>() {}
        fn _assert_skill_index_exists<T: super::skill_index::SkillIndexStore>() {}
        fn _assert_context_breakdown_exists<T: super::context_breakdown::ContextBreakdownStore>() {}
        fn _assert_turn_recorder_exists<T: super::turn_record::TurnRecorder>() {}
    }
}
