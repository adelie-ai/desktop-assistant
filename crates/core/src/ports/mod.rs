/// Inbound ports — trait interfaces that adapters (e.g. D-Bus) call into.
pub mod inbound;

/// Outbound ports — trait interfaces the core uses to reach external services.
pub mod outbound;

/// LLM client port — outbound trait for LLM completion.
pub mod llm;

/// Conversation store port — outbound trait for persistence.
pub mod store;

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
    }
}
