pub mod chunking;
pub mod domain;
pub mod ports;
pub mod service;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("system service error: {0}")]
    SystemService(String),

    #[error("conversation not found: {0}")]
    ConversationNotFound(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("tool execution error: {0}")]
    ToolExecution(String),
}

#[cfg(test)]
mod tests {
    #[test]
    fn core_crate_loads() {
        // Validates that the core crate compiles and its module tree is reachable.
        assert_eq!(1, 1);
    }
}
