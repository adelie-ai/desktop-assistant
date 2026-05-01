pub mod chunking;
pub mod context;
pub mod domain;
pub mod ports;
pub mod prompts;
pub mod sanitize;
pub mod service;
pub mod tools;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("system service error: {0}")]
    SystemService(String),

    #[error("conversation not found: {0}")]
    ConversationNotFound(String),

    #[error("LLM error: {0}")]
    Llm(String),

    /// The prompt exceeded the model's context window. The core service
    /// handles this by truncating the most recent oversized tool result
    /// and retrying (bounded), rather than surfacing a hard failure.
    #[error("LLM context overflow: {detail}")]
    ContextOverflow {
        prompt_tokens: Option<u64>,
        max_tokens: Option<u64>,
        detail: String,
    },

    /// Provider returned a transient throttling error (HTTP 429/529,
    /// "overloaded", service-unavailable). Safe to retry with backoff;
    /// the `RetryingLlmClient` decorator does so on this variant alone.
    /// `retry_after` is populated when the upstream `Retry-After` header
    /// is present and parseable, otherwise `None`.
    #[error("LLM rate limited: {detail}")]
    RateLimited {
        retry_after: Option<std::time::Duration>,
        detail: String,
    },

    /// Permanent quota/billing error. Distinct from [`Self::RateLimited`]:
    /// some providers (notably OpenAI) signal `insufficient_quota` with
    /// HTTP 429, which would otherwise look retryable. This variant is
    /// NOT retried by `RetryingLlmClient` and surfaces a user-visible
    /// message instructing the user to top up or switch keys.
    #[error("LLM quota exceeded: {detail}")]
    QuotaExceeded { detail: String },
    /// Provider reported the configured model is downloading, pulling, or
    /// loading. Today this is Ollama-specific (the daemon ships its own
    /// inference server and may surface "model is currently loading" or
    /// "pull model manifest" messages). Transient setup error rather than
    /// a backend failure — the user can retry shortly.
    #[error("LLM model loading: {detail}")]
    ModelLoading { detail: String },

    /// Provider reported the configured model does not support tool use
    /// (e.g. Ollama models without a tool-calling template). Permanent for
    /// the chosen model — the caller must switch model or disable tools
    /// rather than retrying.
    #[error("LLM tools unsupported: {detail}")]
    ToolsUnsupported { detail: String },

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
