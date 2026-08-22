//! Core domain model, ports, and services for the assistant (hexagonal-architecture core).

pub mod chunking;
pub mod clock;
pub mod context;
pub mod context_window;
pub mod domain;
pub mod error_classify;
pub(crate) mod eviction_class;
pub(crate) mod otel_bridge;
pub mod planning;
pub mod ports;
pub mod prompts;
pub mod recall;
pub mod sanitize;
pub mod service;
pub mod skill_catalog;
pub mod skill_promotion;
pub mod system_id;
pub mod tag_normalize;
pub(crate) mod telemetry;
pub mod tool_advertising;
pub mod tool_provenance;
pub(crate) mod tool_repeat;
pub mod tool_routing;
pub mod tools;
pub mod turn_capture;
pub(crate) mod turn_index;
pub mod verbatim_window;

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

    /// The caller cancelled this `send_prompt` via a
    /// `tokio_util::sync::CancellationToken`. Surfaced when the token is
    /// observed tripped at one of the cooperative checkpoints in the
    /// agentic loop — between turns, before each tool-round dispatch, or
    /// mid-stream inside an LLM adapter's `tokio::select!`. Not retried
    /// by `RetryingLlmClient` (see `is_retryable_error`); the cancellation
    /// is the user's explicit signal to stop.
    #[error("operation cancelled")]
    Cancelled,

    /// A rules-based refusal of a value the caller supplied: the request was
    /// understood, but a fixed rule refuses it, so retrying the identical
    /// input cannot succeed (base rule 8.2 — a decline, not a failure).
    ///
    /// `code` is a stable, machine-readable identifier a transport adapter
    /// mirrors onto the wire's own classification (`api::ErrorCode::Other`)
    /// rather than inventing a second shape; `message` is fit to show the
    /// person who supplied the input. Today only the shared remote-URL
    /// policy (#804, #895) raises this; a general reclassification of the
    /// string-carrying variants above is tracked separately (#972).
    #[error("{description}")]
    InvalidInput {
        code: &'static str,
        description: String,
        message: String,
    },
}

impl CoreError {
    /// A stable, machine-readable name for which error this is.
    ///
    /// This exists so a log line can say what went wrong without saying what
    /// the caller was doing. Every string-carrying variant above quotes
    /// something it was given: a tool error carries whatever an MCP server put
    /// in its message, which is routinely a file path or the argument it
    /// failed on, and an LLM error carries whatever the provider returned.
    /// The `Display` form is therefore conversation content and belongs at
    /// DEBUG; this is not, and belongs beside it at INFO or WARN.
    ///
    /// The strings are part of the operator-facing contract - they are what an
    /// alert or a log query matches on - so treat them as stable and add a new
    /// one rather than renaming an existing one.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SystemService(_) => "system_service",
            Self::ConversationNotFound(_) => "conversation_not_found",
            Self::Llm(_) => "llm",
            Self::ContextOverflow { .. } => "context_overflow",
            Self::RateLimited { .. } => "rate_limited",
            Self::QuotaExceeded { .. } => "quota_exceeded",
            Self::ModelLoading { .. } => "model_loading",
            Self::ToolsUnsupported { .. } => "tools_unsupported",
            Self::Storage(_) => "storage",
            Self::ToolExecution(_) => "tool_execution",
            Self::Cancelled => "cancelled",
            Self::InvalidInput { .. } => "invalid_input",
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn core_crate_loads() {
        // Validates that the core crate compiles and its module tree is reachable.
        assert_eq!(1, 1);
    }
}
