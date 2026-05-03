//! Adapter that pins a fixed [`ReasoningConfig`] onto `stream_completion`
//! calls when the wrapper was configured with a non-empty config.
//!
//! Background tasks (title generation, context summary) call into the
//! `LlmClient` trait with `ReasoningConfig::default()` because they live in
//! the core service layer where the daemon's `purposes` concept doesn't
//! exist. When the user has configured `[purposes.titling].effort`, the
//! daemon needs to substitute that effort's mapped `ReasoningConfig` for
//! the caller's default — without touching the core service code.
//!
//! ## Override semantics
//!
//! - When `configured.is_empty()` (i.e. `ReasoningConfig::default()`), the
//!   wrapper is a transparent passthrough — the caller's reasoning is
//!   forwarded verbatim. This is what we want for the *primary* (interactive)
//!   client, which is wrapped purely for type-uniformity with the backend
//!   stack and must not stomp on the per-turn task-local reasoning installed
//!   by `RoutingConversationHandler`.
//! - When `configured` has any field set, every `stream_completion` call
//!   sees `configured` regardless of what the caller passed. This is the
//!   override path for `[purposes.titling].effort`.
//!
//! Effort changes via `set_purpose` rebuild the daemon's clients on
//! config reload, so the captured value is fixed for the lifetime of the
//! wrapper.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelInfo, ReasoningConfig,
};

/// Wraps an [`LlmClient`] and substitutes a fixed [`ReasoningConfig`]
/// for whatever the caller passes into `stream_completion`. See module
/// docs.
#[derive(Clone)]
pub struct FixedReasoningLlmClient<L> {
    inner: L,
    reasoning: ReasoningConfig,
}

impl<L> FixedReasoningLlmClient<L> {
    pub fn new(inner: L, reasoning: ReasoningConfig) -> Self {
        Self { inner, reasoning }
    }
}

impl<L: LlmClient> LlmClient for FixedReasoningLlmClient<L> {
    fn get_default_model(&self) -> Option<&str> {
        self.inner.get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        self.inner.get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        self.inner.max_context_tokens()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.list_models().await
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.refresh_models().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        caller_reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let effective = if self.reasoning.is_empty() {
            caller_reasoning
        } else {
            self.reasoning
        };
        self.inner
            .stream_completion(messages, tools, effective, on_chunk)
            .await
    }

    fn supports_hosted_tool_search(&self) -> bool {
        self.inner.supports_hosted_tool_search()
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        caller_reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let effective = if self.reasoning.is_empty() {
            caller_reasoning
        } else {
            self.reasoning
        };
        self.inner
            .stream_completion_with_namespaces(
                messages, core_tools, namespaces, effective, on_chunk,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::Role;
    use desktop_assistant_core::ports::llm::{ReasoningLevel, TokenUsage};
    use std::sync::Mutex;

    /// Test double that records the `ReasoningConfig` it receives so we
    /// can prove the wrapper substitutes its own value.
    #[derive(Default)]
    struct CapturingClient {
        last_seen: Mutex<Option<ReasoningConfig>>,
    }

    impl CapturingClient {
        fn last(&self) -> Option<ReasoningConfig> {
            *self.last_seen.lock().unwrap()
        }
    }

    impl LlmClient for CapturingClient {
        fn get_default_model(&self) -> Option<&str> {
            Some("captured")
        }

        fn get_default_base_url(&self) -> Option<&str> {
            Some("http://captured")
        }

        fn max_context_tokens(&self) -> Option<u64> {
            Some(8_000)
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
            Ok(vec![])
        }

        async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
            Ok(vec![])
        }

        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            *self.last_seen.lock().unwrap() = Some(reasoning);
            Ok(LlmResponse {
                text: String::new(),
                tool_calls: vec![],
                usage: Some(TokenUsage::default()),
            })
        }

        fn supports_hosted_tool_search(&self) -> bool {
            false
        }
    }

    fn user_msg(text: &str) -> Vec<Message> {
        vec![Message::new(Role::User, text)]
    }

    #[tokio::test]
    async fn substitutes_caller_reasoning_with_configured_value() {
        // Caller passes default; wrapper must replace it with the
        // configured (Anthropic-flavored) thinking budget.
        let inner = CapturingClient::default();
        let configured = ReasoningConfig::with_thinking_budget(8_000);
        let wrapped = FixedReasoningLlmClient::new(inner, configured);

        wrapped
            .stream_completion(
                user_msg("hi"),
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("inner client returns Ok");

        assert_eq!(wrapped.inner.last(), Some(configured));
    }

    #[tokio::test]
    async fn fixed_default_reasoning_is_passthrough() {
        // Construction with `ReasoningConfig::default()` means "no
        // configured override". The wrapper must forward whatever the
        // caller passes — this is what makes it safe to wrap the primary
        // (interactive) client where the per-turn task-local reasoning
        // is set by `RoutingConversationHandler` and arrives via the
        // caller's `reasoning` argument. Stomping on it would erase the
        // user's effort selection on every interactive turn.
        let inner = CapturingClient::default();
        let wrapped = FixedReasoningLlmClient::new(inner, ReasoningConfig::default());

        let caller = ReasoningConfig::with_reasoning_effort(ReasoningLevel::High);
        wrapped
            .stream_completion(user_msg("hi"), &[], caller, Box::new(|_| true))
            .await
            .expect("inner client returns Ok");

        assert_eq!(
            wrapped.inner.last(),
            Some(caller),
            "default-configured wrapper must forward caller's reasoning untouched"
        );
    }

    #[tokio::test]
    async fn caller_reasoning_passes_through_when_not_configured_even_if_caller_is_default() {
        // Symmetric case: caller is `default()`, configured is `default()`,
        // inner observes `default()`. Confirms the wrapper doesn't wrongly
        // synthesise a non-default value somewhere.
        let inner = CapturingClient::default();
        let wrapped = FixedReasoningLlmClient::new(inner, ReasoningConfig::default());

        wrapped
            .stream_completion(
                user_msg("hi"),
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("inner client returns Ok");

        assert_eq!(wrapped.inner.last(), Some(ReasoningConfig::default()));
    }

    #[tokio::test]
    async fn substitutes_for_with_namespaces_path_too() {
        // The handler's tool-routing dispatch goes through
        // `stream_completion_with_namespaces`. Backend tasks don't
        // currently use namespaces, but the trait surface must stay
        // consistent: the wrapper has to pin reasoning on both methods.
        let inner = CapturingClient::default();
        let configured = ReasoningConfig::with_reasoning_effort(ReasoningLevel::Medium);
        let wrapped = FixedReasoningLlmClient::new(inner, configured);

        // Default `stream_completion_with_namespaces` from the trait
        // delegates to `stream_completion` for clients that haven't
        // overridden it, so the same captured value applies.
        wrapped
            .stream_completion_with_namespaces(
                user_msg("hi"),
                &[],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("inner client returns Ok");

        assert_eq!(wrapped.inner.last(), Some(configured));
    }

    #[test]
    fn forwards_capability_flags_to_inner() {
        let inner = CapturingClient::default();
        let wrapped = FixedReasoningLlmClient::new(inner, ReasoningConfig::default());
        assert!(!wrapped.supports_hosted_tool_search());
        assert_eq!(wrapped.get_default_model(), Some("captured"));
        assert_eq!(wrapped.max_context_tokens(), Some(8_000));
    }
}
