//! Decorator that normalizes opaque backend LLM errors into structured
//! [`CoreError`] variants (epic #178, slice 5a).
//!
//! Connectors surface unrecognized provider failures as `CoreError::Llm(_)` —
//! a bare string the dispatch loop can't act on. This wrapper runs the
//! deterministic tier-1 matchers from [`desktop_assistant_core::error_classify`]
//! over that string and, when they recognize it, swaps in the structured
//! variant (`ContextOverflow`, `RateLimited`, `QuotaExceeded`, …) that the
//! core recovery ladder and `RetryingLlmClient` already know how to handle.
//!
//! ## Placement & loop-safety
//! It wraps the **raw connector** (innermost, in `registry::build_llm_client`),
//! so it sees the connector's own error before `Retrying`/recovery upstream.
//! Putting it *inside* `RetryingLlmClient` is deliberate: `is_retryable_error`
//! retries only `RateLimited`, so a billing error mapped to `QuotaExceeded`
//! (terminal) is never retried, and `ContextOverflow` flows to the bounded
//! recovery ladder — no new loop is introduced.
//!
//! The reclassification is **pure, single-shot, and non-recursive**: it
//! transforms an already-returned error once and returns; it never retries,
//! re-enters, or calls an LLM (tier 1 has no I/O). The reentrancy guard the
//! loop-safety contract calls for becomes relevant only when the LLM-backed
//! tier 3 lands — there is nothing here that can loop.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition, ToolNamespace};
use desktop_assistant_core::error_classify::{ErrorContext, cause_to_core_error, classify_builtin};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelInfo, ReasoningConfig,
};

/// Wraps an [`LlmClient`] and remaps opaque `CoreError::Llm` errors into the
/// structured variant the built-in matchers recognize. See module docs.
#[derive(Clone)]
pub struct ClassifyingLlmClient<L> {
    inner: L,
    connector: String,
}

impl<L> ClassifyingLlmClient<L> {
    pub fn new(inner: L, connector: impl Into<String>) -> Self {
        Self {
            inner,
            connector: connector.into(),
        }
    }

    /// Remap an opaque `CoreError::Llm` into a structured variant when the
    /// tier-1 matchers recognize it. `Ok` and already-structured errors pass
    /// through untouched, and an unrecognized `Llm` error is returned
    /// verbatim — so behavior is unchanged on a miss.
    fn reclassify(&self, result: Result<LlmResponse, CoreError>) -> Result<LlmResponse, CoreError> {
        match result {
            Err(CoreError::Llm(detail)) => {
                let cause = classify_builtin(&ErrorContext {
                    connector: &self.connector,
                    http_status: None,
                    provider_code: None,
                    message: &detail,
                });
                match cause_to_core_error(cause, detail.clone()) {
                    Some(mapped) => {
                        tracing::info!(
                            connector = %self.connector,
                            ?cause,
                            "classified opaque LLM error into a structured cause"
                        );
                        Err(mapped)
                    }
                    None => Err(CoreError::Llm(detail)),
                }
            }
            other => other,
        }
    }
}

#[async_trait::async_trait]
impl<L: LlmClient> LlmClient for ClassifyingLlmClient<L> {
    // --- Faithful passthrough: every capability comes from `inner`. The
    // decorator wraps the connector innermost, so masking any of these (e.g.
    // `max_context_tokens`, which drives budget resolution) would regress
    // behavior. Only the two completion methods are intercepted. ---

    fn get_default_model(&self) -> Option<&str> {
        self.inner.get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        self.inner.get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        self.inner.max_context_tokens()
    }

    fn estimate_tokens(&self, text: &str) -> u64 {
        self.inner.estimate_tokens(text)
    }

    fn supports_hosted_tool_search(&self) -> bool {
        self.inner.supports_hosted_tool_search()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.list_models().await
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.refresh_models().await
    }

    async fn warmup(&self) {
        self.inner.warmup().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let result = self
            .inner
            .stream_completion(messages, tools, reasoning, on_chunk)
            .await;
        self.reclassify(result)
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let result = self
            .inner
            .stream_completion_with_namespaces(
                messages, core_tools, namespaces, reasoning, on_chunk,
            )
            .await;
        self.reclassify(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::Role;
    use std::sync::Mutex;

    enum Behavior {
        Ok,
        ErrLlm(String),
        ErrRateLimited,
    }

    struct StubClient {
        calls: Mutex<u32>,
        behavior: Behavior,
        max_ctx: Option<u64>,
    }

    impl StubClient {
        fn new(behavior: Behavior) -> Self {
            Self {
                calls: Mutex::new(0),
                behavior,
                max_ctx: Some(131_072),
            }
        }
        fn calls(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for StubClient {
        fn max_context_tokens(&self) -> Option<u64> {
            self.max_ctx
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
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            *self.calls.lock().unwrap() += 1;
            match &self.behavior {
                Behavior::Ok => Ok(LlmResponse {
                    text: "ok".into(),
                    tool_calls: vec![],
                    usage: None,
                }),
                Behavior::ErrLlm(s) => Err(CoreError::Llm(s.clone())),
                Behavior::ErrRateLimited => Err(CoreError::RateLimited {
                    retry_after: None,
                    detail: "throttled".into(),
                }),
            }
        }

        fn supports_hosted_tool_search(&self) -> bool {
            false
        }
    }

    fn wrapped(behavior: Behavior) -> ClassifyingLlmClient<StubClient> {
        ClassifyingLlmClient::new(StubClient::new(behavior), "bedrock")
    }

    async fn run(c: &ClassifyingLlmClient<StubClient>) -> Result<LlmResponse, CoreError> {
        c.stream_completion(
            vec![Message::new(Role::User, "hi")],
            &[],
            ReasoningConfig::default(),
            Box::new(|_| true),
        )
        .await
    }

    #[tokio::test]
    async fn opaque_overflow_error_is_reclassified() {
        let c = wrapped(Behavior::ErrLlm(
            "Input is too long for requested model.".into(),
        ));
        let err = run(&c).await.expect_err("must be an error");
        assert!(
            matches!(err, CoreError::ContextOverflow { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn opaque_billing_error_maps_to_quota_exceeded() {
        let c = wrapped(Behavior::ErrLlm(
            "You exceeded your current quota; check your billing details.".into(),
        ));
        let err = run(&c).await.expect_err("must be an error");
        assert!(
            matches!(err, CoreError::QuotaExceeded { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn unrecognized_opaque_error_passes_through_unchanged() {
        let c = wrapped(Behavior::ErrLlm("a wild unfamiliar failure".into()));
        let err = run(&c).await.expect_err("must be an error");
        match err {
            CoreError::Llm(s) => assert_eq!(s, "a wild unfamiliar failure"),
            other => panic!("expected unchanged Llm error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn already_structured_error_is_not_touched() {
        let c = wrapped(Behavior::ErrRateLimited);
        let err = run(&c).await.expect_err("must be an error");
        assert!(matches!(err, CoreError::RateLimited { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn ok_passes_through() {
        let c = wrapped(Behavior::Ok);
        assert!(run(&c).await.is_ok());
    }

    #[tokio::test]
    async fn reclassification_is_single_shot_no_loop() {
        // The decorator transforms the error once and returns — it must never
        // re-invoke the inner client (which would risk a loop).
        let c = wrapped(Behavior::ErrLlm(
            "Input length (479258) exceeds model's maximum context length (131072).".into(),
        ));
        let _ = run(&c).await;
        assert_eq!(
            c.inner.calls(),
            1,
            "inner client must be called exactly once"
        );
    }

    #[test]
    fn delegates_max_context_tokens_to_inner() {
        // Critical: the decorator wraps the connector innermost, so it must
        // not mask the connector's curated context window (issue #176).
        let c = wrapped(Behavior::Ok);
        assert_eq!(c.max_context_tokens(), Some(131_072));
    }
}
