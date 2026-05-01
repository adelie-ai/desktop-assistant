use std::sync::{Arc, Mutex};

use crate::CoreError;
use crate::domain::{Message, ToolCall, ToolDefinition, ToolNamespace};

/// Callback invoked for each chunk of a streaming LLM response.
/// Return `true` to continue, `false` to abort the stream.
pub type ChunkCallback = Box<dyn FnMut(String) -> bool + Send>;

/// Callback invoked to report progress while the assistant is working
/// (e.g. "Searching knowledge base...", "Querying timeclock sessions...").
pub type StatusCallback = Box<dyn FnMut(String) + Send>;

tokio::task_local! {
    /// Per-turn reasoning configuration. Set by the daemon-side routing
    /// handler via [`with_reasoning_config`] before invoking `send_prompt`;
    /// read by [`current_reasoning_config`] inside the dispatch loop and
    /// forwarded to connectors through [`LlmClient::stream_completion`].
    ///
    /// Lives in the task-local slot so each concurrent turn can carry a
    /// distinct reasoning config without any coupling between the routing
    /// wrapper and the core `ConversationHandler`.
    static REASONING_CONFIG: ReasoningConfig;

    /// Per-turn model override (issue #34). Set by the daemon-side routing
    /// handler via [`with_model_override`] before invoking `send_prompt`,
    /// populated from the resolved `(connection_id, model_id, effort)`
    /// selection. Connectors read it via [`current_model_override`] at the
    /// top of `stream_completion` and use it instead of `self.model` when
    /// set. When unset, connectors fall back to `self.model` — preserving
    /// pre-#34 behaviour for callers that don't route through the daemon
    /// (tests, dreaming jobs, etc.).
    ///
    /// Lives in core (next to `REASONING_CONFIG`) precisely so the
    /// connector crates — which can't depend on the daemon — can read it.
    static MODEL_OVERRIDE: String;

    /// Per-turn resolved prompt-token budget for `send_prompt`.
    ///
    /// Lifecycle: populated by the daemon's dispatch wrapper via
    /// [`with_context_budget`] at the start of `send_prompt`; readable for
    /// the duration of that call via [`current_context_budget`]. Read once
    /// at dispatch entry from the three-tier resolution chain (purpose
    /// override → connector curated table → universal fallback) and frozen
    /// for the rest of the turn. The dispatch loop reads it lazily on each
    /// iteration to drive token-pressure compaction.
    ///
    /// Why a task-local: keeps the existing `ConversationService::send_prompt`
    /// signature unchanged while still threading a typed value through
    /// without re-resolving on every turn. Lives in core so the read site
    /// in `service::ConversationHandler` doesn't need to know the daemon's
    /// resolution logic.
    static CONTEXT_BUDGET: ContextBudget;
}

/// Run `fut` with the given reasoning config installed as the current
/// task-local value. All `current_reasoning_config()` calls inside the
/// future (and any sub-tasks that inherit the scope) observe `config`.
pub async fn with_reasoning_config<F, T>(config: ReasoningConfig, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    REASONING_CONFIG.scope(config, fut).await
}

/// Current task-local reasoning config, or `ReasoningConfig::default()`
/// (all `None`) when not set. Safe to call from any async context.
pub fn current_reasoning_config() -> ReasoningConfig {
    REASONING_CONFIG
        .try_with(|c| *c)
        .unwrap_or_default()
}

/// Run `fut` with `model` installed as the current turn's model override.
/// Connectors read it via [`current_model_override`] in `stream_completion`
/// and use it in place of `self.model`. See [`MODEL_OVERRIDE`].
pub async fn with_model_override<F, T>(model: String, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    MODEL_OVERRIDE.scope(model, fut).await
}

/// Current task-local model override, or `None` when not set. Connectors
/// call this at the top of `stream_completion` to determine which model id
/// to send in the request body — falling back to their own `self.model`
/// when unset.
pub fn current_model_override() -> Option<String> {
    MODEL_OVERRIDE.try_with(|m| m.clone()).ok()
}

/// The resolved prompt-token budget for the current `send_prompt` call.
///
/// Resolution happens once at dispatch entry; downstream code reads it
/// via [`current_context_budget`]. The `source` field records which tier
/// of the resolution chain produced the value, for observability —
/// distinguishing "user-authored override" from "connector knows the
/// model" from "fell through to universal fallback".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    /// Maximum input/prompt tokens for the configured model on this turn.
    pub max_input_tokens: u64,
    /// Which resolution tier produced [`Self::max_input_tokens`].
    pub source: BudgetSource,
}

/// Origin tag for a resolved [`ContextBudget`], recorded so operators
/// can tell whether the value came from user config, the connector's
/// curated table, or the universal fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    /// User-authored `purposes.<kind>.max_context_tokens`. Always wins.
    PurposeOverride,
    /// The connector's curated `LlmClient::max_context_tokens()` value
    /// for the configured model (e.g. Anthropic / Bedrock tables).
    ConnectorTable,
    /// Conservative universal fallback used when neither the purpose
    /// nor the connector supplied a value.
    UniversalFallback,
}

/// Run `fut` with `budget` installed as the resolved per-turn context
/// budget. The dispatch loop in [`crate::service::ConversationHandler`]
/// reads this via [`current_context_budget`] to drive token-pressure
/// compaction.
///
/// Why a task-local: see the doc on [`CONTEXT_BUDGET`].
pub async fn with_context_budget<F, T>(budget: ContextBudget, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CONTEXT_BUDGET.scope(budget, fut).await
}

/// Returns the resolved budget for the current dispatch, or `None` if
/// no budget has been installed (e.g. test contexts or background jobs
/// that don't route through the daemon's dispatch wrapper). When `None`,
/// callers should treat this as "no budget known" and skip token-based
/// compaction the same way they would for a connector reporting `None`
/// from `max_context_tokens()`.
pub fn current_context_budget() -> Option<ContextBudget> {
    CONTEXT_BUDGET.try_with(|b| *b).ok()
}

/// Reasoning / extended-thinking level for a single LLM turn.
///
/// Mirrors the tri-state `Effort` knob that the daemon exposes on
/// `SendMessage.override`. Kept in core so the `LlmClient` trait is
/// self-contained and connectors don't take a daemon dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningLevel {
    Low,
    Medium,
    High,
}

impl ReasoningLevel {
    /// Lowercase literal used in OpenAI's `reasoning_effort` request field.
    pub fn as_openai_effort(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Per-turn reasoning configuration threaded from the routing handler
/// through the `LlmClient` trait into per-connector request bodies.
///
/// All fields default to `None`, which means "no reasoning-related fields
/// in the request body" — i.e. the existing behavior. The daemon-side
/// routing handler populates the appropriate field based on the caller's
/// `Effort` hint and the selected connector type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReasoningConfig {
    /// Anthropic extended-thinking budget in tokens. When `Some(N > 0)`,
    /// the Anthropic connector adds `thinking: { type: "enabled",
    /// budget_tokens: N }` to the request. The Bedrock connector forwards
    /// the same shape via `additionalModelRequestFields` for Claude models.
    /// `None` or `Some(0)` disables extended thinking.
    pub thinking_budget_tokens: Option<u32>,
    /// OpenAI `reasoning_effort` literal. When `Some(level)` and the model
    /// supports reasoning (o-series / GPT-5 reasoning), the OpenAI
    /// connector adds `reasoning_effort: "..."` to the request.
    pub reasoning_effort: Option<ReasoningLevel>,
}

impl ReasoningConfig {
    /// Convenience constructor for the Anthropic-flavored side only.
    pub fn with_thinking_budget(budget: u32) -> Self {
        Self {
            thinking_budget_tokens: Some(budget),
            reasoning_effort: None,
        }
    }

    /// Convenience constructor for the OpenAI-flavored side only.
    pub fn with_reasoning_effort(level: ReasoningLevel) -> Self {
        Self {
            thinking_budget_tokens: None,
            reasoning_effort: Some(level),
        }
    }

    /// True when no reasoning-related fields would be added to the
    /// request body. Used by connectors to skip log spam on the fast
    /// path.
    pub fn is_empty(self) -> bool {
        self.thinking_budget_tokens.is_none() && self.reasoning_effort.is_none()
    }
}

/// Token usage statistics from an LLM call.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

/// Capability flags describing what an LLM model supports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCapabilities {
    /// Model supports extended-thinking / reasoning traces.
    #[serde(default)]
    pub reasoning: bool,
    /// Model accepts image input.
    #[serde(default)]
    pub vision: bool,
    /// Model supports tool/function calling.
    #[serde(default)]
    pub tools: bool,
    /// Model is an embedding model (not a chat/completion model).
    #[serde(default)]
    pub embedding: bool,
}

/// Description of a single model exposed by an `LlmClient`.
///
/// Returned by `LlmClient::list_models()` and consumed by the model-picker
/// UI. `context_limit` is optional: connectors should populate it when a
/// reliable value is known (either from a curated static list or a provider
/// API), and leave it `None` otherwise so callers fall back to
/// message-count heuristics instead of bogus token math.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    /// Stable identifier used to invoke the model (e.g.
    /// `claude-sonnet-4-5`, `gpt-5-mini`, `us.anthropic.claude-opus-4-1`).
    pub id: String,
    /// Human-friendly display name for UIs. Defaults to `id` if unknown.
    pub display_name: String,
    /// Maximum prompt-token context window, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_limit: Option<u64>,
    /// Feature flags for this model.
    #[serde(default)]
    pub capabilities: ModelCapabilities,
}

impl ModelInfo {
    /// Convenience constructor using `id` as the display name.
    pub fn new(id: impl Into<String>) -> Self {
        let id: String = id.into();
        Self {
            display_name: id.clone(),
            id,
            context_limit: None,
            capabilities: ModelCapabilities::default(),
        }
    }

    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = name.into();
        self
    }

    pub fn with_context_limit(mut self, limit: u64) -> Self {
        self.context_limit = Some(limit);
        self
    }

    pub fn with_capabilities(mut self, caps: ModelCapabilities) -> Self {
        self.capabilities = caps;
        self
    }
}

/// Response from the LLM, which may contain text, tool calls, or both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmResponse {
    /// The text content of the response (may be empty if only tool calls).
    pub text: String,
    /// Tool calls requested by the LLM (empty if text-only response).
    pub tool_calls: Vec<ToolCall>,
    /// Token usage statistics, if provided by the connector.
    pub usage: Option<TokenUsage>,
}

impl LlmResponse {
    /// Create a text-only response.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tool_calls: Vec::new(),
            usage: None,
        }
    }

    /// Create a response with tool calls.
    pub fn with_tool_calls(text: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            text: text.into(),
            tool_calls,
            usage: None,
        }
    }

    /// Attach token usage statistics.
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Whether this response requests tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Outbound port for LLM completion requests.
pub trait LlmClient: Send + Sync {
    /// Return the connector's built-in default model, if it has one.
    fn get_default_model(&self) -> Option<&str> {
        None
    }

    /// Return the connector's built-in default base URL, if it has one.
    fn get_default_base_url(&self) -> Option<&str> {
        None
    }

    /// Maximum prompt-token budget for the configured model, if known.
    /// Used by the core service to trigger proactive context compaction
    /// before the provider rejects an oversized request.
    fn max_context_tokens(&self) -> Option<u64> {
        None
    }

    /// Stream a completion from the LLM given a message history.
    /// Calls `on_chunk` for each text token/chunk received.
    /// Optionally accepts tool definitions to enable tool calling.
    /// `reasoning` carries optional extended-thinking / reasoning-effort
    /// hints; connectors may ignore it (Ollama) or translate it into a
    /// per-API request field (Anthropic `thinking`, OpenAI
    /// `reasoning_effort`, Bedrock `additionalModelRequestFields`).
    /// Returns an `LlmResponse` which may include tool calls.
    fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> impl std::future::Future<Output = Result<LlmResponse, CoreError>> + Send;

    /// Whether this connector supports server-side hosted tool search
    /// (e.g. OpenAI namespaces with deferred loading).
    fn supports_hosted_tool_search(&self) -> bool {
        false
    }

    /// Stream a completion with namespaced tool definitions.
    ///
    /// Connectors that support hosted tool search (e.g. OpenAI) serialize
    /// namespaces with `defer_loading: true` and append a `tool_search` entry.
    /// The default implementation flattens everything into `stream_completion`.
    fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> impl std::future::Future<Output = Result<LlmResponse, CoreError>> + Send {
        async move {
            let mut all: Vec<ToolDefinition> = core_tools.to_vec();
            for ns in namespaces {
                all.extend(ns.tools.iter().cloned());
            }
            self.stream_completion(messages, &all, reasoning, on_chunk)
                .await
        }
    }

    /// Enumerate the models this connector can serve.
    ///
    /// Connectors should return every model the caller could reasonably
    /// select (chat and embedding). The default implementation returns an
    /// empty list so test mocks and decorators that delegate can opt out;
    /// production connectors override this.
    fn list_models(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<ModelInfo>, CoreError>> + Send {
        async { Ok(Vec::new()) }
    }

    /// Force a fresh fetch of `list_models()`, bypassing any per-connector
    /// cache. Connectors without a cache can delegate to `list_models`.
    fn refresh_models(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<ModelInfo>, CoreError>> + Send {
        async { self.list_models().await }
    }
}

/// True for `CoreError` values that represent a transient backend
/// throttling or overload signal that an automatic retry-with-backoff
/// can recover from. Today that is exactly [`CoreError::RateLimited`].
///
/// Permanent failures that happen to use HTTP 429 — notably OpenAI's
/// `insufficient_quota` — are surfaced as [`CoreError::QuotaExceeded`]
/// at the connector boundary so this classifier never has to tell them
/// apart from genuine rate limits.
pub fn is_retryable_error(e: &CoreError) -> bool {
    matches!(e, CoreError::RateLimited { .. })
}

/// Decorator that wraps any `LlmClient` and retries on transient rate-limit errors
/// with exponential backoff.
pub struct RetryingLlmClient<L> {
    inner: L,
    max_retries: u32,
}

impl<L> RetryingLlmClient<L> {
    pub fn new(inner: L, max_retries: u32) -> Self {
        Self { inner, max_retries }
    }
}

impl<L: LlmClient> LlmClient for RetryingLlmClient<L> {
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
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        // Store the real callback behind Arc<Mutex<Option<...>>> so we can
        // create proxy callbacks for each retry attempt.
        let shared_cb: Arc<Mutex<Option<ChunkCallback>>> = Arc::new(Mutex::new(Some(on_chunk)));

        for attempt in 0..=self.max_retries {
            let cb_ref = Arc::clone(&shared_cb);
            let proxy_cb: ChunkCallback = Box::new(move |chunk: String| -> bool {
                let mut guard = cb_ref.lock().unwrap();
                if let Some(ref mut cb) = *guard {
                    cb(chunk)
                } else {
                    false
                }
            });

            let msgs = messages.clone();
            match self
                .inner
                .stream_completion(msgs, tools, reasoning, proxy_cb)
                .await
            {
                Ok(response) => return Ok(response),
                Err(e) if attempt < self.max_retries && is_retryable_error(&e) => {
                    let delay_secs = 1u64 << attempt;
                    tracing::warn!(
                        "retryable LLM error, retrying in {delay_secs}s (attempt {}/{}): {e}",
                        attempt + 1,
                        self.max_retries
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                }
                Err(e) => return Err(e),
            }
        }

        unreachable!("loop always returns")
    }

    fn supports_hosted_tool_search(&self) -> bool {
        self.inner.supports_hosted_tool_search()
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let shared_cb: Arc<Mutex<Option<ChunkCallback>>> = Arc::new(Mutex::new(Some(on_chunk)));

        for attempt in 0..=self.max_retries {
            let cb_ref = Arc::clone(&shared_cb);
            let proxy_cb: ChunkCallback = Box::new(move |chunk: String| -> bool {
                let mut guard = cb_ref.lock().unwrap();
                if let Some(ref mut cb) = *guard {
                    cb(chunk)
                } else {
                    false
                }
            });

            let msgs = messages.clone();
            match self
                .inner
                .stream_completion_with_namespaces(
                    msgs, core_tools, namespaces, reasoning, proxy_cb,
                )
                .await
            {
                Ok(response) => return Ok(response),
                Err(e) if attempt < self.max_retries && is_retryable_error(&e) => {
                    let delay_secs = 1u64 << attempt;
                    tracing::warn!(
                        "retryable LLM error, retrying in {delay_secs}s (attempt {}/{}): {e}",
                        attempt + 1,
                        self.max_retries
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                }
                Err(e) => return Err(e),
            }
        }

        unreachable!("loop always returns")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Role;

    struct MockLlm {
        chunks: Vec<String>,
    }

    impl LlmClient for MockLlm {
        fn get_default_model(&self) -> Option<&str> {
            Some("mock")
        }

        fn get_default_base_url(&self) -> Option<&str> {
            Some("mock://")
        }

        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let mut full = String::new();
            for chunk in &self.chunks {
                full.push_str(chunk);
                if !on_chunk(chunk.clone()) {
                    return Ok(LlmResponse::text(full));
                }
            }
            Ok(LlmResponse::text(full))
        }
    }

    #[test]
    fn llm_response_text_only() {
        let resp = LlmResponse::text("hello");
        assert_eq!(resp.text, "hello");
        assert!(!resp.has_tool_calls());
    }

    #[test]
    fn llm_response_with_tool_calls() {
        let calls = vec![ToolCall::new("c1", "test", "{}")];
        let resp = LlmResponse::with_tool_calls("", calls);
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls.len(), 1);
    }

    #[tokio::test]
    async fn mock_llm_streams_chunks() {
        use std::sync::{Arc, Mutex};

        let llm = MockLlm {
            chunks: vec!["Hello".into(), " world".into()],
        };
        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received);
        let result = llm
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(move |chunk| {
                    received_clone.lock().unwrap().push(chunk);
                    true
                }),
            )
            .await
            .unwrap();
        assert_eq!(result.text, "Hello world");
        assert!(!result.has_tool_calls());
        assert_eq!(*received.lock().unwrap(), vec!["Hello", " world"]);
    }

    #[tokio::test]
    async fn mock_llm_abort_stops_stream() {
        use std::sync::{Arc, Mutex};

        let llm = MockLlm {
            chunks: vec!["a".into(), "b".into(), "c".into()],
        };
        let count = Arc::new(Mutex::new(0));
        let count_clone = Arc::clone(&count);
        let result = llm
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(move |_chunk| {
                    let mut c = count_clone.lock().unwrap();
                    *c += 1;
                    *c < 2 // abort after second chunk
                }),
            )
            .await
            .unwrap();
        assert_eq!(result.text, "ab");
        assert_eq!(*count.lock().unwrap(), 2);
    }

    // --- is_retryable_error tests ---

    #[test]
    fn retryable_rate_limited_variant() {
        let e = CoreError::RateLimited {
            retry_after: None,
            detail: "HTTP 429 Too Many Requests".into(),
        };
        assert!(is_retryable_error(&e));
    }

    #[test]
    fn rate_limited_with_retry_after_is_retryable() {
        let e = CoreError::RateLimited {
            retry_after: Some(std::time::Duration::from_secs(5)),
            detail: "HTTP 529 overloaded".into(),
        };
        assert!(is_retryable_error(&e));
    }

    #[test]
    fn quota_exceeded_is_not_retryable() {
        let e = CoreError::QuotaExceeded {
            detail: "insufficient_quota".into(),
        };
        assert!(!is_retryable_error(&e));
    }

    #[test]
    fn context_overflow_is_not_retryable() {
        let e = CoreError::ContextOverflow {
            prompt_tokens: Some(200_000),
            max_tokens: Some(180_000),
            detail: "prompt too long".into(),
        };
        assert!(!is_retryable_error(&e));
    }

    #[test]
    fn model_loading_is_not_retryable() {
        let e = CoreError::ModelLoading {
            detail: "loading".into(),
        };
        assert!(!is_retryable_error(&e));
    }

    #[test]
    fn tools_unsupported_is_not_retryable() {
        let e = CoreError::ToolsUnsupported {
            detail: "no tool support".into(),
        };
        assert!(!is_retryable_error(&e));
    }

    #[test]
    fn generic_llm_is_not_retryable() {
        let e = CoreError::Llm("invalid API key".into());
        assert!(!is_retryable_error(&e));
    }

    // --- RetryingLlmClient tests ---

    /// Mock that fails N times with a retryable error, then succeeds.
    struct FailThenSucceedLlm {
        remaining_failures: Mutex<u32>,
    }

    impl LlmClient for FailThenSucceedLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let mut count = self.remaining_failures.lock().unwrap();
            if *count > 0 {
                *count -= 1;
                return Err(CoreError::RateLimited {
                    retry_after: None,
                    detail: "HTTP 429 rate limited".into(),
                });
            }
            on_chunk("ok".into());
            Ok(LlmResponse::text("ok"))
        }
    }

    #[tokio::test]
    async fn retrying_client_succeeds_after_transient_failure() {
        tokio::time::pause();

        let inner = FailThenSucceedLlm {
            remaining_failures: Mutex::new(2),
        };
        let client = RetryingLlmClient::new(inner, 3);

        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received);
        let result = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(move |chunk| {
                    received_clone.lock().unwrap().push(chunk);
                    true
                }),
            )
            .await
            .unwrap();

        assert_eq!(result.text, "ok");
        assert_eq!(*received.lock().unwrap(), vec!["ok"]);
    }

    #[tokio::test]
    async fn retrying_client_passes_through_non_retryable_error() {
        tokio::time::pause();

        struct AlwaysFailLlm;
        impl LlmClient for AlwaysFailLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                Err(CoreError::Llm("invalid API key".into()))
            }
        }

        let client = RetryingLlmClient::new(AlwaysFailLlm, 3);
        let result = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid API key"));
    }

    #[test]
    fn llm_response_usage_defaults_to_none() {
        let resp = LlmResponse::text("hello");
        assert!(resp.usage.is_none());
    }

    #[test]
    fn llm_response_with_usage() {
        let usage = TokenUsage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            cache_creation_input_tokens: Some(10),
            cache_read_input_tokens: Some(20),
        };
        let resp = LlmResponse::text("hello").with_usage(usage.clone());
        assert_eq!(resp.usage, Some(usage));
    }

    #[test]
    fn token_usage_serde_round_trip() {
        let usage = TokenUsage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: Some(20),
        };
        let json = serde_json::to_string(&usage).unwrap();
        let parsed: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage, parsed);
        // cache_creation_input_tokens is None so should be skipped
        assert!(!json.contains("cache_creation_input_tokens"));
    }

    // --- ModelInfo / ModelCapabilities tests ---

    #[test]
    fn model_info_new_defaults_display_name_to_id() {
        let info = ModelInfo::new("claude-sonnet-4-6");
        assert_eq!(info.id, "claude-sonnet-4-6");
        assert_eq!(info.display_name, "claude-sonnet-4-6");
        assert_eq!(info.context_limit, None);
        assert_eq!(info.capabilities, ModelCapabilities::default());
    }

    #[test]
    fn model_info_builder_sets_fields() {
        let caps = ModelCapabilities {
            reasoning: true,
            vision: true,
            tools: true,
            embedding: false,
        };
        let info = ModelInfo::new("gpt-5")
            .with_display_name("GPT-5")
            .with_context_limit(400_000)
            .with_capabilities(caps);
        assert_eq!(info.display_name, "GPT-5");
        assert_eq!(info.context_limit, Some(400_000));
        assert!(info.capabilities.reasoning);
        assert!(info.capabilities.vision);
        assert!(info.capabilities.tools);
        assert!(!info.capabilities.embedding);
    }

    #[test]
    fn model_info_serde_round_trip_full() {
        let info = ModelInfo {
            id: "claude-sonnet-4-6".into(),
            display_name: "Claude Sonnet 4.6".into(),
            context_limit: Some(200_000),
            capabilities: ModelCapabilities {
                reasoning: true,
                vision: true,
                tools: true,
                embedding: false,
            },
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, info);
    }

    #[test]
    fn model_info_context_limit_none_is_skipped_in_json() {
        let info = ModelInfo::new("unknown-model");
        let json = serde_json::to_string(&info).unwrap();
        assert!(!json.contains("context_limit"));
    }

    #[test]
    fn model_capabilities_json_deserializes_missing_flags_as_false() {
        let caps: ModelCapabilities = serde_json::from_str("{}").unwrap();
        assert_eq!(caps, ModelCapabilities::default());
    }

    #[test]
    fn model_capabilities_embedding_flag_isolated() {
        let caps = ModelCapabilities {
            embedding: true,
            ..Default::default()
        };
        assert!(caps.embedding);
        assert!(!caps.reasoning);
        assert!(!caps.tools);
        assert!(!caps.vision);
    }

    #[tokio::test]
    async fn default_list_models_is_empty() {
        struct NoopLlm;
        impl LlmClient for NoopLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                Ok(LlmResponse::text(""))
            }
        }
        let llm = NoopLlm;
        assert!(llm.list_models().await.unwrap().is_empty());
        assert!(llm.refresh_models().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn retrying_client_exhausts_retries() {
        tokio::time::pause();

        let inner = FailThenSucceedLlm {
            remaining_failures: Mutex::new(10), // more failures than retries
        };
        let client = RetryingLlmClient::new(inner, 2);

        let result = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("429"));
    }

    // --- MODEL_OVERRIDE tests (issue #34) ---

    #[tokio::test]
    async fn current_model_override_is_none_outside_scope() {
        assert_eq!(current_model_override(), None);
    }

    #[tokio::test]
    async fn current_model_override_observes_scope() {
        let observed = with_model_override("gpt-5-mini".to_string(), async {
            current_model_override()
        })
        .await;
        assert_eq!(observed, Some("gpt-5-mini".to_string()));
        // After the scope exits the task-local is unset again.
        assert_eq!(current_model_override(), None);
    }

    #[tokio::test]
    async fn nested_model_override_shadows_outer() {
        let inner = with_model_override("outer".into(), async {
            with_model_override("inner".into(), async { current_model_override() }).await
        })
        .await;
        assert_eq!(inner, Some("inner".into()));
    }

    // --- CONTEXT_BUDGET tests (issue #63) ---

    #[tokio::test]
    async fn current_context_budget_returns_none_outside_scope() {
        // No `with_context_budget` wrapper has been installed — typical
        // test context or a background job that doesn't route through
        // the daemon's dispatch wrapper. Read site must observe `None`
        // rather than a misleading default so token-based compaction
        // skips the way it does when a connector reports `None`.
        assert_eq!(current_context_budget(), None);
    }

    #[tokio::test]
    async fn current_context_budget_returns_installed_value() {
        let budget = ContextBudget {
            max_input_tokens: 1_000_000,
            source: BudgetSource::PurposeOverride,
        };
        let observed = with_context_budget(budget, async { current_context_budget() }).await;
        assert_eq!(observed, Some(budget));
        // After the scope exits the task-local is unset again.
        assert_eq!(current_context_budget(), None);
    }

    #[tokio::test]
    async fn nested_context_budget_shadows_outer() {
        let outer = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::UniversalFallback,
        };
        let inner_budget = ContextBudget {
            max_input_tokens: 1_000_000,
            source: BudgetSource::PurposeOverride,
        };
        let observed = with_context_budget(outer, async {
            with_context_budget(inner_budget, async { current_context_budget() }).await
        })
        .await;
        assert_eq!(observed, Some(inner_budget));
    }
}
