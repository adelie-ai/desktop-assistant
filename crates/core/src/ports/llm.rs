use std::sync::{Arc, Mutex};

use crate::CoreError;
use crate::domain::{Message, ToolCall, ToolDefinition};

/// Callback invoked for each chunk of a streaming LLM response.
/// Return `true` to continue, `false` to abort the stream.
pub type ChunkCallback = Box<dyn FnMut(String) -> bool + Send>;

/// Response from the LLM, which may contain text, tool calls, or both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmResponse {
    /// The text content of the response (may be empty if only tool calls).
    pub text: String,
    /// Tool calls requested by the LLM (empty if text-only response).
    pub tool_calls: Vec<ToolCall>,
}

impl LlmResponse {
    /// Create a text-only response.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tool_calls: Vec::new(),
        }
    }

    /// Create a response with tool calls.
    pub fn with_tool_calls(text: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            text: text.into(),
            tool_calls,
        }
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

    /// Stream a completion from the LLM given a message history.
    /// Calls `on_chunk` for each text token/chunk received.
    /// Optionally accepts tool definitions to enable tool calling.
    /// Returns an `LlmResponse` which may include tool calls.
    fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        on_chunk: ChunkCallback,
    ) -> impl std::future::Future<Output = Result<LlmResponse, CoreError>> + Send;
}

/// Check whether a `CoreError` represents a retryable API error (429/529/rate-limit/overloaded).
pub fn is_retryable_error(e: &CoreError) -> bool {
    let normalized = e.to_string().to_ascii_lowercase();
    normalized.contains("429")
        || normalized.contains("rate_limit")
        || normalized.contains("529")
        || normalized.contains("overloaded")
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

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
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
            match self.inner.stream_completion(msgs, tools, proxy_cb).await {
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
    fn retryable_error_429() {
        let e = CoreError::Llm("HTTP 429 Too Many Requests".into());
        assert!(is_retryable_error(&e));
    }

    #[test]
    fn retryable_error_529() {
        let e = CoreError::Llm("HTTP 529 overloaded".into());
        assert!(is_retryable_error(&e));
    }

    #[test]
    fn retryable_error_rate_limit() {
        let e = CoreError::Llm("rate_limit_exceeded".into());
        assert!(is_retryable_error(&e));
    }

    #[test]
    fn retryable_error_overloaded() {
        let e = CoreError::Llm("API is overloaded".into());
        assert!(is_retryable_error(&e));
    }

    #[test]
    fn non_retryable_error() {
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
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let mut count = self.remaining_failures.lock().unwrap();
            if *count > 0 {
                *count -= 1;
                return Err(CoreError::Llm("HTTP 429 rate limited".into()));
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
                Box::new(|_| true),
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid API key"));
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
                Box::new(|_| true),
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("429"));
    }
}
