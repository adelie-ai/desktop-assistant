//! OpenAI Responses API connector implementing the core `LlmClient` port.

use desktop_assistant_core::CoreError;
#[cfg(test)]
use desktop_assistant_core::domain::ToolCall;
use desktop_assistant_core::domain::{Message, Role, ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelCapabilities, ModelInfo, ReasoningConfig,
    TokenUsage, current_model_override,
};
use desktop_assistant_llm_http::{
    STREAM_CONNECT_TIMEOUT, STREAM_EVENT_TIMEOUT, StreamStep, build_response, next_step,
    parse_retry_after_header,
};
use eventsource_stream::Eventsource;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Connection-handshake / per-event stall budgets shared with the other
/// connectors (#214/#220/#302). Kept as local aliases so the streaming loop
/// reads naturally.
const OPENAI_CONNECT_TIMEOUT: std::time::Duration = STREAM_CONNECT_TIMEOUT;
const OPENAI_EVENT_TIMEOUT: std::time::Duration = STREAM_EVENT_TIMEOUT;

/// Overall wall-clock budget for a *non-streaming* request (embeddings).
///
/// Why: streaming completions are bounded by the connect + per-event stall
/// timeouts instead, because a healthy long completion can legitimately run
/// longer than any single fixed cap; but a non-streaming embeddings call has
/// no per-event heartbeat, so a wedged backend would otherwise hang the
/// caller forever. 60s is generous for a batch embed yet still bounded.
const OPENAI_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Return the prompt-token context window for a known OpenAI model id.
///
/// OpenAI exposes context windows through `/v1/models` only inconsistently,
/// so this curated table covers the common families we ship with the model
/// picker. Returns `None` for ids that are unknown or that intentionally
/// defer to the daemon's universal fallback (e.g. the `gpt-5*` family,
/// whose window varies by sub-tier).
///
/// Order matters: more specific prefixes are checked before their general
/// counterparts (e.g. `gpt-4-turbo` before `gpt-4`).
pub fn context_limit_for_model(model: &str) -> Option<u64> {
    if model.starts_with("gpt-4o") || model.starts_with("gpt-4-turbo") {
        return Some(128_000);
    }
    if model.starts_with("gpt-4-32k") {
        return Some(32_768);
    }
    if model.starts_with("gpt-4") {
        return Some(8_192);
    }
    if model.starts_with("gpt-3.5-turbo-16k") {
        return Some(16_384);
    }
    if model.starts_with("gpt-3.5-turbo") {
        return Some(4_096);
    }
    if model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4") {
        return Some(200_000);
    }
    None
}

/// OpenAI-compatible LLM client that streams completions via the Responses API.
pub struct OpenAiClient {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    hosted_tool_search: bool,
    /// First-response (connect) stall budget; defaults to
    /// [`OPENAI_CONNECT_TIMEOUT`], overridable per-connection.
    connect_timeout: std::time::Duration,
    /// Per-chunk stall budget; defaults to [`OPENAI_EVENT_TIMEOUT`].
    event_timeout: std::time::Duration,
    /// Overall budget for non-streaming requests (embeddings); defaults to
    /// [`OPENAI_REQUEST_TIMEOUT`], overridable per-connection.
    request_timeout: std::time::Duration,
    /// Per-connection context-window hard cap, in tokens. `None` = "max
    /// available". Folded with the curated table in `max_context_tokens`.
    context_cap: Option<u64>,
}

impl OpenAiClient {
    pub fn get_default_model() -> Option<&'static str> {
        Some("gpt-5.4")
    }

    pub fn get_default_base_url() -> Option<&'static str> {
        Some("https://api.openai.com/v1")
    }

    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: Self::get_default_model().unwrap_or_default().to_string(),
            base_url: Self::get_default_base_url().unwrap_or_default().to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            hosted_tool_search: false,
            connect_timeout: OPENAI_CONNECT_TIMEOUT,
            event_timeout: OPENAI_EVENT_TIMEOUT,
            request_timeout: OPENAI_REQUEST_TIMEOUT,
            context_cap: None,
        }
    }

    /// Set the per-connection context-window hard cap, in tokens. `None`/
    /// `Some(0)` = "max available". Clamps the budget the daemon packs (no
    /// `num_ctx` to pin — the API enforces its own window), useful for
    /// bounding spend. See `desktop_assistant_llm_http::apply_context_cap`.
    pub fn with_max_context_tokens(mut self, max: Option<u64>) -> Self {
        self.context_cap = max.filter(|m| *m > 0);
        self
    }

    /// Override the first-response (connect) stall budget. `None`/`Some(0)`
    /// keeps the [`OPENAI_CONNECT_TIMEOUT`] default. Seconds.
    pub fn with_connect_timeout(mut self, secs: Option<u64>) -> Self {
        if let Some(s) = secs.filter(|s| *s > 0) {
            self.connect_timeout = std::time::Duration::from_secs(s);
        }
        self
    }

    /// Override the per-chunk stall budget. `None`/`Some(0)` keeps the
    /// [`OPENAI_EVENT_TIMEOUT`] default. Seconds.
    pub fn with_event_timeout(mut self, secs: Option<u64>) -> Self {
        if let Some(s) = secs.filter(|s| *s > 0) {
            self.event_timeout = std::time::Duration::from_secs(s);
        }
        self
    }

    /// Override the overall budget for non-streaming requests (embeddings).
    /// `None`/`Some(0)` keeps the [`OPENAI_REQUEST_TIMEOUT`] default. Seconds.
    pub fn with_request_timeout(mut self, secs: Option<u64>) -> Self {
        if let Some(s) = secs.filter(|s| *s > 0) {
            self.request_timeout = std::time::Duration::from_secs(s);
        }
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_temperature(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: Option<f64>) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_hosted_tool_search(mut self, enabled: bool) -> Self {
        self.hosted_tool_search = enabled;
        self
    }

    /// Return the model name as the stable version identifier.
    pub async fn model_identifier(&self) -> Result<String, CoreError> {
        Ok(self.model.clone())
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// Bounded by [`Self::with_request_timeout`] (default
    /// [`OPENAI_REQUEST_TIMEOUT`]): unlike the streaming path there is no
    /// per-event heartbeat to detect a stall, so the whole request/parse is
    /// wrapped in a wall-clock timeout to keep a wedged backend from hanging
    /// the caller indefinitely.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let request = async {
            let response = self
                .client
                .post(format!("{}/embeddings", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| CoreError::Llm(format!("embedding HTTP request failed: {e}")))?;

            let response = desktop_assistant_llm_http::bail_for_status(
                response,
                "OpenAI embeddings API error",
            )
            .await?;

            let parsed: EmbeddingResponse = response
                .json()
                .await
                .map_err(|e| CoreError::Llm(format!("failed to parse embedding response: {e}")))?;

            Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
        };

        match tokio::time::timeout(self.request_timeout, request).await {
            Ok(result) => result,
            Err(_) => Err(CoreError::Llm(format!(
                "OpenAI embeddings request timed out after {}s",
                self.request_timeout.as_secs()
            ))),
        }
    }

    pub fn from_env() -> Result<Self, CoreError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| CoreError::Llm("OPENAI_API_KEY environment variable not set".into()))?;
        let mut client = Self::new(api_key);
        if let Ok(model) = std::env::var("OPENAI_MODEL") {
            client.model = model;
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            client.base_url = url;
        }
        Ok(client)
    }
}

/// Non-reversible fingerprint of an API key for safe logging.
///
/// Returns the key length plus an FNV-1a64 digest (never the key bytes), so
/// a debug/log line can still tell two keys apart or confirm "the expected
/// key is loaded" without ever printing the secret. Mirrors the daemon's
/// `redacted_secret_audit` fingerprint style.
fn api_key_fingerprint(key: &str) -> String {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in key.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("<redacted len={} fnv1a64:{hash:016x}>", key.len())
}

/// Hand-written `Debug` that fingerprints the API key instead of printing it.
///
/// Why: the derived `Debug` would render `api_key` verbatim, and this client
/// is exactly the kind of value that ends up in a `tracing` field or a
/// wrapping struct's `{:?}`. Redacting at the `Debug` boundary means the
/// secret cannot leak through any formatting path.
impl std::fmt::Debug for OpenAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiClient")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &api_key_fingerprint(&self.api_key))
            .field("connect_timeout", &self.connect_timeout)
            .field("event_timeout", &self.event_timeout)
            .field("request_timeout", &self.request_timeout)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Responses API – request serialization types
// ---------------------------------------------------------------------------

/// OpenAI Responses-API reasoning block (`{ "effort": "low|medium|high" }`).
///
/// Emitted only for reasoning-capable models (see
/// [`model_supports_reasoning`]). For non-reasoning models the field is
/// omitted entirely; sending it there causes a 400 from the API.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct ReasoningBlock {
    effort: &'static str,
}

/// Unified request body for the Responses API (`POST /v1/responses`).
#[derive(Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningBlock>,
}

/// Heterogeneous input items for the Responses API.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(untagged)]
enum InputItem {
    Message(InputMessage),
    FunctionCall(InputFunctionCall),
    FunctionCallOutput(InputFunctionCallOutput),
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct InputMessage {
    role: String,
    content: String,
}

/// A tool call from a previous assistant turn, replayed as input.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct InputFunctionCall {
    r#type: String,
    /// Synthetic ID for the input item (not the correlation key).
    id: String,
    /// The real correlation key matching `function_call_output`.
    call_id: String,
    name: String,
    arguments: String,
}

/// A tool result replayed as input.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct InputFunctionCallOutput {
    r#type: String,
    call_id: String,
    output: String,
}

/// Tool entry in the `tools` array – can be a function, namespace, or tool_search.
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
enum ToolEntry {
    Function(FunctionTool),
    Namespace(NamespaceTool),
    ToolSearch(ToolSearchSentinel),
}

/// Flat function tool (name at top level, not nested under `function`).
#[derive(Serialize, Debug, Clone, PartialEq)]
struct FunctionTool {
    r#type: String,
    name: String,
    description: String,
    parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    defer_loading: Option<bool>,
}

impl FunctionTool {
    fn from_definition(def: &ToolDefinition) -> Self {
        Self {
            r#type: "function".to_string(),
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: def.parameters.clone(),
            defer_loading: None,
        }
    }

    fn from_definition_deferred(def: &ToolDefinition) -> Self {
        Self {
            r#type: "function".to_string(),
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: def.parameters.clone(),
            defer_loading: Some(true),
        }
    }
}

/// A tool namespace entry for hosted tool search.
#[derive(Serialize, Debug, Clone, PartialEq)]
struct NamespaceTool {
    r#type: String,
    name: String,
    description: String,
    tools: Vec<FunctionTool>,
}

impl NamespaceTool {
    fn from_namespace(ns: &ToolNamespace) -> Self {
        // Prefix namespace names to avoid collisions with OpenAI reserved
        // namespaces (e.g. "terminal", "browser").
        let name = format!("adele_{}", ns.name);
        Self {
            r#type: "namespace".to_string(),
            name,
            description: ns.description.clone(),
            tools: ns
                .tools
                .iter()
                .map(FunctionTool::from_definition_deferred)
                .collect(),
        }
    }
}

/// The `tool_search` sentinel that enables OpenAI's hosted tool discovery.
#[derive(Serialize, Debug, Clone, PartialEq)]
struct ToolSearchSentinel {
    r#type: String,
}

// ---------------------------------------------------------------------------
// Embedding response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Responses API – streaming response types
// ---------------------------------------------------------------------------

/// `response.output_text.delta` event payload.
#[derive(Deserialize)]
struct TextDelta {
    delta: String,
}

/// `response.output_item.added` event payload.
#[derive(Deserialize)]
struct OutputItemAdded {
    output_index: usize,
    item: OutputItem,
}

/// An output item (we only care about `function_call` items).
#[derive(Deserialize)]
struct OutputItem {
    r#type: String,
    #[allow(dead_code)]
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

/// `response.function_call_arguments.delta` event payload.
#[derive(Deserialize)]
struct FunctionArgsDelta {
    output_index: usize,
    delta: String,
}

/// `response.function_call_arguments.done` event payload.
#[derive(Deserialize)]
struct FunctionArgsDone {
    output_index: usize,
    arguments: String,
}

/// `response.completed` event payload.
#[derive(Deserialize)]
struct ResponseCompleted {
    response: ResponseCompletedInner,
}

#[derive(Deserialize)]
struct ResponseCompletedInner {
    #[serde(default)]
    usage: Option<ResponseUsage>,
}

#[derive(Deserialize)]
struct ResponseUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

/// OpenAI Responses-API events index tool calls with `output_index`
/// (`usize`). Use the shared accumulator from core (#45).
type ResponseToolAccumulator = desktop_assistant_core::ports::llm::ToolCallAccumulator<usize>;

// ---------------------------------------------------------------------------
// Message conversion: domain Messages → Responses API InputItems
// ---------------------------------------------------------------------------

/// Convert domain messages to Responses API input items plus an optional
/// `instructions` string (extracted from system messages; multiple system
/// messages are concatenated since the Responses API accepts a single string).
fn convert_messages(messages: &[Message]) -> (Vec<InputItem>, Option<String>) {
    let mut items = Vec::new();
    let mut instructions: Option<String> = None;

    for msg in messages {
        match msg.role {
            Role::System => match &mut instructions {
                Some(existing) => {
                    existing.push_str("\n\n");
                    existing.push_str(&msg.content);
                }
                None => {
                    instructions = Some(msg.content.clone());
                }
            },
            Role::User => {
                items.push(InputItem::Message(InputMessage {
                    role: "user".to_string(),
                    content: msg.content.clone(),
                }));
            }
            Role::Assistant => {
                // Emit text content as a message if non-empty
                if !msg.content.is_empty() {
                    items.push(InputItem::Message(InputMessage {
                        role: "assistant".to_string(),
                        content: msg.content.clone(),
                    }));
                }
                // Emit each tool call as a separate FunctionCall item
                for tc in &msg.tool_calls {
                    items.push(InputItem::FunctionCall(InputFunctionCall {
                        r#type: "function_call".to_string(),
                        id: format!("fc_{}", tc.id),
                        call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    }));
                }
            }
            Role::Tool => {
                if let Some(call_id) = &msg.tool_call_id {
                    items.push(InputItem::FunctionCallOutput(InputFunctionCallOutput {
                        r#type: "function_call_output".to_string(),
                        call_id: call_id.clone(),
                        output: msg.content.clone(),
                    }));
                }
            }
        }
    }

    (items, instructions)
}

// ---------------------------------------------------------------------------
// Streaming implementation
// ---------------------------------------------------------------------------

/// Classify a `reqwest` transport error from the initial `send()` into a
/// `CoreError`.
///
/// Connection-phase (`is_connect`) and timeout (`is_timeout`) failures are
/// transient: the request never established a stream, so nothing has been
/// emitted downstream and a retry is idempotency-safe. They map to
/// [`CoreError::RateLimited`], the one variant the core `RetryingLlmClient`
/// decorator retries with backoff (guarded by its stream-not-started check).
/// This brings the reqwest path up to the transport-retry bar that Bedrock
/// gets for free from the AWS SDK. Any other transport error (e.g. a
/// malformed request builder) is a permanent [`CoreError::Llm`].
///
/// `reqwest::Error`'s `Display` never includes request headers, so the
/// bearer token cannot leak through the formatted `detail`.
fn classify_send_error(e: &reqwest::Error) -> CoreError {
    if e.is_connect() || e.is_timeout() {
        CoreError::RateLimited {
            retry_after: None,
            detail: format!("OpenAI connection error: {e}"),
        }
    } else {
        CoreError::Llm(format!("HTTP request failed: {e}"))
    }
}

impl OpenAiClient {
    /// Send a Responses API request and parse the SSE stream into an LlmResponse.
    async fn send_and_stream(
        &self,
        request_json: &str,
        request_body: &impl Serialize,
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let request_bytes = request_json.len();
        tracing::info!(
            request_bytes,
            model = %self.model,
            "LLM request payload"
        );
        tracing::debug!(
            "LLM request body (first 2000 chars): {}",
            &request_json[..request_json.len().min(2000)]
        );

        // Cooperative cancellation token (issue #109): see Anthropic
        // adapter for the threading rationale.
        let cancellation =
            desktop_assistant_core::ports::llm::current_cancellation_token().unwrap_or_default();
        if cancellation.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let send_fut = self
            .client
            .post(format!("{}/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request_body)
            .send();
        // Bound the connection handshake so a stalled connect fails the turn
        // instead of hanging forever (#220).
        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err(CoreError::Cancelled),
            _ = tokio::time::sleep(self.connect_timeout) => {
                tracing::error!(
                    timeout_s = self.connect_timeout.as_secs(),
                    "OpenAI request send() timed out (no response headers)"
                );
                return Err(CoreError::Llm("OpenAI stream stalled".into()));
            }
            r = send_fut => r.map_err(|e| classify_send_error(&e))?,
        };

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after_header(response.headers());
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            // Detect prompt-overflow rejections at the connector boundary
            // and surface them as `CoreError::ContextOverflow` so the
            // core service can truncate the offending tool result and
            // retry. OpenAI returns HTTP 400 with a body shaped like
            // `{ "error": { "code": "context_length_exceeded", "type":
            // "invalid_request_error", "message": "..." } }`.
            if let Some((prompt_tokens, max_tokens)) = detect_openai_context_overflow(&body) {
                tracing::warn!(
                    prompt_tokens = ?prompt_tokens,
                    max_tokens = ?max_tokens,
                    "OpenAI rejected request for context overflow"
                );
                return Err(CoreError::ContextOverflow {
                    prompt_tokens,
                    max_tokens,
                    detail: format!("OpenAI API error (HTTP {status}): {body}"),
                });
            }
            // HTTP 429 is overloaded between two semantically distinct
            // signals on OpenAI: throttling (transient, retryable) and
            // `insufficient_quota` (permanent, billing). Distinguish by
            // parsing the structured error envelope so downstream
            // classifiers don't have to.
            if status.as_u16() == 429 {
                if detect_openai_insufficient_quota(&body) {
                    return Err(CoreError::QuotaExceeded {
                        detail: format!("OpenAI API error (HTTP {status}): {body}"),
                    });
                }
                return Err(CoreError::RateLimited {
                    retry_after,
                    detail: format!("OpenAI API error (HTTP {status}): {body}"),
                });
            }
            // Any 5xx is a transient server-side failure (500 internal, 502
            // bad gateway, 503 overloaded, 504 gateway timeout). These arrive
            // before any stream byte is consumed, so a retry is
            // idempotency-safe; map them to `RateLimited` so the retry
            // decorator backs off and retries rather than hard-failing the
            // turn. 4xx (bad request, auth, unknown model) stay terminal
            // below — they can never succeed on retry.
            if status.is_server_error() {
                return Err(CoreError::RateLimited {
                    retry_after,
                    detail: format!("OpenAI API error (HTTP {status}): {body}"),
                });
            }
            return Err(CoreError::Llm(format!(
                "OpenAI API error (HTTP {status}): {body}"
            )));
        }

        let mut events = response.bytes_stream().eventsource();

        let mut full_response = String::new();
        let mut tool_acc = ResponseToolAccumulator::default();
        let mut token_usage: Option<TokenUsage> = None;
        // Truncation guard (#561): the Responses API always terminates a
        // healthy stream with `response.completed`. If the SSE stream instead
        // ends (`StreamStep::Done`) without it, the connection was dropped
        // mid-response and the accumulated text is a partial answer — surface
        // that as an error rather than a successful completion. A caller-driven
        // stop (the chunk callback returning false) is a deliberate abort, not
        // a truncation, so it is tracked separately.
        let mut saw_completed = false;
        let mut aborted_by_callback = false;

        loop {
            // Race the next SSE event against cancellation and a stall
            // timeout. See the Anthropic adapter for the rationale; same
            // pattern here. The stall window resets on every received
            // event (#220).
            let event = match next_step(&mut events, &cancellation, self.event_timeout).await {
                StreamStep::Item(ev) => ev,
                StreamStep::Done => break,
                StreamStep::Cancelled => {
                    tracing::debug!("OpenAI stream cancelled by token");
                    drop(events);
                    return Err(CoreError::Cancelled);
                }
                StreamStep::Stalled => {
                    tracing::error!(
                        timeout_s = self.event_timeout.as_secs(),
                        "OpenAI stream stalled — no further event"
                    );
                    drop(events);
                    return Err(CoreError::Llm("OpenAI stream stalled".into()));
                }
            };
            let event = event.map_err(|e| CoreError::Llm(format!("stream read error: {e}")))?;
            let data = event.data.as_str();
            match event.event.as_str() {
                "response.output_text.delta" => {
                    if let Ok(td) = serde_json::from_str::<TextDelta>(data) {
                        full_response.push_str(&td.delta);
                        if !on_chunk(td.delta) {
                            tracing::debug!("streaming aborted by callback");
                            aborted_by_callback = true;
                            break;
                        }
                    }
                }
                "response.output_item.added" => {
                    if let Ok(added) = serde_json::from_str::<OutputItemAdded>(data)
                        && added.item.r#type == "function_call"
                    {
                        tool_acc.start(
                            added.output_index,
                            added.item.call_id.unwrap_or_default(),
                            added.item.name.unwrap_or_default(),
                        );
                    }
                }
                "response.function_call_arguments.delta" => {
                    if let Ok(d) = serde_json::from_str::<FunctionArgsDelta>(data) {
                        tool_acc.append(d.output_index, &d.delta);
                    }
                }
                "response.function_call_arguments.done" => {
                    if let Ok(d) = serde_json::from_str::<FunctionArgsDone>(data) {
                        tool_acc.finalize(d.output_index, &d.arguments);
                    }
                }
                "response.tool_search_call.searching" => {
                    tracing::info!("tool search initiated");
                }
                "response.tool_search_call.in_progress" => {
                    // Tool search still running — nothing to do.
                }
                "response.tool_search_call.completed" => {
                    tracing::info!(data, "tool search completed");
                }
                "response.failed" => {
                    tracing::warn!("OpenAI response failed: {data}");
                    // Extract a concise error message if possible.
                    let msg = serde_json::from_str::<serde_json::Value>(data)
                        .ok()
                        .and_then(|v| {
                            v.get("response")
                                .and_then(|r| r.get("error"))
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str())
                                .map(String::from)
                        })
                        .unwrap_or_else(|| "response.failed".into());
                    return Err(CoreError::Llm(format!("OpenAI server_error: {msg}")));
                }
                "error" => {
                    tracing::warn!("OpenAI stream error: {data}");
                }
                "response.completed" => {
                    if let Ok(rc) = serde_json::from_str::<ResponseCompleted>(data)
                        && let Some(u) = rc.response.usage
                    {
                        token_usage = Some(TokenUsage {
                            input_tokens: u.input_tokens,
                            output_tokens: u.output_tokens,
                            ..Default::default()
                        });
                    }
                    saw_completed = true;
                    break;
                }
                other => {
                    tracing::debug!("ignoring SSE event: {:?}", other);
                }
            }
        }

        // A stream that ended without `response.completed` and was not
        // deliberately aborted by the caller is a dropped/truncated response.
        if !saw_completed && !aborted_by_callback {
            tracing::error!(
                text_len = full_response.len(),
                "OpenAI stream ended before response.completed (truncated)"
            );
            return Err(CoreError::Llm(
                "OpenAI stream truncated: connection ended before response.completed".into(),
            ));
        }

        let tool_calls = tool_acc.into_tool_calls();
        tracing::debug!(
            text_len = full_response.len(),
            tool_call_count = tool_calls.len(),
            "OpenAI response parsed"
        );
        Ok(build_response(full_response, tool_calls, token_usage))
    }
}

/// OpenAI error envelope used to detect prompt-overflow rejections.
///
/// The Responses API and Chat Completions API both surface structured
/// errors as `{ "error": { "code": "...", "type": "...", "message":
/// "..." } }`; we only deserialize the inner `error` shape since that's
/// the only thing this connector inspects.
#[derive(Deserialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorBody,
}

#[derive(Deserialize, Default)]
struct OpenAiErrorBody {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: String,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
}

/// Detect an OpenAI context-overflow rejection in an HTTP error body.
///
/// Returns `Some((prompt_tokens, max_tokens))` (each may itself be `None`
/// when the wording doesn't carry numbers) when the body parses as the
/// OpenAI error envelope and `error.code == "context_length_exceeded"`.
/// Returns `None` for any other shape so the caller can fall through to
/// a generic `CoreError::Llm`.
///
/// Why: pattern-matching on error message strings is normally banned
/// (see `AGENTS.md`), but at the connector boundary this is the only
/// signal OpenAI provides for context-window rejections — converting
/// it into structured `CoreError::ContextOverflow` here is exactly what
/// the rule carves out, since downstream code never has to.
fn detect_openai_context_overflow(body: &str) -> Option<(Option<u64>, Option<u64>)> {
    let envelope: OpenAiErrorEnvelope = serde_json::from_str(body).ok()?;
    if envelope.error.code.as_deref() != Some("context_length_exceeded") {
        return None;
    }
    Some(parse_openai_context_length_message(&envelope.error.message))
}

/// Detect OpenAI's `insufficient_quota` billing error in an HTTP error
/// body. Returns true when the structured error envelope has either
/// `error.code == "insufficient_quota"` or `error.type == "insufficient_quota"`.
///
/// Why: OpenAI uses HTTP 429 for two semantically distinct signals —
/// transient rate-limit throttling (retryable) and permanent
/// `insufficient_quota` (NOT retryable). Distinguishing them at the
/// connector boundary lets `is_retryable_error` stay a flat
/// `matches!(CoreError::RateLimited)` downstream. This is the same
/// connector-boundary carve-out that `detect_openai_context_overflow`
/// uses — see that function's docstring.
fn detect_openai_insufficient_quota(body: &str) -> bool {
    let Ok(envelope): Result<OpenAiErrorEnvelope, _> = serde_json::from_str(body) else {
        return false;
    };
    envelope.error.code.as_deref() == Some("insufficient_quota")
        || envelope.error.error_type.as_deref() == Some("insufficient_quota")
}

/// Parse OpenAI's `"This model's maximum context length is 128000
/// tokens. However, your messages resulted in 153827 tokens. ..."`
/// wording into `(prompt_tokens, max_tokens)`.
///
/// OpenAI lists the numbers in the order `(max, prompt)` — opposite to
/// Bedrock/Anthropic — so this swaps them before returning. If either
/// value is missing the corresponding tuple slot is `None`; the core's
/// overflow recovery path tolerates absent measurements.
fn parse_openai_context_length_message(message: &str) -> (Option<u64>, Option<u64>) {
    let nums: Vec<u64> = message
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    match nums.as_slice() {
        [max, prompt, ..] => (Some(*prompt), Some(*max)),
        [max] => (None, Some(*max)),
        _ => (None, None),
    }
}

/// Curated list of OpenAI models exposed through this connector.
///
/// OpenAI's `/v1/models` endpoint returns a firehose of model IDs
/// (including retired and fine-tune variants) with no capability metadata,
/// so this hand-maintained table is the set the model picker should offer.
///
/// `list_models` routes this table through the shared
/// [`merge_curated_with_live`](desktop_assistant_llm_http::merge_curated_with_live)
/// resolver (issue #304). Today it passes an empty live list, so the curated
/// table is returned unchanged; when we start fetching `/v1/models` the live
/// ids slot in as the resolver's `live` argument — curated metadata wins on
/// overlap, unknown live ids are appended.
///
/// To extend: append a `ModelInfo` entry here with the right
/// `ModelCapabilities` flags. `reasoning: true` belongs on the o-series
/// (o1, o3, o4) and any GPT-5 variant that exposes reasoning traces.
/// True when `model` is one of the curated OpenAI models flagged as
/// reasoning-capable. Used to gate the `reasoning.effort` field on
/// requests — sending it for non-reasoning models (e.g. `gpt-4o`) is a
/// 400 from the API, so we silently drop it with a debug log.
///
/// Matches by exact id AND common prefixes (e.g. `gpt-5-mini-2025…`), so
/// custom pinned versions resolve the same way as their family name.
fn model_supports_reasoning(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    // Exact id match against the curated capabilities table.
    if curated_openai_models()
        .iter()
        .any(|c| c.id.eq_ignore_ascii_case(model) && c.capabilities.reasoning)
    {
        return true;
    }
    // Prefix heuristics for pinned/versioned ids not in the curated table.
    //   o-series: `o1`, `o1-mini`, `o3`, `o3-mini`, `o4-mini`, …
    //   GPT-5 reasoning: `gpt-5`, `gpt-5-mini`, `gpt-5.4`, …
    let is_o_series = m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4");
    let is_gpt5 = m.starts_with("gpt-5");
    is_o_series || is_gpt5
}

/// Build a `ReasoningBlock` from the per-turn reasoning hint, honoring
/// the per-model capability gate. Returns `None` for non-reasoning
/// models (debug-logged) and when no effort is requested.
fn reasoning_for(model: &str, reasoning: ReasoningConfig) -> Option<ReasoningBlock> {
    let level = reasoning.reasoning_effort?;
    if !model_supports_reasoning(model) {
        tracing::debug!(
            model,
            requested_effort = ?level,
            "OpenAI reasoning_effort requested but model is not reasoning-capable; dropping field"
        );
        return None;
    }
    Some(ReasoningBlock {
        effort: level.as_openai_effort(),
    })
}

fn curated_openai_models() -> Vec<ModelInfo> {
    let chat_caps = ModelCapabilities {
        reasoning: false,
        vision: true,
        tools: true,
        embedding: false,
    };
    let reasoning_caps = ModelCapabilities {
        reasoning: true,
        vision: true,
        tools: true,
        embedding: false,
    };
    let embedding_caps = ModelCapabilities {
        reasoning: false,
        vision: false,
        tools: false,
        embedding: true,
    };

    vec![
        // --- GPT-5 family ---
        ModelInfo::new("gpt-5")
            .with_display_name("GPT-5")
            .with_context_limit(400_000)
            .with_capabilities(reasoning_caps),
        ModelInfo::new("gpt-5-mini")
            .with_display_name("GPT-5 Mini")
            .with_context_limit(400_000)
            .with_capabilities(reasoning_caps),
        ModelInfo::new("gpt-5.4")
            .with_display_name("GPT-5.4")
            .with_context_limit(400_000)
            .with_capabilities(reasoning_caps),
        // --- o-series reasoning models ---
        ModelInfo::new("o4-mini")
            .with_display_name("o4-mini")
            .with_context_limit(200_000)
            .with_capabilities(reasoning_caps),
        ModelInfo::new("o3")
            .with_display_name("o3")
            .with_context_limit(200_000)
            .with_capabilities(reasoning_caps),
        ModelInfo::new("o3-mini")
            .with_display_name("o3-mini")
            .with_context_limit(200_000)
            .with_capabilities(reasoning_caps),
        // --- GPT-4.1 / 4o family (fallback general chat) ---
        ModelInfo::new("gpt-4.1")
            .with_display_name("GPT-4.1")
            .with_context_limit(1_000_000)
            .with_capabilities(chat_caps),
        ModelInfo::new("gpt-4o")
            .with_display_name("GPT-4o")
            .with_context_limit(128_000)
            .with_capabilities(chat_caps),
        ModelInfo::new("gpt-4o-mini")
            .with_display_name("GPT-4o mini")
            .with_context_limit(128_000)
            .with_capabilities(chat_caps),
        // --- Embedding models ---
        ModelInfo::new("text-embedding-3-large")
            .with_display_name("Text Embedding 3 Large")
            .with_capabilities(embedding_caps),
        ModelInfo::new("text-embedding-3-small")
            .with_display_name("Text Embedding 3 Small")
            .with_capabilities(embedding_caps),
    ]
}

#[async_trait::async_trait]
impl LlmClient for OpenAiClient {
    fn get_default_model(&self) -> Option<&str> {
        Self::get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        Self::get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        let model = current_model_override().unwrap_or_else(|| self.model.clone());
        // Fold the per-connection hard cap into the curated window so the
        // daemon budgets against the capped value (e.g. to bound spend).
        desktop_assistant_llm_http::apply_context_cap(
            self.context_cap,
            context_limit_for_model(&model),
        )
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // No live `/v1/models` fetch yet; the empty live list returns the
        // curated table unchanged. Routing through the shared resolver keeps
        // the merge policy in one place for when a live fetch lands (#304).
        Ok(desktop_assistant_llm_http::merge_curated_with_live(
            curated_openai_models(),
            Vec::new(),
        ))
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let (input, instructions) = convert_messages(&messages);
        let tool_entries: Vec<ToolEntry> = tools
            .iter()
            .map(|t| ToolEntry::Function(FunctionTool::from_definition(t)))
            .collect();

        // Per-turn model override (issue #34): when the daemon-side routing
        // layer has set `MODEL_OVERRIDE`, dispatch the user-chosen model
        // instead of the connector's baked-in `self.model`. The reasoning
        // gating below is keyed on the dispatched model so e.g. a request
        // for `gpt-5` carries `reasoning_effort` even when the connection
        // was built with a non-reasoning default.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());

        let reasoning_block = reasoning_for(&model, reasoning);

        let request = ResponsesRequest {
            model,
            input,
            instructions,
            stream: true,
            tools: tool_entries,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_tokens,
            reasoning: reasoning_block,
        };

        let request_json =
            serde_json::to_string(&request).unwrap_or_else(|_| "<serialization error>".into());

        self.send_and_stream(&request_json, &request, on_chunk)
            .await
    }

    fn supports_hosted_tool_search(&self) -> bool {
        self.hosted_tool_search
    }

    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let (input, instructions) = convert_messages(&messages);

        let mut tool_entries: Vec<ToolEntry> = core_tools
            .iter()
            .map(|t| ToolEntry::Function(FunctionTool::from_definition(t)))
            .collect();

        for ns in namespaces {
            tool_entries.push(ToolEntry::Namespace(NamespaceTool::from_namespace(ns)));
        }

        tool_entries.push(ToolEntry::ToolSearch(ToolSearchSentinel {
            r#type: "tool_search".to_string(),
        }));

        // Per-turn model override (issue #34); see `stream_completion`.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());

        let reasoning_block = reasoning_for(&model, reasoning);

        let request = ResponsesRequest {
            model,
            input,
            instructions,
            stream: true,
            tools: tool_entries,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_tokens,
            reasoning: reasoning_block,
        };

        let request_json =
            serde_json::to_string(&request).unwrap_or_else(|_| "<serialization error>".into());

        tracing::info!(
            namespace_count = namespaces.len(),
            core_tool_count = core_tools.len(),
            "using hosted tool search with namespaces"
        );

        self.send_and_stream(&request_json, &request, on_chunk)
            .await
    }
}

#[async_trait::async_trait]
impl desktop_assistant_core::ports::embedding::EmbeddingClient for OpenAiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        OpenAiClient::embed(self, texts).await
    }

    async fn model_identifier(&self) -> Result<String, CoreError> {
        OpenAiClient::model_identifier(self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::ports::llm::ReasoningLevel;

    // The stall-loop primitive (`StreamStep` / `next_step`) and its
    // `StallingStream` harness now live in `desktop-assistant-llm-http` and are
    // tested there (#302).

    // --- convert_messages tests ---

    #[test]
    fn convert_messages_user_only() {
        let msgs = vec![Message::new(Role::User, "hello")];
        let (items, instructions) = convert_messages(&msgs);
        assert!(instructions.is_none());
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0],
            InputItem::Message(InputMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            })
        );
    }

    #[test]
    fn convert_messages_system_becomes_instructions() {
        let msgs = vec![
            Message::new(Role::System, "You are helpful."),
            Message::new(Role::User, "hi"),
        ];
        let (items, instructions) = convert_messages(&msgs);
        assert_eq!(instructions.as_deref(), Some("You are helpful."));
        // System message should NOT appear in items
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn convert_messages_concatenates_system_messages() {
        let msgs = vec![
            Message::new(Role::System, "first"),
            Message::new(Role::System, "second"),
            Message::new(Role::User, "hi"),
        ];
        let (_, instructions) = convert_messages(&msgs);
        assert_eq!(instructions.as_deref(), Some("first\n\nsecond"));
    }

    #[test]
    fn convert_messages_assistant_text() {
        let msgs = vec![Message::new(Role::Assistant, "I can help")];
        let (items, _) = convert_messages(&msgs);
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0],
            InputItem::Message(InputMessage {
                role: "assistant".to_string(),
                content: "I can help".to_string(),
            })
        );
    }

    #[test]
    fn convert_messages_assistant_with_tool_calls() {
        let calls = vec![ToolCall::new("c1", "read_file", r#"{"path": "/tmp/a"}"#)];
        let msg = Message::assistant_with_tool_calls(calls);
        let (items, _) = convert_messages(&[msg]);
        // Empty content → no InputMessage, just the FunctionCall
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::FunctionCall(fc) => {
                assert_eq!(fc.r#type, "function_call");
                assert_eq!(fc.call_id, "c1");
                assert_eq!(fc.id, "fc_c1");
                assert_eq!(fc.name, "read_file");
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }

    #[test]
    fn convert_messages_tool_result() {
        let msg = Message::tool_result("c1", "file contents");
        let (items, _) = convert_messages(&[msg]);
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::FunctionCallOutput(out) => {
                assert_eq!(out.r#type, "function_call_output");
                assert_eq!(out.call_id, "c1");
                assert_eq!(out.output, "file contents");
            }
            other => panic!("expected FunctionCallOutput, got {other:?}"),
        }
    }

    #[test]
    fn convert_messages_mixed_history() {
        let msgs = vec![
            Message::new(Role::System, "Be concise."),
            Message::new(Role::User, "Read /tmp/a"),
            Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1",
                "read_file",
                r#"{"path":"/tmp/a"}"#,
            )]),
            Message::tool_result("c1", "contents of a"),
            Message::new(Role::Assistant, "Here are the contents."),
        ];
        let (items, instructions) = convert_messages(&msgs);
        assert_eq!(instructions.as_deref(), Some("Be concise."));
        assert_eq!(items.len(), 4); // user, function_call, function_call_output, assistant text
    }

    // --- Tool serialization tests ---

    #[test]
    fn function_tool_serialization_flat_name() {
        let def = ToolDefinition::new("test", "A test tool", serde_json::json!({"type": "object"}));
        let tool = FunctionTool::from_definition(&def);
        let json: serde_json::Value = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "test");
        assert_eq!(json["description"], "A test tool");
        // Name is at top level, NOT nested under "function"
        assert!(json.get("function").is_none());
    }

    #[test]
    fn function_tool_deferred_has_defer_loading() {
        let def = ToolDefinition::new("t", "d", serde_json::json!({}));
        let tool = FunctionTool::from_definition_deferred(&def);
        let json: serde_json::Value = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["defer_loading"], true);
    }

    #[test]
    fn namespace_tool_serialization() {
        let ns = ToolNamespace::new(
            "jira",
            "Jira project tools",
            vec![
                ToolDefinition::new("jira__list", "List issues", serde_json::json!({})),
                ToolDefinition::new("jira__create", "Create issue", serde_json::json!({})),
            ],
        );
        let entry = ToolEntry::Namespace(NamespaceTool::from_namespace(&ns));
        let json: serde_json::Value = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["type"], "namespace");
        assert_eq!(json["name"], "adele_jira");
        assert_eq!(json["description"], "Jira project tools");
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["type"], "function");
        assert!(tools[0]["defer_loading"].as_bool().unwrap());
        assert_eq!(tools[0]["name"], "jira__list");
        // Flat format: no "function" wrapper
        assert!(tools[0].get("function").is_none());
    }

    #[test]
    fn tool_search_sentinel_serialization() {
        let entry = ToolEntry::ToolSearch(ToolSearchSentinel {
            r#type: "tool_search".to_string(),
        });
        let json: serde_json::Value = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["type"], "tool_search");
    }

    // --- Request serialization tests ---

    #[test]
    fn response_request_uses_max_output_tokens() {
        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: Some(1024),
            reasoning: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("max_output_tokens"));
        assert!(!json.contains("\"max_tokens\""));
    }

    #[test]
    fn response_request_uses_instructions() {
        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: Some("Be helpful.".into()),
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };
        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(json["instructions"], "Be helpful.");
    }

    #[test]
    fn response_request_omits_empty_tools() {
        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
    }

    #[test]
    fn response_request_with_tools() {
        let def = ToolDefinition::new("test", "desc", serde_json::json!({"type": "object"}));
        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![ToolEntry::Function(FunctionTool::from_definition(&def))],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("\"test\""));
    }

    #[test]
    fn request_with_namespaces_serialization() {
        let core_tool = ToolDefinition::new("mcp_control", "Control MCP", serde_json::json!({}));
        let ns = ToolNamespace::new(
            "kb",
            "Knowledge base tools",
            vec![ToolDefinition::new(
                "kb_write",
                "Write to KB",
                serde_json::json!({}),
            )],
        );

        let mut entries: Vec<ToolEntry> = vec![ToolEntry::Function(FunctionTool::from_definition(
            &core_tool,
        ))];
        entries.push(ToolEntry::Namespace(NamespaceTool::from_namespace(&ns)));
        entries.push(ToolEntry::ToolSearch(ToolSearchSentinel {
            r#type: "tool_search".to_string(),
        }));

        let req = ResponsesRequest {
            model: "gpt-5.4".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: entries,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };

        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[1]["type"], "namespace");
        assert_eq!(tools[2]["type"], "tool_search");
    }

    // --- Reasoning / effort tests ----------------------------------------

    #[test]
    fn model_supports_reasoning_curated_ids() {
        assert!(model_supports_reasoning("gpt-5"));
        assert!(model_supports_reasoning("gpt-5-mini"));
        assert!(model_supports_reasoning("gpt-5.4"));
        assert!(model_supports_reasoning("o3"));
        assert!(model_supports_reasoning("o3-mini"));
        assert!(model_supports_reasoning("o4-mini"));
    }

    #[test]
    fn model_supports_reasoning_rejects_chat_models() {
        assert!(!model_supports_reasoning("gpt-4o"));
        assert!(!model_supports_reasoning("gpt-4o-mini"));
        assert!(!model_supports_reasoning("gpt-4.1"));
    }

    #[test]
    fn model_supports_reasoning_prefix_heuristic_versioned_ids() {
        // Pinned versions not in the curated list still resolve correctly.
        assert!(model_supports_reasoning("o1-2024-12-17"));
        assert!(model_supports_reasoning("gpt-5-mini-2025-09-01"));
        assert!(!model_supports_reasoning("gpt-4o-2024-11-20"));
    }

    #[test]
    fn reasoning_for_omits_when_no_effort_requested() {
        assert!(reasoning_for("gpt-5", ReasoningConfig::default()).is_none());
    }

    #[test]
    fn reasoning_for_emits_block_on_reasoning_model() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::High);
        let block = reasoning_for("gpt-5.4", cfg).expect("reasoning block expected");
        assert_eq!(block.effort, "high");
    }

    #[test]
    fn reasoning_for_drops_block_on_non_reasoning_model() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::High);
        // gpt-4o does not support reasoning — field should be dropped, not sent.
        assert!(reasoning_for("gpt-4o", cfg).is_none());
    }

    #[test]
    fn request_includes_reasoning_for_supported_model() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::Medium);
        let block = reasoning_for("gpt-5", cfg);
        let req = ResponsesRequest {
            model: "gpt-5".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: block,
        };
        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(json["reasoning"]["effort"], "medium");
    }

    #[test]
    fn request_omits_reasoning_field_when_none() {
        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input: vec![],
            instructions: None,
            stream: true,
            tools: vec![],
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("reasoning"),
            "reasoning field must be omitted when None; got: {json}"
        );
    }

    // Tool accumulator unit tests moved to
    // `desktop_assistant_core::ports::llm` (#45) along with the type.

    // --- Client builder test ---

    #[test]
    fn client_builder() {
        let client = OpenAiClient::new("test-key".into())
            .with_model("gpt-3.5-turbo")
            .with_base_url("http://localhost:8080");
        assert_eq!(client.model, "gpt-3.5-turbo");
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn from_env_missing_key() {
        // SAFETY: single-threaded test scope; the env var is the one this
        // test owns, no other test in this binary touches `OPENAI_API_KEY`.
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let result = OpenAiClient::from_env();
        assert!(matches!(result, Err(CoreError::Llm(_))));
    }

    /// The shared-resolver path with an empty live list must return exactly
    /// the curated table — proves the #304 refactor is behavior-preserving.
    #[tokio::test]
    async fn list_models_matches_curated_table_via_resolver() {
        let client = OpenAiClient::new("key".into());
        let models = client.list_models().await.unwrap();
        assert_eq!(models, curated_openai_models());
    }

    #[tokio::test]
    async fn list_models_returns_curated_openai_models() {
        let client = OpenAiClient::new("key".into());
        let models = client.list_models().await.unwrap();
        assert!(!models.is_empty());

        // GPT-5 family present with reasoning.
        let gpt5 = models.iter().find(|m| m.id == "gpt-5").unwrap();
        assert!(gpt5.capabilities.reasoning);
        assert!(gpt5.capabilities.tools);
        assert!(!gpt5.capabilities.embedding);
        assert_eq!(gpt5.context_limit, Some(400_000));

        // o-series reasoning flag.
        let o3 = models.iter().find(|m| m.id == "o3").unwrap();
        assert!(o3.capabilities.reasoning);

        // Embedding model has embedding flag, no chat flags.
        let embed = models
            .iter()
            .find(|m| m.id == "text-embedding-3-large")
            .unwrap();
        assert!(embed.capabilities.embedding);
        assert!(!embed.capabilities.reasoning);
        assert!(!embed.capabilities.tools);
        assert!(!embed.capabilities.vision);

        // Non-reasoning chat model example.
        let gpt4o = models.iter().find(|m| m.id == "gpt-4o").unwrap();
        assert!(!gpt4o.capabilities.reasoning);
        assert!(gpt4o.capabilities.tools);
    }

    // --- MODEL_OVERRIDE wiring (issue #34) -------------------------------

    /// Minimal SSE body that lets `send_and_stream` reach `break` on
    /// `response.completed` — empty `response` object is enough.
    const STUB_SSE_BODY: &str = "event: response.completed\ndata: {\"response\":{}}\n\n";

    #[tokio::test]
    async fn stream_completion_uses_self_model_when_override_unset() {
        let server = httpmock::MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/responses")
                .body_includes(r#""model":"gpt-5.4""#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let _ = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await;
        m.assert_calls(1);
    }

    #[tokio::test]
    async fn stream_completion_uses_model_override_when_set() {
        use desktop_assistant_core::ports::llm::with_model_override;

        let server = httpmock::MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/responses")
                .body_includes(r#""model":"gpt-4o-mini""#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        with_model_override("gpt-4o-mini".into(), async {
            let _ = client
                .stream_completion(
                    vec![Message::new(Role::User, "hi")],
                    &[],
                    ReasoningConfig::default(),
                    Box::new(|_| true),
                )
                .await;
        })
        .await;
        m.assert_calls(1);
    }

    // --- context_limit_for_model tests ---

    #[test]
    fn context_limit_for_known_models() {
        assert_eq!(context_limit_for_model("gpt-4o"), Some(128_000));
        assert_eq!(context_limit_for_model("gpt-4o-mini"), Some(128_000));
        assert_eq!(
            context_limit_for_model("gpt-4-turbo-2024-04-09"),
            Some(128_000)
        );
        assert_eq!(context_limit_for_model("gpt-4-32k-0613"), Some(32_768));
        assert_eq!(context_limit_for_model("gpt-4-0613"), Some(8_192));
        assert_eq!(
            context_limit_for_model("gpt-3.5-turbo-16k-0613"),
            Some(16_384)
        );
        assert_eq!(context_limit_for_model("gpt-3.5-turbo"), Some(4_096));
        assert_eq!(context_limit_for_model("o1-preview"), Some(200_000));
        assert_eq!(context_limit_for_model("o3-mini"), Some(200_000));
        assert_eq!(context_limit_for_model("o4-mini"), Some(200_000));
    }

    #[test]
    fn context_limit_specific_prefix_wins_over_general() {
        // gpt-4o is checked before the bare gpt-4 fallback, so we get
        // 128k rather than 8k.
        assert_eq!(context_limit_for_model("gpt-4o-2024-08-06"), Some(128_000));
        // Likewise gpt-4-turbo is 128k, not 8k.
        assert_eq!(context_limit_for_model("gpt-4-turbo"), Some(128_000));
        // Likewise gpt-4-32k is 32k, not 8k.
        assert_eq!(context_limit_for_model("gpt-4-32k"), Some(32_768));
    }

    #[test]
    fn context_limit_for_gpt5_returns_none() {
        // GPT-5 family is intentionally excluded — the daemon's universal
        // fallback handles it until we curate per-tier values.
        assert_eq!(context_limit_for_model("gpt-5"), None);
        assert_eq!(context_limit_for_model("gpt-5-mini"), None);
        assert_eq!(context_limit_for_model("gpt-5.4"), None);
    }

    #[test]
    fn context_limit_for_unknown_returns_none() {
        assert_eq!(context_limit_for_model("davinci"), None);
        assert_eq!(context_limit_for_model("text-embedding-3-large"), None);
        assert_eq!(context_limit_for_model("totally-fake-model"), None);
    }

    #[test]
    fn max_context_tokens_uses_configured_model() {
        let client = OpenAiClient::new("k".into()).with_model("gpt-4o");
        assert_eq!(client.max_context_tokens(), Some(128_000));
    }

    #[tokio::test]
    async fn max_context_tokens_consults_model_override() {
        use desktop_assistant_core::ports::llm::with_model_override;

        let client = OpenAiClient::new("k".into()).with_model("totally-fake-model");
        assert_eq!(client.max_context_tokens(), None);
        let observed =
            with_model_override("gpt-4o-mini".into(), async { client.max_context_tokens() }).await;
        assert_eq!(observed, Some(128_000));
    }

    // --- Context-overflow detection (issue #59) --------------------------

    #[test]
    fn detect_openai_context_overflow_extracts_token_counts() {
        // Numbers appear as (max, prompt) in OpenAI's wording — note the
        // helper swaps them so the returned tuple is (prompt, max).
        let body = r#"{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 153827 tokens. Please reduce the length of the messages."}}"#;
        let (prompt, max) =
            detect_openai_context_overflow(body).expect("should detect context overflow");
        assert_eq!(prompt, Some(153_827));
        assert_eq!(max, Some(128_000));
    }

    #[test]
    fn detect_openai_context_overflow_returns_none_for_other_codes() {
        // Different code → not a context overflow.
        let body = r#"{"error":{"code":"invalid_api_key","type":"invalid_request_error","message":"Invalid API key"}}"#;
        assert!(detect_openai_context_overflow(body).is_none());

        // No code field at all.
        let body = r#"{"error":{"type":"server_error","message":"upstream timeout"}}"#;
        assert!(detect_openai_context_overflow(body).is_none());

        // Unrelated 400 with a different validation issue.
        let body = r#"{"error":{"code":"missing_required_parameter","type":"invalid_request_error","message":"missing 'model'"}}"#;
        assert!(detect_openai_context_overflow(body).is_none());

        // Non-JSON garbage.
        assert!(detect_openai_context_overflow("HTTP 502 bad gateway").is_none());
    }

    #[test]
    fn detect_openai_context_overflow_tolerates_missing_numbers() {
        // The code triggers detection even if the message lacks numbers.
        let body = r#"{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"context length exceeded"}}"#;
        let (prompt, max) =
            detect_openai_context_overflow(body).expect("should still trigger overflow");
        assert_eq!(prompt, None);
        assert_eq!(max, None);
    }

    #[tokio::test]
    async fn http_400_context_length_exceeded_emits_context_overflow() {
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(400)
                .header("content-type", "application/json")
                .body(
                    r#"{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 153827 tokens. Please reduce the length of the messages."}}"#,
                );
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("400 context_length_exceeded must fail");

        match err {
            CoreError::ContextOverflow {
                prompt_tokens,
                max_tokens,
                detail,
            } => {
                assert_eq!(prompt_tokens, Some(153_827));
                assert_eq!(max_tokens, Some(128_000));
                assert!(detail.contains("context_length_exceeded"));
            }
            other => panic!("expected ContextOverflow, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_400_other_error_remains_generic_llm() {
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(400)
                .header("content-type", "application/json")
                .body(
                    r#"{"error":{"code":"missing_required_parameter","type":"invalid_request_error","message":"missing model"}}"#,
                );
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("400 must fail");

        assert!(
            matches!(err, CoreError::Llm(_)),
            "non-overflow 400 should stay generic; got {err:?}"
        );
    }

    #[tokio::test]
    async fn http_429_rate_limit_emits_rate_limited() {
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(429)
                .header("content-type", "application/json")
                .header("retry-after", "20")
                .body(
                    r#"{"error":{"code":"rate_limit_exceeded","type":"rate_limit_error","message":"Rate limit reached"}}"#,
                );
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("429 must fail");

        match err {
            CoreError::RateLimited {
                retry_after,
                detail,
            } => {
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(20)));
                assert!(detail.contains("rate_limit_exceeded"));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_429_insufficient_quota_emits_quota_exceeded() {
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(429)
                .header("content-type", "application/json")
                .body(
                    r#"{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"You exceeded your current quota, please check your plan and billing details."}}"#,
                );
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("429 must fail");

        match err {
            CoreError::QuotaExceeded { detail } => {
                assert!(detail.contains("insufficient_quota"));
            }
            other => panic!("expected QuotaExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_503_service_unavailable_emits_rate_limited() {
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(503)
                .header("content-type", "application/json")
                .body(r#"{"error":{"message":"Service overloaded"}}"#);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("503 must fail");

        assert!(
            matches!(err, CoreError::RateLimited { .. }),
            "503 should map to RateLimited; got {err:?}"
        );
    }

    #[test]
    fn detect_openai_insufficient_quota_matches_envelope() {
        let body = r#"{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"You exceeded your current quota"}}"#;
        assert!(detect_openai_insufficient_quota(body));
    }

    #[test]
    fn detect_openai_insufficient_quota_rejects_other_errors() {
        let body = r#"{"error":{"code":"rate_limit_exceeded","type":"rate_limit_error","message":"slow down"}}"#;
        assert!(!detect_openai_insufficient_quota(body));
    }

    #[test]
    fn detect_openai_insufficient_quota_rejects_garbage() {
        assert!(!detect_openai_insufficient_quota("not json at all"));
    }

    // `parse_retry_after_header` now lives in `desktop-assistant-llm-http`
    // and is tested there (#302).

    // --- Cancellation (issue #109) ---------------------------------------

    /// Drive the streaming entrypoint against a local stub that holds the
    /// connection open for several seconds. Cancellation must surface
    /// `CoreError::Cancelled` and unblock well before the stub's delay
    /// completes — that's the "stream is dropped" assertion.
    #[tokio::test]
    async fn openai_stream_aborts_on_cancellation() {
        use desktop_assistant_core::ports::llm::with_cancellation_token;
        use std::time::Duration;
        use tokio_util::sync::CancellationToken;

        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(200)
                .header("content-type", "text/event-stream")
                .delay(Duration::from_secs(5))
                .body(STUB_SSE_BODY);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let token = CancellationToken::new();
        let cancel_handle = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_handle.cancel();
        });

        let start = std::time::Instant::now();
        let result = with_cancellation_token(token, async {
            client
                .stream_completion(
                    vec![Message::new(Role::User, "hi")],
                    &[],
                    ReasoningConfig::default(),
                    Box::new(|_| true),
                )
                .await
        })
        .await;
        let elapsed = start.elapsed();

        assert!(
            matches!(result, Err(CoreError::Cancelled)),
            "expected Cancelled, got {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "stream should abort promptly on cancellation; took {elapsed:?}"
        );
    }

    // --- Timeouts configurable (#561) ------------------------------------

    #[test]
    fn openai_timeouts_configurable() {
        use std::time::Duration;
        let c = OpenAiClient::new("k".into())
            .with_connect_timeout(Some(5))
            .with_event_timeout(Some(7))
            .with_request_timeout(Some(9));
        assert_eq!(c.connect_timeout, Duration::from_secs(5));
        assert_eq!(c.event_timeout, Duration::from_secs(7));
        assert_eq!(c.request_timeout, Duration::from_secs(9));

        // `None` and `Some(0)` are both "keep the default" — a zero timeout
        // would mean "fail instantly", which is never what a caller wants.
        let d = OpenAiClient::new("k".into())
            .with_connect_timeout(Some(0))
            .with_event_timeout(None)
            .with_request_timeout(Some(0));
        assert_eq!(d.connect_timeout, OPENAI_CONNECT_TIMEOUT);
        assert_eq!(d.event_timeout, OPENAI_EVENT_TIMEOUT);
        assert_eq!(d.request_timeout, OPENAI_REQUEST_TIMEOUT);
    }

    #[tokio::test]
    async fn openai_connect_timeout_configurable() {
        use std::time::Duration;
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(200)
                .header("content-type", "text/event-stream")
                .delay(Duration::from_secs(3))
                .body(STUB_SSE_BODY);
        });

        let mut client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        // Reach past the public builder (min 1s) to keep the test fast; the
        // production path uses `with_connect_timeout`.
        client.connect_timeout = Duration::from_millis(50);

        let start = std::time::Instant::now();
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("a stalled connect must fail, not hang");
        let elapsed = start.elapsed();

        assert!(
            matches!(err, CoreError::Llm(ref d) if d.contains("stalled")),
            "expected a stall error, got {err:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "connect timeout should fire near its configured value; took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn openai_embed_times_out_when_backend_stalls() {
        use std::time::Duration;
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .header("content-type", "application/json")
                .delay(Duration::from_secs(3))
                .body(r#"{"data":[{"embedding":[0.1,0.2]}]}"#);
        });

        let mut client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        client.request_timeout = Duration::from_millis(50);

        let start = std::time::Instant::now();
        let err = client
            .embed(vec!["hello".into()])
            .await
            .expect_err("a wedged embeddings backend must time out, not hang");
        let elapsed = start.elapsed();

        assert!(
            matches!(err, CoreError::Llm(ref d) if d.contains("timed out")),
            "expected a timeout error, got {err:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "embed timeout should fire near its configured value; took {elapsed:?}"
        );
    }

    // --- Streaming truncation (#561) -------------------------------------

    #[tokio::test]
    async fn openai_stream_truncation_surfaces_error() {
        // A 200 response whose SSE body ends *without* `response.completed`
        // is a dropped/truncated stream. Returning the partial text as a
        // successful completion would silently feed a half-answer downstream,
        // so it must surface as a clear error instead.
        let truncated = "event: response.output_text.delta\n\
                         data: {\"delta\":\"partial ans\"}\n\n";
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(truncated);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("a truncated stream must fail");

        assert!(
            matches!(err, CoreError::Llm(ref d) if d.contains("truncat")),
            "expected a truncation error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn openai_stream_completed_is_not_truncation() {
        // The happy path (body carries `response.completed`) must still
        // succeed — the truncation guard must not fire on a clean close.
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let resp = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("a completed stream must succeed");
        assert!(resp.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn openai_stream_callback_abort_is_not_truncation() {
        // When the caller's chunk callback returns false it is an intentional
        // client-side stop, not a dropped stream — the partial response is
        // returned as success, never re-labelled as truncation.
        let body = "event: response.output_text.delta\n\
                    data: {\"delta\":\"first\"}\n\n\
                    event: response.output_text.delta\n\
                    data: {\"delta\":\"second\"}\n\n";
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(body);
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let resp = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                // Abort after the very first chunk.
                Box::new(|_| false),
            )
            .await
            .expect("callback abort must return the partial response, not error");
        assert_eq!(resp.text, "first");
    }

    // --- Transient-failure classification / retries (#561) ---------------

    #[tokio::test]
    async fn openai_http_5xx_maps_to_rate_limited() {
        // 500 / 502 / 504 are transient server-side failures. They arrive
        // before any stream byte is consumed, so retrying is idempotency-safe;
        // mapping them to `RateLimited` is what makes the core retry decorator
        // back off and retry rather than hard-failing the turn.
        for status in [500u16, 502, 504] {
            let server = httpmock::MockServer::start();
            let _m = server.mock(|when, then| {
                when.method(httpmock::Method::POST).path("/responses");
                then.status(status)
                    .header("content-type", "application/json")
                    .body(r#"{"error":{"message":"internal error"}}"#);
            });

            let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
            let err = client
                .stream_completion(
                    vec![Message::new(Role::User, "hi")],
                    &[],
                    ReasoningConfig::default(),
                    Box::new(|_| true),
                )
                .await
                .unwrap_err();

            assert!(
                matches!(err, CoreError::RateLimited { .. }),
                "HTTP {status} should map to RateLimited (retryable); got {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn openai_http_404_model_not_found_is_terminal() {
        // An unknown/unavailable model id is a permanent condition — it must
        // NOT be swept into the retryable 5xx bucket, or the daemon would
        // burn its whole retry budget on a request that can never succeed.
        let server = httpmock::MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/responses");
            then.status(404)
                .header("content-type", "application/json")
                .body(
                    r#"{"error":{"code":"model_not_found","type":"invalid_request_error","message":"The model 'gpt-nope' does not exist"}}"#,
                );
        });

        let client = OpenAiClient::new("key".into()).with_base_url(server.url(""));
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .unwrap_err();

        match err {
            CoreError::Llm(detail) => {
                assert!(
                    detail.contains("model_not_found"),
                    "detail should carry the reason: {detail}"
                );
            }
            other => panic!("404 model-not-found should stay terminal Llm; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn openai_connection_error_is_retryable() {
        // A connection that never establishes (nothing listening) is a
        // transport-phase failure — no request reached the server and nothing
        // streamed, so it is safe to retry. It must map to `RateLimited`, not
        // a terminal `Llm`, to match the AWS-SDK transport-retry bar Bedrock
        // gets for free.
        let client = OpenAiClient::new("key".into()).with_base_url("http://127.0.0.1:1");
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("connection refused must fail");

        assert!(
            matches!(err, CoreError::RateLimited { .. }),
            "a connection error should be retryable; got {err:?}"
        );
    }

    // --- API-key redaction (#561) ----------------------------------------

    #[test]
    fn openai_api_key_redacted_in_display() {
        let secret = "sk-supersecret-ABC123XYZ";
        let client = OpenAiClient::new(secret.into());
        let rendered = format!("{client:?}");
        assert!(
            !rendered.contains(secret),
            "raw API key leaked through Debug: {rendered}"
        );
        assert!(
            rendered.contains("redacted"),
            "expected a redaction marker in Debug output: {rendered}"
        );
    }

    #[test]
    fn api_key_fingerprint_hides_key_but_is_stable() {
        let a = api_key_fingerprint("sk-abc");
        let b = api_key_fingerprint("sk-abc");
        let c = api_key_fingerprint("sk-different");
        assert!(!a.contains("sk-abc"), "fingerprint leaked the key: {a}");
        assert_eq!(a, b, "fingerprint must be deterministic");
        assert_ne!(a, c, "different keys must fingerprint differently");
    }
}
