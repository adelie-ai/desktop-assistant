//! OpenRouter connector implementing the core [`LlmClient`] port.
//!
//! OpenRouter is an OpenAI-compatible aggregator that routes one API to many
//! model vendors (`anthropic/*`, `openai/*`, `google/*`, `meta-llama/*`, ...).
//! This crate owns only what is OpenRouter-specific: the request envelope, the
//! auth + attribution headers, the base-URL / endpoint shaping, the
//! `cache_control` decision, model addressing, the live `/models` catalog
//! fetch, and the error classifier. Everything shared with other Chat
//! Completions consumers -- message/tool conversion, the tool-schema and
//! empty-key sanitizers, SSE `choices[].delta` parsing, usage parsing, and the
//! base error mapping -- comes from
//! [`desktop_assistant_llm_openai_compat`], and the connect race / stall
//! constants / model-merge helpers come from `desktop_assistant_llm_http`.

use std::time::Duration;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelCapabilities, ModelInfo, ReasoningConfig,
    current_cancellation_token, current_model_override,
};
use desktop_assistant_llm_http::{
    STREAM_CONNECT_TIMEOUT, STREAM_EVENT_TIMEOUT, apply_context_cap, bail_for_status,
    merge_curated_with_live,
};
use desktop_assistant_llm_openai_compat::{
    ChatMessage, ChatTool, classify_error, consume_chat_stream, mark_system_cache_breakpoint,
    to_chat_messages, to_chat_tools,
};
use reqwest::Client;
use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};

/// Connection-handshake / per-event stall budgets shared with the other
/// connectors. Kept as local aliases so the streaming loop reads naturally.
const OPENROUTER_CONNECT_TIMEOUT: Duration = STREAM_CONNECT_TIMEOUT;
const OPENROUTER_EVENT_TIMEOUT: Duration = STREAM_EVENT_TIMEOUT;

/// Fixed, public attribution identifiers OpenRouter surfaces on the account's
/// activity page (via `HTTP-Referer` / `X-Title`). These are a constant Adele
/// project identity, deliberately NOT derived from user, session, or internal
/// data, so they can never leak private context to the aggregator.
const ADELE_REFERER: &str = "https://github.com/adelie-ai/desktop-assistant";
const ADELE_TITLE: &str = "Adele";

/// Upper bound on an attribution header value. A generous cap that the fixed
/// constants sit well under; it exists only so a future bad edit cannot ship an
/// unbounded header.
const MAX_ATTRIBUTION_LEN: usize = 256;

/// True when `value` is safe to send as an HTTP header value: non-empty, within
/// [`MAX_ATTRIBUTION_LEN`], and free of ASCII control characters (notably CR/LF,
/// which would enable header splitting). Guards the fixed attribution constants
/// as defense in depth -- an invalid value is dropped rather than sent.
fn is_valid_attribution(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_ATTRIBUTION_LEN
        && !value.chars().any(|c| c.is_control())
}

/// OpenRouter LLM client that streams completions via the OpenAI-compatible
/// Chat Completions endpoint (`{base_url}/chat/completions`).
pub struct OpenRouterClient {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    /// Stored per the builder contract; OpenRouter's routed API does not expose
    /// hosted tool search uniformly, so [`Self::supports_hosted_tool_search`]
    /// returns `false` in v1 regardless of this flag.
    hosted_tool_search: bool,
    /// First-response (connect) stall budget; defaults to
    /// [`OPENROUTER_CONNECT_TIMEOUT`], overridable per-connection.
    connect_timeout: Duration,
    /// Per-chunk stall budget; defaults to [`OPENROUTER_EVENT_TIMEOUT`].
    event_timeout: Duration,
    /// Per-connection context-window hard cap, in tokens. `None` = "max
    /// available". Folded with the curated table in `max_context_tokens`.
    context_cap: Option<u64>,
}

/// Redacting `Debug` so the API key can never leak through a `{:?}` render --
/// e.g. if a caller embeds the client in a `#[derive(Debug)]` struct or an
/// error context. The key field renders as `<redacted; len=N>`, exposing only
/// its length (per the uniform cloud-connector contract).
impl std::fmt::Debug for OpenRouterClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterClient")
            .field(
                "api_key",
                &format_args!("<redacted; len={}>", self.api_key.len()),
            )
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("max_tokens", &self.max_tokens)
            .field("hosted_tool_search", &self.hosted_tool_search)
            .field("connect_timeout", &self.connect_timeout)
            .field("event_timeout", &self.event_timeout)
            .field("context_cap", &self.context_cap)
            .finish()
    }
}

impl OpenRouterClient {
    /// The connector's built-in default model.
    ///
    /// Verify the exact `vendor/model` slug against the live `/models` catalog
    /// at ship time -- slugs are version-fragile; a stale default is recoverable
    /// via the model picker.
    pub fn get_default_model() -> Option<&'static str> {
        Some("anthropic/claude-sonnet-4-6")
    }

    /// The connector's built-in default base URL.
    pub fn get_default_base_url() -> Option<&'static str> {
        Some("https://openrouter.ai/api/v1")
    }

    /// Construct a client with the default model and base URL.
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
            connect_timeout: OPENROUTER_CONNECT_TIMEOUT,
            event_timeout: OPENROUTER_EVENT_TIMEOUT,
            context_cap: None,
        }
    }

    /// Set the logical `vendor/model` id sent in the request body.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the API base URL (e.g. to point tests at a mock server).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the sampling temperature. Omitted from the request body when `None`.
    pub fn with_temperature(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set nucleus-sampling `top_p`. Omitted from the request body when `None`.
    pub fn with_top_p(mut self, top_p: Option<f64>) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set the completion token cap. Omitted from the request body when `None`.
    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override the first-response (connect) stall budget. `None`/`Some(0)`
    /// keeps the [`OPENROUTER_CONNECT_TIMEOUT`] default. Seconds.
    pub fn with_connect_timeout(mut self, secs: Option<u64>) -> Self {
        if let Some(s) = secs.filter(|s| *s > 0) {
            self.connect_timeout = Duration::from_secs(s);
        }
        self
    }

    /// Override the per-chunk stall budget. `None`/`Some(0)` keeps the
    /// [`OPENROUTER_EVENT_TIMEOUT`] default. Seconds.
    pub fn with_event_timeout(mut self, secs: Option<u64>) -> Self {
        if let Some(s) = secs.filter(|s| *s > 0) {
            self.event_timeout = Duration::from_secs(s);
        }
        self
    }

    /// Set the per-connection context-window hard cap, in tokens. `None`/
    /// `Some(0)` = "max available". Clamps the daemon's input budget (no
    /// `num_ctx` to pin -- the API enforces its own window), useful for bounding
    /// spend. See [`desktop_assistant_llm_http::apply_context_cap`].
    pub fn with_max_context_tokens(mut self, max: Option<u64>) -> Self {
        self.context_cap = max.filter(|m| *m > 0);
        self
    }

    /// Record the hosted-tool-search preference from the builder. Stored for
    /// forward compatibility only; [`Self::supports_hosted_tool_search`] returns
    /// `false` in v1, so this never enables the namespace path.
    pub fn with_hosted_tool_search(mut self, enabled: bool) -> Self {
        self.hosted_tool_search = enabled;
        self
    }

    /// Build a client from the environment: `OPENROUTER_API_KEY` (required),
    /// `OPENROUTER_MODEL` and `OPENROUTER_BASE_URL` (both optional).
    pub fn from_env() -> Result<Self, CoreError> {
        let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
            CoreError::Llm("OPENROUTER_API_KEY environment variable not set".into())
        })?;
        let mut client = Self::new(api_key);
        if let Ok(model) = std::env::var("OPENROUTER_MODEL") {
            client.model = model;
        }
        if let Ok(url) = std::env::var("OPENROUTER_BASE_URL") {
            client.base_url = url;
        }
        Ok(client)
    }
}

// ---------------------------------------------------------------------------
// Request envelope (OpenRouter-specific; the wire shapes it wraps are shared)
// ---------------------------------------------------------------------------

/// The Chat Completions request body OpenRouter accepts. The `messages` and
/// `tools` arrays use the shared compat wire types; the top-level shaping
/// (`stream`, `reasoning`, `usage:{include:true}`) is OpenRouter-owned.
#[derive(Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<ChatMessage>,
    /// Omitted entirely when there are no tools (an empty `tools` array trips
    /// some routed backends).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    /// Unified reasoning control; omitted when no reasoning was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningBlock>,
    /// OpenRouter's documented streaming-usage mechanism (`usage:{include:true}`),
    /// NOT the OpenAI `stream_options` form -- so the final chunk carries token
    /// and cache-activity totals.
    usage: UsageAccounting,
}

/// OpenRouter's unified `reasoning` object. `effort` and `max_tokens` are
/// mutually informative -- OpenRouter normalizes whichever is present per routed
/// model. Both are optional; the block is omitted entirely when empty.
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Default)]
struct ReasoningBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

/// The `usage` request object: `{ include: true }` opts into streamed usage
/// accounting on the terminating chunk.
#[derive(Serialize)]
struct UsageAccounting {
    include: bool,
}

/// Build the `reasoning` block from the per-turn [`ReasoningConfig`].
///
/// Maps `reasoning_effort` -> `effort` and `thinking_budget_tokens` ->
/// `max_tokens`, and returns `None` (omit the field) when neither is set. A
/// `thinking_budget_tokens` of `0` is treated as "no budget". Unlike the OpenAI
/// connector there is no per-model reasoning gate here: OpenRouter normalizes
/// (or ignores) the field per routed model, so the connector emits whatever the
/// daemon's OpenRouter arm resolved.
fn reasoning_block_for(_reasoning: ReasoningConfig) -> Option<ReasoningBlock> {
    // TODO(impl): map reasoning_effort -> effort, thinking_budget_tokens ->
    // max_tokens; omit when empty. Stubbed to establish the failing spec.
    None
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

impl OpenRouterClient {
    /// POST the request and stream the SSE response into an [`LlmResponse`].
    ///
    /// The connect handshake is raced against cancellation and
    /// [`Self::connect_timeout`] so a stalled connect fails the turn instead of
    /// hanging. On a non-2xx status the body is classified via
    /// [`classify_openrouter_error`]; on success the raw byte stream is handed to
    /// the shared [`consume_chat_stream`], which owns all SSE parsing.
    async fn send_and_stream(
        &self,
        model: &str,
        request_body: &ChatCompletionsRequest,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let request_json =
            serde_json::to_string(request_body).unwrap_or_else(|_| "<serialization error>".into());
        tracing::info!(
            request_bytes = request_json.len(),
            model = %model,
            "OpenRouter request payload"
        );
        tracing::debug!(
            "OpenRouter request body (first 2000 chars): {}",
            &request_json[..request_json.len().min(2000)]
        );

        // Cooperative cancellation (issue #109): bail before dialing out, then
        // race the connect against the token and the connect timeout.
        let cancellation = current_cancellation_token().unwrap_or_default();
        if cancellation.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let mut builder = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");
        // Fixed public attribution; validated (defense in depth) and dropped if
        // ever made invalid, never user/session-derived.
        if is_valid_attribution(ADELE_REFERER) {
            builder = builder.header("HTTP-Referer", ADELE_REFERER);
        }
        if is_valid_attribution(ADELE_TITLE) {
            builder = builder.header("X-Title", ADELE_TITLE);
        }
        let send_fut = builder.json(request_body).send();

        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err(CoreError::Cancelled),
            _ = tokio::time::sleep(self.connect_timeout) => {
                tracing::error!(
                    timeout_s = self.connect_timeout.as_secs(),
                    "OpenRouter request send() timed out (no response headers)"
                );
                return Err(CoreError::Llm("OpenRouter stream stalled".into()));
            }
            r = send_fut => r.map_err(|e| CoreError::Llm(format!("HTTP request failed: {e}")))?,
        };

        if !response.status().is_success() {
            let status = response.status();
            let headers = response.headers().clone();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(classify_openrouter_error(status, &headers, &body));
        }

        // NOTE (follow-up): a routed backend that rejects tool use while
        // streaming should be classified to `CoreError::ToolsUnsupported`, with
        // a non-streaming retry + per-model memo (the Bedrock-proven pattern).
        // v1 relies on the base classifier's generic `Llm` mapping for that
        // case; the streaming-only path here does not yet fall back.
        consume_chat_stream(
            response.bytes_stream(),
            &cancellation,
            self.event_timeout,
            on_chunk,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/// Classify an OpenRouter HTTP error into a [`CoreError`].
///
/// OpenRouter-specific arms run first, then the shared base classifier:
///
/// 1. HTTP 402 (Payment Required) or an out-of-credits body ->
///    [`CoreError::QuotaExceeded`] (permanent billing; not retried). The base
///    classifier does not know 402, so this must precede it.
/// 2. everything else -> [`classify_error`], which handles context overflow,
///    `insufficient_quota`, 429 (with `Retry-After`), and 5xx.
fn classify_openrouter_error(status: StatusCode, headers: &HeaderMap, body: &str) -> CoreError {
    // TODO(impl): add the HTTP 402 arm before delegating. Stubbed to establish
    // the failing spec; only the credits-body detection is wired here.
    if detect_openrouter_insufficient_credits(body) {
        return CoreError::QuotaExceeded {
            detail: format!("OpenRouter API error (HTTP {status}): {body}"),
        };
    }
    classify_error(status, headers, body)
}

/// Detect OpenRouter's exhausted-credits signal in an HTTP error body.
///
/// OpenRouter reports depleted credits with HTTP 402 and/or a body whose
/// message mentions credits (e.g. "insufficient credits", "requires more
/// credits"). This is the sanctioned connector-boundary string match (see the
/// base detectors in the compat crate): a plain case-insensitive substring scan
/// so it is robust to OpenRouter's numeric `error.code` (which would break a
/// string-typed envelope parse) and to non-JSON bodies from an upstream proxy.
fn detect_openrouter_insufficient_credits(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("insufficient credit")
        || lower.contains("insufficient_credits")
        || lower.contains("requires more credit")
        || lower.contains("more credits")
        || lower.contains("negative credit")
}

// ---------------------------------------------------------------------------
// Model catalog: curated table merged with the live `/models` endpoint
// ---------------------------------------------------------------------------

/// A small curated set of common `vendor/model` ids with capability flags.
///
/// The live `/models` endpoint fills the long tail (merged via
/// [`merge_curated_with_live`], curated metadata winning on overlap). Slugs are
/// version-fragile; treat this as a convenience seed, not an authority.
fn curated_openrouter_models() -> Vec<ModelInfo> {
    // TODO(impl): seed a small curated vendor/model table with capability flags
    // and context windows. Stubbed empty to establish the failing spec.
    Vec::new()
}

/// The curated prompt-token window for a `vendor/model` id, if known. Exact-id
/// lookup against [`curated_openrouter_models`]; the live catalog is not
/// consulted here (that path is async and cached by the picker).
fn curated_context_limit(model: &str) -> Option<u64> {
    curated_openrouter_models()
        .into_iter()
        .find(|m| m.id == model)
        .and_then(|m| m.context_limit)
}

/// One entry of the OpenRouter `/models` response `data` array (only the fields
/// this connector reads).
#[derive(Deserialize)]
struct LiveModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    architecture: Option<Architecture>,
    #[serde(default)]
    supported_parameters: Vec<String>,
}

/// The `architecture` sub-object of a [`LiveModel`]; used only to infer vision
/// support from the input modalities.
#[derive(Deserialize, Default)]
struct Architecture {
    #[serde(default)]
    input_modalities: Vec<String>,
}

/// The top-level `/models` response envelope.
#[derive(Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<LiveModel>,
}

/// Convert a live `/models` entry into a [`ModelInfo`], deriving capability
/// flags from `supported_parameters` (tools / reasoning) and `architecture`
/// (image input -> vision). Embeddings are never flagged: OpenRouter's
/// embedding coverage is excluded from this connector in v1.
fn live_model_to_info(m: LiveModel) -> ModelInfo {
    let caps = ModelCapabilities {
        reasoning: m
            .supported_parameters
            .iter()
            .any(|p| p == "reasoning" || p == "include_reasoning"),
        vision: m
            .architecture
            .as_ref()
            .is_some_and(|a| a.input_modalities.iter().any(|x| x == "image")),
        tools: m
            .supported_parameters
            .iter()
            .any(|p| p == "tools" || p == "tool_choice"),
        embedding: false,
    };
    let mut info = ModelInfo::new(m.id);
    if let Some(name) = m.name {
        info = info.with_display_name(name);
    }
    if let Some(ctx) = m.context_length {
        info = info.with_context_limit(ctx);
    }
    info.with_capabilities(caps)
}

impl OpenRouterClient {
    /// Fetch and parse the live `{base_url}/models` catalog. Bubbles up a
    /// [`CoreError`] on transport / non-2xx / parse failure so the caller can
    /// degrade to the curated table.
    async fn fetch_live_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("OpenRouter models HTTP request failed: {e}")))?;
        let response = bail_for_status(response, "OpenRouter models API error").await?;
        let parsed: ModelsResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse OpenRouter models: {e}")))?;
        Ok(parsed.data.into_iter().map(live_model_to_info).collect())
    }
}

#[async_trait::async_trait]
impl LlmClient for OpenRouterClient {
    fn get_default_model(&self) -> Option<&str> {
        Self::get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        Self::get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        // Honour a per-turn model override so the budget tracks the dispatched
        // model, not the connection's baked-in default.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());
        apply_context_cap(self.context_cap, curated_context_limit(&model))
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        // Per-turn model override (issue #34): dispatch the user-chosen model
        // instead of the connector's baked-in default when set.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());

        // Shared wire conversion (tool-schema + empty-key sanitization applied
        // inside these helpers), then mark the system block for caching so
        // OpenRouter can normalize a breakpoint per routed provider.
        let mut chat_messages = to_chat_messages(&messages);
        mark_system_cache_breakpoint(&mut chat_messages);
        let chat_tools = to_chat_tools(tools);

        let request = ChatCompletionsRequest {
            model: model.clone(),
            messages: chat_messages,
            tools: chat_tools,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stream: true,
            reasoning: reasoning_block_for(reasoning),
            usage: UsageAccounting { include: true },
        };

        self.send_and_stream(&model, &request, on_chunk).await
    }

    fn supports_hosted_tool_search(&self) -> bool {
        // Off in v1 -- OpenRouter's routed API does not expose hosted tool
        // search uniformly. Namespaces flatten into the standard `tools` array
        // via the trait's default `stream_completion_with_namespaces`.
        false
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // Always degrade to the curated table rather than surfacing an error to
        // the picker when the live listing is unavailable.
        let live = self.fetch_live_models().await.unwrap_or_else(|e| {
            tracing::warn!("OpenRouter live /models fetch failed, using curated table only: {e}");
            Vec::new()
        });
        Ok(merge_curated_with_live(curated_openrouter_models(), live))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::Role;
    use desktop_assistant_core::ports::llm::{
        ReasoningLevel, with_cancellation_token, with_model_override,
    };
    use httpmock::Method::{GET, POST};
    use httpmock::MockServer;
    use std::sync::{Arc, Mutex};

    /// Minimal happy-path SSE body: one text delta then the terminator.
    const STUB_SSE_BODY: &str =
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n";

    fn client_for(server: &MockServer) -> OpenRouterClient {
        OpenRouterClient::new("test-key".into()).with_base_url(server.url(""))
    }

    // --- Builder / defaults ---------------------------------------------

    #[test]
    fn defaults_are_openrouter_shaped() {
        assert_eq!(
            OpenRouterClient::get_default_model(),
            Some("anthropic/claude-sonnet-4-6")
        );
        assert_eq!(
            OpenRouterClient::get_default_base_url(),
            Some("https://openrouter.ai/api/v1")
        );
        let client = OpenRouterClient::new("k".into());
        assert_eq!(client.model, "anthropic/claude-sonnet-4-6");
        assert_eq!(client.base_url, "https://openrouter.ai/api/v1");
    }

    #[test]
    fn builder_sets_fields() {
        let client = OpenRouterClient::new("k".into())
            .with_model("openai/gpt-4o")
            .with_base_url("http://localhost:9")
            .with_temperature(Some(0.5))
            .with_top_p(Some(0.9))
            .with_max_tokens(Some(1024))
            .with_hosted_tool_search(true);
        assert_eq!(client.model, "openai/gpt-4o");
        assert_eq!(client.base_url, "http://localhost:9");
        assert_eq!(client.temperature, Some(0.5));
        assert_eq!(client.top_p, Some(0.9));
        assert_eq!(client.max_tokens, Some(1024));
        // Stored, but the trait still reports no hosted tool search in v1.
        assert!(client.hosted_tool_search);
        assert!(!client.supports_hosted_tool_search());
    }

    #[test]
    fn timeout_and_cap_zero_is_treated_as_default() {
        let client = OpenRouterClient::new("k".into())
            .with_connect_timeout(Some(0))
            .with_event_timeout(Some(0))
            .with_max_context_tokens(Some(0));
        assert_eq!(client.connect_timeout, OPENROUTER_CONNECT_TIMEOUT);
        assert_eq!(client.event_timeout, OPENROUTER_EVENT_TIMEOUT);
        assert_eq!(client.context_cap, None);
    }

    #[test]
    fn from_env_missing_key() {
        // SAFETY: single-threaded test scope; this test owns the env var and no
        // other test in this binary touches `OPENROUTER_API_KEY`.
        unsafe { std::env::remove_var("OPENROUTER_API_KEY") };
        let result = OpenRouterClient::from_env();
        assert!(matches!(result, Err(CoreError::Llm(_))));
    }

    // --- Redacting Debug -------------------------------------------------

    #[test]
    fn api_key_is_redacted_in_debug() {
        let secret = "sk-or-supersecret-DO-NOT-LEAK-0123456789";
        let client = OpenRouterClient::new(secret.into()).with_model("openai/gpt-4o");
        let rendered = format!("{client:?}");
        assert!(
            !rendered.contains(secret),
            "raw API key leaked in Debug output: {rendered}"
        );
        assert!(
            !rendered.contains("supersecret"),
            "a substring of the API key leaked in Debug output: {rendered}"
        );
        assert!(
            rendered.contains("redacted"),
            "key field missing: {rendered}"
        );
        assert!(
            rendered.contains("openai/gpt-4o"),
            "model should still be visible: {rendered}"
        );
    }

    // --- Attribution header validation ----------------------------------

    #[test]
    fn attribution_constants_are_valid() {
        assert!(is_valid_attribution(ADELE_REFERER));
        assert!(is_valid_attribution(ADELE_TITLE));
    }

    #[test]
    fn attribution_rejects_control_chars_and_empty() {
        assert!(!is_valid_attribution(""));
        assert!(!is_valid_attribution("has\nnewline"));
        assert!(!is_valid_attribution("has\rcarriage"));
        assert!(!is_valid_attribution("tab\tinside"));
        assert!(!is_valid_attribution(&"x".repeat(MAX_ATTRIBUTION_LEN + 1)));
    }

    // --- Reasoning mapping ----------------------------------------------

    #[test]
    fn reasoning_block_maps_effort() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::High);
        let block = reasoning_block_for(cfg).expect("effort block");
        assert_eq!(block.effort, Some("high"));
        assert_eq!(block.max_tokens, None);
    }

    #[test]
    fn reasoning_block_maps_budget_to_max_tokens() {
        let cfg = ReasoningConfig::with_thinking_budget(4096);
        let block = reasoning_block_for(cfg).expect("budget block");
        assert_eq!(block.max_tokens, Some(4096));
        assert_eq!(block.effort, None);
    }

    #[test]
    fn reasoning_block_omitted_when_empty() {
        assert!(reasoning_block_for(ReasoningConfig::default()).is_none());
    }

    #[test]
    fn reasoning_block_omitted_when_budget_is_zero() {
        assert!(reasoning_block_for(ReasoningConfig::with_thinking_budget(0)).is_none());
    }

    #[test]
    fn reasoning_block_serializes_effort_only() {
        let block = ReasoningBlock {
            effort: Some("medium"),
            max_tokens: None,
        };
        let json = serde_json::to_value(&block).expect("serialize");
        assert_eq!(json["effort"], "medium");
        assert!(
            json.get("max_tokens").is_none(),
            "max_tokens must be omitted"
        );
    }

    // --- Request envelope serialization ---------------------------------

    fn build_request(
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
    ) -> ChatCompletionsRequest {
        let mut msgs = to_chat_messages(&[
            Message::new(Role::System, "sys prompt"),
            Message::new(Role::User, "hi"),
        ]);
        mark_system_cache_breakpoint(&mut msgs);
        ChatCompletionsRequest {
            model: "openai/gpt-4o".into(),
            messages: msgs,
            tools: to_chat_tools(tools),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: true,
            reasoning: reasoning_block_for(reasoning),
            usage: UsageAccounting { include: true },
        }
    }

    #[test]
    fn request_omits_tools_when_empty() {
        let req = build_request(&[], ReasoningConfig::default());
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(!json.contains("\"tools\""), "empty tools must be omitted");
    }

    #[test]
    fn request_includes_tools_when_present() {
        let tools = vec![ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type":"object"}),
        )];
        let req = build_request(&tools, ReasoningConfig::default());
        let json: serde_json::Value = serde_json::to_value(&req).expect("serialize");
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "read_file");
    }

    #[test]
    fn request_always_requests_streamed_usage() {
        let req = build_request(&[], ReasoningConfig::default());
        let json: serde_json::Value = serde_json::to_value(&req).expect("serialize");
        assert_eq!(json["stream"], true);
        assert_eq!(json["usage"]["include"], true);
    }

    #[test]
    fn request_marks_system_cache_breakpoint() {
        let req = build_request(&[], ReasoningConfig::default());
        let json: serde_json::Value = serde_json::to_value(&req).expect("serialize");
        // The system message content is the multi-part array with the marker.
        assert_eq!(
            json["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn request_includes_reasoning_effort_when_set() {
        let req = build_request(
            &[],
            ReasoningConfig::with_reasoning_effort(ReasoningLevel::Low),
        );
        let json: serde_json::Value = serde_json::to_value(&req).expect("serialize");
        assert_eq!(json["reasoning"]["effort"], "low");
    }

    #[test]
    fn request_omits_reasoning_when_empty() {
        let req = build_request(&[], ReasoningConfig::default());
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(
            !json.contains("reasoning"),
            "reasoning must be omitted: {json}"
        );
    }

    // --- MODEL_OVERRIDE routes the wire model ---------------------------

    #[tokio::test]
    async fn stream_uses_self_model_when_override_unset() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .body_includes(r#""model":"anthropic/claude-sonnet-4-6""#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let client = client_for(&server);
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
    async fn stream_uses_model_override_when_set() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .body_includes(r#""model":"google/gemini-2.5-pro""#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let client = client_for(&server);
        with_model_override("google/gemini-2.5-pro".into(), async {
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

    #[tokio::test]
    async fn request_body_carries_cache_and_usage_over_the_wire() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .body_includes(r#""cache_control""#)
                .body_includes(r#""ephemeral""#)
                .body_includes(r#""include":true"#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let client = client_for(&server);
        let _ = client
            .stream_completion(
                vec![
                    Message::new(Role::System, "you are helpful"),
                    Message::new(Role::User, "hi"),
                ],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await;
        m.assert_calls(1);
    }

    // --- Streaming happy path -------------------------------------------

    #[tokio::test]
    async fn stream_returns_text_and_invokes_callback() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(concat!(
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n",
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\", world\"}}]}\n\n",
                    "data: [DONE]\n\n",
                ));
        });
        let client = client_for(&server);
        let received = Arc::new(Mutex::new(String::new()));
        let received_cl = Arc::clone(&received);
        let result = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(move |c| {
                    received_cl.lock().expect("lock").push_str(&c);
                    true
                }),
            )
            .await
            .expect("stream ok");
        assert_eq!(result.text, "Hello, world");
        assert_eq!(*received.lock().expect("lock"), "Hello, world");
    }

    #[tokio::test]
    async fn stream_accumulates_tool_call() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(concat!(
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n",
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"rust\\\"}\"}}]}}]}\n\n",
                    "data: [DONE]\n\n",
                ));
        });
        let client = client_for(&server);
        let result = client
            .stream_completion(
                vec![Message::new(Role::User, "find rust")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("stream ok");
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_1");
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[0].arguments, r#"{"q":"rust"}"#);
    }

    #[tokio::test]
    async fn stream_parses_usage_with_cache_activity() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(concat!(
                    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
                    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":7,\"prompt_tokens_details\":{\"cached_tokens\":8,\"cache_write_tokens\":16}}}\n\n",
                    "data: [DONE]\n\n",
                ));
        });
        let client = client_for(&server);
        let result = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("stream ok");
        let usage = result.usage.expect("usage present");
        assert_eq!(usage.input_tokens, Some(42));
        assert_eq!(usage.output_tokens, Some(7));
        assert_eq!(usage.cache_read_input_tokens, Some(8));
        assert_eq!(usage.cache_creation_input_tokens, Some(16));
    }

    // --- Error paths -----------------------------------------------------

    #[tokio::test]
    async fn http_400_context_overflow_maps_to_context_overflow() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(400).header("content-type", "application/json").body(
                r#"{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 153827 tokens."}}"#,
            );
        });
        let client = client_for(&server);
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("must fail");
        match err {
            CoreError::ContextOverflow {
                prompt_tokens,
                max_tokens,
                ..
            } => {
                assert_eq!(prompt_tokens, Some(153_827));
                assert_eq!(max_tokens, Some(128_000));
            }
            other => panic!("expected ContextOverflow, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_402_maps_to_quota_exceeded() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(402)
                .header("content-type", "application/json")
                .body(r#"{"error":{"code":402,"message":"Payment required"}}"#);
        });
        let client = client_for(&server);
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("must fail");
        assert!(
            matches!(err, CoreError::QuotaExceeded { .. }),
            "402 must map to QuotaExceeded; got {err:?}"
        );
    }

    #[tokio::test]
    async fn insufficient_credits_body_maps_to_quota_exceeded_even_without_402() {
        // Some upstream proxies signal exhausted credits with a non-402 status;
        // the body detector must still catch it.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(400).header("content-type", "application/json").body(
                r#"{"error":{"code":"insufficient_credits","message":"This request requires more credits than you have available."}}"#,
            );
        });
        let client = client_for(&server);
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("must fail");
        assert!(
            matches!(err, CoreError::QuotaExceeded { .. }),
            "credits body must map to QuotaExceeded; got {err:?}"
        );
    }

    #[tokio::test]
    async fn http_429_maps_to_rate_limited_with_retry_after() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(429)
                .header("content-type", "application/json")
                .header("retry-after", "20")
                .body(r#"{"error":{"message":"Rate limit reached"}}"#);
        });
        let client = client_for(&server);
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("must fail");
        match err {
            CoreError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(Duration::from_secs(20)));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_503_maps_to_rate_limited() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(503)
                .header("content-type", "application/json")
                .body("upstream overloaded");
        });
        let client = client_for(&server);
        let err = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("must fail");
        assert!(
            matches!(err, CoreError::RateLimited { .. }),
            "503 must map to RateLimited; got {err:?}"
        );
    }

    // --- Direct classifier / detector unit tests ------------------------

    #[test]
    fn classify_402_is_quota_regardless_of_body() {
        let err = classify_openrouter_error(StatusCode::PAYMENT_REQUIRED, &HeaderMap::new(), "{}");
        assert!(matches!(err, CoreError::QuotaExceeded { .. }));
    }

    #[test]
    fn classify_delegates_non_openrouter_cases() {
        // A plain 401 is not an OpenRouter-specific case; it delegates to the
        // base classifier's generic `Llm` mapping.
        let err = classify_openrouter_error(
            StatusCode::UNAUTHORIZED,
            &HeaderMap::new(),
            r#"{"error":{"message":"bad key"}}"#,
        );
        assert!(matches!(err, CoreError::Llm(_)));
    }

    #[test]
    fn detect_credits_matches_wordings_and_rejects_others() {
        assert!(detect_openrouter_insufficient_credits(
            r#"{"error":{"message":"insufficient credits"}}"#
        ));
        assert!(detect_openrouter_insufficient_credits(
            "This request requires more credits than available"
        ));
        assert!(detect_openrouter_insufficient_credits(
            r#"{"code":"insufficient_credits"}"#
        ));
        assert!(!detect_openrouter_insufficient_credits(
            r#"{"error":{"message":"rate limit reached"}}"#
        ));
        assert!(!detect_openrouter_insufficient_credits("Bad Gateway"));
    }

    // --- Cancellation ----------------------------------------------------

    #[tokio::test]
    async fn stream_aborts_on_cancellation() {
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .delay(Duration::from_secs(5))
                .body(STUB_SSE_BODY);
        });
        let client = client_for(&server);
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

    // --- max_context_tokens ---------------------------------------------

    #[test]
    fn max_context_tokens_uses_curated_model() {
        let client = OpenRouterClient::new("k".into()).with_model("openai/gpt-4o");
        assert_eq!(client.max_context_tokens(), Some(128_000));
    }

    #[test]
    fn max_context_tokens_unknown_model_is_none() {
        let client = OpenRouterClient::new("k".into()).with_model("vendor/totally-unknown");
        assert_eq!(client.max_context_tokens(), None);
    }

    #[test]
    fn max_context_tokens_honours_cap() {
        let client = OpenRouterClient::new("k".into())
            .with_model("openai/gpt-4o")
            .with_max_context_tokens(Some(32_000));
        assert_eq!(client.max_context_tokens(), Some(32_000));
    }

    #[tokio::test]
    async fn max_context_tokens_consults_model_override() {
        let client = OpenRouterClient::new("k".into()).with_model("vendor/unknown");
        assert_eq!(client.max_context_tokens(), None);
        let observed = with_model_override("openai/gpt-4o".into(), async {
            client.max_context_tokens()
        })
        .await;
        assert_eq!(observed, Some(128_000));
    }

    // --- list_models: curated + live merge, degrade on failure ----------

    #[tokio::test]
    async fn list_models_merges_curated_with_live() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/models");
            then.status(200).header("content-type", "application/json").body(
                r#"{"data":[
                    {"id":"openai/gpt-4o","name":"live gpt-4o","context_length":9999},
                    {"id":"cohere/command-r","name":"Command R","context_length":128000,"supported_parameters":["tools"],"architecture":{"input_modalities":["text"]}}
                ]}"#,
            );
        });
        let client = client_for(&server);
        let models = client.list_models().await.expect("ok");

        // Curated metadata wins on the overlapping id (context 128k, not 9999).
        let gpt4o = models
            .iter()
            .find(|m| m.id == "openai/gpt-4o")
            .expect("curated gpt-4o present");
        assert_eq!(gpt4o.context_limit, Some(128_000));
        // The unknown live id is appended with its parsed metadata.
        let cmd = models
            .iter()
            .find(|m| m.id == "cohere/command-r")
            .expect("live model appended");
        assert_eq!(cmd.context_limit, Some(128_000));
        assert!(cmd.capabilities.tools);
        assert!(!cmd.capabilities.vision);
        // Curated entries come first.
        assert_eq!(models[0].id, "anthropic/claude-sonnet-4-6");
    }

    #[tokio::test]
    async fn list_models_degrades_to_curated_on_fetch_failure() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/models");
            then.status(500).body("boom");
        });
        let client = client_for(&server);
        let models = client.list_models().await.expect("must degrade, not error");
        assert_eq!(models, curated_openrouter_models());
    }

    #[test]
    fn live_model_infers_capabilities() {
        let m = LiveModel {
            id: "x/y".into(),
            name: Some("Y".into()),
            context_length: Some(4096),
            architecture: Some(Architecture {
                input_modalities: vec!["text".into(), "image".into()],
            }),
            supported_parameters: vec!["tools".into(), "reasoning".into()],
        };
        let info = live_model_to_info(m);
        assert_eq!(info.id, "x/y");
        assert_eq!(info.display_name, "Y");
        assert_eq!(info.context_limit, Some(4096));
        assert!(info.capabilities.tools);
        assert!(info.capabilities.reasoning);
        assert!(info.capabilities.vision);
        assert!(!info.capabilities.embedding);
    }
}
