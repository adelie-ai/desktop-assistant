//! Azure OpenAI (Microsoft Foundry) connector implementing the core
//! [`LlmClient`] port on the OpenAI **Chat Completions** dialect shared with
//! OpenRouter ([`desktop_assistant_llm_openai_compat`]).
//!
//! One connector points at one Azure resource and selects a model on top, like
//! Bedrock -- the wrinkle is that the "model" is an operator-provisioned
//! **deployment** whose name need not equal the base model. See
//! `docs/connectors/azure.md`.
//!
//! ## Surfaces
//!
//! - **v1 GA (default, [`ApiSurface::V1`]):**
//!   `POST {resource}/openai/v1/chat/completions`, deployment in the request
//!   body `model`, no `api-version` query.
//! - **Legacy ([`ApiSurface::Classic`]):**
//!   `{resource}/openai/deployments/{deployment}/chat/completions?api-version={ver}`,
//!   deployment in the URL path.
//!
//! ## Auth
//!
//! - [`AuthMode::ApiKey`] (default): the `api-key: <key>` header.
//! - [`AuthMode::Entra`]: `Authorization: Bearer <token>` from a refreshing
//!   [`TokenProvider`] (Entra ID / managed identity, scope
//!   `https://ai.azure.com/.default`). This pass ships the seam and a mock;
//!   real token acquisition is a documented follow-up (see [`TokenProvider`]).
//!
//! Credentials travel in headers only -- never in a URL, log, or error message
//! -- and the client renders a redacting [`Debug`].

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition};
use desktop_assistant_core::ports::embedding::EmbeddingClient;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelCapabilities, ModelInfo, ReasoningConfig,
    current_cancellation_token, current_model_override,
};
use desktop_assistant_llm_http::{
    Clock, ModelCache, STREAM_CONNECT_TIMEOUT, STREAM_EVENT_TIMEOUT, apply_context_cap,
    bail_for_status, merge_curated_with_live,
};
use desktop_assistant_llm_openai_compat::{
    ChatMessage, ChatTool, StreamingDispatchError, classify_error, consume_chat_stream,
    dispatch_non_streaming, send_chat_request, to_chat_messages, to_chat_tools,
};
use reqwest::header::HeaderMap;
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// Default `api-version` for the legacy ([`ApiSurface::Classic`]) surface. A GA
/// value; the operator overrides it via [`AzureClient::with_api_version`] when
/// their resource needs a different one. Unused on the default `v1` surface,
/// which removed the `api-version` query entirely.
const DEFAULT_CLASSIC_API_VERSION: &str = "2024-10-21";

/// Which Azure REST surface the connector targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApiSurface {
    /// The v1 GA surface: `/openai/v1/chat/completions`, deployment in the body
    /// `model`, no `api-version` query. The default.
    #[default]
    V1,
    /// The legacy surface:
    /// `/openai/deployments/{deployment}/chat/completions?api-version={ver}`,
    /// deployment in the URL path. Offered for resources not yet on v1.
    Classic,
}

impl std::str::FromStr for ApiSurface {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "v1" => Ok(Self::V1),
            "classic" | "legacy" => Ok(Self::Classic),
            other => Err(CoreError::Llm(format!(
                "unknown Azure api_surface '{other}'; expected 'v1' or 'classic'"
            ))),
        }
    }
}

/// How the connector authenticates to the Azure resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AuthMode {
    /// Static resource key sent as the `api-key` header. The default.
    #[default]
    ApiKey,
    /// A short-lived Entra ID / managed-identity bearer token from a
    /// [`TokenProvider`], sent as `Authorization: Bearer <token>`.
    Entra,
}

impl std::str::FromStr for AuthMode {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "api_key" | "apikey" | "api-key" | "key" => Ok(Self::ApiKey),
            "entra" | "entra_id" | "managed_identity" | "aad" => Ok(Self::Entra),
            other => Err(CoreError::Llm(format!(
                "unknown Azure auth_mode '{other}'; expected 'api_key' or 'entra'"
            ))),
        }
    }
}

/// Seam for acquiring a short-lived Entra ID / managed-identity bearer token
/// (scope `https://ai.azure.com/.default`).
///
/// The implementation MUST refresh transparently and MUST NOT log the token.
/// This crate ships the trait plus a test mock; a production implementation
/// (reusing the workspace `jsonwebtoken` crate and the existing OAuth /
/// keyring machinery, gated on a CVE scan before first build) is a tracked
/// follow-up -- until it lands, `auth_mode = entra` requires an injected
/// provider via [`AzureClient::with_token_provider`].
#[async_trait::async_trait]
pub trait TokenProvider: Send + Sync {
    /// Return a currently-valid bearer token, refreshing as needed. Errors map
    /// to a [`CoreError::Llm`] at the call site; the token is never included in
    /// the error.
    async fn token(&self) -> Result<String, CoreError>;
}

/// Azure OpenAI client that streams Chat Completions and serves embeddings.
pub struct AzureClient {
    client: Client,
    /// Resource key for [`AuthMode::ApiKey`]; empty when using Entra.
    api_key: String,
    /// The deployment name -- sent as the request body `model` (v1) or in the
    /// URL path (classic).
    model: String,
    /// The resource endpoint, e.g. `https://<name>.openai.azure.com`. No
    /// default host: resource-specific and required.
    base_url: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    hosted_tool_search: bool,
    connect_timeout: Duration,
    event_timeout: Duration,
    context_cap: Option<u64>,
    api_surface: ApiSurface,
    api_version: String,
    auth_mode: AuthMode,
    token_provider: Option<Arc<dyn TokenProvider>>,
    /// TTL cache for `list_models()`. Azure has no live listing endpoint yet
    /// (deployment enumeration is an ARM control-plane operation), so the cache
    /// currently holds the curated table; it keeps the connector uniform with
    /// OpenRouter/Google and is where a future live listing slots in (#620).
    model_cache: ModelCache,
    /// Per-deployment memo of backends that reject tool use in streaming (#619).
    /// A deployment recorded here skips the stream attempt and goes straight to
    /// the non-streaming `/chat/completions` path on the next tools turn.
    /// Populated at runtime from the provider's tools-unsupported-in-streaming
    /// error; the guard is never held across an `.await`.
    non_streaming_tools_models: Arc<Mutex<HashSet<String>>>,
}

/// Redacting [`Debug`] so neither the API key nor a bearer token can leak
/// through a `{:?}` render. The key shows only its length; the token provider
/// shows presence, never its value. Mirrors the Anthropic connector's posture.
impl std::fmt::Debug for AzureClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureClient")
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
            .field("api_surface", &self.api_surface)
            .field("api_version", &self.api_version)
            .field("auth_mode", &self.auth_mode)
            .field(
                "token_provider",
                &self.token_provider.as_ref().map(|_| "<TokenProvider>"),
            )
            .field("model_cache", &self.model_cache)
            .field("non_streaming_tools_models", &"<memo>")
            .finish()
    }
}

impl AzureClient {
    /// Azure deployments are operator-named, so there is no sensible default
    /// model -- the deployment must be configured explicitly.
    pub fn get_default_model() -> Option<&'static str> {
        None
    }

    /// The resource endpoint is resource-specific, so there is no default host.
    /// A missing `base_url` surfaces a clear error at request time rather than
    /// defaulting to some wrong host.
    pub fn get_default_base_url() -> Option<&'static str> {
        None
    }

    /// Construct with a resource key (used only under [`AuthMode::ApiKey`]).
    /// For Entra, pass an empty key and set [`Self::with_auth_mode`] +
    /// [`Self::with_token_provider`].
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
            connect_timeout: STREAM_CONNECT_TIMEOUT,
            event_timeout: STREAM_EVENT_TIMEOUT,
            context_cap: None,
            api_surface: ApiSurface::V1,
            api_version: DEFAULT_CLASSIC_API_VERSION.to_string(),
            auth_mode: AuthMode::ApiKey,
            token_provider: None,
            model_cache: ModelCache::new(),
            non_streaming_tools_models: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Set the deployment name (sent as the body `model`, or the URL path
    /// segment on the classic surface).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the resource endpoint, e.g. `https://<name>.openai.azure.com`. A
    /// trailing slash is tolerated.
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

    /// Set the completion-token cap. For reasoning deployments this is sent as
    /// `max_completion_tokens`; for others as `max_tokens`.
    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override the first-response (connect) stall budget. `None`/`Some(0)`
    /// keeps the [`STREAM_CONNECT_TIMEOUT`] default. Seconds.
    pub fn with_connect_timeout(mut self, secs: Option<u64>) -> Self {
        if let Some(s) = secs.filter(|s| *s > 0) {
            self.connect_timeout = Duration::from_secs(s);
        }
        self
    }

    /// Override the per-chunk stall budget. `None`/`Some(0)` keeps the
    /// [`STREAM_EVENT_TIMEOUT`] default. Seconds.
    pub fn with_event_timeout(mut self, secs: Option<u64>) -> Self {
        if let Some(s) = secs.filter(|s| *s > 0) {
            self.event_timeout = Duration::from_secs(s);
        }
        self
    }

    /// Set the per-connection context-window hard cap, in tokens. `None`/
    /// `Some(0)` = "max available". Folded with the resolved window in
    /// [`LlmClient::max_context_tokens`] via
    /// [`apply_context_cap`](desktop_assistant_llm_http::apply_context_cap).
    pub fn with_max_context_tokens(mut self, max: Option<u64>) -> Self {
        self.context_cap = max.filter(|m| *m > 0);
        self
    }

    /// Kept for factory-shape parity; Azure Chat Completions does not expose
    /// hosted tool search, so v1 always passes `false`.
    pub fn with_hosted_tool_search(mut self, enabled: bool) -> Self {
        self.hosted_tool_search = enabled;
        self
    }

    /// Select the REST surface ([`ApiSurface::V1`] by default).
    pub fn with_api_surface(mut self, surface: ApiSurface) -> Self {
        self.api_surface = surface;
        self
    }

    /// Set the `api-version` used by the classic surface only. Ignored on v1.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Select the auth mechanism ([`AuthMode::ApiKey`] by default).
    pub fn with_auth_mode(mut self, mode: AuthMode) -> Self {
        self.auth_mode = mode;
        self
    }

    /// Inject the [`TokenProvider`] used under [`AuthMode::Entra`].
    pub fn with_token_provider(mut self, provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = Some(provider);
        self
    }

    /// Override the `list_models()` cache TTL (default: 1h). `Duration::ZERO` disables
    /// caching (every entry is immediately stale).
    pub fn with_model_cache_ttl(mut self, ttl: Duration) -> Self {
        self.model_cache.set_ttl(ttl);
        self
    }

    /// Inject a [`Clock`] for deterministic cache-TTL tests. Production uses the
    /// default [`SystemClock`](desktop_assistant_llm_http::SystemClock).
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.model_cache.set_clock(clock);
        self
    }

    /// Build from environment variables: `AZURE_OPENAI_API_KEY` (required),
    /// and optionally `AZURE_OPENAI_MODEL` (the deployment) and
    /// `AZURE_OPENAI_BASE_URL` (the resource endpoint).
    pub fn from_env() -> Result<Self, CoreError> {
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").map_err(|_| {
            CoreError::Llm("AZURE_OPENAI_API_KEY environment variable not set".into())
        })?;
        let mut client = Self::new(api_key);
        if let Ok(model) = std::env::var("AZURE_OPENAI_MODEL") {
            client.model = model;
        }
        if let Ok(url) = std::env::var("AZURE_OPENAI_BASE_URL") {
            client.base_url = url;
        }
        Ok(client)
    }

    /// Return the deployment name as the stable embedding-model version id.
    pub async fn model_identifier(&self) -> Result<String, CoreError> {
        Ok(self.model.clone())
    }

    /// The resource endpoint with any trailing slash removed.
    fn base(&self) -> &str {
        self.base_url.trim_end_matches('/')
    }

    /// Shape the chat-completions URL for the configured surface.
    fn chat_completions_url(&self, deployment: &str) -> String {
        match self.api_surface {
            ApiSurface::V1 => format!("{}/openai/v1/chat/completions", self.base()),
            ApiSurface::Classic => format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                self.base(),
                deployment,
                self.api_version
            ),
        }
    }

    /// Shape the embeddings URL for the configured surface.
    fn embeddings_url(&self, deployment: &str) -> String {
        match self.api_surface {
            ApiSurface::V1 => format!("{}/openai/v1/embeddings", self.base()),
            ApiSurface::Classic => format!(
                "{}/openai/deployments/{}/embeddings?api-version={}",
                self.base(),
                deployment,
                self.api_version
            ),
        }
    }

    /// Attach the auth header for the configured mode. The credential is only
    /// ever placed in a header. Fails loudly (never silently unauthenticated)
    /// when the selected mode is missing its credential.
    async fn apply_auth(&self, rb: RequestBuilder) -> Result<RequestBuilder, CoreError> {
        match self.auth_mode {
            AuthMode::ApiKey => {
                if self.api_key.is_empty() {
                    return Err(CoreError::Llm(
                        "Azure api-key auth selected but no API key is configured".into(),
                    ));
                }
                Ok(rb.header("api-key", &self.api_key))
            }
            AuthMode::Entra => {
                let provider = self.token_provider.as_ref().ok_or_else(|| {
                    CoreError::Llm(
                        "Azure Entra auth selected but no TokenProvider is configured".into(),
                    )
                })?;
                let token = provider.token().await?;
                Ok(rb.header("Authorization", format!("Bearer {token}")))
            }
        }
    }

    /// Assemble the Chat Completions request envelope for `deployment`.
    ///
    /// Reasoning deployments carry `max_completion_tokens` (not `max_tokens`);
    /// tools are omitted when empty; `stream_options.include_usage` requests
    /// the final `usage` chunk. `mark_system_cache_breakpoint` is deliberately
    /// NOT called -- Azure caches automatically.
    fn build_request_body(
        &self,
        deployment: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
    ) -> ChatRequest {
        let (max_tokens, max_completion_tokens) = if model_supports_reasoning(deployment) {
            (None, self.max_tokens)
        } else {
            (self.max_tokens, None)
        };
        ChatRequest {
            model: deployment.to_string(),
            messages: to_chat_messages(messages),
            tools: to_chat_tools(tools),
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens,
            max_completion_tokens,
            reasoning_effort: reasoning_for(deployment, reasoning),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
        }
    }

    /// Flip a streaming request envelope to its non-streaming form: `stream`
    /// off and `stream_options` dropped (invalid when not streaming). Used for
    /// the tools-in-streaming fallback (#619).
    fn to_non_streaming(mut request: ChatRequest) -> ChatRequest {
        request.stream = false;
        request.stream_options = None;
        request
    }

    /// Build the `/chat/completions` request: the field-aware preflight, the
    /// request-size log, the URL for the configured surface, and the auth
    /// header. Shared by the streaming and non-streaming send paths.
    async fn build_request(
        &self,
        deployment: &str,
        request_body: &ChatRequest,
    ) -> Result<RequestBuilder, CoreError> {
        // Field-aware preflight ("no silent failures"): surface the missing
        // piece by name instead of a malformed-URL 404 on the first turn.
        if self.base().is_empty() {
            return Err(CoreError::Llm(
                "Azure needs the resource endpoint, e.g. https://<resource>.openai.azure.com"
                    .into(),
            ));
        }
        if deployment.is_empty() {
            return Err(CoreError::Llm(
                "Azure needs a deployment name (sent as the request body `model`); \
                 set the connection's model/deployment"
                    .into(),
            ));
        }

        let request_json =
            serde_json::to_string(request_body).unwrap_or_else(|_| "<serialization error>".into());
        tracing::info!(
            request_bytes = request_json.len(),
            deployment = %deployment,
            stream = request_body.stream,
            "Azure LLM request payload"
        );
        tracing::debug!(
            "Azure request body (first 2000 chars): {}",
            &request_json[..request_json.len().min(2000)]
        );

        let url = self.chat_completions_url(deployment);
        let rb = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(request_body);
        self.apply_auth(rb).await
    }

    /// POST the request (with a bounded connect race via the shared
    /// [`send_chat_request`]) and consume the SSE stream into an
    /// [`LlmResponse`], mapping any HTTP error via [`Self::classify_azure_error`].
    ///
    /// A deployment that rejects tools-in-streaming classifies to
    /// [`CoreError::ToolsUnsupported`]; that arm hands the (still-unconsumed)
    /// callback back through [`StreamingDispatchError::ToolsUnsupported`] so the
    /// caller can retry non-streaming without rebuilding it (#619).
    async fn send_and_stream(
        &self,
        deployment: &str,
        request_body: &ChatRequest,
        on_chunk: ChunkCallback,
        cancellation: &CancellationToken,
    ) -> Result<LlmResponse, StreamingDispatchError> {
        let request = match self.build_request(deployment, request_body).await {
            Ok(rb) => rb,
            Err(e) => return Err(StreamingDispatchError::Other(e)),
        };
        let response = match send_chat_request(
            request,
            cancellation,
            self.connect_timeout,
            "Azure stream stalled",
            |status, headers, body| self.classify_azure_error(status, headers, body),
        )
        .await
        {
            Ok(r) => r,
            Err(CoreError::ToolsUnsupported { detail }) => {
                return Err(StreamingDispatchError::ToolsUnsupported { on_chunk, detail });
            }
            Err(e) => return Err(StreamingDispatchError::Other(e)),
        };

        consume_chat_stream(
            response.bytes_stream(),
            cancellation,
            self.event_timeout,
            on_chunk,
        )
        .await
        .map_err(StreamingDispatchError::Other)
    }

    /// POST the request with `stream: false` and parse the single JSON response
    /// into an [`LlmResponse`] via the shared [`dispatch_non_streaming`]. Used
    /// as the fallback for a deployment that rejects tools-in-streaming (#619).
    async fn send_non_streaming(
        &self,
        deployment: &str,
        request_body: &ChatRequest,
        on_chunk: ChunkCallback,
        cancellation: &CancellationToken,
    ) -> Result<LlmResponse, CoreError> {
        let request = self.build_request(deployment, request_body).await?;
        let response = send_chat_request(
            request,
            cancellation,
            self.connect_timeout,
            "Azure stream stalled",
            |status, headers, body| self.classify_azure_error(status, headers, body),
        )
        .await?;
        dispatch_non_streaming(response, cancellation, on_chunk).await
    }

    /// Map an HTTP error to a [`CoreError`], adding Azure specifics on top of
    /// the shared OpenAI-compatible [`classify_error`].
    ///
    /// Order: content-filter decline (never echoes the flagged body), then the
    /// classic invalid/retired `api-version`, then a clear 401, else delegate.
    fn classify_azure_error(
        &self,
        status: StatusCode,
        headers: &HeaderMap,
        body: &str,
    ) -> CoreError {
        // A content-filter block is a business decline, not a technical
        // failure: non-retryable, logged at info (not error), surfaced with a
        // specific reason -- and NEVER the raw body, which echoes the flagged
        // user content.
        //
        // NOTE: there is no dedicated `CoreError::Declined` variant yet, so this
        //       maps to `Llm` with a clean reason; a future decline variant
        //       should carry `retryable = false` and a machine-readable code.
        // NOTE: content_filter can also arrive as a streaming `finish_reason`
        //       on a 200 response. The shared `consume_chat_stream` does not
        //       surface `finish_reason`, so only the request-level (HTTP-error)
        //       filter is mapped here; the streaming case is a tracked follow-up.
        if let Some(reason) = detect_content_filter(body) {
            tracing::info!("Azure content filter declined the request (HTTP {status})");
            return CoreError::Llm(reason);
        }

        if matches!(self.api_surface, ApiSurface::Classic) && detect_invalid_api_version(body) {
            return CoreError::Llm(format!(
                "Azure api-version '{}' is invalid or retired (HTTP {status}); \
                 update the connection's api_version to a supported value",
                self.api_version
            ));
        }

        if status.as_u16() == 401 {
            // The token/key lives only in the request headers and is never in
            // the response body, so including `body` here cannot leak it.
            return CoreError::Llm(format!(
                "Azure authentication failed (HTTP 401): verify the api-key or Entra token \
                 for resource {}. Detail: {body}",
                self.base()
            ));
        }

        classify_error(status, headers, body)
    }

    /// Generate embeddings for a batch of texts against `/openai/v1/embeddings`
    /// (or the classic deployments path), using the configured deployment as
    /// the embedding `model`.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        if self.base().is_empty() {
            return Err(CoreError::Llm(
                "Azure needs the resource endpoint, e.g. https://<resource>.openai.azure.com"
                    .into(),
            ));
        }
        if self.model.is_empty() {
            return Err(CoreError::Llm(
                "Azure embeddings need an embedding deployment name (the `model`); \
                 set the connection's model/deployment"
                    .into(),
            ));
        }

        let body = serde_json::json!({ "model": self.model, "input": texts });
        let url = self.embeddings_url(&self.model);
        let rb = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);
        let rb = self.apply_auth(rb).await?;
        let response = rb
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("Azure embedding HTTP request failed: {e}")))?;

        let response = bail_for_status(response, "Azure embeddings API error").await?;

        let parsed: EmbeddingResponse = response.json().await.map_err(|e| {
            CoreError::Llm(format!("failed to parse Azure embedding response: {e}"))
        })?;

        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

// ---------------------------------------------------------------------------
// Request / response wire types
// ---------------------------------------------------------------------------

/// The Chat Completions request envelope. Reuses the compat message/tool types;
/// only the top-level shaping (reasoning field selection, `stream_options`) is
/// Azure's.
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    stream: bool,
    /// Only valid while streaming: asks Azure to append the final `usage`
    /// chunk. Omitted on the non-streaming path -- the API rejects
    /// `stream_options` when `stream` is `false` (usage is returned inline
    /// there anyway).
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

/// `stream_options` -- asks Azure to append the final `usage` chunk.
#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Free helpers: reasoning gating, model resolution, error detection
// ---------------------------------------------------------------------------

/// True when a deployment name resolves to a reasoning-capable model.
///
/// Deployment names are operator-defined and need not equal the base model, so
/// this is a conservative name heuristic: the GPT-5 family (`gpt-5` / `gpt5`),
/// and o-series tokens (`o1` / `o3` / `o4`, e.g. `o4-mini`, `my-o3-deploy`).
/// When a name gives no clear signal it returns `false`, so the connector
/// omits `reasoning_effort` rather than sending an unsupported field.
pub fn model_supports_reasoning(deployment: &str) -> bool {
    let m = deployment.to_ascii_lowercase();
    if m.contains("gpt-5") || m.contains("gpt5") {
        return true;
    }
    // An o-series token: an alphanumeric run starting with 'o' followed by one
    // of the reasoning generations (1/3/4). Tokenising on non-alphanumerics
    // avoids matching the trailing 'o' of "gpt-4o" (which tokenises to "4o").
    m.split(|c: char| !c.is_ascii_alphanumeric()).any(|tok| {
        let b = tok.as_bytes();
        b.len() >= 2 && b[0] == b'o' && matches!(b[1], b'1' | b'3' | b'4')
    })
}

/// The `reasoning_effort` literal for a turn, gated by the resolved model.
/// Returns `None` (and debug-logs) for non-reasoning deployments and when no
/// effort was requested, so an unsupported field is never sent.
fn reasoning_for(deployment: &str, reasoning: ReasoningConfig) -> Option<&'static str> {
    let level = reasoning.reasoning_effort?;
    if !model_supports_reasoning(deployment) {
        tracing::debug!(
            deployment,
            requested_effort = ?level,
            "Azure reasoning_effort requested but deployment does not resolve to a \
             reasoning model; dropping field"
        );
        return None;
    }
    Some(level.as_openai_effort())
}

/// Best-effort prompt-token window for a deployment, inferred from the base
/// model embedded in its name. Returns `None` when unknown (caller falls back
/// to the universal default) or for GPT-5 (window varies by tier). Specific
/// families are checked before their general counterparts.
pub fn context_limit_for_deployment(deployment: &str) -> Option<u64> {
    let m = deployment.to_ascii_lowercase();
    // o-series reasoning models: 200k.
    let is_o_series = m.split(|c: char| !c.is_ascii_alphanumeric()).any(|tok| {
        let b = tok.as_bytes();
        b.len() >= 2 && b[0] == b'o' && matches!(b[1], b'1' | b'3' | b'4')
    });
    if is_o_series {
        return Some(200_000);
    }
    // GPT-5 family: window varies by tier; defer to the universal fallback.
    if m.contains("gpt-5") || m.contains("gpt5") {
        return None;
    }
    if m.contains("gpt-4.1") || m.contains("gpt-41") || m.contains("gpt41") {
        return Some(1_000_000);
    }
    if m.contains("gpt-4o") || m.contains("gpt4o") || m.contains("gpt-4-turbo") {
        return Some(128_000);
    }
    if m.contains("gpt-4-32k") {
        return Some(32_768);
    }
    if m.contains("gpt-4") {
        return Some(8_192);
    }
    if m.contains("gpt-35") || m.contains("gpt-3.5") {
        return Some(16_384);
    }
    None
}

/// Detect an Azure content-filter block and build a clean, informative reason
/// that names the flagged policy categories WITHOUT echoing any user content.
///
/// Returns `None` for any non-content-filter body. The category names and
/// severities are policy metadata (not the flagged text), so they are safe to
/// surface; the raw body is never included.
fn detect_content_filter(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    let code = error.get("code").and_then(|c| c.as_str());
    let inner = error.get("innererror");
    let inner_code = inner.and_then(|i| i.get("code")).and_then(|c| c.as_str());

    let is_filter =
        code == Some("content_filter") || inner_code == Some("ResponsibleAIPolicyViolation");
    if !is_filter {
        return None;
    }

    let mut categories: Vec<String> = inner
        .and_then(|i| i.get("content_filter_result"))
        .and_then(|r| r.as_object())
        .map(|obj| {
            obj.iter()
                .filter(|(_, v)| v.get("filtered").and_then(|f| f.as_bool()) == Some(true))
                .map(|(k, v)| match v.get("severity").and_then(|s| s.as_str()) {
                    Some(sev) => format!("{k} ({sev})"),
                    None => k.clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    categories.sort();

    let detail = if categories.is_empty() {
        "Azure content filter blocked this request (no category detail provided)".to_string()
    } else {
        format!(
            "Azure content filter blocked this request; flagged categories: {}",
            categories.join(", ")
        )
    };
    Some(detail)
}

/// Detect an invalid/retired `api-version` rejection (classic surface). True
/// when the error code is `invalid_api_version` or the message references the
/// api version. Only consulted on the classic surface.
fn detect_invalid_api_version(body: &str) -> bool {
    let Ok(value): Result<serde_json::Value, _> = serde_json::from_str(body) else {
        return false;
    };
    let error = value.get("error");
    let code = error.and_then(|e| e.get("code")).and_then(|c| c.as_str());
    if code == Some("invalid_api_version") {
        return true;
    }
    let message = error
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    message.contains("api version") || message.contains("api-version")
}

/// Curated base-model table exposed through this connector.
///
/// The model picker must NOT offer these as selectable Azure ids (a curated
/// `gpt-4o` when the deployment is `my-gpt4` yields a 404); they carry the
/// capability/context metadata used to resolve a deployment's base model.
fn curated_azure_models() -> Vec<ModelInfo> {
    let chat = ModelCapabilities {
        reasoning: false,
        vision: true,
        tools: true,
        embedding: false,
    };
    let reasoning = ModelCapabilities {
        reasoning: true,
        vision: true,
        tools: true,
        embedding: false,
    };
    let embedding = ModelCapabilities {
        reasoning: false,
        vision: false,
        tools: false,
        embedding: true,
    };

    vec![
        // --- GPT-5 reasoning family ---
        ModelInfo::new("gpt-5")
            .with_display_name("GPT-5")
            .with_capabilities(reasoning),
        ModelInfo::new("gpt-5-mini")
            .with_display_name("GPT-5 Mini")
            .with_capabilities(reasoning),
        // --- o-series reasoning models ---
        ModelInfo::new("o3")
            .with_display_name("o3")
            .with_context_limit(200_000)
            .with_capabilities(reasoning),
        ModelInfo::new("o4-mini")
            .with_display_name("o4-mini")
            .with_context_limit(200_000)
            .with_capabilities(reasoning),
        // --- GPT-4.1 / 4o general chat ---
        ModelInfo::new("gpt-4.1")
            .with_display_name("GPT-4.1")
            .with_context_limit(1_000_000)
            .with_capabilities(chat),
        ModelInfo::new("gpt-4o")
            .with_display_name("GPT-4o")
            .with_context_limit(128_000)
            .with_capabilities(chat),
        ModelInfo::new("gpt-4o-mini")
            .with_display_name("GPT-4o mini")
            .with_context_limit(128_000)
            .with_capabilities(chat),
        // --- Embedding models ---
        ModelInfo::new("text-embedding-3-large")
            .with_display_name("Text Embedding 3 Large")
            .with_capabilities(embedding),
        ModelInfo::new("text-embedding-3-small")
            .with_display_name("Text Embedding 3 Small")
            .with_capabilities(embedding),
    ]
}

impl AzureClient {
    /// Build the model listing and cache it. Bypasses the TTL: the shared tail
    /// of both `list_models` (on a cache miss) and `refresh_models`.
    ///
    /// There is no live deployment enumeration yet (an ARM control-plane
    /// operation, `management.azure.com`), so this resolves to the curated
    /// table. When it lands it slots in as the resolver's `live` argument
    /// (curated metadata wins on overlap, unknown deployments appended); the
    /// caching shape here does not change.
    async fn fetch_merge_store(&self) -> Vec<ModelInfo> {
        let merged = merge_curated_with_live(curated_azure_models(), Vec::new());
        self.model_cache.store(merged.clone());
        merged
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl LlmClient for AzureClient {
    fn get_default_model(&self) -> Option<&str> {
        Self::get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        Self::get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        let deployment = current_model_override().unwrap_or_else(|| self.model.clone());
        apply_context_cap(self.context_cap, context_limit_for_deployment(&deployment))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // Serve a warm, unexpired listing from the cache; on a miss, (re)build
        // it and cache the result. See [`Self::fetch_merge_store`] for why this
        // resolves to the curated table today.
        if let Some(cached) = self.model_cache.cached() {
            return Ok(cached);
        }
        Ok(self.fetch_merge_store().await)
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        // Force a rebuild, bypassing the TTL, and re-populate the cache.
        Ok(self.fetch_merge_store().await)
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        // Per-turn model override (issue #34): the daemon routes the chosen
        // deployment via MODEL_OVERRIDE; reasoning gating and URL shaping key on
        // the dispatched deployment, not the baked-in default.
        let deployment = current_model_override().unwrap_or_else(|| self.model.clone());
        let cancellation = current_cancellation_token().unwrap_or_default();
        let has_tools = !tools.is_empty();

        // Memo (#619): a deployment known to reject tools-in-streaming skips the
        // stream attempt when this turn carries tools. A no-tools turn always
        // streams. The guard is dropped before any `.await`.
        let skip_streaming = has_tools && {
            let memo = self
                .non_streaming_tools_models
                .lock()
                .expect("non_streaming_tools_models mutex poisoned");
            memo.contains(&deployment)
        };
        if skip_streaming {
            tracing::debug!(
                deployment = %deployment,
                "skipping stream: deployment memoized as tools-in-streaming-unsupported"
            );
            let request = Self::to_non_streaming(self.build_request_body(
                &deployment,
                &messages,
                tools,
                reasoning,
            ));
            return self
                .send_non_streaming(&deployment, &request, on_chunk, &cancellation)
                .await;
        }

        // Try streaming first; on the tools-in-streaming rejection, memoize the
        // deployment and retry once via non-streaming with the handed-back
        // callback. A non-streaming failure surfaces as-is -- it never loops
        // back to the stream attempt (#619).
        let stream_request = self.build_request_body(&deployment, &messages, tools, reasoning);
        match self
            .send_and_stream(&deployment, &stream_request, on_chunk, &cancellation)
            .await
        {
            Ok(response) => Ok(response),
            Err(StreamingDispatchError::ToolsUnsupported { on_chunk, detail }) => {
                tracing::warn!(
                    deployment = %deployment,
                    detail,
                    "Azure rejected tools in streaming; retrying non-streaming and memoizing \
                     the deployment so future turns skip the stream attempt"
                );
                {
                    let mut memo = self
                        .non_streaming_tools_models
                        .lock()
                        .expect("non_streaming_tools_models mutex poisoned");
                    memo.insert(deployment.clone());
                }
                let request = Self::to_non_streaming(self.build_request_body(
                    &deployment,
                    &messages,
                    tools,
                    reasoning,
                ));
                self.send_non_streaming(&deployment, &request, on_chunk, &cancellation)
                    .await
            }
            Err(StreamingDispatchError::Other(e)) => Err(e),
        }
    }

    fn supports_hosted_tool_search(&self) -> bool {
        self.hosted_tool_search
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for AzureClient {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        AzureClient::embed(self, texts).await
    }

    async fn model_identifier(&self) -> Result<String, CoreError> {
        AzureClient::model_identifier(self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::Role;
    use desktop_assistant_core::ports::llm::{
        ReasoningLevel, with_cancellation_token, with_model_override,
    };
    use httpmock::Method::POST;
    use httpmock::MockServer;

    // A data-only Chat Completions SSE body: a text delta, then a usage-only
    // final chunk (with cached_tokens), then the terminator.
    const STUB_SSE_BODY: &str = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,",
        "\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n",
        "data: [DONE]\n\n",
    );

    // An SSE body carrying a single complete tool call.
    const STUB_SSE_TOOLS_BODY: &str = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"id\":\"call_1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    struct MockTokenProvider {
        token: String,
    }

    #[async_trait::async_trait]
    impl TokenProvider for MockTokenProvider {
        async fn token(&self) -> Result<String, CoreError> {
            Ok(self.token.clone())
        }
    }

    fn client_for(server: &MockServer) -> AzureClient {
        AzureClient::new("secret-key".into())
            .with_base_url(server.url(""))
            .with_model("dep")
    }

    // --- URL shaping -----------------------------------------------------

    #[test]
    fn v1_chat_url_has_openai_v1_path_and_no_api_version() {
        let c = AzureClient::new("k".into())
            .with_base_url("https://foo.openai.azure.com")
            .with_model("dep");
        let url = c.chat_completions_url("dep");
        assert_eq!(
            url,
            "https://foo.openai.azure.com/openai/v1/chat/completions"
        );
        assert!(
            !url.contains("api-version"),
            "v1 must not carry api-version"
        );
    }

    #[test]
    fn v1_chat_url_trims_trailing_slash() {
        let c = AzureClient::new("k".into()).with_base_url("https://foo.openai.azure.com/");
        assert_eq!(
            c.chat_completions_url("dep"),
            "https://foo.openai.azure.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn classic_chat_url_has_deployment_and_api_version() {
        let c = AzureClient::new("k".into())
            .with_base_url("https://foo.openai.azure.com")
            .with_api_surface(ApiSurface::Classic)
            .with_api_version("2024-10-21");
        let url = c.chat_completions_url("mydep");
        assert_eq!(
            url,
            "https://foo.openai.azure.com/openai/deployments/mydep/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn v1_embeddings_url_shape() {
        let c = AzureClient::new("k".into()).with_base_url("https://foo.openai.azure.com");
        assert_eq!(
            c.embeddings_url("emb"),
            "https://foo.openai.azure.com/openai/v1/embeddings"
        );
    }

    #[test]
    fn classic_embeddings_url_shape() {
        let c = AzureClient::new("k".into())
            .with_base_url("https://foo.openai.azure.com")
            .with_api_surface(ApiSurface::Classic)
            .with_api_version("2024-10-21");
        assert_eq!(
            c.embeddings_url("emb"),
            "https://foo.openai.azure.com/openai/deployments/emb/embeddings?api-version=2024-10-21"
        );
    }

    // --- Builder / defaults ----------------------------------------------

    #[test]
    fn defaults_have_no_model_or_base_url() {
        assert_eq!(AzureClient::get_default_model(), None);
        assert_eq!(AzureClient::get_default_base_url(), None);
        let c = AzureClient::new("k".into());
        assert_eq!(c.model, "");
        assert_eq!(c.base_url, "");
        assert_eq!(c.api_surface, ApiSurface::V1);
        assert_eq!(c.auth_mode, AuthMode::ApiKey);
    }

    #[test]
    fn builder_sets_fields() {
        let c = AzureClient::new("k".into())
            .with_model("my-dep")
            .with_base_url("https://x.openai.azure.com")
            .with_api_surface(ApiSurface::Classic)
            .with_api_version("2025-01-01")
            .with_auth_mode(AuthMode::Entra);
        assert_eq!(c.model, "my-dep");
        assert_eq!(c.base_url, "https://x.openai.azure.com");
        assert_eq!(c.api_surface, ApiSurface::Classic);
        assert_eq!(c.api_version, "2025-01-01");
        assert_eq!(c.auth_mode, AuthMode::Entra);
    }

    #[test]
    fn api_surface_and_auth_mode_parse_from_str() {
        assert_eq!("v1".parse::<ApiSurface>().unwrap(), ApiSurface::V1);
        assert_eq!(
            "CLASSIC".parse::<ApiSurface>().unwrap(),
            ApiSurface::Classic
        );
        assert!("responses".parse::<ApiSurface>().is_err());
        assert_eq!("api_key".parse::<AuthMode>().unwrap(), AuthMode::ApiKey);
        assert_eq!("entra".parse::<AuthMode>().unwrap(), AuthMode::Entra);
        assert!("basic".parse::<AuthMode>().is_err());
    }

    // --- Request serialization -------------------------------------------

    #[test]
    fn reasoning_deployment_uses_max_completion_tokens() {
        let c = AzureClient::new("k".into())
            .with_model("o3")
            .with_max_tokens(Some(1024));
        let req = c.build_request_body("o3", &[], &[], ReasoningConfig::default());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["max_completion_tokens"], 1024);
        assert!(
            json.get("max_tokens").is_none(),
            "reasoning models must not send max_tokens"
        );
    }

    #[test]
    fn non_reasoning_deployment_uses_max_tokens() {
        let c = AzureClient::new("k".into())
            .with_model("gpt-4o")
            .with_max_tokens(Some(512));
        let req = c.build_request_body("gpt-4o", &[], &[], ReasoningConfig::default());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["max_tokens"], 512);
        assert!(json.get("max_completion_tokens").is_none());
    }

    #[test]
    fn tools_omitted_when_empty() {
        let c = AzureClient::new("k".into()).with_model("gpt-4o");
        let req = c.build_request_body("gpt-4o", &[], &[], ReasoningConfig::default());
        let s = serde_json::to_string(&req).unwrap();
        assert!(!s.contains("\"tools\""), "empty tools must be omitted: {s}");
    }

    #[test]
    fn tools_included_when_present() {
        let c = AzureClient::new("k".into()).with_model("gpt-4o");
        let tools = vec![ToolDefinition::new(
            "lookup",
            "look things up",
            serde_json::json!({"type":"object"}),
        )];
        let req = c.build_request_body("gpt-4o", &[], &tools, ReasoningConfig::default());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn request_sets_stream_and_include_usage() {
        let c = AzureClient::new("k".into()).with_model("gpt-4o");
        let req = c.build_request_body("gpt-4o", &[], &[], ReasoningConfig::default());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
    }

    #[test]
    fn request_model_is_the_deployment() {
        let c = AzureClient::new("k".into()).with_model("default");
        let req = c.build_request_body("custom-deploy", &[], &[], ReasoningConfig::default());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "custom-deploy");
    }

    #[test]
    fn reasoning_effort_emitted_for_reasoning_deployment() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::High);
        let c = AzureClient::new("k".into()).with_model("o3");
        let req = c.build_request_body("o3", &[], &[], cfg);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["reasoning_effort"], "high");
    }

    #[test]
    fn reasoning_effort_dropped_for_non_reasoning_deployment() {
        let cfg = ReasoningConfig::with_reasoning_effort(ReasoningLevel::High);
        let c = AzureClient::new("k".into()).with_model("gpt-4o");
        let req = c.build_request_body("gpt-4o", &[], &[], cfg);
        let s = serde_json::to_string(&req).unwrap();
        assert!(
            !s.contains("reasoning_effort"),
            "unsupported reasoning_effort must be dropped: {s}"
        );
    }

    #[test]
    fn does_not_stamp_cache_breakpoint() {
        // Azure caches automatically; the system message must stay a plain
        // string (no cache_control markers).
        let c = AzureClient::new("k".into()).with_model("gpt-4o");
        let req = c.build_request_body(
            "gpt-4o",
            &[Message::new(Role::System, "sys")],
            &[],
            ReasoningConfig::default(),
        );
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["messages"][0]["content"], "sys");
        assert!(
            !serde_json::to_string(&req)
                .unwrap()
                .contains("cache_control")
        );
    }

    // --- Reasoning / context helpers -------------------------------------

    #[test]
    fn reasoning_gating_by_deployment_name() {
        assert!(model_supports_reasoning("o3"));
        assert!(model_supports_reasoning("o4-mini"));
        assert!(model_supports_reasoning("o1-preview"));
        assert!(model_supports_reasoning("gpt-5"));
        assert!(model_supports_reasoning("gpt5-turbo"));
        assert!(model_supports_reasoning("my-o3-prod")); // operator-prefixed
        assert!(!model_supports_reasoning("gpt-4o"));
        assert!(!model_supports_reasoning("gpt-4o-mini"));
        assert!(!model_supports_reasoning("gpt-4.1"));
        assert!(!model_supports_reasoning("gpt-35-turbo"));
    }

    #[test]
    fn context_limit_resolves_from_deployment_name() {
        assert_eq!(context_limit_for_deployment("o3"), Some(200_000));
        assert_eq!(context_limit_for_deployment("o4-mini"), Some(200_000));
        assert_eq!(context_limit_for_deployment("gpt-4o"), Some(128_000));
        assert_eq!(
            context_limit_for_deployment("my-gpt-4o-prod"),
            Some(128_000)
        );
        assert_eq!(context_limit_for_deployment("gpt-4.1"), Some(1_000_000));
        assert_eq!(context_limit_for_deployment("gpt-35-turbo"), Some(16_384));
        // GPT-5 window varies by tier -> defer to the universal fallback.
        assert_eq!(context_limit_for_deployment("gpt-5"), None);
        assert_eq!(context_limit_for_deployment("totally-custom"), None);
    }

    #[test]
    fn max_context_tokens_folds_cap_and_override() {
        let c = AzureClient::new("k".into()).with_model("gpt-4o");
        assert_eq!(c.max_context_tokens(), Some(128_000));
        let capped = AzureClient::new("k".into())
            .with_model("gpt-4o")
            .with_max_context_tokens(Some(32_000));
        assert_eq!(capped.max_context_tokens(), Some(32_000));
    }

    #[tokio::test]
    async fn max_context_tokens_consults_model_override() {
        let c = AzureClient::new("k".into()).with_model("totally-custom");
        assert_eq!(c.max_context_tokens(), None);
        let observed = with_model_override("o3".into(), async { c.max_context_tokens() }).await;
        assert_eq!(observed, Some(200_000));
    }

    #[tokio::test]
    async fn list_models_returns_curated_table() {
        let c = AzureClient::new("k".into());
        let models = c.list_models().await.unwrap();
        assert!(!models.is_empty());
        let o3 = models.iter().find(|m| m.id == "o3").unwrap();
        assert!(o3.capabilities.reasoning);
        let gpt4o = models.iter().find(|m| m.id == "gpt-4o").unwrap();
        assert!(!gpt4o.capabilities.reasoning);
        assert!(gpt4o.capabilities.tools);
        let embed = models
            .iter()
            .find(|m| m.id == "text-embedding-3-large")
            .unwrap();
        assert!(embed.capabilities.embedding);
        assert!(!embed.capabilities.tools);
    }

    // --- Content-filter / api-version detection --------------------------

    #[test]
    fn detect_content_filter_names_categories_not_body() {
        let body = r#"{"error":{"code":"content_filter","message":"FLAGGED_USER_TEXT_DO_NOT_LEAK","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"hate":{"filtered":true,"severity":"high"},"sexual":{"filtered":false,"severity":"safe"}}}}}"#;
        let reason = detect_content_filter(body).expect("content filter detected");
        assert!(reason.to_lowercase().contains("content filter"));
        assert!(reason.contains("hate"), "filtered category named: {reason}");
        assert!(
            !reason.contains("FLAGGED_USER_TEXT_DO_NOT_LEAK"),
            "must not echo the raw body: {reason}"
        );
        assert!(
            !reason.contains("sexual"),
            "un-filtered categories must not be named: {reason}"
        );
    }

    #[test]
    fn detect_content_filter_none_for_other_errors() {
        assert!(detect_content_filter(r#"{"error":{"code":"invalid_api_key"}}"#).is_none());
        assert!(detect_content_filter("not json").is_none());
    }

    #[test]
    fn detect_invalid_api_version_by_code_and_message() {
        assert!(detect_invalid_api_version(
            r#"{"error":{"code":"invalid_api_version","message":"x"}}"#
        ));
        assert!(detect_invalid_api_version(
            r#"{"error":{"message":"Unsupported api-version '1999-01-01'"}}"#
        ));
        assert!(!detect_invalid_api_version(r#"{"error":{"code":"other"}}"#));
    }

    // --- Streaming (httpmock) --------------------------------------------

    #[tokio::test]
    async fn stream_completion_text_happy_path_with_cached_tokens() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST).path("/openai/v1/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let c = client_for(&server);
        let resp = c
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("stream ok");
        m.assert_calls(1);
        assert_eq!(resp.text, "Hi");
        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.cache_read_input_tokens, Some(4));
    }

    #[tokio::test]
    async fn stream_completion_tool_calls_happy_path() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/openai/v1/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_TOOLS_BODY);
        });
        let c = client_for(&server);
        let resp = c
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("stream ok");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_1");
        assert_eq!(resp.tool_calls[0].name, "lookup");
    }

    // --- Auth header selection -------------------------------------------

    #[tokio::test]
    async fn api_key_mode_sends_api_key_header() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/openai/v1/chat/completions")
                .header("api-key", "secret-key");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let c = client_for(&server);
        let _ = c
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
    async fn entra_mode_sends_bearer_token() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/openai/v1/chat/completions")
                .header("Authorization", "Bearer test-token");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let c = AzureClient::new(String::new())
            .with_base_url(server.url(""))
            .with_model("dep")
            .with_auth_mode(AuthMode::Entra)
            .with_token_provider(Arc::new(MockTokenProvider {
                token: "test-token".into(),
            }));
        let _ = c
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
    async fn entra_without_token_provider_errors_clearly() {
        let c = AzureClient::new(String::new())
            .with_base_url("https://foo.openai.azure.com")
            .with_model("dep")
            .with_auth_mode(AuthMode::Entra);
        let err = c
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("entra without provider must fail");
        assert!(err.to_string().contains("TokenProvider"), "{err}");
    }

    // --- MODEL_OVERRIDE routing ------------------------------------------

    #[tokio::test]
    async fn model_override_routes_deployment_in_v1_body() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/openai/v1/chat/completions")
                .body_includes(r#""model":"override-dep""#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let c = client_for(&server).with_model("default-dep");
        with_model_override("override-dep".into(), async {
            let _ = c
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
    async fn model_override_routes_deployment_in_classic_url() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/openai/deployments/override-dep/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let c = AzureClient::new("secret-key".into())
            .with_base_url(server.url(""))
            .with_model("default-dep")
            .with_api_surface(ApiSurface::Classic)
            .with_api_version("2024-10-21");
        with_model_override("override-dep".into(), async {
            let _ = c
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

    // --- Error paths ------------------------------------------------------

    #[tokio::test]
    async fn http_400_context_overflow_maps_to_context_overflow() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/openai/v1/chat/completions");
            then.status(400).header("content-type", "application/json").body(
                r#"{"error":{"code":"context_length_exceeded","type":"invalid_request_error","message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 153827 tokens."}}"#,
            );
        });
        let err = client_for(&server)
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("overflow must fail");
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
    async fn http_401_names_the_fix_without_leaking() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/openai/v1/chat/completions");
            then.status(401).header("content-type", "application/json").body(
                r#"{"error":{"code":"401","message":"Access denied due to invalid subscription key."}}"#,
            );
        });
        let err = client_for(&server)
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("401 must fail");
        let CoreError::Llm(detail) = err else {
            panic!("expected Llm, got {err:?}");
        };
        assert!(detail.contains("401"), "{detail}");
        assert!(
            detail.to_lowercase().contains("api-key"),
            "names the fix: {detail}"
        );
        assert!(
            !detail.contains("secret-key"),
            "must never echo the key: {detail}"
        );
    }

    #[tokio::test]
    async fn http_429_maps_to_rate_limited_with_retry_after() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/openai/v1/chat/completions");
            then.status(429)
                .header("content-type", "application/json")
                .header("retry-after", "20")
                .body(r#"{"error":{"code":"rate_limit_exceeded","message":"Rate limit reached"}}"#);
        });
        let err = client_for(&server)
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("429 must fail");
        match err {
            CoreError::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(Duration::from_secs(20)));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_400_content_filter_maps_to_clean_decline() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/openai/v1/chat/completions");
            then.status(400).header("content-type", "application/json").body(
                r#"{"error":{"code":"content_filter","message":"FLAGGED_USER_TEXT_DO_NOT_LEAK","innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"violence":{"filtered":true,"severity":"medium"}}}}}"#,
            );
        });
        let err = client_for(&server)
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("content filter must fail");
        let CoreError::Llm(detail) = err else {
            panic!("expected Llm decline, got {err:?}");
        };
        assert!(detail.contains("violence"), "names the filter: {detail}");
        assert!(
            !detail.contains("FLAGGED_USER_TEXT_DO_NOT_LEAK"),
            "must not echo the flagged body: {detail}"
        );
    }

    #[tokio::test]
    async fn http_500_maps_to_rate_limited() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/openai/v1/chat/completions");
            then.status(500).body("boom");
        });
        let err = client_for(&server)
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("500 must fail");
        assert!(matches!(err, CoreError::RateLimited { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn classic_invalid_api_version_names_the_fix() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST)
                .path("/openai/deployments/dep/chat/completions");
            then.status(400).header("content-type", "application/json").body(
                r#"{"error":{"code":"invalid_api_version","message":"The api-version '1999-01-01' is not supported."}}"#,
            );
        });
        let c = AzureClient::new("secret-key".into())
            .with_base_url(server.url(""))
            .with_model("dep")
            .with_api_surface(ApiSurface::Classic)
            .with_api_version("1999-01-01");
        let err = c
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("bad api-version must fail");
        let CoreError::Llm(detail) = err else {
            panic!("expected Llm, got {err:?}");
        };
        assert!(detail.contains("api_version"), "{detail}");
        assert!(detail.contains("1999-01-01"), "{detail}");
    }

    // --- Embeddings -------------------------------------------------------

    #[tokio::test]
    async fn embeddings_round_trip_via_v1_path() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(POST)
                .path("/openai/v1/embeddings")
                .header("api-key", "secret-key")
                .body_includes(r#""model":"text-embedding-3-small""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"data":[{"embedding":[0.1,0.2,0.3]},{"embedding":[0.4,0.5,0.6]}]}"#);
        });
        let c = AzureClient::new("secret-key".into())
            .with_base_url(server.url(""))
            .with_model("text-embedding-3-small");
        let vectors = c
            .embed(vec!["a".into(), "b".into()])
            .await
            .expect("embed ok");
        m.assert_calls(1);
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![0.1, 0.2, 0.3]);
    }

    // --- Preflight / unhappy paths ---------------------------------------

    #[tokio::test]
    async fn empty_base_url_errors_clearly() {
        let c = AzureClient::new("k".into()).with_model("dep");
        let err = c
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("missing base_url must fail");
        assert!(err.to_string().contains("resource endpoint"), "{err}");
    }

    #[tokio::test]
    async fn empty_deployment_errors_clearly() {
        let c = AzureClient::new("k".into()).with_base_url("https://foo.openai.azure.com");
        let err = c
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("missing deployment must fail");
        assert!(err.to_string().contains("deployment"), "{err}");
    }

    #[tokio::test]
    async fn embeddings_empty_deployment_errors_clearly() {
        let c = AzureClient::new("k".into()).with_base_url("https://foo.openai.azure.com");
        let err = c.embed(vec!["a".into()]).await.expect_err("must fail");
        assert!(err.to_string().contains("deployment"), "{err}");
    }

    // --- Cancellation -----------------------------------------------------

    #[tokio::test]
    async fn stream_aborts_on_cancellation() {
        use tokio_util::sync::CancellationToken;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/openai/v1/chat/completions");
            then.status(200)
                .header("content-type", "text/event-stream")
                .delay(Duration::from_secs(5))
                .body(STUB_SSE_BODY);
        });
        let c = client_for(&server);
        let token = CancellationToken::new();
        let cancel_handle = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_handle.cancel();
        });

        let start = std::time::Instant::now();
        let result = with_cancellation_token(token, async {
            c.stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
        })
        .await;
        assert!(
            matches!(result, Err(CoreError::Cancelled)),
            "got {result:?}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "cancellation must abort promptly"
        );
    }

    // --- from_env ---------------------------------------------------------

    #[test]
    fn from_env_reads_azure_vars() {
        // SAFETY: single-threaded test that owns the AZURE_OPENAI_* vars; no
        // other test in this binary reads or writes them.
        unsafe {
            std::env::remove_var("AZURE_OPENAI_API_KEY");
            std::env::remove_var("AZURE_OPENAI_MODEL");
            std::env::remove_var("AZURE_OPENAI_BASE_URL");
        }
        assert!(
            matches!(AzureClient::from_env(), Err(CoreError::Llm(_))),
            "missing key must error"
        );

        // SAFETY: same single-threaded ownership as above.
        unsafe {
            std::env::set_var("AZURE_OPENAI_API_KEY", "envkey");
            std::env::set_var("AZURE_OPENAI_MODEL", "env-dep");
            std::env::set_var("AZURE_OPENAI_BASE_URL", "https://env.openai.azure.com");
        }
        let c = AzureClient::from_env().expect("from_env ok");
        assert_eq!(c.model, "env-dep");
        assert_eq!(c.base_url, "https://env.openai.azure.com");

        // SAFETY: clean up so a later run starts fresh.
        unsafe {
            std::env::remove_var("AZURE_OPENAI_API_KEY");
            std::env::remove_var("AZURE_OPENAI_MODEL");
            std::env::remove_var("AZURE_OPENAI_BASE_URL");
        }
    }

    // --- Redaction --------------------------------------------------------

    #[test]
    fn debug_redacts_api_key() {
        let secret = "azure-supersecret-DO-NOT-LEAK-0123456789";
        let c = AzureClient::new(secret.into())
            .with_model("gpt-4o")
            .with_base_url("https://foo.openai.azure.com");
        let rendered = format!("{c:?}");
        assert!(!rendered.contains(secret), "raw key leaked: {rendered}");
        assert!(
            !rendered.contains("supersecret"),
            "key substring leaked: {rendered}"
        );
        assert!(
            rendered.contains("redacted"),
            "key must be present-but-redacted"
        );
        assert!(
            rendered.contains("gpt-4o"),
            "non-secret fields stay visible"
        );
    }

    // --- TTL model cache (issue #620) -----------------------------------

    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    /// Mock clock: an atomic second-offset from a fixed origin, advanced by the
    /// tests so TTL behaviour is deterministic (no sleeping, no real network).
    struct MockClock {
        origin: Instant,
        offset_secs: AtomicU64,
    }

    impl MockClock {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                origin: Instant::now(),
                offset_secs: AtomicU64::new(0),
            })
        }
        fn advance_secs(&self, secs: u64) {
            self.offset_secs.fetch_add(secs, Ordering::SeqCst);
        }
    }

    impl Clock for MockClock {
        fn now(&self) -> Instant {
            self.origin + Duration::from_secs(self.offset_secs.load(Ordering::SeqCst))
        }
    }

    #[tokio::test]
    async fn list_models_serves_curated_through_the_cache() {
        // Azure has no live listing endpoint yet (deployment enumeration is an
        // ARM control-plane operation), so the cache always serves the curated
        // table. Exercise the cached path with an injected clock: the result is
        // stable within the TTL and rebuilt (identically) once it lapses.
        let clock = MockClock::new();
        let expected = merge_curated_with_live(curated_azure_models(), Vec::new());
        let client = AzureClient::new("k".into())
            .with_clock(clock.clone())
            .with_model_cache_ttl(Duration::from_secs(3600));

        assert_eq!(client.list_models().await.unwrap(), expected); // fills cache
        clock.advance_secs(30 * 60);
        assert_eq!(client.list_models().await.unwrap(), expected); // within TTL
        clock.advance_secs(3600);
        assert_eq!(client.list_models().await.unwrap(), expected); // past TTL, rebuilt
    }

    #[tokio::test]
    async fn refresh_models_returns_curated_table() {
        let client = AzureClient::new("k".into());
        let expected = merge_curated_with_live(curated_azure_models(), Vec::new());
        assert_eq!(client.refresh_models().await.unwrap(), expected);
    }

    // --- Non-streaming fallback + per-model memo (#619) ------------------

    const TOOLS_UNSUPPORTED_BODY: &str = r#"{"error":{"code":"invalid_request_error","type":"invalid_request_error","message":"This model does not support tool use in streaming mode. Disable streaming to use tools."}}"#;

    const NONSTREAMING_TOOL_RESPONSE: &str = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"On it","tool_calls":[{"id":"call_9","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"rust\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":11,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}"#;

    fn a_tool() -> ToolDefinition {
        ToolDefinition::new(
            "lookup",
            "look things up",
            serde_json::json!({"type":"object"}),
        )
    }

    fn streaming_rejects_tools(server: &MockServer) -> httpmock::Mock<'_> {
        server.mock(|when, then| {
            when.method(POST)
                .path("/openai/v1/chat/completions")
                .body_includes(r#""stream":true"#);
            then.status(400)
                .header("content-type", "application/json")
                .body(TOOLS_UNSUPPORTED_BODY);
        })
    }

    fn non_streaming_returns<'a>(
        server: &'a MockServer,
        status: u16,
        body: &str,
    ) -> httpmock::Mock<'a> {
        let owned = body.to_string();
        server.mock(move |when, then| {
            when.method(POST)
                .path("/openai/v1/chat/completions")
                .body_includes(r#""stream":false"#);
            then.status(status)
                .header("content-type", "application/json")
                .body(&owned);
        })
    }

    #[tokio::test]
    async fn tools_unsupported_streaming_falls_back_to_non_streaming() {
        let server = MockServer::start();
        let stream_mock = streaming_rejects_tools(&server);
        let ns_mock = non_streaming_returns(&server, 200, NONSTREAMING_TOOL_RESPONSE);
        let c = client_for(&server);

        let received = Arc::new(std::sync::Mutex::new(String::new()));
        let rc = Arc::clone(&received);
        let resp = c
            .stream_completion(
                vec![Message::new(Role::User, "find rust")],
                &[a_tool()],
                ReasoningConfig::default(),
                Box::new(move |ch| {
                    rc.lock().expect("lock").push_str(&ch);
                    true
                }),
            )
            .await
            .expect("non-streaming fallback ok");

        stream_mock.assert_calls(1);
        ns_mock.assert_calls(1);
        assert_eq!(resp.text, "On it");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "lookup");
        assert_eq!(resp.tool_calls[0].arguments, r#"{"q":"rust"}"#);
        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.input_tokens, Some(11));
        assert_eq!(usage.cache_read_input_tokens, Some(2));
        assert_eq!(*received.lock().expect("lock"), "On it");
    }

    #[tokio::test]
    async fn memo_skips_streaming_on_the_second_call() {
        let server = MockServer::start();
        let stream_mock = streaming_rejects_tools(&server);
        let ns_mock = non_streaming_returns(&server, 200, NONSTREAMING_TOOL_RESPONSE);
        let c = client_for(&server);

        for _ in 0..2 {
            c.stream_completion(
                vec![Message::new(Role::User, "find rust")],
                &[a_tool()],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect("ok");
        }

        stream_mock.assert_calls(1);
        ns_mock.assert_calls(2);
    }

    #[tokio::test]
    async fn model_without_error_stays_on_streaming() {
        let server = MockServer::start();
        let stream_mock = server.mock(|when, then| {
            when.method(POST)
                .path("/openai/v1/chat/completions")
                .body_includes(r#""stream":true"#);
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(STUB_SSE_BODY);
        });
        let ns_mock = non_streaming_returns(&server, 200, NONSTREAMING_TOOL_RESPONSE);
        let c = client_for(&server);

        for _ in 0..2 {
            let resp = c
                .stream_completion(
                    vec![Message::new(Role::User, "hi")],
                    &[a_tool()],
                    ReasoningConfig::default(),
                    Box::new(|_| true),
                )
                .await
                .expect("stream ok");
            assert_eq!(resp.text, "Hi");
        }

        stream_mock.assert_calls(2);
        ns_mock.assert_calls(0);
    }

    #[tokio::test]
    async fn non_streaming_failure_surfaces_without_looping() {
        let server = MockServer::start();
        let stream_mock = streaming_rejects_tools(&server);
        let ns_mock = non_streaming_returns(&server, 500, "upstream boom");
        let c = client_for(&server);

        let err = c
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[a_tool()],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("non-streaming failure must surface");

        stream_mock.assert_calls(1);
        ns_mock.assert_calls(1);
        assert!(
            matches!(err, CoreError::RateLimited { .. }),
            "500 must map to RateLimited; got {err:?}"
        );
    }

    #[tokio::test]
    async fn cancellation_honoured_on_non_streaming_path() {
        let server = MockServer::start();
        streaming_rejects_tools(&server);
        server.mock(|when, then| {
            when.method(POST)
                .path("/openai/v1/chat/completions")
                .body_includes(r#""stream":false"#);
            then.status(200)
                .header("content-type", "application/json")
                .delay(Duration::from_secs(5))
                .body(NONSTREAMING_TOOL_RESPONSE);
        });
        let c = client_for(&server);
        let token = tokio_util::sync::CancellationToken::new();
        let handle = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            handle.cancel();
        });

        let start = std::time::Instant::now();
        let result = with_cancellation_token(token, async {
            c.stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[a_tool()],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
        })
        .await;
        let elapsed = start.elapsed();

        assert!(
            matches!(result, Err(CoreError::Cancelled)),
            "expected Cancelled on the non-streaming path, got {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "cancellation must abort the non-streaming retry promptly; took {elapsed:?}"
        );
    }
}
