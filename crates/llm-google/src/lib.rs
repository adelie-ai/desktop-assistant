//! Google Vertex AI (Gemini) connector implementing the core `LlmClient` port.
//!
//! Targets the native Gemini `generateContent` schema on two surfaces that
//! speak the identical wire format and differ only in host, path prefix, and
//! auth:
//!
//! - **Vertex AI** (`auth_mode = vertex`, the default): project/region-scoped,
//!   OAuth2-bearer authenticated (a [`TokenProvider`] mints the token from a
//!   GCP service account). URL:
//!   `POST {base}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse`
//!   with `base = https://{location}-aiplatform.googleapis.com`.
//! - **Gemini API / AI Studio** (`auth_mode = api_key`): a single API key sent
//!   in the `x-goog-api-key` header (never `?key=` in the URL). URL:
//!   `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse`.
//!
//! Structural reference: `llm-bedrock` (the other custom-schema connector).
//! Shared streaming/model/error primitives come from `llm-http`.

mod convert;
mod errors;
mod models;
mod schema;
mod token;
mod wire;

pub use models::{context_limit_for_model, curated_gemini_models, model_supports_thinking};
pub use token::{
    CLOUD_PLATFORM_SCOPE, NoopTokenProvider, ServiceAccountKey, ServiceAccountTokenProvider,
    StaticTokenProvider, TokenProvider,
};

use std::sync::Arc;
use std::time::{Duration, Instant};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition};
use desktop_assistant_core::ports::embedding::EmbeddingClient;
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelInfo, ReasoningConfig, TokenUsage,
    ToolCallAccumulator, current_cancellation_token, current_model_override,
};
use desktop_assistant_llm_http::{
    STREAM_CONNECT_TIMEOUT, STREAM_EVENT_TIMEOUT, StreamStep, apply_context_cap, bail_for_status,
    build_response, merge_curated_with_live, next_step,
};
use eventsource_stream::Eventsource;
use serde::Deserialize;
use tokio::sync::Mutex;

/// Built-in default chat model.
const DEFAULT_MODEL: &str = "gemini-2.5-pro";
/// Cheaper backend default (exposed for the daemon's `default_backend_chat_model`).
const DEFAULT_BACKEND_MODEL: &str = "gemini-2.5-flash";
/// Default Vertex region.
const DEFAULT_LOCATION: &str = "us-central1";
/// Default embedding model (served on both surfaces).
const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-004";
/// Composed default base URL for Vertex at [`DEFAULT_LOCATION`].
const DEFAULT_BASE_URL: &str = "https://us-central1-aiplatform.googleapis.com";
/// Host for the Gemini API (AI Studio) surface.
const GEMINI_API_HOST: &str = "https://generativelanguage.googleapis.com";

/// Vertex REST API version.
///
/// Pinned to `v1` (the live GA surface). `thinkingConfig` is GA on Vertex `v1`
/// for the Gemini 2.5 family, so it does not force `v1beta1`. Kept as a single
/// constant so a future preview feature that requires `v1beta1` is a one-line
/// bump rather than a scattered change.
const VERTEX_API_VERSION: &str = "v1";
/// Gemini API (AI Studio) version; this surface uses the `v1beta` prefix, which
/// is where AI Studio exposes `thinkingConfig`.
const GEMINI_API_VERSION: &str = "v1beta";

/// Default TTL for the `list_models()` cache (mirrors the Bedrock connector).
const DEFAULT_MODEL_CACHE_TTL: Duration = Duration::from_secs(60 * 60);

/// Which surface (and therefore host + auth) this client targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AuthMode {
    /// Vertex AI: project/region-scoped, OAuth2 bearer.
    #[default]
    Vertex,
    /// Gemini API (AI Studio): single API key in the `x-goog-api-key` header.
    ApiKey,
}

impl AuthMode {
    /// Canonical lowercase string (matches the `daemon.toml` `auth_mode` value).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::ApiKey => "api_key",
        }
    }

    /// Parse from config text. Unknown values default to `Vertex` (the decided
    /// Google target), so a typo fails on Vertex preflight (missing project /
    /// credential) rather than silently switching auth mechanism.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "api_key" | "apikey" | "gemini" | "aistudio" => Self::ApiKey,
            _ => Self::Vertex,
        }
    }
}

/// Abstraction over `Instant::now()` so the model-cache TTL test can advance
/// time without sleeping. Mirrors `llm-bedrock::ModelClock`.
pub trait ModelClock: Send + Sync {
    fn now(&self) -> Instant;
}

/// Default clock reading the monotonic OS clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl ModelClock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[derive(Default)]
struct ModelCache {
    entry: Option<(Instant, Vec<ModelInfo>)>,
}

/// Vertex AI / Gemini connector.
pub struct GoogleClient {
    http: reqwest::Client,
    model: String,
    /// Explicit base-URL override; when `None` the URL is composed from
    /// `location` (Vertex) or the fixed Gemini API host (api-key mode).
    base_url: Option<String>,
    /// API key; used only in [`AuthMode::ApiKey`].
    api_key: String,
    auth_mode: AuthMode,
    project: Option<String>,
    location: String,
    credentials_path: Option<String>,
    token_provider: Arc<dyn TokenProvider>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    connect_timeout: Duration,
    event_timeout: Duration,
    context_cap: Option<u64>,
    model_cache: Arc<Mutex<ModelCache>>,
    model_cache_ttl: Duration,
    clock: Arc<dyn ModelClock>,
}

/// Redacting `Debug`: the API key renders as its length only, and the token
/// provider / clock / model cache render as opaque placeholders so no
/// credential material can leak through a `{:?}`.
impl std::fmt::Debug for GoogleClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleClient")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field(
                "api_key",
                &format_args!("<redacted; len={}>", self.api_key.len()),
            )
            .field("auth_mode", &self.auth_mode)
            .field("project", &self.project)
            .field("location", &self.location)
            .field("credentials_path", &self.credentials_path)
            .field("token_provider", &format_args!("<dyn TokenProvider>"))
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("max_tokens", &self.max_tokens)
            .field("connect_timeout", &self.connect_timeout)
            .field("event_timeout", &self.event_timeout)
            .field("context_cap", &self.context_cap)
            .field("model_cache_ttl", &self.model_cache_ttl)
            .finish()
    }
}

impl GoogleClient {
    /// Built-in default chat model.
    pub fn get_default_model() -> Option<&'static str> {
        Some(DEFAULT_MODEL)
    }

    /// Cheaper backend default chat model.
    pub fn get_default_backend_model() -> Option<&'static str> {
        Some(DEFAULT_BACKEND_MODEL)
    }

    /// Composed default base URL (Vertex at [`DEFAULT_LOCATION`]).
    pub fn get_default_base_url() -> Option<&'static str> {
        Some(DEFAULT_BASE_URL)
    }

    /// Default embedding model.
    pub fn get_default_embedding_model() -> Option<&'static str> {
        Some(DEFAULT_EMBEDDING_MODEL)
    }

    /// Construct a Vertex client with default model/location. `api_key` is used
    /// only in [`AuthMode::ApiKey`]; in the default Vertex mode a credential is
    /// supplied via [`Self::with_credentials_path`] / [`Self::with_token_provider`].
    pub fn new(api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            model: DEFAULT_MODEL.to_string(),
            base_url: None,
            api_key,
            auth_mode: AuthMode::Vertex,
            project: None,
            location: DEFAULT_LOCATION.to_string(),
            credentials_path: None,
            token_provider: Arc::new(NoopTokenProvider),
            temperature: None,
            top_p: None,
            max_tokens: None,
            connect_timeout: STREAM_CONNECT_TIMEOUT,
            event_timeout: STREAM_EVENT_TIMEOUT,
            context_cap: None,
            model_cache: Arc::new(Mutex::new(ModelCache::default())),
            model_cache_ttl: DEFAULT_MODEL_CACHE_TTL,
            clock: Arc::new(SystemClock),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the base URL. Empty/whitespace clears the override (falls back
    /// to composing from `location`).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let b = base_url.into();
        self.base_url = if b.trim().is_empty() { None } else { Some(b) };
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
    /// `Some(0)` = "max available". Folded with the curated table in
    /// [`LlmClient::max_context_tokens`].
    pub fn with_max_context_tokens(mut self, max: Option<u64>) -> Self {
        self.context_cap = max.filter(|m| *m > 0);
        self
    }

    /// Set the GCP project (Vertex).
    pub fn with_project(mut self, project: Option<String>) -> Self {
        self.project = project.filter(|p| !p.trim().is_empty());
        self
    }

    /// Set the Vertex location/region (e.g. `us-central1` or `global`). Empty
    /// values keep the default.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        let l = location.into();
        if !l.trim().is_empty() {
            self.location = l;
        }
        self
    }

    /// Select the auth surface. Does not overload `new(api_key)`.
    pub fn with_auth_mode(mut self, mode: AuthMode) -> Self {
        self.auth_mode = mode;
        self
    }

    /// Point Vertex at a service-account JSON key file. Installs a
    /// [`ServiceAccountTokenProvider`] over that path. `None`/empty leaves the
    /// current provider in place.
    pub fn with_credentials_path(mut self, path: Option<String>) -> Self {
        if let Some(p) = path.filter(|p| !p.trim().is_empty()) {
            self.token_provider = Arc::new(ServiceAccountTokenProvider::from_credentials_path(
                p.clone(),
            ));
            self.credentials_path = Some(p);
        }
        self
    }

    /// Inject a [`TokenProvider`] directly (the test seam; also usable when an
    /// outer layer already resolves ADC).
    pub fn with_token_provider(mut self, provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = provider;
        self
    }

    /// Override the `list_models()` cache TTL (default 1h).
    pub fn with_model_cache_ttl(mut self, ttl: Duration) -> Self {
        self.model_cache_ttl = ttl;
        self
    }

    /// Inject a custom clock for deterministic cache-TTL tests.
    pub fn with_clock(mut self, clock: Arc<dyn ModelClock>) -> Self {
        self.clock = clock;
        self
    }

    /// Return the embedding model id as the stable version identifier.
    pub async fn model_identifier(&self) -> Result<String, CoreError> {
        Ok(self.model.clone())
    }

    /// Build `generationConfig` for `model` and the per-turn reasoning hint.
    /// Returns `None` when no knob is set so the field is omitted entirely.
    /// `thinkingConfig` is included only when a positive budget is requested
    /// **and** the model is thinking-capable (Gemini 2.5+).
    pub(crate) fn build_generation_config(
        &self,
        model: &str,
        reasoning: ReasoningConfig,
    ) -> Option<wire::GenerationConfig> {
        let mut cfg = wire::GenerationConfig {
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_tokens,
            thinking_config: None,
        };
        if let Some(budget) = reasoning.thinking_budget_tokens
            && budget > 0
            && model_supports_thinking(model)
        {
            cfg.thinking_config = Some(wire::ThinkingConfig {
                thinking_budget: budget,
                // Keep thought traces out of the user-visible stream in v1;
                // `thoughtsTokenCount` still reports the reasoning spend.
                include_thoughts: false,
            });
        }
        (!cfg.is_empty()).then_some(cfg)
    }

    /// The effective base URL: an explicit override, else the Gemini API host
    /// (api-key mode) or the region-composed Vertex host.
    fn effective_base_url(&self) -> String {
        if let Some(base) = &self.base_url {
            return base.trim_end_matches('/').to_string();
        }
        match self.auth_mode {
            AuthMode::ApiKey => GEMINI_API_HOST.to_string(),
            AuthMode::Vertex => format!("https://{}-aiplatform.googleapis.com", self.location),
        }
    }

    /// The Vertex project, or a specific `Unavailable`-style error naming the fix.
    fn require_project(&self) -> Result<&str, CoreError> {
        self.project
            .as_deref()
            .filter(|p| !p.is_empty())
            .ok_or_else(|| {
                CoreError::Llm(
                    "Vertex needs a GCP project; set GOOGLE_CLOUD_PROJECT or project= in the \
                     connection config"
                        .into(),
                )
            })
    }

    /// The `:streamGenerateContent?alt=sse` URL for `model` on this surface.
    fn stream_url(&self, model: &str) -> Result<String, CoreError> {
        let base = self.effective_base_url();
        match self.auth_mode {
            AuthMode::Vertex => {
                let project = self.require_project()?;
                Ok(format!(
                    "{base}/{VERTEX_API_VERSION}/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse",
                    location = self.location,
                ))
            }
            AuthMode::ApiKey => Ok(format!(
                "{base}/{GEMINI_API_VERSION}/models/{model}:streamGenerateContent?alt=sse"
            )),
        }
    }

    /// The model-listing URL for this surface.
    fn models_list_url(&self) -> String {
        let base = self.effective_base_url();
        match self.auth_mode {
            AuthMode::Vertex => format!("{base}/{VERTEX_API_VERSION}/publishers/google/models"),
            AuthMode::ApiKey => format!("{base}/{GEMINI_API_VERSION}/models"),
        }
    }

    /// The embeddings URL for `self.model` on this surface.
    fn embed_url(&self) -> Result<String, CoreError> {
        let base = self.effective_base_url();
        match self.auth_mode {
            AuthMode::Vertex => {
                let project = self.require_project()?;
                Ok(format!(
                    "{base}/{VERTEX_API_VERSION}/projects/{project}/locations/{location}/publishers/google/models/{model}:predict",
                    location = self.location,
                    model = self.model,
                ))
            }
            AuthMode::ApiKey => Ok(format!(
                "{base}/{GEMINI_API_VERSION}/models/{model}:embedContent",
                model = self.model
            )),
        }
    }

    /// Attach the surface-appropriate auth to a request. Vertex adds a bearer
    /// token from the [`TokenProvider`]; api-key mode adds the `x-goog-api-key`
    /// header. The credential travels in a header only, never the URL.
    async fn authorize(
        &self,
        req: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, CoreError> {
        match self.auth_mode {
            AuthMode::Vertex => {
                let token = self.token_provider.token().await?;
                Ok(req.bearer_auth(token))
            }
            AuthMode::ApiKey => Ok(req.header("x-goog-api-key", &self.api_key)),
        }
    }

    /// Send a `generateContent` request and parse the SSE stream into an
    /// [`LlmResponse`]. Reuses the shared connect-race / stall-loop scaffolding.
    async fn send_and_stream(
        &self,
        model: &str,
        request: &wire::GenerateContentRequest,
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let cancellation = current_cancellation_token().unwrap_or_default();
        if cancellation.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let url = self.stream_url(model)?;
        let request_json = serde_json::to_string(request)
            .map_err(|e| CoreError::Llm(format!("failed to serialize request: {e}")))?;
        tracing::info!(request_bytes = request_json.len(), model = %model, "LLM request payload");

        let req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .body(request_json);
        let req = self.authorize(req).await?;

        let send_fut = req.send();
        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err(CoreError::Cancelled),
            _ = tokio::time::sleep(self.connect_timeout) => {
                tracing::error!(
                    timeout_s = self.connect_timeout.as_secs(),
                    "Google request send() timed out (no response headers)"
                );
                return Err(CoreError::Llm("Google Vertex stream stalled (no response headers)".into()));
            }
            r = send_fut => r.map_err(|e| CoreError::Llm(format!("HTTP request failed: {e}")))?,
        };

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".into());
            return Err(errors::classify_http_error(status, &headers, &body));
        }

        let mut events = response.bytes_stream().eventsource();
        let mut text = String::new();
        let mut tool_acc: ToolCallAccumulator<i64> = ToolCallAccumulator::new();
        let mut fn_index: i64 = 0;
        let mut usage: Option<TokenUsage> = None;

        'outer: loop {
            let event = match next_step(&mut events, &cancellation, self.event_timeout).await {
                StreamStep::Item(ev) => ev,
                StreamStep::Done => break,
                StreamStep::Cancelled => {
                    drop(events);
                    return Err(CoreError::Cancelled);
                }
                StreamStep::Stalled => {
                    drop(events);
                    return Err(CoreError::Llm(
                        "Google Vertex stream stalled (no events)".into(),
                    ));
                }
            };
            let event = event.map_err(|e| CoreError::Llm(format!("stream read error: {e}")))?;
            let data = event.data.as_str();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let frame: wire::GenerateContentResponse = match serde_json::from_str(data) {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!("failed to parse Gemini SSE frame: {e}");
                    continue;
                }
            };

            // A safety block is a business decline: surface an informative,
            // non-retryable reason and never dump the flagged body.
            if let Some(category) = errors::safety_block(&frame) {
                tracing::info!(category = %category, "Gemini declined the request via its safety filter");
                drop(events);
                return Err(errors::safety_decline_error(&category));
            }

            if let Some(u) = &frame.usage_metadata {
                usage = Some(map_usage(u));
            }

            for candidate in frame.candidates {
                let Some(content) = candidate.content else {
                    continue;
                };
                for part in content.parts {
                    // Keep thinking-trace parts out of the user-visible stream.
                    if part.thought {
                        continue;
                    }
                    if let Some(chunk) = part.text {
                        if chunk.is_empty() {
                            continue;
                        }
                        text.push_str(&chunk);
                        // Break (not return) on abort so accumulated tool calls
                        // and usage are still assembled below.
                        if !on_chunk(chunk) {
                            tracing::debug!("Google stream aborted by callback");
                            break 'outer;
                        }
                    } else if let Some(call) = part.function_call {
                        // Each functionCall part is a whole call; register it as
                        // start+finalize keyed by a running index.
                        let key = fn_index;
                        fn_index += 1;
                        let id = format!("call_{key}");
                        let args =
                            serde_json::to_string(&call.args).unwrap_or_else(|_| "{}".to_string());
                        tool_acc.start(key, id, call.name);
                        tool_acc.finalize(key, args);
                    }
                }
            }
        }

        Ok(build_response(text, tool_acc.into_tool_calls(), usage))
    }

    // --- Model listing --------------------------------------------------

    async fn list_models_cached(&self) -> Result<Vec<ModelInfo>, CoreError> {
        {
            let cache = self.model_cache.lock().await;
            if let Some((fetched_at, entry)) = cache.entry.as_ref()
                && self.clock.now().saturating_duration_since(*fetched_at) < self.model_cache_ttl
            {
                return Ok(entry.clone());
            }
        }
        self.refresh_models_internal().await
    }

    async fn refresh_models_internal(&self) -> Result<Vec<ModelInfo>, CoreError> {
        let fresh = self.fetch_models().await;
        let now = self.clock.now();
        let mut cache = self.model_cache.lock().await;
        cache.entry = Some((now, fresh.clone()));
        Ok(fresh)
    }

    /// Curated table merged with a best-effort live listing. A live-listing
    /// failure degrades to the curated table rather than surfacing an error.
    async fn fetch_models(&self) -> Vec<ModelInfo> {
        let live = match self.fetch_live_models().await {
            Ok(live) => live,
            Err(e) => {
                tracing::debug!("Google live model listing failed; using curated table: {e}");
                Vec::new()
            }
        };
        merge_curated_with_live(curated_gemini_models(), live)
    }

    async fn fetch_live_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        let url = self.models_list_url();
        let req = self.authorize(self.http.get(&url)).await?;
        let response = req
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("Google list-models request failed: {e}")))?;
        let response = bail_for_status(response, "Google list-models").await?;
        let listing: ModelListing = response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse Google model listing: {e}")))?;
        Ok(listing.into_model_infos())
    }

    // --- Embeddings -----------------------------------------------------

    async fn embed_impl(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        match self.auth_mode {
            AuthMode::Vertex => self.embed_vertex(texts).await,
            AuthMode::ApiKey => self.embed_gemini_api(texts).await,
        }
    }

    /// Vertex `:predict` embeddings: one batched `instances` request.
    async fn embed_vertex(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        let url = self.embed_url()?;
        let instances: Vec<serde_json::Value> = texts
            .iter()
            .map(|t| serde_json::json!({ "content": t }))
            .collect();
        let body = serde_json::json!({ "instances": instances });
        let req = self.authorize(self.http.post(&url).json(&body)).await?;
        let response = req
            .send()
            .await
            .map_err(|e| CoreError::Llm(format!("Vertex embeddings request failed: {e}")))?;
        let response = bail_for_status(response, "Vertex embeddings").await?;
        let parsed: VertexEmbedResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Llm(format!("failed to parse Vertex embeddings: {e}")))?;
        Ok(parsed
            .predictions
            .into_iter()
            .map(|p| p.embeddings.values)
            .collect())
    }

    /// Gemini API `:embedContent` embeddings: one request per text.
    async fn embed_gemini_api(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        let url = self.embed_url()?;
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let body = serde_json::json!({
                "model": format!("models/{}", self.model),
                "content": { "parts": [ { "text": text } ] },
            });
            let req = self.authorize(self.http.post(&url).json(&body)).await?;
            let response = req.send().await.map_err(|e| {
                CoreError::Llm(format!("Gemini API embeddings request failed: {e}"))
            })?;
            let response = bail_for_status(response, "Gemini API embeddings").await?;
            let parsed: GeminiEmbedResponse = response.json().await.map_err(|e| {
                CoreError::Llm(format!("failed to parse Gemini API embeddings: {e}"))
            })?;
            out.push(parsed.embedding.values);
        }
        Ok(out)
    }
}

/// Map a Gemini `usageMetadata` to the core `TokenUsage`.
fn map_usage(u: &wire::UsageMetadata) -> TokenUsage {
    if let Some(thoughts) = u.thoughts_token_count {
        tracing::debug!(
            thoughts_token_count = thoughts,
            "Gemini reasoning-token usage"
        );
    }
    TokenUsage {
        input_tokens: u.prompt_token_count,
        output_tokens: u.candidates_token_count,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: u.cached_content_token_count,
    }
}

// --- Live model listing (lenient, both surfaces) ---------------------------

#[derive(Deserialize, Default)]
struct ModelListing {
    #[serde(default)]
    models: Vec<LiveModel>,
    #[serde(default, rename = "publisherModels")]
    publisher_models: Vec<LiveModel>,
}

#[derive(Deserialize)]
struct LiveModel {
    #[serde(default)]
    name: String,
}

impl ModelListing {
    /// Reduce a listing to bare `ModelInfo`s (id only); curated metadata wins
    /// on merge. The `name` is `models/<id>` or `publishers/google/models/<id>`,
    /// so the id is its last path segment.
    fn into_model_infos(self) -> Vec<ModelInfo> {
        self.models
            .into_iter()
            .chain(self.publisher_models)
            .filter_map(|m| {
                let id = m.name.rsplit('/').next().unwrap_or_default().to_string();
                (!id.is_empty()).then(|| ModelInfo::new(id))
            })
            .collect()
    }
}

// --- Embedding response shapes ---------------------------------------------

#[derive(Deserialize)]
struct VertexEmbedResponse {
    #[serde(default)]
    predictions: Vec<VertexPrediction>,
}

#[derive(Deserialize)]
struct VertexPrediction {
    embeddings: EmbeddingValues,
}

#[derive(Deserialize)]
struct GeminiEmbedResponse {
    embedding: EmbeddingValues,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    #[serde(default)]
    values: Vec<f32>,
}

#[async_trait::async_trait]
impl LlmClient for GoogleClient {
    fn get_default_model(&self) -> Option<&str> {
        Self::get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        Self::get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        let model = current_model_override().unwrap_or_else(|| self.model.clone());
        apply_context_cap(self.context_cap, context_limit_for_model(&model))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.list_models_cached().await
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.refresh_models_internal().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        // Per-turn model override (issue #34) drives the URL path segment.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());
        let (system_instruction, contents) = convert::convert_messages(&messages);
        let tools_wire = convert::build_tools(tools);
        let generation_config = self.build_generation_config(&model, reasoning);

        let request = wire::GenerateContentRequest {
            contents,
            system_instruction,
            tools: tools_wire,
            generation_config,
        };
        self.send_and_stream(&model, &request, on_chunk).await
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for GoogleClient {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        self.embed_impl(texts).await
    }

    async fn model_identifier(&self) -> Result<String, CoreError> {
        GoogleClient::model_identifier(self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_mode_parse_and_as_str() {
        assert_eq!(AuthMode::parse("vertex"), AuthMode::Vertex);
        assert_eq!(AuthMode::parse("api_key"), AuthMode::ApiKey);
        assert_eq!(AuthMode::parse("apikey"), AuthMode::ApiKey);
        assert_eq!(AuthMode::parse("gemini"), AuthMode::ApiKey);
        // Unknown defaults to Vertex.
        assert_eq!(AuthMode::parse("nonsense"), AuthMode::Vertex);
        assert_eq!(AuthMode::Vertex.as_str(), "vertex");
        assert_eq!(AuthMode::ApiKey.as_str(), "api_key");
    }

    #[test]
    fn defaults() {
        let c = GoogleClient::new(String::new());
        assert_eq!(c.model, DEFAULT_MODEL);
        assert_eq!(c.location, DEFAULT_LOCATION);
        assert_eq!(c.auth_mode, AuthMode::Vertex);
        assert_eq!(GoogleClient::get_default_model(), Some("gemini-2.5-pro"));
        assert_eq!(
            GoogleClient::get_default_base_url(),
            Some("https://us-central1-aiplatform.googleapis.com")
        );
        assert_eq!(
            GoogleClient::get_default_embedding_model(),
            Some("text-embedding-004")
        );
    }

    #[test]
    fn builder_sets_fields() {
        let c = GoogleClient::new("k".into())
            .with_model("gemini-2.5-flash")
            .with_project(Some("proj-1".into()))
            .with_location("europe-west4")
            .with_auth_mode(AuthMode::ApiKey)
            .with_temperature(Some(0.5))
            .with_top_p(Some(0.9))
            .with_max_tokens(Some(2048));
        assert_eq!(c.model, "gemini-2.5-flash");
        assert_eq!(c.project.as_deref(), Some("proj-1"));
        assert_eq!(c.location, "europe-west4");
        assert_eq!(c.auth_mode, AuthMode::ApiKey);
        assert_eq!(c.temperature, Some(0.5));
        assert_eq!(c.top_p, Some(0.9));
        assert_eq!(c.max_tokens, Some(2048));
    }

    #[test]
    fn connect_and_event_timeout_ignore_zero_and_none() {
        let base = GoogleClient::new("k".into());
        assert_eq!(base.connect_timeout, STREAM_CONNECT_TIMEOUT);
        assert_eq!(base.event_timeout, STREAM_EVENT_TIMEOUT);

        let over = GoogleClient::new("k".into())
            .with_connect_timeout(Some(3))
            .with_event_timeout(Some(7));
        assert_eq!(over.connect_timeout, Duration::from_secs(3));
        assert_eq!(over.event_timeout, Duration::from_secs(7));

        let zero = GoogleClient::new("k".into())
            .with_connect_timeout(Some(0))
            .with_event_timeout(Some(0));
        assert_eq!(zero.connect_timeout, STREAM_CONNECT_TIMEOUT);
        assert_eq!(zero.event_timeout, STREAM_EVENT_TIMEOUT);
    }

    #[test]
    fn max_context_tokens_folds_cap_with_curated_window() {
        // Known model, no cap -> curated window.
        let c = GoogleClient::new("k".into()).with_model("gemini-2.5-pro");
        assert_eq!(c.max_context_tokens(), Some(1_048_576));
        // Cap below the window clamps down.
        let capped = GoogleClient::new("k".into())
            .with_model("gemini-2.5-pro")
            .with_max_context_tokens(Some(200_000));
        assert_eq!(capped.max_context_tokens(), Some(200_000));
        // Unknown model, no cap -> None.
        let unknown = GoogleClient::new("k".into()).with_model("no-such-model");
        assert_eq!(unknown.max_context_tokens(), None);
    }

    #[tokio::test]
    async fn max_context_tokens_consults_model_override() {
        use desktop_assistant_core::ports::llm::with_model_override;
        let c = GoogleClient::new("k".into()).with_model("no-such-model");
        assert_eq!(c.max_context_tokens(), None);
        let observed =
            with_model_override("gemini-1.5-pro".into(), async { c.max_context_tokens() }).await;
        assert_eq!(observed, Some(2_097_152));
    }

    #[test]
    fn generation_config_includes_thinking_for_capable_model() {
        let c = GoogleClient::new("k".into());
        let cfg = c
            .build_generation_config(
                "gemini-2.5-pro",
                ReasoningConfig::with_thinking_budget(2048),
            )
            .expect("config present");
        let t = cfg.thinking_config.expect("thinkingConfig present");
        assert_eq!(t.thinking_budget, 2048);
    }

    #[test]
    fn generation_config_omits_thinking_when_budget_zero() {
        let c = GoogleClient::new("k".into());
        assert!(
            c.build_generation_config("gemini-2.5-pro", ReasoningConfig::with_thinking_budget(0))
                .is_none(),
            "budget 0 and no other knobs -> whole generationConfig omitted"
        );
    }

    #[test]
    fn generation_config_omits_thinking_for_non_capable_model() {
        let c = GoogleClient::new("k".into());
        let cfg = c.build_generation_config(
            "gemini-1.5-pro",
            ReasoningConfig::with_thinking_budget(2048),
        );
        assert!(
            cfg.as_ref()
                .and_then(|c| c.thinking_config.as_ref())
                .is_none(),
            "1.5 is not thinking-capable; thinkingConfig must be omitted"
        );
    }

    #[test]
    fn generation_config_carries_sampling_knobs() {
        let c = GoogleClient::new("k".into())
            .with_temperature(Some(0.4))
            .with_top_p(Some(0.8))
            .with_max_tokens(Some(1024));
        let cfg = c
            .build_generation_config("gemini-2.5-pro", ReasoningConfig::default())
            .expect("config present");
        assert_eq!(cfg.temperature, Some(0.4));
        assert_eq!(cfg.top_p, Some(0.8));
        assert_eq!(cfg.max_output_tokens, Some(1024));
        assert!(cfg.thinking_config.is_none());
    }

    #[test]
    fn generation_config_none_when_nothing_set() {
        let c = GoogleClient::new("k".into());
        assert!(
            c.build_generation_config("gemini-2.5-pro", ReasoningConfig::default())
                .is_none()
        );
    }

    #[test]
    fn debug_redacts_api_key() {
        let secret = "AIza-super-secret-key-DO-NOT-LEAK";
        let c = GoogleClient::new(secret.into()).with_model("gemini-2.5-pro");
        let rendered = format!("{c:?}");
        assert!(!rendered.contains(secret), "api key leaked: {rendered}");
        assert!(rendered.contains("redacted"));
        assert!(rendered.contains("gemini-2.5-pro"));
    }

    // --- TTL model cache (issue #620) -----------------------------------

    use httpmock::Method::GET;
    use httpmock::MockServer;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Mock clock: an atomic second-offset from a fixed origin, advanced by the
    /// tests so TTL expiry is deterministic (no sleeping, no real wall clock).
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

    /// A Gemini-API (AI Studio) client pointed at a mock server; the api-key
    /// surface lists models at `{base}/v1beta/models` and needs no project.
    fn api_key_client(server: &MockServer, clock: Arc<MockClock>) -> GoogleClient {
        GoogleClient::new("k".into())
            .with_auth_mode(AuthMode::ApiKey)
            .with_base_url(server.url(""))
            .with_clock(clock)
    }

    /// A `/v1beta/models` mock returning one bare live id; returns the handle so
    /// a test can assert the exact number of upstream calls.
    fn live_models_mock(server: &MockServer) -> httpmock::Mock<'_> {
        server.mock(|when, then| {
            when.method(GET).path("/v1beta/models");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models":[{"name":"models/gemini-live-xyz"}]}"#);
        })
    }

    #[tokio::test]
    async fn list_models_served_from_cache_within_ttl() {
        let server = MockServer::start();
        let m = live_models_mock(&server);
        let clock = MockClock::new();
        let client = api_key_client(&server, clock.clone());

        let first = client.list_models().await.expect("first fetch");
        clock.advance_secs(30 * 60); // < 1h TTL
        let second = client.list_models().await.expect("served from cache");

        assert_eq!(first, second);
        assert!(first.iter().any(|m| m.id == "gemini-live-xyz"));
        m.assert_calls(1); // the second call did NOT hit the endpoint
    }

    #[tokio::test]
    async fn list_models_refetches_after_ttl_expiry() {
        let server = MockServer::start();
        let m = live_models_mock(&server);
        let clock = MockClock::new();
        let client =
            api_key_client(&server, clock.clone()).with_model_cache_ttl(Duration::from_secs(3600));

        client.list_models().await.expect("first fetch");
        clock.advance_secs(3601); // past the TTL
        client.list_models().await.expect("refetch");
        m.assert_calls(2);
    }

    #[tokio::test]
    async fn list_models_refetches_at_exact_ttl_boundary() {
        let server = MockServer::start();
        let m = live_models_mock(&server);
        let clock = MockClock::new();
        let client =
            api_key_client(&server, clock.clone()).with_model_cache_ttl(Duration::from_secs(3600));

        client.list_models().await.expect("first fetch");
        clock.advance_secs(3600); // age == TTL → expired
        client.list_models().await.expect("refetch at boundary");
        m.assert_calls(2);
    }

    #[tokio::test]
    async fn refresh_models_bypasses_cache() {
        let server = MockServer::start();
        let m = live_models_mock(&server);
        let clock = MockClock::new();
        let client = api_key_client(&server, clock.clone());

        client.list_models().await.expect("prime cache"); // call 1
        client.refresh_models().await.expect("forced refetch"); // call 2, ignores TTL
        m.assert_calls(2);

        // refresh re-stored, so a subsequent list within TTL is served warm.
        client.list_models().await.expect("served from refreshed cache");
        m.assert_calls(2);
    }

    #[tokio::test]
    async fn failure_degrades_to_curated_without_poisoning_then_success_caches() {
        let clock = MockClock::new();
        let server = MockServer::start();
        let failing = server.mock(|when, then| {
            when.method(GET).path("/v1beta/models");
            then.status(500).body("boom");
        });
        let client = api_key_client(&server, clock.clone());

        // Live failure → degrade to the curated table, and do NOT cache it.
        let degraded = client.list_models().await.expect("degrade, not error");
        assert_eq!(degraded, curated_gemini_models());
        failing.assert_calls(1);
        failing.delete();

        // The failure did not poison the cache, so the next call re-fetches —
        // and a success now populates the cache.
        let ok = server.mock(|when, then| {
            when.method(GET).path("/v1beta/models");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"models":[{"name":"models/gemini-back"}]}"#);
        });
        let recovered = client.list_models().await.expect("success");
        assert!(recovered.iter().any(|m| m.id == "gemini-back"));
        ok.assert_calls(1);

        // Now served warm (no clock advance): still exactly one success call.
        let warm = client.list_models().await.expect("served from cache");
        assert_eq!(recovered, warm);
        ok.assert_calls(1);
    }
}
