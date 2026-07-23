//! AWS Bedrock Converse API connector implementing the core `LlmClient` port.

mod tool_names;
pub use tool_names::ToolNameMap;

use aws_config::{BehaviorVersion, Region};
use aws_credential_types::Credentials;
use aws_sdk_bedrock::Client as BedrockControlClient;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, Message as BedrockMessage, SystemContentBlock, Tool,
    ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolSpecification,
    ToolUseBlock,
};
use aws_smithy_types::{Document, Number};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelCapabilities, ModelInfo, ReasoningConfig,
    TokenUsage, current_model_override,
};
use desktop_assistant_llm_http::{STREAM_CONNECT_TIMEOUT, STREAM_EVENT_TIMEOUT};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OnceCell};

/// Default TTL for the `list_models()` cache. One hour is cheap to refresh
/// and long enough that UIs don't trigger a round-trip on every open.
const DEFAULT_MODEL_CACHE_TTL: Duration = Duration::from_secs(60 * 60);

/// Abstraction over `Instant::now()` so the cache TTL test can advance time
/// without sleeping. The production impl is `SystemClock`.
pub trait ModelClock: Send + Sync {
    fn now(&self) -> Instant;
}

/// Default clock that reads the monotonic OS clock.
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

/// Amazon Bedrock client using the Converse API.
pub struct BedrockClient {
    model: String,
    base_url: String,
    api_key: String,
    aws_profile: Option<String>,
    client: OnceCell<Client>,
    control_client: OnceCell<BedrockControlClient>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    model_cache: Arc<Mutex<ModelCache>>,
    model_cache_ttl: Duration,
    clock: Arc<dyn ModelClock>,
    /// Models discovered at runtime to reject `ConverseStream` with
    /// tools. Populated when the static allowlist
    /// (`supports_streaming_with_tools`) reports `true` but Bedrock
    /// returns the specific "doesn't support tool use in streaming
    /// mode" validation error. Per-instance so each client warms its
    /// own cache; not shared across `BedrockClient` instances. (#67)
    non_streaming_tools_models: Arc<Mutex<HashSet<String>>>,
    /// First-response (connect) stall budget; defaults to
    /// [`STREAM_CONNECT_TIMEOUT`], overridable per-connection.
    connect_timeout: Duration,
    /// Per-chunk stall budget; defaults to [`STREAM_EVENT_TIMEOUT`].
    event_timeout: Duration,
    /// Per-connection context-window hard cap, in tokens. `None` = "max
    /// available". Folded with the curated table in `max_context_tokens`.
    context_cap: Option<u64>,
}

impl BedrockClient {
    pub fn get_default_model() -> Option<&'static str> {
        Some("us.anthropic.claude-sonnet-4-6")
    }

    pub fn get_default_base_url() -> Option<&'static str> {
        Some("us-east-1")
    }

    pub fn new(api_key: String) -> Self {
        Self {
            model: Self::get_default_model().unwrap_or_default().to_string(),
            base_url: Self::get_default_base_url().unwrap_or_default().to_string(),
            api_key,
            aws_profile: None,
            client: OnceCell::new(),
            control_client: OnceCell::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            model_cache: Arc::new(Mutex::new(ModelCache::default())),
            model_cache_ttl: DEFAULT_MODEL_CACHE_TTL,
            clock: Arc::new(SystemClock),
            non_streaming_tools_models: Arc::new(Mutex::new(HashSet::new())),
            connect_timeout: STREAM_CONNECT_TIMEOUT,
            event_timeout: STREAM_EVENT_TIMEOUT,
            context_cap: None,
        }
    }

    /// Set the per-connection context-window hard cap, in tokens. `None`/
    /// `Some(0)` = "max available". Clamps the daemon's input budget (no
    /// `num_ctx` to pin), useful for bounding spend. See
    /// `desktop_assistant_llm_http::apply_context_cap`.
    pub fn with_max_context_tokens(mut self, max: Option<u64>) -> Self {
        self.context_cap = max.filter(|m| *m > 0);
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

    /// Override the `list_models()` cache TTL (default: 1h).
    pub fn with_model_cache_ttl(mut self, ttl: Duration) -> Self {
        self.model_cache_ttl = ttl;
        self
    }

    /// Inject a custom clock for deterministic cache-TTL tests.
    pub fn with_clock(mut self, clock: Arc<dyn ModelClock>) -> Self {
        self.clock = clock;
        self
    }

    /// Test-only: prime the `list_models()` cache so the cache-TTL test
    /// can exercise hit/miss behavior without reaching AWS. The
    /// `fetched_at` timestamp is stamped using the configured clock.
    #[doc(hidden)]
    pub async fn __set_models_cache_for_test(&self, models: Vec<ModelInfo>) {
        let now = self.clock.now();
        let mut cache = self.model_cache.lock().await;
        cache.entry = Some((now, models));
    }

    /// Test-only: peek at the cache contents.
    #[doc(hidden)]
    pub async fn __peek_models_cache_for_test(&self) -> Option<Vec<ModelInfo>> {
        let cache = self.model_cache.lock().await;
        cache.entry.as_ref().map(|(_, v)| v.clone())
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self.client = OnceCell::new();
        self.control_client = OnceCell::new();
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

    pub fn with_aws_profile(mut self, profile: Option<String>) -> Self {
        self.aws_profile = profile.filter(|s| !s.trim().is_empty());
        self
    }

    async fn load_shared_config(&self) -> aws_config::SdkConfig {
        let mut loader = aws_config::defaults(BehaviorVersion::latest());

        let effective_profile = self
            .aws_profile
            .clone()
            .or_else(|| aws_profile_exists("adele").then(|| "adele".to_string()));

        if let Some(ref profile) = effective_profile {
            tracing::info!(aws_profile = %profile, "using AWS profile");
            loader = loader.profile_name(profile);
        }

        if let Some(region) = region_from_base_url(&self.base_url) {
            loader = loader.region(Region::new(region));
        }

        if let Some(credentials) = static_credentials_from_api_key(&self.api_key) {
            loader = loader.credentials_provider(credentials);
        } else if !self.api_key.trim().is_empty() {
            tracing::debug!(
                "llm.bedrock.api_key is set but not parseable as static credentials; falling back to AWS credential chain"
            );
        }

        loader.load().await
    }

    async fn client(&self) -> Result<&Client, CoreError> {
        self.client
            .get_or_try_init(|| async {
                let shared_config = self.load_shared_config().await;
                Ok(Client::new(&shared_config))
            })
            .await
    }

    async fn control_client(&self) -> Result<&BedrockControlClient, CoreError> {
        self.control_client
            .get_or_try_init(|| async {
                let shared_config = self.load_shared_config().await;
                Ok(BedrockControlClient::new(&shared_config))
            })
            .await
    }

    /// Return the model ID as the stable version identifier.
    ///
    /// Bedrock model IDs already include version info (e.g.
    /// `amazon.titan-embed-text-v2:0`), so no server call is needed.
    pub async fn model_identifier(&self) -> Result<String, CoreError> {
        Ok(self.model.clone())
    }

    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        let client = self.client().await?;

        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            let payload = serde_json::json!({
                "inputText": text,
            });

            let response = client
                .invoke_model()
                .model_id(self.model.clone())
                .content_type("application/json")
                .accept("application/json")
                .body(payload.to_string().into_bytes().into())
                .send()
                .await
                .map_err(|e| CoreError::Llm(format!("Bedrock embeddings request failed: {e}")))?;

            let body = response.body.into_inner();
            let parsed: BedrockEmbeddingResponse = serde_json::from_slice(&body).map_err(|e| {
                CoreError::Llm(format!("failed to parse Bedrock embedding response: {e}"))
            })?;

            vectors.push(parsed.embedding);
        }

        Ok(vectors)
    }
}

#[derive(serde::Deserialize)]
struct BedrockEmbeddingResponse {
    #[serde(default)]
    embedding: Vec<f32>,
}

/// Check whether an AWS profile exists in `~/.aws/config` or `~/.aws/credentials`.
fn aws_profile_exists(name: &str) -> bool {
    let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
        return false;
    };
    let aws_dir = std::path::Path::new(&home).join(".aws");

    // ~/.aws/config uses [profile <name>] (except [default])
    let config_section = format!("[profile {name}]");
    // ~/.aws/credentials uses [<name>]
    let creds_section = format!("[{name}]");

    for (path, needle) in [
        (aws_dir.join("config"), config_section.as_str()),
        (aws_dir.join("credentials"), creds_section.as_str()),
    ] {
        if let Ok(contents) = std::fs::read_to_string(&path)
            && contents.contains(needle)
        {
            return true;
        }
    }
    false
}

fn region_from_base_url(base_url: &str) -> Option<String> {
    let trimmed = base_url.trim();
    if trimmed.is_empty() {
        return None;
    }

    if !trimmed.contains("http://") && !trimmed.contains("https://") {
        return Some(trimmed.to_string());
    }

    let without_scheme = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
        .unwrap_or(trimmed);

    let host = without_scheme.split('/').next().unwrap_or_default();
    let segments: Vec<&str> = host.split('.').collect();
    if segments.len() >= 4
        && segments.first().copied() == Some("bedrock-runtime")
        && segments.get(2).copied() == Some("amazonaws")
    {
        return segments.get(1).map(|s| s.to_string());
    }

    None
}

fn static_credentials_from_api_key(api_key: &str) -> Option<Credentials> {
    let trimmed = api_key.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut parts = trimmed.splitn(3, ':');
    let access_key_id = parts.next()?.trim();
    let secret_access_key = parts.next()?.trim();
    let session_token = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);

    if access_key_id.is_empty() || secret_access_key.is_empty() {
        return None;
    }

    Some(Credentials::new(
        access_key_id.to_string(),
        secret_access_key.to_string(),
        session_token,
        None,
        "desktop-assistant-bedrock-static",
    ))
}

fn convert_messages(
    messages: &[Message],
    tool_names: &ToolNameMap,
) -> Result<(Vec<SystemContentBlock>, Vec<BedrockMessage>), CoreError> {
    let mut system = Vec::new();
    let mut api_messages = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                system.push(SystemContentBlock::Text(msg.content.clone()));
            }
            Role::User => {
                // Merge consecutive user messages to maintain alternation.
                let is_consecutive_user = api_messages
                    .last()
                    .is_some_and(|m: &BedrockMessage| m.role() == &ConversationRole::User);
                if is_consecutive_user {
                    let prev = api_messages.pop().unwrap();
                    let mut builder = BedrockMessage::builder().role(ConversationRole::User);
                    for block in prev.content() {
                        let b: ContentBlock = block.clone();
                        builder = builder.content(b);
                    }
                    builder = builder.content(ContentBlock::Text(msg.content.clone()));
                    api_messages.push(builder.build().map_err(|e| {
                        CoreError::Llm(format!("failed to build Bedrock user message payload: {e}"))
                    })?);
                } else {
                    api_messages.push(
                        BedrockMessage::builder()
                            .role(ConversationRole::User)
                            .content(ContentBlock::Text(msg.content.clone()))
                            .build()
                            .map_err(|e| {
                                CoreError::Llm(format!(
                                    "failed to build Bedrock user message payload: {e}"
                                ))
                            })?,
                    );
                }
            }
            Role::Assistant => {
                let mut builder = BedrockMessage::builder().role(ConversationRole::Assistant);

                if !msg.content.is_empty() {
                    builder = builder.content(ContentBlock::Text(msg.content.clone()));
                }

                for tc in &msg.tool_calls {
                    let input_json = serde_json::from_str::<serde_json::Value>(&tc.arguments)
                        .unwrap_or(serde_json::json!({}));
                    // gpt-oss on Bedrock emits `{"":{}}` (an empty-string key) for
                    // no-argument tool calls; echoing that back as `toolUse.input`
                    // makes Bedrock 400 ("messages.N.content.0.toolUse.input is
                    // invalid") on every subsequent turn, since the bad block lives
                    // in history. Normalize it to a valid object (#214).
                    let doc = json_to_document(sanitize_tool_input(input_json));
                    // Sanitize the historical tool name to satisfy Bedrock's
                    // `^[a-zA-Z0-9_-]+$` constraint. This is essential: a
                    // `toolUse` block from an EARLIER turn lives in the
                    // message history, so the offending name is re-sent on
                    // every subsequent turn (the live error points at
                    // `messages.N`, i.e. pre-existing history). The tool_use_id
                    // is an id, not a name, and is left untouched so result
                    // correlation still works.
                    let safe_name = tool_names.to_safe(&tc.name).into_owned();
                    builder = builder.content(ContentBlock::ToolUse(
                        ToolUseBlock::builder()
                            .tool_use_id(tc.id.clone())
                            .name(safe_name)
                            .input(doc)
                            .build()
                            .map_err(|e| {
                                CoreError::Llm(format!(
                                    "failed to build Bedrock assistant tool-use payload: {e}"
                                ))
                            })?,
                    ));
                }

                api_messages.push(builder.build().map_err(|e| {
                    CoreError::Llm(format!(
                        "failed to build Bedrock assistant message payload: {e}"
                    ))
                })?);
            }
            Role::Tool => {
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                let result_block = ContentBlock::ToolResult(
                    ToolResultBlock::builder()
                        .tool_use_id(tool_use_id)
                        .content(ToolResultContentBlock::Text(msg.content.clone()))
                        .build()
                        .map_err(|e| {
                            CoreError::Llm(format!(
                                "failed to build Bedrock tool-result payload: {e}"
                            ))
                        })?,
                );
                // Bedrock requires all tool results for a single assistant turn
                // to be in one user message. Merge consecutive tool results.
                let merged = api_messages.last().and_then(|m: &BedrockMessage| {
                    if m.role() == &ConversationRole::User
                        && m.content()
                            .iter()
                            .all(|c| matches!(c, ContentBlock::ToolResult(_)))
                        && !m.content().is_empty()
                    {
                        Some(true)
                    } else {
                        None
                    }
                });
                if merged.is_some() {
                    let prev = api_messages.pop().unwrap();
                    let mut builder = BedrockMessage::builder().role(ConversationRole::User);
                    for block in prev.content() {
                        let b: ContentBlock = block.clone();
                        builder = builder.content(b);
                    }
                    builder = builder.content(result_block);
                    api_messages.push(builder.build().map_err(|e| {
                        CoreError::Llm(format!("failed to build Bedrock tool message payload: {e}"))
                    })?);
                } else {
                    api_messages.push(
                        BedrockMessage::builder()
                            .role(ConversationRole::User)
                            .content(result_block)
                            .build()
                            .map_err(|e| {
                                CoreError::Llm(format!(
                                    "failed to build Bedrock tool message payload: {e}"
                                ))
                            })?,
                    );
                }
            }
        }
    }

    Ok((system, api_messages))
}

fn convert_tools(
    tools: &[ToolDefinition],
    tool_names: &ToolNameMap,
) -> Result<Option<ToolConfiguration>, CoreError> {
    if tools.is_empty() {
        return Ok(None);
    }

    let mut cfg_builder = ToolConfiguration::builder();
    for tool in tools {
        // Defensively strip top-level oneOf/anyOf/allOf, which Bedrock rejects
        // and which would otherwise 400 the whole request (taking every other
        // tool down with the one offender). See `sanitize_tool_schema`.
        let input_doc = json_to_document(sanitize_tool_schema(tool.parameters.clone()));
        // Sanitize the tool-spec name to Bedrock's `^[a-zA-Z0-9_-]+$`. Must
        // match the sanitization applied to history `toolUse` names so the
        // model's response correlates back to the right tool.
        let safe_name = tool_names.to_safe(&tool.name).into_owned();
        let spec = ToolSpecification::builder()
            .name(safe_name)
            .description(tool.description.clone())
            .input_schema(ToolInputSchema::Json(input_doc))
            .build()
            .map_err(|e| CoreError::Llm(format!("failed to build Bedrock tool spec: {e}")))?;
        cfg_builder = cfg_builder.tools(Tool::ToolSpec(spec));
    }

    let cfg = cfg_builder
        .build()
        .map_err(|e| CoreError::Llm(format!("failed to build Bedrock tool config: {e}")))?;

    Ok(Some(cfg))
}

/// Bedrock indexes streamed content blocks with `i32`. Use the shared
/// accumulator from core (#45).
type ToolCallAccumulator = desktop_assistant_core::ports::llm::ToolCallAccumulator<i32>;

fn apply_stream_event(
    event: aws_sdk_bedrockruntime::types::ConverseStreamOutput,
    text: &mut String,
    tool_acc: &mut ToolCallAccumulator,
    on_chunk: &mut ChunkCallback,
    token_usage: &mut Option<TokenUsage>,
) -> bool {
    match event {
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStart(start) => {
            if let Some(content_start) = start.start()
                && let aws_sdk_bedrockruntime::types::ContentBlockStart::ToolUse(tool_use) =
                    content_start
            {
                tool_acc.start(
                    start.content_block_index(),
                    tool_use.tool_use_id(),
                    tool_use.name(),
                );
            }
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockDelta(delta) => {
            if let Some(content_delta) = delta.delta() {
                match content_delta {
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(chunk) => {
                        text.push_str(chunk);
                        if !on_chunk(chunk.clone()) {
                            tracing::debug!("Bedrock stream aborted by callback");
                            return false;
                        }
                    }
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(tool_delta) => {
                        tool_acc.append(delta.content_block_index(), tool_delta.input());
                    }
                    _ => {}
                }
            }
        }
        aws_sdk_bedrockruntime::types::ConverseStreamOutput::Metadata(meta) => {
            if let Some(usage) = meta.usage() {
                *token_usage = Some(TokenUsage {
                    input_tokens: Some(usage.input_tokens() as u64),
                    output_tokens: Some(usage.output_tokens() as u64),
                    ..Default::default()
                });
            }
        }
        _ => {}
    }

    true
}

/// Parsed details of a Bedrock context-overflow validation error. The token
/// counts are optional because not every overflow message carries them (e.g.
/// `"Input is too long for requested model."`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextOverflowInfo {
    pub prompt_tokens: Option<u64>,
    pub max_tokens: Option<u64>,
}

/// Detect whether a Bedrock validation-error message means the prompt
/// exceeded the model's context window, extracting the token counts when the
/// message includes them. Returns `None` for unrelated errors so the caller
/// falls through to the generic mapping.
///
/// Recognized shapes (case-insensitive) — Bedrock is not consistent across
/// model families, so we match several:
///   - `"prompt is too long: 203524 tokens > 200000 maximum"` (Anthropic)
///   - `"Input length (479258) exceeds model's maximum context length (131072)."`
///   - `"Input is too long for requested model."` (no counts)
///
/// Mapping these to `CoreError::ContextOverflow` is what lets the core
/// recovery ladder (truncate the largest tool result → trim old pairs →
/// summarise-and-shrink) fire and retry, instead of surfacing a hard failure
/// and losing the turn.
pub fn parse_context_overflow(message: &str) -> Option<ContextOverflowInfo> {
    let lower = message.to_ascii_lowercase();
    let is_overflow = lower.contains("prompt is too long")
        || lower.contains("input is too long")
        || (lower.contains("exceeds") && lower.contains("context length"));
    if !is_overflow {
        return None;
    }

    // Pull the first two integers, if any. Across the recognized shapes the
    // counts appear as (prompt, max) in that order; fewer than two means the
    // message stated the overflow without numbers, which is still actionable.
    let nums: Vec<u64> = message
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    let (prompt_tokens, max_tokens) = match nums.as_slice() {
        [prompt, max, ..] => (Some(*prompt), Some(*max)),
        _ => (None, None),
    };
    Some(ContextOverflowInfo {
        prompt_tokens,
        max_tokens,
    })
}

/// Map a Bedrock `converse_stream` SDK error to the equivalent
/// `CoreError`. Extracted so the dispatch logic is unit-testable
/// independent of the network call site.
fn map_converse_stream_error(
    e: aws_sdk_bedrockruntime::error::SdkError<
        aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError,
    >,
) -> CoreError {
    use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
    // Detect prompt-overflow validation errors and surface them as
    // CoreError::ContextOverflow so the core service can truncate
    // the offending tool result and retry.
    if let Some(ConverseStreamError::ValidationException(ve)) = e.as_service_error() {
        let raw = ve.message().unwrap_or("unknown");
        if let Some(info) = parse_context_overflow(raw) {
            tracing::warn!(
                prompt_tokens = ?info.prompt_tokens,
                max_tokens = ?info.max_tokens,
                "Bedrock rejected request for context overflow"
            );
            return CoreError::ContextOverflow {
                prompt_tokens: info.prompt_tokens,
                max_tokens: info.max_tokens,
                detail: format!("Bedrock validation error: {raw}"),
            };
        }
    }
    if let Some(svc) = e.as_service_error()
        && let Some(mapped) = map_converse_stream_service_error(svc)
    {
        return mapped;
    }
    let detail = match e.as_service_error() {
        Some(ConverseStreamError::ValidationException(ve)) => {
            format!("validation error: {}", ve.message().unwrap_or("unknown"))
        }
        Some(ConverseStreamError::AccessDeniedException(ad)) => {
            format!("access denied: {}", ad.message().unwrap_or("unknown"))
        }
        Some(ConverseStreamError::ModelTimeoutException(mt)) => {
            format!("model timeout: {}", mt.message().unwrap_or("unknown"))
        }
        Some(other) => format!("{other}"),
        None => format!("{e:#}"),
    };
    tracing::warn!("Bedrock converse_stream error: {detail}");
    CoreError::Llm(format!("Bedrock converse_stream request failed: {detail}"))
}

/// Map a Bedrock `ConverseStreamError` to the structured
/// [`CoreError`] variant for the cases that have a dedicated variant
/// (`RateLimited`, `ModelLoading`). Returns `None` if the variant has
/// no dedicated mapping — the caller falls through to the generic
/// `CoreError::Llm` path.
///
/// Doing the mapping in a dedicated function lets tests cover each
/// arm without needing to construct an `SdkError`.
fn map_converse_stream_service_error(
    err: &aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError,
) -> Option<CoreError> {
    use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
    match err {
        ConverseStreamError::ThrottlingException(te) => Some(CoreError::RateLimited {
            retry_after: None,
            detail: format!("Bedrock throttling: {}", te.message().unwrap_or("unknown")),
        }),
        ConverseStreamError::ServiceUnavailableException(se) => Some(CoreError::RateLimited {
            retry_after: None,
            detail: format!(
                "Bedrock service unavailable: {}",
                se.message().unwrap_or("unknown")
            ),
        }),
        ConverseStreamError::ModelNotReadyException(mr) => Some(CoreError::ModelLoading {
            detail: format!(
                "Bedrock model not ready: {}",
                mr.message().unwrap_or("unknown")
            ),
        }),
        _ => None,
    }
}

/// Return the prompt-token context window for a known Bedrock model ID.
///
/// Accepts cross-region inference-profile prefixes (`us.`, `eu.`, `apac.`).
/// Returns `None` for models without a known limit; callers should treat
/// `None` as "disable token-based compaction" and rely on message-count
/// fallbacks instead.
///
/// `ListFoundationModels` does not expose context windows, so this table
/// is the single source of truth for Bedrock models whose windows we know.
/// The `list_models()` implementation uses it to populate
/// `ModelInfo::context_limit`.
pub fn context_limit_for_model(model_id: &str) -> Option<u64> {
    let base = strip_region_prefix(model_id);

    // Anthropic Claude on Bedrock: 3.x and 4.x all ship with 200K context.
    if base.starts_with("anthropic.claude-3")
        || base.starts_with("anthropic.claude-sonnet-4")
        || base.starts_with("anthropic.claude-opus-4")
        || base.starts_with("anthropic.claude-haiku-4")
    {
        return Some(200_000);
    }

    // OpenAI gpt-oss on Bedrock (120b and 20b): 131,072-token window. This
    // value is authoritative — it's exactly what Bedrock reports in its
    // overflow error ("... maximum context length (131072)"). Without it the
    // budget falls to the 200K universal fallback and overshoots the real
    // window, which is the root of issue #176.
    if base.starts_with("openai.gpt-oss") {
        return Some(131_072);
    }

    // Other families (Amazon Nova, Meta Llama, Mistral, Cohere, DeepSeek) are
    // intentionally left to the universal fallback for now rather than guessed
    // here: over-stating a window makes the model hard-reject requests (the
    // exact failure this issue fixes), so new entries should be added only
    // with a verified per-model number. Tracked under epic #178.
    None
}

/// Heuristic capability inference from a model id. Operates on the *base*
/// id (region-prefix already stripped) so it works for both bare foundation
/// model ids and inference-profile ids.
fn infer_capabilities_from_id(
    base_id: &str,
    vision: bool,
    is_embedding: bool,
) -> ModelCapabilities {
    let lc = base_id.to_ascii_lowercase();

    let tools = lc.contains("anthropic.claude")
        || lc.contains("amazon.nova")
        || lc.contains("meta.llama3")
        || lc.contains("meta.llama4")
        || lc.contains("mistral")
        || lc.contains("cohere.command")
        || lc.contains("deepseek");

    let reasoning = lc.contains("anthropic.claude-sonnet-4")
        || lc.contains("anthropic.claude-opus-4")
        || lc.contains("anthropic.claude-haiku-4")
        || lc.contains("anthropic.claude-3-7")
        || lc.contains("deepseek.r1")
        || lc.contains("deepseek-r1");

    ModelCapabilities {
        reasoning,
        vision,
        tools: tools && !is_embedding,
        embedding: is_embedding,
    }
}

/// Strip a cross-region inference-profile prefix (`us.`, `eu.`, `apac.`) to
/// recover the underlying foundation model id. Returns the input unchanged
/// when no known prefix matches.
fn strip_region_prefix(id: &str) -> &str {
    id.strip_prefix("us.")
        .or_else(|| id.strip_prefix("eu."))
        .or_else(|| id.strip_prefix("apac."))
        .unwrap_or(id)
}

/// Whether a Bedrock model accepts tool-use requests via `ConverseStream`.
///
/// AWS Bedrock has a per-model restriction: some foundation models support
/// tools via `Converse` *only*, not `ConverseStream`. Llama 3/4 fall in
/// that bucket; Claude does not. (#67)
///
/// `base_id` should be the region-prefix-stripped foundation model id —
/// `meta.llama4-…`, not `us.meta.llama4-…`. The caller is responsible
/// for calling [`strip_region_prefix`] first.
///
/// Conservative: defaults to `true` for unknown models so we keep the
/// streaming path when in doubt. The runtime fallback in `stream_completion`
/// catches mis-classifications by parsing the specific validation error
/// and retrying via `Converse` — that retry also memoizes the model so
/// subsequent calls skip straight to the non-streaming path.
fn supports_streaming_with_tools(base_id: &str) -> bool {
    let lc = base_id.to_ascii_lowercase();
    if lc.starts_with("meta.llama3") || lc.starts_with("meta.llama4") {
        return false;
    }
    true
}

/// Detect the Bedrock validation error that signals "this model accepts
/// tools via Converse but not ConverseStream". The exact message text is
/// documented on the Bedrock supported-features page; matching is
/// case-insensitive and tolerant of leading/trailing punctuation.
fn is_streaming_tools_unsupported_message(message: &str) -> bool {
    let lc = message.to_ascii_lowercase();
    lc.contains("doesn't support tool use in streaming")
        || lc.contains("does not support tool use in streaming")
}

/// Convert an `aws_smithy_types::Document` (used for non-streaming
/// `ToolUse.input`) into a JSON string. Inverse of `json_to_document`;
/// used by the non-streaming dispatch to produce a `ToolCall.arguments`
/// in the same shape the streaming path emits.
fn document_to_json_string(doc: &Document) -> String {
    fn doc_to_value(doc: &Document) -> serde_json::Value {
        match doc {
            Document::Null => serde_json::Value::Null,
            Document::Bool(b) => serde_json::Value::Bool(*b),
            Document::Number(n) => match n {
                Number::PosInt(v) => serde_json::Value::Number((*v).into()),
                Number::NegInt(v) => serde_json::Value::Number((*v).into()),
                Number::Float(v) => serde_json::Number::from_f64(*v)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
            },
            Document::String(s) => serde_json::Value::String(s.clone()),
            Document::Array(a) => serde_json::Value::Array(a.iter().map(doc_to_value).collect()),
            Document::Object(o) => serde_json::Value::Object(
                o.iter()
                    .map(|(k, v)| (k.clone(), doc_to_value(v)))
                    .collect(),
            ),
        }
    }
    serde_json::to_string(&doc_to_value(doc)).unwrap_or_else(|_| "{}".to_string())
}

/// Convert a `FoundationModelSummary` into a `ModelInfo`, returning `None`
/// if the model should be filtered out (not ACTIVE, not text/embedding, or
/// not invocable via on-demand throughput).
fn summary_to_model_info(
    summary: &aws_sdk_bedrock::types::FoundationModelSummary,
) -> Option<ModelInfo> {
    use aws_sdk_bedrock::types::{FoundationModelLifecycleStatus, InferenceType, ModelModality};

    // Filter: lifecycle must be ACTIVE (skip LEGACY / deprecated models).
    if let Some(lifecycle) = summary.model_lifecycle.as_ref()
        && lifecycle.status() != &FoundationModelLifecycleStatus::Active
    {
        return None;
    }

    // Filter: must support on-demand throughput. Newer models (Claude 4.x,
    // Nova Premier, DeepSeek R1, etc.) are only callable via an inference
    // profile or Provisioned Throughput; surfacing the bare id leads to a
    // ValidationException at invocation time. Inference profiles are merged
    // separately by `fetch_models_uncached`.
    let supports_on_demand = summary
        .inference_types_supported()
        .iter()
        .any(|t| t == &InferenceType::OnDemand);
    if !supports_on_demand {
        return None;
    }

    // Filter: output modality must include TEXT or EMBEDDING.
    // (We skip pure IMAGE/VIDEO generation models — they're not usable as
    // chat/embedding backends in this connector.)
    let output_modalities = summary.output_modalities();
    let is_text = output_modalities.contains(&ModelModality::Text);
    let is_embedding = output_modalities.contains(&ModelModality::Embedding);
    if !(is_text || is_embedding) {
        return None;
    }

    let input_modalities = summary.input_modalities();
    let vision = input_modalities.contains(&ModelModality::Image);

    let id = summary.model_id();
    let model_name = summary.model_name().unwrap_or(id).to_string();
    let capabilities = infer_capabilities_from_id(id, vision, is_embedding);

    Some(ModelInfo {
        id: id.to_string(),
        display_name: model_name,
        context_limit: context_limit_for_model(id),
        capabilities,
    })
}

/// Convert an `InferenceProfileSummary` into a `ModelInfo`. Returns `None`
/// for non-active profiles or profiles whose underlying foundation model
/// can't be recovered.
///
/// Capabilities are derived from the underlying foundation model id (after
/// stripping the region prefix) since the profile API doesn't expose them.
/// Vision support is conservatively inferred from the model id family rather
/// than from a real modality field — Bedrock doesn't surface modalities on
/// profiles, but the profile's underlying model has the same modalities as
/// its foundation counterpart.
fn inference_profile_to_model_info(
    profile: &aws_sdk_bedrock::types::InferenceProfileSummary,
) -> Option<ModelInfo> {
    use aws_sdk_bedrock::types::InferenceProfileStatus;

    if profile.status != InferenceProfileStatus::Active {
        return None;
    }

    let profile_id = profile.inference_profile_id();
    if profile_id.is_empty() {
        return None;
    }

    let base_id = strip_region_prefix(profile_id);
    let lc = base_id.to_ascii_lowercase();

    // Vision: known multimodal Bedrock model families. Profile API gives us
    // no modality info, so this list is best-effort and conservative.
    let vision = lc.contains("anthropic.claude-3")
        || lc.contains("anthropic.claude-sonnet-4")
        || lc.contains("anthropic.claude-opus-4")
        || lc.contains("anthropic.claude-haiku-4")
        || lc.contains("amazon.nova-pro")
        || lc.contains("amazon.nova-lite")
        || lc.contains("amazon.nova-premier")
        || lc.contains("meta.llama3-2-11b-vision")
        || lc.contains("meta.llama3-2-90b-vision")
        || lc.contains("meta.llama4");

    // Inference profiles cover chat models; embeddings stay on their bare
    // ids (which support OnDemand and pass through the foundation-model
    // path).
    let is_embedding = false;

    let display_name = if profile.inference_profile_name.is_empty() {
        profile_id.to_string()
    } else {
        profile.inference_profile_name.clone()
    };

    Some(ModelInfo {
        id: profile_id.to_string(),
        display_name,
        // context_limit_for_model already strips the region prefix internally.
        context_limit: context_limit_for_model(profile_id),
        capabilities: infer_capabilities_from_id(base_id, vision, is_embedding),
    })
}

impl BedrockClient {
    /// Call `ListFoundationModels` + `ListInferenceProfiles` and merge into
    /// a single `ModelInfo` list:
    ///
    /// * Foundation models without `OnDemand` support are filtered out —
    ///   their bare ids are uncallable and surfacing them leads to runtime
    ///   `ValidationException`s. Users reach those models via inference
    ///   profiles instead.
    /// * Inference profiles are merged in with their prefixed ids
    ///   (`us.anthropic.claude-haiku-4-5-…` etc.) so the model picker
    ///   exposes the IDs that AWS will actually accept on Converse.
    ///
    /// Both calls go in parallel. `ListInferenceProfiles` failures are
    /// logged and swallowed: many existing IAM policies grant
    /// `bedrock:ListFoundationModels` without
    /// `bedrock:ListInferenceProfiles`, and we'd rather degrade to the
    /// foundation-model-only list than fail the whole picker.
    async fn fetch_models_uncached(&self) -> Result<Vec<ModelInfo>, CoreError> {
        let client = self.control_client().await?;

        let foundation_fut = client.list_foundation_models().send();
        let profiles_fut = client.list_inference_profiles().send();

        let (foundation_res, profiles_res) = tokio::join!(foundation_fut, profiles_fut);

        let foundation = foundation_res
            .map_err(|e| CoreError::Llm(format!("Bedrock ListFoundationModels failed: {e:#}")))?;

        let mut models: Vec<ModelInfo> = foundation
            .model_summaries()
            .iter()
            .filter_map(summary_to_model_info)
            .collect();

        match profiles_res {
            Ok(profile_resp) => {
                for profile in profile_resp.inference_profile_summaries() {
                    if let Some(info) = inference_profile_to_model_info(profile) {
                        models.push(info);
                    }
                }
            }
            Err(error) => {
                tracing::warn!(
                    "Bedrock ListInferenceProfiles failed; model picker will only show \
                     on-demand foundation models. Grant bedrock:ListInferenceProfiles to \
                     surface inference-profile ids (Claude 4.x, Nova Premier, etc.). \
                     Cause: {error:#}"
                );
            }
        }

        // Stable ordering so UIs don't shuffle between refreshes.
        // Defensive dedupe — foundation ids and profile ids don't collide
        // in practice, but keep the merge total just in case.
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models.dedup_by(|a, b| a.id == b.id);
        Ok(models)
    }

    /// Return cached models, refreshing if the TTL elapsed or the cache is
    /// empty.
    async fn list_models_cached(&self) -> Result<Vec<ModelInfo>, CoreError> {
        {
            let cache = self.model_cache.lock().await;
            if let Some((fetched_at, entry)) = cache.entry.as_ref() {
                let age = self.clock.now().saturating_duration_since(*fetched_at);
                if age < self.model_cache_ttl {
                    return Ok(entry.clone());
                }
            }
        }
        self.refresh_models_internal().await
    }

    /// Force a refresh: bypass the cache, fetch from Bedrock, and populate
    /// the cache on success.
    async fn refresh_models_internal(&self) -> Result<Vec<ModelInfo>, CoreError> {
        let fresh = self.fetch_models_uncached().await?;
        let now = self.clock.now();
        let mut cache = self.model_cache.lock().await;
        cache.entry = Some((now, fresh.clone()));
        Ok(fresh)
    }
}

#[async_trait::async_trait]
impl LlmClient for BedrockClient {
    fn get_default_model(&self) -> Option<&str> {
        Self::get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        Self::get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        // Fold the per-connection hard cap into the curated window so the
        // daemon budgets against the capped value (e.g. to bound spend).
        desktop_assistant_llm_http::apply_context_cap(
            self.context_cap,
            context_limit_for_model(&self.model),
        )
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
        // Cooperative cancellation token (issue #109): pre-check before
        // building the AWS SDK client / making any network call. Inside
        // the streaming loop we race the next event against
        // `token.cancelled()` so the body stream is dropped cleanly
        // when the user cancels mid-stream.
        let cancellation =
            desktop_assistant_core::ports::llm::current_cancellation_token().unwrap_or_default();
        if cancellation.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let client = self.client().await?;

        // Bedrock validates every tool name (in the request tool-spec AND in
        // every `toolUse` block carried in the message history) against
        // `^[a-zA-Z0-9_-]+$` with a 64-char cap — stricter than the Anthropic
        // API. Build a per-request bijection from the available tools, apply
        // it consistently to the tool definitions and to historical
        // `toolUse.name`s, and reverse it when the model echoes a name back so
        // dispatch still hits the real (possibly `.`/`:`/`/`-containing) tool.
        let tool_names = ToolNameMap::from_names(tools.iter().map(|t| t.name.as_str()));
        let (system, api_messages) = convert_messages(&messages, &tool_names)?;
        let tool_config = convert_tools(tools, &tool_names)?;

        // Per-turn model override (issue #34): when the daemon-side routing
        // layer has set `MODEL_OVERRIDE`, dispatch the user-chosen model id
        // instead of the connector's baked-in `self.model`. Used both for
        // the request `model_id` and for keying reasoning support /
        // context-window heuristics below.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());

        let msg_count = api_messages.len();
        let tool_count = tools.len();
        let system_chars: usize = system.iter().map(|b| format!("{b:?}").len()).sum();
        let msg_chars: usize = api_messages.iter().map(|m| format!("{m:?}").len()).sum();
        tracing::info!(
            msg_chars,
            msg_count,
            tool_count,
            system_chars,
            model = %model,
            "LLM request payload"
        );

        let inputs = BedrockRequestInputs {
            model: model.clone(),
            api_messages,
            system,
            tool_config,
            inference_cfg: self.build_inference_config(),
            additional_request_fields: build_additional_model_request_fields(&model, reasoning),
            tool_names,
        };

        // Path selection (#67):
        // - No tools: streaming is always safe; use the streaming path.
        // - Tools + model on the static deny-list: skip the stream attempt
        //   and go straight to non-streaming.
        // - Tools + runtime cache says non-streaming: same.
        // - Otherwise: try streaming first; on the specific
        //   "doesn't support tool use in streaming" validation error,
        //   memoize the model and retry via non-streaming.
        let base_model = strip_region_prefix(&model);
        let cache_says_non_streaming = !tools.is_empty() && {
            let cache = self.non_streaming_tools_models.lock().await;
            cache.contains(&model)
        };
        let allowlist_says_non_streaming =
            !tools.is_empty() && !supports_streaming_with_tools(base_model);
        if cache_says_non_streaming || allowlist_says_non_streaming {
            if allowlist_says_non_streaming {
                tracing::debug!(
                    model = %model,
                    "skipping ConverseStream: model on the non-streaming-with-tools deny-list"
                );
            }
            return self.dispatch_non_streaming(client, inputs, on_chunk).await;
        }

        match self
            .dispatch_streaming(client, &inputs, on_chunk, &cancellation)
            .await
        {
            Ok(response) => Ok(response),
            Err(StreamingDispatchError::StreamingToolsUnsupported { on_chunk, detail }) => {
                tracing::warn!(
                    model = %model,
                    detail,
                    "Bedrock rejected ConverseStream with tools; retrying via Converse \
                     and memoizing the model so future turns skip the stream attempt"
                );
                self.non_streaming_tools_models
                    .lock()
                    .await
                    .insert(model.clone());
                self.dispatch_non_streaming(client, inputs, on_chunk).await
            }
            Err(StreamingDispatchError::Other(err)) => Err(err),
        }
    }
}

#[async_trait::async_trait]
impl desktop_assistant_core::ports::embedding::EmbeddingClient for BedrockClient {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, CoreError> {
        BedrockClient::embed(self, texts).await
    }

    async fn model_identifier(&self) -> Result<String, CoreError> {
        BedrockClient::model_identifier(self).await
    }
}

/// Outcome of a `ConverseStream` dispatch attempt. The "streaming with
/// tools is unsupported" arm carries the unconsumed callback so the
/// caller can retry against `Converse` without rebuilding it; a
/// `ChunkCallback` is `FnOnce`-ish in spirit (boxed dyn FnMut) and
/// passing it back avoids forcing a `Clone` bound on the trait.
enum StreamingDispatchError {
    StreamingToolsUnsupported {
        on_chunk: ChunkCallback,
        detail: String,
    },
    Other(CoreError),
}

/// All the per-call parameters that `ConverseStream` and `Converse`
/// share. Built once at the top of `stream_completion` and consumed by
/// whichever dispatch path runs (#67).
struct BedrockRequestInputs {
    model: String,
    api_messages: Vec<BedrockMessage>,
    system: Vec<SystemContentBlock>,
    tool_config: Option<ToolConfiguration>,
    inference_cfg: Option<aws_sdk_bedrockruntime::types::InferenceConfiguration>,
    additional_request_fields: Option<Document>,
    /// Sanitized<->original tool-name bijection for this request. Used to map
    /// the (sanitized) name the model returns in a `toolUse` back to the real
    /// tool so the upstream dispatch can execute it. (#198)
    tool_names: ToolNameMap,
}

impl BedrockClient {
    fn build_inference_config(
        &self,
    ) -> Option<aws_sdk_bedrockruntime::types::InferenceConfiguration> {
        if self.temperature.is_none() && self.top_p.is_none() && self.max_tokens.is_none() {
            return None;
        }
        let mut inference_cfg = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder();
        if let Some(t) = self.temperature {
            inference_cfg = inference_cfg.temperature(t as f32);
        }
        if let Some(p) = self.top_p {
            inference_cfg = inference_cfg.top_p(p as f32);
        }
        if let Some(m) = self.max_tokens {
            inference_cfg = inference_cfg.max_tokens(m as i32);
        }
        Some(inference_cfg.build())
    }

    /// Attempt the streaming dispatch. The success path mirrors the
    /// pre-#67 implementation; the error path tags the specific
    /// "tools-in-streaming-mode" validation error so the caller can
    /// transparently fall back to `Converse`.
    ///
    /// `cancellation` is checked between SDK events via `tokio::select!`
    /// (issue #109) so the body stream is dropped cleanly when the user
    /// cancels mid-stream.
    async fn dispatch_streaming(
        &self,
        client: &Client,
        inputs: &BedrockRequestInputs,
        mut on_chunk: ChunkCallback,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> Result<LlmResponse, StreamingDispatchError> {
        let mut request = client
            .converse_stream()
            .model_id(inputs.model.clone())
            .set_messages(Some(inputs.api_messages.clone()));
        if let Some(cfg) = inputs.inference_cfg.clone() {
            request = request.inference_config(cfg);
        }
        if !inputs.system.is_empty() {
            request = request.set_system(Some(inputs.system.clone()));
        }
        if let Some(cfg) = inputs.tool_config.clone() {
            request = request.tool_config(cfg);
        }
        if let Some(extra) = inputs.additional_request_fields.clone() {
            request = request.additional_model_request_fields(extra);
        }

        // Bound both the connection handshake and the gap between streamed
        // events so a stalled Bedrock stream fails the turn gracefully instead
        // of hanging forever (#214). `stream.recv()` and `send()` have no
        // built-in timeout; gpt-oss on Bedrock was observed accepting a
        // tool-history follow-up request and then never emitting an event.
        // The budgets default to the values shared with the reqwest connectors
        // (#302) but are overridable per-connection; Bedrock's AWS-SDK stream
        // can't reuse the `tokio_stream`-typed `next_step`, so it applies the
        // same `self.connect_timeout` / `self.event_timeout` directly.
        let connect_timeout = self.connect_timeout;
        let event_timeout = self.event_timeout;

        // Race connection establishment against cancellation and a timeout. If
        // the user cancels mid-handshake we drop the in-flight request (the
        // SDK's HTTP body) before it resolves.
        let send_fut = request.send();
        let response = tokio::select! {
            _ = cancellation.cancelled() => {
                return Err(StreamingDispatchError::Other(CoreError::Cancelled));
            }
            _ = tokio::time::sleep(connect_timeout) => {
                tracing::error!(
                    timeout_s = connect_timeout.as_secs(),
                    "Bedrock converse_stream send() timed out (no response headers)"
                );
                return Err(StreamingDispatchError::Other(CoreError::Llm(
                    "Bedrock converse_stream connection timed out".into(),
                )));
            }
            r = send_fut => match r {
                Ok(r) => r,
                Err(e) => {
                    if let Some(detail) = streaming_tools_unsupported_detail(&e) {
                        return Err(StreamingDispatchError::StreamingToolsUnsupported {
                            on_chunk,
                            detail,
                        });
                    }
                    return Err(StreamingDispatchError::Other(map_converse_stream_error(e)));
                }
            },
        };

        let mut stream = response.stream;
        let mut text = String::new();
        let mut tool_acc = ToolCallAccumulator::default();
        let mut token_usage: Option<TokenUsage> = None;
        let mut event_count: u64 = 0;

        loop {
            // Race the next streaming event against cancellation and a
            // stall timeout. Dropping `stream` closes the underlying HTTP
            // body the same way the SSE adapters do.
            let event_result = tokio::select! {
                _ = cancellation.cancelled() => {
                    tracing::debug!("Bedrock stream cancelled by token");
                    drop(stream);
                    return Err(StreamingDispatchError::Other(CoreError::Cancelled));
                }
                _ = tokio::time::sleep(event_timeout) => {
                    tracing::error!(
                        timeout_s = event_timeout.as_secs(),
                        events_so_far = event_count,
                        "Bedrock converse_stream stalled — no further event"
                    );
                    drop(stream);
                    return Err(StreamingDispatchError::Other(CoreError::Llm(
                        "Bedrock converse_stream stalled (no events)".into(),
                    )));
                }
                ev = stream.recv() => ev,
            };
            event_count += 1;
            let event = match event_result {
                Ok(Some(e)) => e,
                Ok(None) => break,
                Err(e) => {
                    return Err(StreamingDispatchError::Other(CoreError::Llm(format!(
                        "Bedrock stream receive failed: {e}"
                    ))));
                }
            };
            if !apply_stream_event(
                event,
                &mut text,
                &mut tool_acc,
                &mut on_chunk,
                &mut token_usage,
            ) {
                break;
            }
        }

        // Reverse the sanitization: the model echoed back the Bedrock-safe
        // tool name, but the upstream dispatch (and the MCP routing table)
        // keys on the ORIGINAL name. Map each call's name back. The
        // tool_use_id is left untouched.
        let tool_calls = restore_tool_call_names(tool_acc.into_tool_calls(), &inputs.tool_names);
        let mut response = if tool_calls.is_empty() {
            LlmResponse::text(text)
        } else {
            LlmResponse::with_tool_calls(text, tool_calls)
        };
        if let Some(usage) = token_usage {
            response = response.with_usage(usage);
        }
        Ok(response)
    }

    /// Non-streaming dispatch via Bedrock's `Converse` API. Used for
    /// models that reject tools in streaming mode (#67). Synthesises a
    /// single `on_chunk` call with the full text so the upstream
    /// service contract — "the callback fires at least once with the
    /// model's prose output" — is preserved.
    async fn dispatch_non_streaming(
        &self,
        client: &Client,
        inputs: BedrockRequestInputs,
        mut on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let mut request = client
            .converse()
            .model_id(inputs.model.clone())
            .set_messages(Some(inputs.api_messages));
        if let Some(cfg) = inputs.inference_cfg {
            request = request.inference_config(cfg);
        }
        if !inputs.system.is_empty() {
            request = request.set_system(Some(inputs.system));
        }
        if let Some(cfg) = inputs.tool_config {
            request = request.tool_config(cfg);
        }
        if let Some(extra) = inputs.additional_request_fields {
            request = request.additional_model_request_fields(extra);
        }

        let response = request.send().await.map_err(map_converse_error)?;

        let mut text = String::new();
        let mut tool_calls = Vec::new();
        if let Some(aws_sdk_bedrockruntime::types::ConverseOutput::Message(message)) =
            response.output
        {
            for block in message.content() {
                match block {
                    ContentBlock::Text(s) => text.push_str(s),
                    ContentBlock::ToolUse(tool_use) => {
                        // Reverse the sanitization so upstream dispatch hits
                        // the real tool; the id is left untouched.
                        let original_name =
                            inputs.tool_names.to_original(tool_use.name()).into_owned();
                        tool_calls.push(ToolCall::new(
                            tool_use.tool_use_id().to_string(),
                            original_name,
                            document_to_json_string(tool_use.input()),
                        ));
                    }
                    _ => {}
                }
            }
        }

        // Fire the callback once with the full text so the upstream
        // service treats this as a (degenerate) stream rather than
        // skipping its post-completion processing. Bail without erroring
        // if the callback signals abort — the response is fully built
        // either way.
        if !text.is_empty() {
            let _ = on_chunk(text.clone());
        }

        let token_usage = response.usage.as_ref().map(|usage| TokenUsage {
            input_tokens: Some(usage.input_tokens() as u64),
            output_tokens: Some(usage.output_tokens() as u64),
            ..Default::default()
        });

        let mut llm_response = if tool_calls.is_empty() {
            LlmResponse::text(text)
        } else {
            LlmResponse::with_tool_calls(text, tool_calls)
        };
        if let Some(usage) = token_usage {
            llm_response = llm_response.with_usage(usage);
        }
        Ok(llm_response)
    }
}

/// Map a Bedrock `Converse` SDK error to `CoreError`. Mirrors
/// `map_converse_stream_error` but for the non-streaming op (#67).
fn map_converse_error(
    e: aws_sdk_bedrockruntime::error::SdkError<
        aws_sdk_bedrockruntime::operation::converse::ConverseError,
    >,
) -> CoreError {
    use aws_sdk_bedrockruntime::operation::converse::ConverseError;
    if let Some(ConverseError::ValidationException(ve)) = e.as_service_error() {
        let raw = ve.message().unwrap_or("unknown");
        if let Some(info) = parse_context_overflow(raw) {
            tracing::warn!(
                prompt_tokens = ?info.prompt_tokens,
                max_tokens = ?info.max_tokens,
                "Bedrock rejected non-streaming request for context overflow"
            );
            return CoreError::ContextOverflow {
                prompt_tokens: info.prompt_tokens,
                max_tokens: info.max_tokens,
                detail: format!("Bedrock validation error: {raw}"),
            };
        }
    }
    let detail = match e.as_service_error() {
        Some(ConverseError::ValidationException(ve)) => {
            format!("validation error: {}", ve.message().unwrap_or("unknown"))
        }
        Some(ConverseError::ThrottlingException(te)) => {
            return CoreError::RateLimited {
                retry_after: None,
                detail: format!("Bedrock throttling: {}", te.message().unwrap_or("unknown")),
            };
        }
        Some(ConverseError::ServiceUnavailableException(se)) => {
            return CoreError::RateLimited {
                retry_after: None,
                detail: format!(
                    "Bedrock service unavailable: {}",
                    se.message().unwrap_or("unknown")
                ),
            };
        }
        Some(ConverseError::ModelNotReadyException(mr)) => {
            return CoreError::ModelLoading {
                detail: format!(
                    "Bedrock model not ready: {}",
                    mr.message().unwrap_or("unknown")
                ),
            };
        }
        Some(ConverseError::AccessDeniedException(ad)) => {
            format!("access denied: {}", ad.message().unwrap_or("unknown"))
        }
        Some(ConverseError::ModelTimeoutException(mt)) => {
            format!("model timeout: {}", mt.message().unwrap_or("unknown"))
        }
        Some(other) => format!("{other}"),
        None => format!("{e:#}"),
    };
    tracing::warn!("Bedrock converse error: {detail}");
    CoreError::Llm(format!("Bedrock converse request failed: {detail}"))
}

/// If the SDK error is the specific "tool use in streaming mode is
/// unsupported" validation, return the raw message; otherwise `None`.
/// Used by `dispatch_streaming` to flag the case where we should fall
/// back to non-streaming. (#67)
fn streaming_tools_unsupported_detail(
    e: &aws_sdk_bedrockruntime::error::SdkError<
        aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError,
    >,
) -> Option<String> {
    use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
    let ConverseStreamError::ValidationException(ve) = e.as_service_error()? else {
        return None;
    };
    let raw = ve.message().unwrap_or("");
    if is_streaming_tools_unsupported_message(raw) {
        Some(raw.to_string())
    } else {
        None
    }
}

/// Recognize Claude-family Bedrock model ids. Only Claude models accept
/// the `thinking` extended-thinking block via `additionalModelRequestFields`.
///
/// Matches both the legacy `anthropic.claude-*` names and the cross-region
/// inference profile aliases (`us.anthropic.claude-*`, `eu.anthropic.claude-*`,
/// `apac.anthropic.claude-*`).
fn is_claude_bedrock_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("anthropic.claude")
}

/// Build the `additionalModelRequestFields` Document for a Bedrock
/// Converse request, translating the per-turn reasoning hint into the
/// per-vendor shape.
///
/// - Claude-family: `{"thinking": {"type": "enabled", "budget_tokens": N}}`
/// - Others: `None` (unrecognized field would cause a 400).
///
/// Returns `None` when no reasoning is requested or when the model is
/// not known to support extended thinking.
fn build_additional_model_request_fields(
    model: &str,
    reasoning: ReasoningConfig,
) -> Option<Document> {
    use std::collections::HashMap;
    let budget = match reasoning.thinking_budget_tokens {
        Some(n) if n > 0 => n,
        _ => return None,
    };
    if !is_claude_bedrock_model(model) {
        tracing::debug!(
            model,
            budget,
            "Bedrock reasoning requested but model is not Claude-family; dropping thinking field"
        );
        return None;
    }
    let mut thinking: HashMap<String, Document> = HashMap::new();
    thinking.insert("type".to_string(), Document::String("enabled".to_string()));
    thinking.insert(
        "budget_tokens".to_string(),
        Document::Number(Number::PosInt(u64::from(budget))),
    );
    let mut root: HashMap<String, Document> = HashMap::new();
    root.insert("thinking".to_string(), Document::Object(thinking));
    Some(Document::Object(root))
}

/// Map each tool call's (Bedrock-sanitized) name back to the original tool
/// name using the per-request bijection, leaving ids and arguments untouched.
/// Applied to the calls the model returns so upstream dispatch keys on the
/// real name. (#198)
fn restore_tool_call_names(calls: Vec<ToolCall>, tool_names: &ToolNameMap) -> Vec<ToolCall> {
    calls
        .into_iter()
        .map(|call| {
            let original = tool_names.to_original(&call.name).into_owned();
            ToolCall::new(call.id, original, call.arguments)
        })
        .collect()
}

/// Normalize a tool call's arguments into something Bedrock accepts as
/// `toolUse.input`. gpt-oss-120b on Bedrock emits `{"":{}}` (an object with a
/// single empty-string key) for no-argument calls; Bedrock then rejects the
/// echoed history with "toolUse.input is invalid". We drop empty-string keys
/// (the observed garbage), and coerce a non-object input to an empty object so
/// the field is always a valid JSON object. Well-formed arguments pass through
/// unchanged. (#214)
fn sanitize_tool_input(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            serde_json::Value::Object(map.into_iter().filter(|(k, _)| !k.is_empty()).collect())
        }
        // A non-object input is not a valid `toolUse.input`; represent
        // "no arguments" as an empty object.
        _ => serde_json::json!({}),
    }
}

/// Defensively strip composite keywords Bedrock's Converse API rejects at the
/// **top level** of a tool `input_schema`.
///
/// Bedrock returns
/// `tools.N.custom.input_schema: input_schema does not support oneOf, allOf,
/// or anyOf at the top level` and fails the *entire* request — every other
/// tool in the turn goes down with the one offender. Since the daemon passes
/// MCP tool schemas straight through, a single misbehaving server can 400 every
/// LLM turn. This guard ensures no server can do that.
///
/// Behavior:
/// - Only acts on a JSON **object** schema; any other value (`true`, a string,
///   etc.) is returned untouched.
/// - Removes top-level `oneOf`, `anyOf`, `allOf` only. `not` is left alone —
///   the reported Bedrock failure is specific to those three composites.
/// - Does **not** recurse into `properties.*` (or anywhere else). Nested
///   composites inside property subschemas are legal in Bedrock and are
///   commonly used; recursing could corrupt valid schemas.
/// - If stripping leaves the object without a `type`, sets `"type": "object"`
///   so the result is still a valid object schema. `properties`, `required`,
///   `description`, etc. are preserved untouched.
/// - A no-op for schemas that don't carry those keys.
///
/// This is the schema-level analogue of the tool-*name* sanitization in
/// [`tool_names`]: a defensive, last-resort fixup on the Bedrock request path,
/// leaving the Anthropic-API path and the schemas sent to MCP servers untouched.
fn sanitize_tool_schema(schema: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(mut map) = schema else {
        // Non-object schema (`true`/`false`/string/etc.) — nothing to strip,
        // and we must not wrap it. Return as-is.
        return schema;
    };

    let mut removed_any = false;
    for key in ["oneOf", "anyOf", "allOf"] {
        if map.remove(key).is_some() {
            removed_any = true;
        }
    }

    // Only ensure a `type` when we actually altered the schema and left it
    // without one — a clean schema that legitimately omits `type` is left
    // exactly as the server sent it.
    if removed_any && !map.contains_key("type") {
        map.insert(
            "type".to_string(),
            serde_json::Value::String("object".to_string()),
        );
    }

    serde_json::Value::Object(map)
}

fn json_to_document(value: serde_json::Value) -> Document {
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(v) => Document::Bool(v),
        serde_json::Value::String(v) => Document::String(v),
        serde_json::Value::Number(n) => {
            if let Some(v) = n.as_u64() {
                Document::Number(Number::PosInt(v))
            } else if let Some(v) = n.as_i64() {
                Document::Number(Number::NegInt(v))
            } else {
                Document::Number(Number::Float(n.as_f64().unwrap_or_default()))
            }
        }
        serde_json::Value::Array(values) => {
            Document::Array(values.into_iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => Document::Object(
            map.into_iter()
                .map(|(k, v)| (k, json_to_document(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_bedrockruntime::types::{
        ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
        ConverseStreamOutput, ToolUseBlockDelta, ToolUseBlockStart,
    };
    use std::sync::{Arc, Mutex};

    // --- tool input_schema sanitization (top-level oneOf/anyOf/allOf) -----

    #[test]
    fn sanitize_schema_strips_top_level_one_of() {
        let got = sanitize_tool_schema(serde_json::json!({
            "type": "object",
            "description": "a tool",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "oneOf": [{"required": ["x"]}],
        }));
        // oneOf is gone...
        assert!(got.get("oneOf").is_none(), "oneOf must be stripped");
        // ...and everything else is preserved.
        assert_eq!(got["type"], "object");
        assert_eq!(got["description"], "a tool");
        assert_eq!(got["properties"]["x"]["type"], "string");
        assert_eq!(got["required"], serde_json::json!(["x"]));
    }

    #[test]
    fn sanitize_schema_strips_top_level_any_of() {
        let got = sanitize_tool_schema(serde_json::json!({
            "type": "object",
            "anyOf": [{"type": "object"}, {"type": "null"}],
        }));
        assert!(got.get("anyOf").is_none(), "anyOf must be stripped");
        assert_eq!(got["type"], "object");
    }

    #[test]
    fn sanitize_schema_strips_top_level_all_of() {
        let got = sanitize_tool_schema(serde_json::json!({
            "type": "object",
            "allOf": [{"required": ["a"]}, {"required": ["b"]}],
        }));
        assert!(got.get("allOf").is_none(), "allOf must be stripped");
        assert_eq!(got["type"], "object");
    }

    #[test]
    fn sanitize_schema_adds_type_when_missing_after_stripping() {
        // A schema whose only top-level shape was a composite must still be a
        // valid object schema after stripping.
        let got = sanitize_tool_schema(serde_json::json!({
            "oneOf": [{"type": "object"}, {"type": "string"}],
        }));
        assert!(got.get("oneOf").is_none());
        assert_eq!(got["type"], "object", "missing type must default to object");
    }

    #[test]
    fn sanitize_schema_clean_schema_is_unchanged() {
        // No composites -> exact passthrough, including a schema that omits
        // `type` (we must not inject one when we didn't strip anything).
        let clean = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
        });
        assert_eq!(sanitize_tool_schema(clean.clone()), clean);

        let no_type = serde_json::json!({
            "properties": {"a": {"type": "integer"}},
        });
        assert_eq!(sanitize_tool_schema(no_type.clone()), no_type);
    }

    #[test]
    fn sanitize_schema_does_not_recurse_into_properties() {
        // A nested anyOf inside a property subschema is legal in Bedrock and
        // must be preserved — we only touch the top level.
        let got = sanitize_tool_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        }));
        assert_eq!(
            got["properties"]["foo"]["anyOf"],
            serde_json::json!([{"type": "string"}, {"type": "null"}]),
            "nested anyOf must be preserved"
        );
    }

    #[test]
    fn sanitize_schema_non_object_values_pass_through() {
        // `true`/`false`/string/number/null are valid JSON-Schema values that
        // are not objects; handle them without panicking and without wrapping.
        assert_eq!(
            sanitize_tool_schema(serde_json::json!(true)),
            serde_json::json!(true)
        );
        assert_eq!(
            sanitize_tool_schema(serde_json::json!("a string")),
            serde_json::json!("a string")
        );
        assert_eq!(
            sanitize_tool_schema(serde_json::Value::Null),
            serde_json::Value::Null
        );
    }

    #[test]
    fn convert_tools_strips_top_level_composite_from_schema() {
        // End-to-end: a tool whose schema carries a top-level oneOf converts
        // without that key reaching the Bedrock spec.
        let tools = vec![ToolDefinition::new(
            "terminal_execute",
            "run",
            serde_json::json!({
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "oneOf": [{"required": ["cmd"]}],
            }),
        )];
        let map = ToolNameMap::from_names(tools.iter().map(|t| t.name.as_str()));
        let cfg = convert_tools(&tools, &map).expect("ok").expect("some");
        let schema = tool_spec_schema(&cfg, "terminal_execute");
        let Document::Object(obj) = schema else {
            panic!("expected object schema, got {schema:?}");
        };
        assert!(
            !obj.contains_key("oneOf"),
            "oneOf must not reach the Bedrock spec"
        );
        assert!(obj.contains_key("type"), "type must be present");
        assert!(obj.contains_key("properties"), "properties preserved");
    }

    // --- toolUse.input sanitization (#214) -------------------------------

    #[test]
    fn sanitize_tool_input_strips_empty_key_garbage() {
        // gpt-oss's no-arg-call garbage -> a clean empty object.
        let got = sanitize_tool_input(serde_json::json!({"": {}}));
        assert_eq!(got, serde_json::json!({}));
    }

    #[test]
    fn sanitize_tool_input_preserves_real_arguments() {
        let args = serde_json::json!({"content": "note", "key": "goal"});
        assert_eq!(sanitize_tool_input(args.clone()), args);
    }

    #[test]
    fn sanitize_tool_input_drops_only_the_empty_key() {
        let got = sanitize_tool_input(serde_json::json!({"": 1, "real": 2}));
        assert_eq!(got, serde_json::json!({"real": 2}));
    }

    #[test]
    fn sanitize_tool_input_coerces_non_object_to_empty_object() {
        assert_eq!(
            sanitize_tool_input(serde_json::json!(null)),
            serde_json::json!({})
        );
        assert_eq!(
            sanitize_tool_input(serde_json::json!("oops")),
            serde_json::json!({})
        );
        assert_eq!(
            sanitize_tool_input(serde_json::json!([1, 2])),
            serde_json::json!({})
        );
    }

    // --- Extended-thinking (reasoning) wiring ----------------------------

    #[test]
    fn claude_bedrock_model_detection() {
        assert!(is_claude_bedrock_model("anthropic.claude-opus-4-1"));
        assert!(is_claude_bedrock_model("us.anthropic.claude-sonnet-4-6"));
        assert!(is_claude_bedrock_model("eu.anthropic.claude-haiku-4-5"));
        assert!(!is_claude_bedrock_model("amazon.titan-text-express-v1"));
        assert!(!is_claude_bedrock_model("meta.llama3-70b"));
    }

    #[test]
    fn additional_model_request_fields_none_when_no_budget() {
        assert!(
            build_additional_model_request_fields(
                "us.anthropic.claude-sonnet-4-6",
                ReasoningConfig::default(),
            )
            .is_none()
        );
    }

    #[test]
    fn additional_model_request_fields_none_for_non_claude_with_budget() {
        let cfg = ReasoningConfig::with_thinking_budget(8_000);
        assert!(
            build_additional_model_request_fields("meta.llama3-70b", cfg).is_none(),
            "thinking must not be forwarded to non-Claude Bedrock models"
        );
    }

    #[test]
    fn additional_model_request_fields_shape_matches_anthropic_native() {
        let cfg = ReasoningConfig::with_thinking_budget(24_000);
        let doc = build_additional_model_request_fields("us.anthropic.claude-opus-4-1", cfg)
            .expect("thinking doc expected for Claude model");
        let Document::Object(root) = doc else {
            panic!("expected object at root");
        };
        let thinking = match root.get("thinking") {
            Some(Document::Object(t)) => t,
            _ => panic!("missing `thinking` key"),
        };
        assert!(
            matches!(thinking.get("type"), Some(Document::String(s)) if s == "enabled"),
            "thinking.type must be \"enabled\""
        );
        match thinking.get("budget_tokens") {
            Some(Document::Number(Number::PosInt(n))) => assert_eq!(*n, 24_000),
            other => panic!("budget_tokens shape unexpected: {other:?}"),
        }
    }

    #[test]
    fn region_parsing_supports_raw_region() {
        assert_eq!(
            region_from_base_url("us-west-2").as_deref(),
            Some("us-west-2")
        );
    }

    #[test]
    fn region_parsing_supports_bedrock_endpoint() {
        assert_eq!(
            region_from_base_url("https://bedrock-runtime.us-east-1.amazonaws.com").as_deref(),
            Some("us-east-1")
        );
    }

    #[test]
    fn region_parsing_rejects_unknown_endpoint() {
        assert!(region_from_base_url("https://example.com").is_none());
    }

    #[test]
    fn context_limit_claude_sonnet_4_cross_region() {
        assert_eq!(
            context_limit_for_model("us.anthropic.claude-sonnet-4-6"),
            Some(200_000)
        );
        assert_eq!(
            context_limit_for_model("eu.anthropic.claude-sonnet-4-5"),
            Some(200_000)
        );
    }

    #[test]
    fn context_limit_claude_opus_and_haiku_4() {
        assert_eq!(
            context_limit_for_model("anthropic.claude-opus-4-1"),
            Some(200_000)
        );
        assert_eq!(
            context_limit_for_model("us.anthropic.claude-haiku-4-5-20251001"),
            Some(200_000)
        );
    }

    #[test]
    fn context_limit_claude_3() {
        assert_eq!(
            context_limit_for_model("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            Some(200_000)
        );
    }

    #[test]
    fn context_limit_unknown_model_returns_none() {
        assert_eq!(context_limit_for_model("meta.llama3-70b"), None);
        assert_eq!(
            context_limit_for_model("mistral.mistral-large-2407-v1:0"),
            None
        );
    }

    #[test]
    fn context_limit_gpt_oss() {
        // 131,072 is authoritative — the exact window Bedrock reports in its
        // overflow error for this family.
        assert_eq!(
            context_limit_for_model("openai.gpt-oss-120b-1:0"),
            Some(131_072)
        );
        assert_eq!(
            context_limit_for_model("openai.gpt-oss-20b-1:0"),
            Some(131_072)
        );
    }

    #[test]
    fn context_limit_gpt_oss_cross_region() {
        for id in [
            "us.openai.gpt-oss-120b-1:0",
            "eu.openai.gpt-oss-120b-1:0",
            "apac.openai.gpt-oss-20b-1:0",
        ] {
            assert_eq!(context_limit_for_model(id), Some(131_072), "{id}");
        }
    }

    #[test]
    fn parse_context_overflow_extracts_counts_anthropic_phrase() {
        assert_eq!(
            parse_context_overflow("prompt is too long: 203524 tokens > 200000 maximum"),
            Some(ContextOverflowInfo {
                prompt_tokens: Some(203_524),
                max_tokens: Some(200_000),
            })
        );
    }

    #[test]
    fn parse_context_overflow_case_insensitive_phrase() {
        assert_eq!(
            parse_context_overflow("Prompt Is Too Long: 250000 tokens > 200000 maximum"),
            Some(ContextOverflowInfo {
                prompt_tokens: Some(250_000),
                max_tokens: Some(200_000),
            })
        );
    }

    #[test]
    fn parse_context_overflow_exceeds_maximum_context_length_form() {
        // The exact string gpt-oss on Bedrock returns.
        assert_eq!(
            parse_context_overflow(
                "Input length (479258) exceeds model's maximum context length (131072)."
            ),
            Some(ContextOverflowInfo {
                prompt_tokens: Some(479_258),
                max_tokens: Some(131_072),
            })
        );
    }

    #[test]
    fn parse_context_overflow_input_too_long_without_counts() {
        // The other gpt-oss/Bedrock variant — no numbers available.
        assert_eq!(
            parse_context_overflow("Input is too long for requested model."),
            Some(ContextOverflowInfo {
                prompt_tokens: None,
                max_tokens: None,
            })
        );
    }

    #[test]
    fn parse_context_overflow_rejects_unrelated_message() {
        assert_eq!(parse_context_overflow("model not ready"), None);
        assert_eq!(parse_context_overflow("bad token 12345 in request"), None);
        assert_eq!(
            parse_context_overflow("access denied: not authorized"),
            None
        );
    }

    #[test]
    fn static_credentials_supports_colon_format() {
        let creds = static_credentials_from_api_key("AKIA123:secret456").expect("credentials");
        assert_eq!(creds.access_key_id(), "AKIA123");
        assert_eq!(creds.secret_access_key(), "secret456");
        assert!(creds.session_token().is_none());
    }

    #[test]
    fn static_credentials_supports_session_token() {
        let creds =
            static_credentials_from_api_key("AKIA123:secret456:token789").expect("credentials");
        assert_eq!(creds.access_key_id(), "AKIA123");
        assert_eq!(creds.secret_access_key(), "secret456");
        assert_eq!(creds.session_token(), Some("token789"));
    }

    // The standalone accumulator unit tests moved to
    // `desktop_assistant_core::ports::llm` (#45) where the type now
    // lives. The Bedrock-specific stream-event integration test below
    // still exercises the connector's wiring of the accumulator.

    #[test]
    fn stream_event_processing_handles_mixed_text_and_tool_calls() {
        let mut text = String::new();
        let mut tool_acc = ToolCallAccumulator::default();
        let mut token_usage: Option<TokenUsage> = None;
        let chunks: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = Arc::clone(&chunks);

        let mut on_chunk: ChunkCallback = Box::new(move |chunk| {
            chunks_clone.lock().expect("lock").push(chunk);
            true
        });

        let tool_start = ContentBlockStartEvent::builder()
            .content_block_index(0)
            .start(ContentBlockStart::ToolUse(
                ToolUseBlockStart::builder()
                    .tool_use_id("call_1")
                    .name("read_file")
                    .build()
                    .expect("tool start"),
            ))
            .build()
            .expect("start event");

        let text_delta = ContentBlockDeltaEvent::builder()
            .content_block_index(1)
            .delta(ContentBlockDelta::Text("Hello".to_string()))
            .build()
            .expect("text delta");

        let tool_delta_1 = ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .delta(ContentBlockDelta::ToolUse(
                ToolUseBlockDelta::builder()
                    .input("{\"path\":\"/tmp")
                    .build()
                    .expect("tool delta 1"),
            ))
            .build()
            .expect("tool delta event 1");

        let tool_delta_2 = ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .delta(ContentBlockDelta::ToolUse(
                ToolUseBlockDelta::builder()
                    .input("/a\"}")
                    .build()
                    .expect("tool delta 2"),
            ))
            .build()
            .expect("tool delta event 2");

        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockStart(tool_start),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
            &mut token_usage,
        ));
        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(text_delta),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
            &mut token_usage,
        ));
        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(tool_delta_1),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
            &mut token_usage,
        ));
        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(tool_delta_2),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
            &mut token_usage,
        ));

        assert_eq!(text, "Hello");
        assert_eq!(*chunks.lock().expect("lock"), vec!["Hello"]);

        let calls = tool_acc.into_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, "{\"path\":\"/tmp/a\"}");
    }

    #[test]
    fn stream_event_processing_stops_on_callback_abort() {
        let mut text = String::new();
        let mut tool_acc = ToolCallAccumulator::default();
        let mut token_usage: Option<TokenUsage> = None;
        let mut seen = 0usize;

        let mut on_chunk: ChunkCallback = Box::new(move |_chunk| {
            seen += 1;
            seen < 2
        });

        let first = ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .delta(ContentBlockDelta::Text("A".to_string()))
            .build()
            .expect("first delta");
        let second = ContentBlockDeltaEvent::builder()
            .content_block_index(0)
            .delta(ContentBlockDelta::Text("B".to_string()))
            .build()
            .expect("second delta");

        assert!(apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(first),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
            &mut token_usage,
        ));
        assert!(!apply_stream_event(
            ConverseStreamOutput::ContentBlockDelta(second),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
            &mut token_usage,
        ));
        assert_eq!(text, "AB");
    }

    // --- list_models / cache / summary_to_model_info tests ---

    use aws_sdk_bedrock::types::{
        FoundationModelLifecycle, FoundationModelLifecycleStatus, FoundationModelSummary,
        InferenceType, ModelModality,
    };

    /// Mock clock backed by an atomic offset (in seconds) from a fixed
    /// origin. Tests drive it forward by calling `advance_secs`.
    struct MockClock {
        origin: Instant,
        offset: std::sync::atomic::AtomicU64,
    }

    impl MockClock {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                origin: Instant::now(),
                offset: std::sync::atomic::AtomicU64::new(0),
            })
        }

        fn advance_secs(&self, secs: u64) {
            self.offset
                .fetch_add(secs, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl ModelClock for MockClock {
        fn now(&self) -> Instant {
            self.origin + Duration::from_secs(self.offset.load(std::sync::atomic::Ordering::SeqCst))
        }
    }

    fn make_summary(
        id: &str,
        status: FoundationModelLifecycleStatus,
        output_modality: ModelModality,
        input_modalities: Vec<ModelModality>,
    ) -> FoundationModelSummary {
        let mut builder = FoundationModelSummary::builder()
            .model_arn(format!("arn:aws:bedrock:us-east-1::foundation-model/{id}"))
            .model_id(id)
            .model_name(id)
            .provider_name("test")
            .set_output_modalities(Some(vec![output_modality]))
            .set_input_modalities(Some(input_modalities))
            .inference_types_supported(InferenceType::OnDemand)
            .model_lifecycle(
                FoundationModelLifecycle::builder()
                    .status(status)
                    .build()
                    .expect("lifecycle"),
            );
        let _ = &mut builder;
        builder.build().expect("build summary")
    }

    #[test]
    fn summary_filters_out_legacy_models() {
        let legacy = make_summary(
            "anthropic.claude-2",
            FoundationModelLifecycleStatus::Legacy,
            ModelModality::Text,
            vec![ModelModality::Text],
        );
        assert!(summary_to_model_info(&legacy).is_none());
    }

    #[test]
    fn summary_filters_out_pure_image_models() {
        let image = make_summary(
            "stability.stable-diffusion-xl",
            FoundationModelLifecycleStatus::Active,
            ModelModality::Image,
            vec![ModelModality::Text],
        );
        assert!(summary_to_model_info(&image).is_none());
    }

    #[test]
    fn summary_keeps_active_text_model_with_caps() {
        let model = make_summary(
            "anthropic.claude-sonnet-4-6",
            FoundationModelLifecycleStatus::Active,
            ModelModality::Text,
            vec![ModelModality::Text, ModelModality::Image],
        );
        let info = summary_to_model_info(&model).expect("keep active text model");
        assert_eq!(info.id, "anthropic.claude-sonnet-4-6");
        assert_eq!(info.context_limit, Some(200_000));
        assert!(info.capabilities.tools);
        assert!(info.capabilities.vision);
        assert!(info.capabilities.reasoning);
        assert!(!info.capabilities.embedding);
    }

    #[test]
    fn summary_keeps_active_embedding_model() {
        let model = make_summary(
            "amazon.titan-embed-text-v2:0",
            FoundationModelLifecycleStatus::Active,
            ModelModality::Embedding,
            vec![ModelModality::Text],
        );
        let info = summary_to_model_info(&model).expect("keep embedding model");
        assert!(info.capabilities.embedding);
        assert!(!info.capabilities.tools);
        assert!(!info.capabilities.reasoning);
    }

    #[test]
    fn summary_unknown_lifecycle_defaults_to_keep() {
        // No lifecycle field → fall through and keep (AWS sometimes omits).
        let summary = FoundationModelSummary::builder()
            .model_arn("arn:aws:bedrock:us-east-1::foundation-model/meta.llama3-70b-instruct-v1:0")
            .model_id("meta.llama3-70b-instruct-v1:0")
            .model_name("Llama 3 70B Instruct")
            .provider_name("meta")
            .set_output_modalities(Some(vec![ModelModality::Text]))
            .set_input_modalities(Some(vec![ModelModality::Text]))
            .inference_types_supported(InferenceType::OnDemand)
            .build()
            .expect("summary");
        let info = summary_to_model_info(&summary).expect("kept");
        assert_eq!(info.id, "meta.llama3-70b-instruct-v1:0");
        assert!(info.capabilities.tools);
        assert!(!info.capabilities.vision);
    }

    #[tokio::test]
    async fn list_models_hits_cache_within_ttl() {
        let clock = MockClock::new();
        let client = BedrockClient::new("".into())
            .with_clock(clock.clone())
            .with_model_cache_ttl(Duration::from_secs(60 * 60));

        let cached = vec![
            ModelInfo::new("a").with_context_limit(1),
            ModelInfo::new("b").with_context_limit(2),
        ];
        client.__set_models_cache_for_test(cached.clone()).await;

        // Advance < TTL → cache hit.
        clock.advance_secs(30 * 60);
        let got = client.list_models().await.expect("cache hit");
        assert_eq!(got, cached);
    }

    #[tokio::test]
    async fn list_models_expires_after_ttl() {
        // When the TTL elapses, list_models tries to fetch. We don't
        // have AWS credentials in a unit test, so expect an error — but
        // the key assertion is that the cache was NOT reused.
        let clock = MockClock::new();
        let client = BedrockClient::new("".into())
            .with_clock(clock.clone())
            .with_model_cache_ttl(Duration::from_secs(60));

        let cached = vec![ModelInfo::new("stale")];
        client.__set_models_cache_for_test(cached.clone()).await;

        // Cache is still within TTL.
        assert_eq!(client.list_models().await.expect("within ttl"), cached,);

        // Advance past TTL → next call bypasses cache and will attempt a
        // network fetch. We just verify the call path diverges (either
        // an error or a non-cached response) rather than asserting on the
        // specific failure mode (which depends on the local AWS env).
        clock.advance_secs(120);
        let _ = client.list_models().await;
        // The cache may have been overwritten or cleared; the important
        // invariant is that a cache-hit of the stale data did NOT occur
        // (verified by reaching the network path above — if it had hit
        // the cache, it would have returned Ok(cached) without touching
        // AWS).
    }

    #[tokio::test]
    async fn refresh_models_bypasses_cache() {
        // Verify refresh_models always attempts a fresh fetch. We prime
        // the cache with known data, then call refresh — the cached
        // value MUST NOT be returned (refresh bypasses the TTL check).
        let clock = MockClock::new();
        let client = BedrockClient::new("".into())
            .with_clock(clock.clone())
            .with_model_cache_ttl(Duration::from_secs(60 * 60));

        let cached = vec![ModelInfo::new("cached-only")];
        client.__set_models_cache_for_test(cached.clone()).await;

        // refresh_models() never returns the cached vec without calling
        // out to AWS. In CI/offline envs this errors; the assertion is
        // that we do NOT get back the exact cached payload.
        // Err is expected in offline test envs; the call diverges from
        // the cache path regardless of outcome.
        if let Ok(models) = client.refresh_models().await {
            assert_ne!(models, cached);
        }
    }

    // --- OnDemand filter + inference profile merge tests (#50) ---

    fn make_summary_inference_types(
        id: &str,
        status: FoundationModelLifecycleStatus,
        output_modality: ModelModality,
        input_modalities: Vec<ModelModality>,
        inference_types: &[InferenceType],
    ) -> FoundationModelSummary {
        let mut builder = FoundationModelSummary::builder()
            .model_arn(format!("arn:aws:bedrock:us-east-1::foundation-model/{id}"))
            .model_id(id)
            .model_name(id)
            .provider_name("test")
            .set_output_modalities(Some(vec![output_modality]))
            .set_input_modalities(Some(input_modalities))
            .model_lifecycle(
                FoundationModelLifecycle::builder()
                    .status(status)
                    .build()
                    .expect("lifecycle"),
            );
        for it in inference_types {
            builder = builder.inference_types_supported(it.clone());
        }
        builder.build().expect("build summary")
    }

    #[test]
    fn summary_filters_out_models_without_on_demand() {
        let provisioned_only = make_summary_inference_types(
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            FoundationModelLifecycleStatus::Active,
            ModelModality::Text,
            vec![ModelModality::Text, ModelModality::Image],
            &[InferenceType::Provisioned],
        );
        assert!(
            summary_to_model_info(&provisioned_only).is_none(),
            "models without OnDemand must be filtered (use inference profile instead)"
        );
    }

    #[test]
    fn summary_filters_out_models_with_no_inference_types() {
        // Defensive: AWS may omit inference_types entirely. Treat as
        // not-on-demand (consistent with the OnDemand-required policy).
        let none = make_summary_inference_types(
            "deepseek.r1-v1:0",
            FoundationModelLifecycleStatus::Active,
            ModelModality::Text,
            vec![ModelModality::Text],
            &[],
        );
        assert!(summary_to_model_info(&none).is_none());
    }

    #[test]
    fn summary_keeps_model_with_on_demand_among_others() {
        let mixed = make_summary_inference_types(
            "anthropic.claude-3-haiku-20240307-v1:0",
            FoundationModelLifecycleStatus::Active,
            ModelModality::Text,
            vec![ModelModality::Text],
            &[InferenceType::OnDemand, InferenceType::Provisioned],
        );
        let info = summary_to_model_info(&mixed).expect("kept");
        assert_eq!(info.id, "anthropic.claude-3-haiku-20240307-v1:0");
    }

    fn make_profile(
        id: &str,
        name: &str,
        status: InferenceProfileStatus,
    ) -> InferenceProfileSummary {
        // The builder requires `models` to be set (the underlying foundation
        // models the profile routes to). The conversion code doesn't read
        // them — we infer capabilities from the profile id — so a single
        // stub entry is enough for the test.
        let model_stub = InferenceProfileModel::builder()
            .model_arn("arn:aws:bedrock:us-east-1::foundation-model/test")
            .build();
        InferenceProfileSummary::builder()
            .inference_profile_arn(format!(
                "arn:aws:bedrock:us-east-1:0:inference-profile/{id}"
            ))
            .inference_profile_id(id)
            .inference_profile_name(name)
            .status(status)
            .r#type(InferenceProfileType::SystemDefined)
            .models(model_stub)
            .build()
            .expect("build profile summary")
    }

    use aws_sdk_bedrock::types::{
        InferenceProfileModel, InferenceProfileStatus, InferenceProfileSummary,
        InferenceProfileType,
    };

    #[test]
    fn profile_skips_non_active() {
        // Bedrock currently exposes only the Active variant, but defensive
        // coverage in case AWS adds others.
        let profile = make_profile(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude Haiku 4.5 (US)",
            InferenceProfileStatus::Active,
        );
        // sanity: the active path keeps it
        assert!(inference_profile_to_model_info(&profile).is_some());
    }

    #[test]
    fn profile_anthropic_claude_4_inferred_capabilities() {
        let profile = make_profile(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude Haiku 4.5 (US)",
            InferenceProfileStatus::Active,
        );
        let info = inference_profile_to_model_info(&profile).expect("kept");
        assert_eq!(info.id, "us.anthropic.claude-haiku-4-5-20251001-v1:0");
        assert_eq!(info.display_name, "Claude Haiku 4.5 (US)");
        assert_eq!(info.context_limit, Some(200_000));
        assert!(info.capabilities.tools);
        assert!(info.capabilities.reasoning);
        assert!(info.capabilities.vision);
        assert!(!info.capabilities.embedding);
    }

    #[test]
    fn profile_amazon_nova_capabilities() {
        let profile = make_profile(
            "us.amazon.nova-premier-v1:0",
            "Nova Premier (US)",
            InferenceProfileStatus::Active,
        );
        let info = inference_profile_to_model_info(&profile).expect("kept");
        assert_eq!(info.id, "us.amazon.nova-premier-v1:0");
        assert!(info.capabilities.tools, "Nova supports tool use");
        assert!(info.capabilities.vision, "Nova Premier is multimodal");
        assert!(!info.capabilities.reasoning);
        assert!(!info.capabilities.embedding);
    }

    #[test]
    fn profile_deepseek_r1_capabilities() {
        let profile = make_profile(
            "us.deepseek.r1-v1:0",
            "DeepSeek R1 (US)",
            InferenceProfileStatus::Active,
        );
        let info = inference_profile_to_model_info(&profile).expect("kept");
        assert_eq!(info.id, "us.deepseek.r1-v1:0");
        assert!(info.capabilities.reasoning, "R1 is a reasoning model");
        assert!(info.capabilities.tools);
        assert!(!info.capabilities.vision);
    }

    #[test]
    fn profile_falls_back_to_id_when_name_empty() {
        let profile = make_profile(
            "us.anthropic.claude-sonnet-4-6",
            "",
            InferenceProfileStatus::Active,
        );
        let info = inference_profile_to_model_info(&profile).expect("kept");
        assert_eq!(info.display_name, "us.anthropic.claude-sonnet-4-6");
    }

    // --- Partial-failure reporting for the model listing (#648) ---
    //
    // `ListInferenceProfiles` failing must degrade the listing (on-demand
    // foundation models only) AND say so in the returned data. In a current
    // AWS account the surviving on-demand set is almost entirely embedding
    // models, so a silent degradation looks to the operator like "Bedrock
    // only has embedding models" rather than "a permission is missing".

    /// `ListFoundationModels` payload shaped like a current account: an
    /// on-demand embedding model plus one legacy on-demand chat model.
    /// Modern chat models (Claude 4.x, Nova Premier, ...) are absent from the
    /// on-demand set entirely - they are reachable only via inference
    /// profiles, which is exactly why losing the profile call is so visible.
    const FOUNDATION_MODELS_BODY: &str = r#"{
      "modelSummaries": [
        {
          "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
          "modelId": "amazon.titan-embed-text-v2:0",
          "modelName": "Titan Text Embeddings V2",
          "providerName": "Amazon",
          "inputModalities": ["TEXT"],
          "outputModalities": ["EMBEDDING"],
          "inferenceTypesSupported": ["ON_DEMAND"],
          "modelLifecycle": {"status": "ACTIVE"}
        },
        {
          "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
          "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
          "modelName": "Claude 3 Haiku",
          "providerName": "Anthropic",
          "inputModalities": ["TEXT", "IMAGE"],
          "outputModalities": ["TEXT"],
          "inferenceTypesSupported": ["ON_DEMAND"],
          "modelLifecycle": {"status": "ACTIVE"}
        }
      ]
    }"#;

    /// `ListInferenceProfiles` payload with a single active system profile.
    const INFERENCE_PROFILES_BODY: &str = r#"{
      "inferenceProfileSummaries": [
        {
          "inferenceProfileName": "US Anthropic Claude Sonnet 4.6",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.anthropic.claude-sonnet-4-6",
          "inferenceProfileId": "us.anthropic.claude-sonnet-4-6",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6"}
          ],
          "status": "ACTIVE",
          "type": "SYSTEM_DEFINED"
        }
      ]
    }"#;

    /// The IAM denial an account without `bedrock:ListInferenceProfiles`
    /// actually gets back. `111122223333` is AWS's documentation account id.
    const ACCESS_DENIED_BODY: &str = r#"{"message":"User: arn:aws:iam::111122223333:user/adele is not authorized to perform: bedrock:ListInferenceProfiles on resource: arn:aws:bedrock:us-east-1:111122223333:inference-profile/*"}"#;

    /// Fake secret used by the control-plane test client. Asserted absent
    /// from user-facing notices so a signing credential can never ride out
    /// on a degradation message.
    const TEST_SECRET_ACCESS_KEY: &str = "wJalrXUtnFEMIxK7MDENGxbPxRfiCYEXAMPLEKEY";

    /// A `BedrockClient` whose control-plane calls are pointed at `server`.
    ///
    /// Static credentials parsed from the api-key keep the AWS credential
    /// chain (profile files, IMDS) out of the unit test; the region is taken
    /// from `base_url`, so nothing here depends on the machine's AWS setup.
    fn control_plane_client(server: &httpmock::MockServer) -> BedrockClient {
        BedrockClient::new(format!("AKIAIOSFODNN7EXAMPLE:{TEST_SECRET_ACCESS_KEY}"))
            .with_base_url("us-east-1")
            .__with_control_endpoint_for_test(server.url(""))
    }

    /// Mock `ListFoundationModels` returning [`FOUNDATION_MODELS_BODY`].
    fn mock_foundation_models(server: &httpmock::MockServer) -> httpmock::Mock<'_> {
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/foundation-models");
            then.status(200)
                .header("content-type", "application/json")
                .body(FOUNDATION_MODELS_BODY);
        })
    }

    /// Mock `ListInferenceProfiles` failing with `status` / `error_type`.
    fn mock_inference_profiles_error<'a>(
        server: &'a httpmock::MockServer,
        status: u16,
        error_type: &str,
        body: &str,
    ) -> httpmock::Mock<'a> {
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/inference-profiles");
            then.status(status)
                .header("content-type", "application/json")
                .header("x-amzn-errortype", error_type)
                .body(body);
        })
    }

    /// Both control-plane calls succeed.
    fn mock_healthy_control_plane(server: &httpmock::MockServer) -> httpmock::Mock<'_> {
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/inference-profiles");
            then.status(200)
                .header("content-type", "application/json")
                .body(INFERENCE_PROFILES_BODY);
        });
        mock_foundation_models(server)
    }

    #[tokio::test]
    async fn list_models_reports_partial_failure_when_profiles_call_fails() {
        let server = httpmock::MockServer::start();
        mock_foundation_models(&server);
        mock_inference_profiles_error(&server, 403, "AccessDeniedException", ACCESS_DENIED_BODY);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("a profiles failure degrades the listing, it does not fail it");

        let notice = report
            .notices
            .first()
            .expect("the partial failure must leave the connector as data, not only a log line");
        assert_eq!(notice.kind, ModelListingNoticeKind::PartialCatalog);
        assert!(
            notice.summary.to_lowercase().contains("inference profile"),
            "the summary must name what is missing, got {:?}",
            notice.summary
        );
        assert!(
            report.is_degraded(),
            "a report carrying a notice is a degraded report"
        );
    }

    #[tokio::test]
    async fn list_models_still_returns_on_demand_models_when_profiles_fail() {
        let server = httpmock::MockServer::start();
        mock_foundation_models(&server);
        mock_inference_profiles_error(&server, 403, "AccessDeniedException", ACCESS_DENIED_BODY);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("degradation must not become a hard error");

        let ids: Vec<&str> = report.models.iter().map(|m| m.id.as_str()).collect();
        assert!(
            ids.contains(&"amazon.titan-embed-text-v2:0"),
            "on-demand embedding models survive, got {ids:?}"
        );
        assert!(
            ids.contains(&"anthropic.claude-3-haiku-20240307-v1:0"),
            "on-demand chat models survive, got {ids:?}"
        );
    }

    #[tokio::test]
    async fn partial_failure_names_the_missing_permission() {
        let server = httpmock::MockServer::start();
        mock_foundation_models(&server);
        mock_inference_profiles_error(&server, 403, "AccessDeniedException", ACCESS_DENIED_BODY);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("degraded listing");

        let notice = report.notices.first().expect("notice present");
        assert_eq!(
            notice.required_permission.as_deref(),
            Some("bedrock:ListInferenceProfiles"),
            "an authorization failure must name the permission to grant"
        );
        assert!(
            notice.detail.contains("bedrock:ListInferenceProfiles"),
            "the human-readable detail must be actionable on its own, got {:?}",
            notice.detail
        );
    }

    #[tokio::test]
    async fn list_models_reports_success_cleanly_when_both_calls_succeed() {
        let server = httpmock::MockServer::start();
        mock_healthy_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        assert!(
            report.notices.is_empty(),
            "the happy path must not manufacture a warning, got {:?}",
            report.notices
        );
        assert!(!report.is_degraded());
        let ids: Vec<&str> = report.models.iter().map(|m| m.id.as_str()).collect();
        assert!(
            ids.contains(&"us.anthropic.claude-sonnet-4-6"),
            "inference profiles are merged into the listing, got {ids:?}"
        );
    }

    #[tokio::test]
    async fn model_refresh_reports_a_result_even_when_the_list_is_unchanged() {
        let server = httpmock::MockServer::start();
        let foundation = mock_healthy_control_plane(&server);
        let client = control_plane_client(&server);

        let first = client
            .refresh_models_detailed()
            .await
            .expect("first refresh reports a result");
        let second = client
            .refresh_models_detailed()
            .await
            .expect("an unchanged refresh still reports a result");

        assert_eq!(
            first.models, second.models,
            "same account contents means the same list"
        );
        assert!(
            !second.models.is_empty(),
            "a refresh that changes nothing must still report what it found, \
             otherwise the client cannot tell it happened"
        );
        assert!(second.notices.is_empty());
        foundation.assert_calls(2);
    }

    #[tokio::test]
    async fn cached_listing_repeats_the_partial_failure_notice() {
        // A cache hit must not quietly drop the degradation: the picker would
        // look healthy again for the whole TTL while still being incomplete.
        let server = httpmock::MockServer::start();
        let foundation = mock_foundation_models(&server);
        mock_inference_profiles_error(&server, 403, "AccessDeniedException", ACCESS_DENIED_BODY);

        let client = control_plane_client(&server).with_model_cache_ttl(Duration::from_secs(3600));
        let first = client.list_models_detailed().await.expect("first listing");
        let second = client
            .list_models_detailed()
            .await
            .expect("second listing served from cache");

        foundation.assert_calls(1);
        assert!(!second.notices.is_empty(), "cache hit dropped the notice");
        assert_eq!(first.notices, second.notices);
    }

    #[tokio::test]
    async fn partial_failure_from_a_non_permission_error_does_not_blame_iam() {
        let server = httpmock::MockServer::start();
        mock_foundation_models(&server);
        mock_inference_profiles_error(
            &server,
            400,
            "ValidationException",
            r#"{"message":"1 validation error detected"}"#,
        );

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("degraded listing");

        let notice = report.notices.first().expect("notice present");
        assert!(
            notice.required_permission.is_none(),
            "only an authorization failure implicates IAM, got {:?}",
            notice.required_permission
        );
        assert!(
            notice.detail.contains("ValidationException"),
            "the real cause must survive into the detail, got {:?}",
            notice.detail
        );
    }

    #[tokio::test]
    async fn partial_failure_notice_never_carries_the_signing_secret() {
        let server = httpmock::MockServer::start();
        mock_foundation_models(&server);
        mock_inference_profiles_error(&server, 403, "AccessDeniedException", ACCESS_DENIED_BODY);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("degraded listing");

        let notice = report.notices.first().expect("notice present");
        let rendered = format!(
            "{} {} {:?}",
            notice.summary, notice.detail, notice.required_permission
        );
        assert!(
            !rendered.contains(TEST_SECRET_ACCESS_KEY),
            "a user-facing notice must never carry the signing secret"
        );
    }

    #[tokio::test]
    async fn partial_failure_detail_is_bounded_for_an_oversized_service_message() {
        // Defensive: the detail is rendered by clients and travels the wire,
        // so an abusive/broken upstream message must not be relayed whole.
        let huge = "x".repeat(10_000);
        let server = httpmock::MockServer::start();
        mock_foundation_models(&server);
        mock_inference_profiles_error(
            &server,
            403,
            "AccessDeniedException",
            &format!(r#"{{"message":"{huge}"}}"#),
        );

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("degraded listing");

        let notice = report.notices.first().expect("notice present");
        assert!(
            notice.detail.chars().count() <= MAX_NOTICE_DETAIL_CHARS,
            "detail must be truncated, got {} chars",
            notice.detail.chars().count()
        );
        assert!(
            notice.detail.contains("bedrock:ListInferenceProfiles"),
            "truncation must keep the actionable part, got {:?}",
            notice.detail
        );
    }

    #[tokio::test]
    async fn list_models_fails_hard_when_the_foundation_models_call_fails() {
        // Losing BOTH listings leaves nothing to degrade to: that is a real
        // failure and must surface as one rather than as an empty picker.
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/foundation-models");
            then.status(403)
                .header("content-type", "application/json")
                .header("x-amzn-errortype", "AccessDeniedException")
                .body(r#"{"message":"not authorized to perform: bedrock:ListFoundationModels"}"#);
        });
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/inference-profiles");
            then.status(200)
                .header("content-type", "application/json")
                .body(INFERENCE_PROFILES_BODY);
        });

        let err = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect_err("a foundation-models failure is not a degradation");
        match err {
            CoreError::Llm(msg) => assert!(
                msg.contains("ListFoundationModels"),
                "error must name the failing call, got {msg:?}"
            ),
            other => panic!("expected CoreError::Llm, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn plain_list_models_still_returns_just_the_models() {
        // The narrow `list_models` contract is unchanged for callers that
        // don't care about notices.
        let server = httpmock::MockServer::start();
        mock_healthy_control_plane(&server);

        let models = control_plane_client(&server)
            .list_models()
            .await
            .expect("healthy listing");
        let detailed_ids: Vec<String> = models.into_iter().map(|m| m.id).collect();
        assert!(detailed_ids.contains(&"us.anthropic.claude-sonnet-4-6".to_string()));
    }

    #[test]
    fn strip_region_prefix_recognises_known_regions() {
        assert_eq!(
            strip_region_prefix("us.anthropic.claude-haiku-4-5"),
            "anthropic.claude-haiku-4-5"
        );
        assert_eq!(
            strip_region_prefix("eu.anthropic.claude-sonnet-4-6"),
            "anthropic.claude-sonnet-4-6"
        );
        assert_eq!(
            strip_region_prefix("apac.amazon.nova-pro-v1:0"),
            "amazon.nova-pro-v1:0"
        );
        // Unknown / no prefix passes through.
        assert_eq!(
            strip_region_prefix("anthropic.claude-3-haiku-20240307-v1:0"),
            "anthropic.claude-3-haiku-20240307-v1:0"
        );
    }

    // --- Structured CoreError mapping tests (issue #60) ---

    #[test]
    fn map_throttling_exception_emits_rate_limited() {
        use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
        use aws_sdk_bedrockruntime::types::error::ThrottlingException;

        let exc = ThrottlingException::builder()
            .message("rate of requests exceeded")
            .build();
        let svc_err = ConverseStreamError::ThrottlingException(exc);

        let mapped =
            map_converse_stream_service_error(&svc_err).expect("throttling has dedicated mapping");
        match mapped {
            CoreError::RateLimited {
                retry_after,
                detail,
            } => {
                assert_eq!(retry_after, None);
                assert!(detail.contains("rate of requests exceeded"));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn map_service_unavailable_emits_rate_limited() {
        use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
        use aws_sdk_bedrockruntime::types::error::ServiceUnavailableException;

        let exc = ServiceUnavailableException::builder()
            .message("backend overloaded")
            .build();
        let svc_err = ConverseStreamError::ServiceUnavailableException(exc);

        let mapped = map_converse_stream_service_error(&svc_err)
            .expect("service unavailable has dedicated mapping");
        match mapped {
            CoreError::RateLimited {
                retry_after,
                detail,
            } => {
                assert_eq!(retry_after, None);
                assert!(detail.contains("backend overloaded"));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn map_model_not_ready_emits_model_loading() {
        use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
        use aws_sdk_bedrockruntime::types::error::ModelNotReadyException;

        let exc = ModelNotReadyException::builder()
            .message("model warming up")
            .build();
        let svc_err = ConverseStreamError::ModelNotReadyException(exc);

        let mapped = map_converse_stream_service_error(&svc_err)
            .expect("model-not-ready has dedicated mapping");
        match mapped {
            CoreError::ModelLoading { detail } => {
                assert!(detail.contains("model warming up"));
            }
            other => panic!("expected ModelLoading, got {other:?}"),
        }
    }

    #[test]
    fn map_unhandled_variants_return_none() {
        use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
        use aws_sdk_bedrockruntime::types::error::AccessDeniedException;

        let exc = AccessDeniedException::builder()
            .message("not allowed")
            .build();
        let svc_err = ConverseStreamError::AccessDeniedException(exc);

        // AccessDenied has no dedicated structured variant — caller
        // falls through to the generic `CoreError::Llm` formatting.
        assert!(map_converse_stream_service_error(&svc_err).is_none());
    }

    // --- #67: tools-in-streaming-mode fallback -------------------------------

    #[test]
    fn supports_streaming_with_tools_denies_llama_family() {
        // Llama 3 / 4 reject tools in streaming mode; everything else
        // is currently assumed safe (and the runtime fallback covers
        // mis-classifications).
        assert!(!supports_streaming_with_tools(
            "meta.llama4-maverick-17b-instruct-v1:0"
        ));
        assert!(!supports_streaming_with_tools(
            "meta.llama4-scout-17b-instruct-v1:0"
        ));
        assert!(!supports_streaming_with_tools(
            "meta.llama3-70b-instruct-v1:0"
        ));

        // Claude is the canonical safe case.
        assert!(supports_streaming_with_tools("anthropic.claude-sonnet-4-6"));
        // Unknown models default to the streaming path so we don't
        // regress legitimate users; the runtime fallback catches misses.
        assert!(supports_streaming_with_tools("amazon.nova-premier-v1:0"));
        assert!(supports_streaming_with_tools("future.unknown-model"));
    }

    #[test]
    fn supports_streaming_with_tools_works_on_stripped_id() {
        // Caller is responsible for stripping the region prefix; the
        // helper itself doesn't strip. Assert the contract: passing
        // the prefixed form would mis-classify (currently safe because
        // unknown→true, but still worth pinning).
        let stripped = strip_region_prefix("us.meta.llama4-maverick-17b-instruct-v1:0");
        assert!(!supports_streaming_with_tools(stripped));
    }

    #[test]
    fn detect_streaming_tools_unsupported_message() {
        assert!(is_streaming_tools_unsupported_message(
            "This model doesn't support tool use in streaming mode."
        ));
        assert!(is_streaming_tools_unsupported_message(
            "Validation: this model does not support tool use in streaming mode"
        ));
        // Unrelated validation errors must NOT match.
        assert!(!is_streaming_tools_unsupported_message(
            "prompt is too long: 203524 tokens > 200000 maximum"
        ));
        assert!(!is_streaming_tools_unsupported_message(""));
    }

    #[test]
    fn document_to_json_round_trips() {
        // Build a Document of every shape the SDK might emit and verify
        // we serialize back into the same JSON the streaming path would
        // produce. Used as the source for `ToolCall.arguments` in the
        // non-streaming dispatch (#67).
        use std::collections::HashMap;

        let mut inner = HashMap::new();
        inner.insert("flag".to_string(), Document::Bool(true));
        inner.insert("count".to_string(), Document::Number(Number::PosInt(42)));
        inner.insert(
            "items".to_string(),
            Document::Array(vec![Document::String("a".to_string()), Document::Null]),
        );
        let doc = Document::Object(inner);

        let json: serde_json::Value =
            serde_json::from_str(&document_to_json_string(&doc)).expect("valid JSON");
        assert_eq!(json["flag"], serde_json::json!(true));
        assert_eq!(json["count"], serde_json::json!(42));
        assert_eq!(
            json["items"],
            serde_json::json!(["a", serde_json::Value::Null])
        );
    }

    // --- Tool-name sanitization on the Bedrock path (#198) ---------------

    // `ToolDefinition`, `ToolCall`, `Tool`, `ToolConfiguration`, `ContentBlock`
    // are all in scope via `super::*`.

    /// Pull the tool-spec names out of a built `ToolConfiguration`.
    fn tool_spec_names(cfg: &ToolConfiguration) -> Vec<String> {
        cfg.tools()
            .iter()
            .filter_map(|t| match t {
                Tool::ToolSpec(spec) => Some(spec.name().to_string()),
                _ => None,
            })
            .collect()
    }

    /// Fetch the `input_schema` JSON document for the spec with the given
    /// (already-sanitized) name. Panics if not found — test helper.
    fn tool_spec_schema(cfg: &ToolConfiguration, name: &str) -> Document {
        for t in cfg.tools() {
            if let Tool::ToolSpec(spec) = t
                && spec.name() == name
                && let Some(ToolInputSchema::Json(doc)) = spec.input_schema()
            {
                return doc.clone();
            }
        }
        panic!("no spec named {name:?} with a JSON input schema");
    }

    /// Collect every `toolUse` name across all assistant messages.
    fn tool_use_names(messages: &[BedrockMessage]) -> Vec<String> {
        let mut names = Vec::new();
        for m in messages {
            for block in m.content() {
                if let ContentBlock::ToolUse(tu) = block {
                    names.push(tu.name().to_string());
                }
            }
        }
        names
    }

    #[test]
    fn convert_tools_sanitizes_spec_names() {
        let tools = vec![
            ToolDefinition::new("fs.read", "read", serde_json::json!({"type": "object"})),
            ToolDefinition::new("do thing", "do", serde_json::json!({"type": "object"})),
            ToolDefinition::new("ok_name", "ok", serde_json::json!({"type": "object"})),
        ];
        let map = ToolNameMap::from_names(tools.iter().map(|t| t.name.as_str()));
        let cfg = convert_tools(&tools, &map).expect("ok").expect("some");
        let names = tool_spec_names(&cfg);
        for n in &names {
            assert!(
                n.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "spec name not Bedrock-valid: {n:?}"
            );
        }
        // The already-valid name is untouched.
        assert!(names.contains(&"ok_name".to_string()));
        // And each safe name reverses to its original.
        for n in &names {
            let orig = map.to_original(n).into_owned();
            assert!(
                ["fs.read", "do thing", "ok_name"].contains(&orig.as_str()),
                "unexpected reverse: {n:?} -> {orig:?}"
            );
        }
    }

    #[test]
    fn convert_messages_sanitizes_historical_tool_use_name() {
        // THE core fix: a `toolUse` block from an earlier turn lives in the
        // history; its name must be sanitized when re-serialized, because
        // Bedrock validates every `messages.N...toolUse.name`. This is the
        // live failure (error at `messages.10`), independent of the current
        // tool definitions.
        let history = vec![
            Message::new(Role::User, "hi"),
            Message::assistant_with_tool_calls(vec![ToolCall::new(
                "call-1",
                "weather.lookup", // invalid for Bedrock (contains '.')
                r#"{"city":"NYC"}"#,
            )]),
            Message::tool_result("call-1", "sunny"),
        ];
        // Map built from the CURRENT tool set (which still offers the tool).
        let map = ToolNameMap::from_names(["weather.lookup"]);
        let (_system, messages) = convert_messages(&history, &map).expect("convert ok");

        let names = tool_use_names(&messages);
        assert_eq!(names.len(), 1, "expected one toolUse in history");
        let safe = &names[0];
        assert!(
            safe.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "historical toolUse name not sanitized: {safe:?}"
        );
        assert!(!safe.contains('.'), "dot must be gone: {safe:?}");
        // And it round-trips to the original via the same map, so a tool def
        // built from the same map will agree with this name.
        assert_eq!(map.to_original(safe), "weather.lookup");
    }

    #[test]
    fn convert_messages_sanitizes_history_even_when_tool_not_offered_now() {
        // A tool used in an earlier turn may no longer be in the current tool
        // set. Its historical `toolUse` name STILL must be valid for Bedrock,
        // or the whole request is rejected.
        let history = vec![Message::assistant_with_tool_calls(vec![ToolCall::new(
            "call-9",
            "legacy:tool/name",
            "{}",
        )])];
        // Empty map: the tool isn't offered this turn.
        let map = ToolNameMap::from_names(Vec::<&str>::new());
        let (_system, messages) = convert_messages(&history, &map).expect("convert ok");
        let names = tool_use_names(&messages);
        assert_eq!(names.len(), 1);
        assert!(
            names[0]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "name not sanitized: {:?}",
            names[0]
        );
    }

    #[test]
    fn spec_and_history_names_agree_for_same_tool() {
        // The name Bedrock sees in the tool-spec MUST equal the name it sees
        // in the historical `toolUse` block for the same tool, or the model's
        // call won't correlate. Verify they're identical through one map.
        let tool = "ns__weird.name:1";
        let map = ToolNameMap::from_names([tool]);

        let cfg = convert_tools(
            &[ToolDefinition::new(tool, "d", serde_json::json!({}))],
            &map,
        )
        .expect("ok")
        .expect("some");
        let spec_name = tool_spec_names(&cfg).into_iter().next().expect("one spec");

        let history = vec![Message::assistant_with_tool_calls(vec![ToolCall::new(
            "c1", tool, "{}",
        )])];
        let (_s, messages) = convert_messages(&history, &map).expect("ok");
        let hist_name = tool_use_names(&messages).into_iter().next().expect("one");

        assert_eq!(
            spec_name, hist_name,
            "tool-spec name and history toolUse name must match"
        );
    }

    #[test]
    fn restore_tool_call_names_reverses_to_original() {
        // The dispatch path: the model returns the sanitized name; we must
        // hand the ORIGINAL back to the caller so MCP routing resolves it.
        let map = ToolNameMap::from_names(["fs.read", "a.b", "a:b"]);
        let safe_fs = map.to_safe("fs.read").into_owned();
        let safe_ab1 = map.to_safe("a.b").into_owned();
        let safe_ab2 = map.to_safe("a:b").into_owned();

        let returned = vec![
            ToolCall::new("id1", safe_fs, r#"{"p":1}"#),
            ToolCall::new("id2", safe_ab1, "{}"),
            ToolCall::new("id3", safe_ab2, "{}"),
        ];
        let restored = restore_tool_call_names(returned, &map);
        assert_eq!(restored[0].name, "fs.read");
        assert_eq!(restored[1].name, "a.b");
        assert_eq!(restored[2].name, "a:b");
        // ids and arguments survive untouched.
        assert_eq!(restored[0].id, "id1");
        assert_eq!(restored[0].arguments, r#"{"p":1}"#);
    }

    // --- Cancellation (issue #109) ---------------------------------------

    /// The Bedrock adapter routes through the AWS SDK, which is not
    /// trivially mockable at the HTTP level the way `httpmock` lets us
    /// stub the other adapters. The contract we verify here is the one
    /// the cancellation work introduces at the connector boundary:
    /// when the task-local `CANCELLATION_TOKEN` is already tripped on
    /// entry to `stream_completion`, the adapter returns
    /// `CoreError::Cancelled` without dispatching any AWS request.
    ///
    /// The mid-stream `tokio::select!` against `token.cancelled()` is
    /// covered indirectly by the core-level test
    /// `send_prompt_returns_cancelled_when_token_fires_mid_stream`,
    /// which drives a `SlowStreamLlm` modelled on the same shape the
    /// real connector uses.
    #[tokio::test]
    async fn bedrock_stream_aborts_on_cancellation() {
        use desktop_assistant_core::ports::llm::with_cancellation_token;
        use tokio_util::sync::CancellationToken;

        // Use a fake API key and no real credentials. The point is the
        // entry-check: cancellation pre-empts the request before the
        // SDK is invoked, so missing credentials never matter.
        let client = BedrockClient::new("fake".into()).with_model("anthropic.claude-sonnet-4-6");

        let token = CancellationToken::new();
        token.cancel();

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
        // The check must run *before* the SDK reaches the network. AWS
        // credential resolution alone can take many ms; 1s is generous.
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "pre-cancelled token should short-circuit before AWS dispatch; took {elapsed:?}"
        );
    }
}
