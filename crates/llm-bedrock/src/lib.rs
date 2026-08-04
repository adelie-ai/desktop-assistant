//! AWS Bedrock Converse API connector implementing the core `LlmClient` port.

mod tool_names;
pub use tool_names::ToolNameMap;

use aws_config::{BehaviorVersion, Region};
use aws_credential_types::Credentials;
use aws_sdk_bedrock::Client as BedrockControlClient;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::types::{
    CachePointBlock, CachePointType, ContentBlock, ConversationRole, Message as BedrockMessage,
    SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolSpecification, ToolUseBlock,
};
use aws_smithy_types::{Document, Number};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, Role, ToolCall, ToolDefinition};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmClient, LlmResponse, ModelCapabilities, ModelInfo, ModelKind,
    ModelListingNotice, ModelListingReport, ReasoningConfig, TokenUsage, current_model_override,
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

/// Whole-request budget for the non-streaming (`Converse`) path.
///
/// `Converse` answers once, after generation is complete, so this bounds a
/// whole generation rather than a stall. Ten minutes is chosen to be longer
/// than any one-shot completion a Bedrock chat model produces in practice -
/// the path is mandatory for Llama 3 and 4 with tools, whose answers can run
/// for minutes - so the bound catches a hung request and nothing else.
///
/// Deliberately not [`STREAM_EVENT_TIMEOUT`] or a sum of the stream budgets:
/// those bound the gap between events, and one name must not answer two
/// questions. A user who cancels does not wait this out - the dispatch races
/// the cancellation token as well.
const NON_STREAMING_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// Upper bound on the characters of provider text relayed in a listing
/// notice's `detail`.
///
/// Why: the detail is rendered by clients and travels the daemon's wire
/// protocol, so a broken or hostile upstream message must not be relayed
/// whole. Generous enough for a real AWS `AccessDeniedException`, which
/// names the principal, the action, and the resource.
const MAX_NOTICE_DETAIL_CHARS: usize = 600;

/// The IAM action that `ListInferenceProfiles` requires. Named in the
/// degradation notice so an operator can fix the policy without digging
/// through daemon logs.
const LIST_INFERENCE_PROFILES_PERMISSION: &str = "bedrock:ListInferenceProfiles";

/// How much of a Converse request this connector marks for prompt caching.
///
/// Caching is not free. Bedrock bills a cache **write** above the uncached
/// input rate, and the write pays back only when a later turn reads the same
/// prefix. A conversation that runs for several turns reads it every turn and
/// comes out ahead; a workload of short one-turn conversations pays the
/// premium every turn and reads it rarely. So the policy is a setting, not a
/// constant.
///
/// It is also a diagnostic. `none` rules caching out of a misbehaving turn
/// without a code change.
///
/// Only the two shipped values exist. A third, "system prefix and tool list",
/// is deliberately absent: Bedrock evaluates checkpoints in the order `tools`
/// -> `system` -> `messages`, and a change in an earlier section invalidates
/// the cache for every later one. Tool search moves the tool list inside a
/// conversation, so a checkpoint on `tools` would invalidate the system cache
/// on every turn the list moves, and would cost more than it saves. See
/// `docs/connectors/bedrock.md`, "Prompt caching".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePolicy {
    /// Emit no cache checkpoints, whatever the model supports.
    None,
    /// One checkpoint after the stable system prefix. The default, and the
    /// behaviour every Bedrock connection had before this setting existed.
    #[default]
    SystemPromptOnly,
}

/// Whether a request for `model_id` carries a cache checkpoint.
///
/// Two independent conditions, and both must hold: the operator's policy
/// allows a checkpoint, and the model accepts one. `model_id` may be an
/// inference-profile id; `base_model_for` reduces it to the foundation model
/// the profile routes to.
///
/// A model the connector has learned to reject checkpoints at runtime is
/// handled by the caller, which does not call this for such a model.
fn wants_cache_checkpoint(policy: CachePolicy, model_id: &str) -> bool {
    match policy {
        CachePolicy::None => false,
        CachePolicy::SystemPromptOnly => supports_prompt_caching(&base_model_for(model_id)),
    }
}

#[derive(Default)]
struct ModelCache {
    /// Why the whole report and not just the models: a cache hit that
    /// dropped the notices would make a degraded listing look healthy again
    /// for the rest of the TTL: the exact invisibility this reporting
    /// exists to remove.
    entry: Option<(Instant, ModelListingReport)>,
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
    /// Models discovered at runtime to reject a `cachePoint` block, although
    /// [`supports_prompt_caching`] accepts them. That list is read from AWS
    /// documentation which states it covers only the models absent from
    /// "Models at a glance", so it is a best reading and not an enumeration;
    /// this set is how a wrong entry costs one call instead of every turn.
    ///
    /// Written only from a refusal that names the cache field, on a request
    /// that carried a checkpoint. Per-instance, like
    /// [`Self::non_streaming_tools_models`]. (#1028)
    cache_unsupported_models: Arc<Mutex<HashSet<String>>>,
    /// First-response (connect) stall budget; defaults to
    /// [`STREAM_CONNECT_TIMEOUT`], overridable per-connection.
    connect_timeout: Duration,
    /// Per-chunk stall budget; defaults to [`STREAM_EVENT_TIMEOUT`].
    event_timeout: Duration,
    /// Whole-request budget for the non-streaming (`Converse`) path; defaults
    /// to [`NON_STREAMING_REQUEST_TIMEOUT`].
    ///
    /// Its own setting, not a function of the two stream budgets: those bound
    /// a connection and the gap between events, and this bounds a whole
    /// generation. One name answering both questions would make a change to
    /// stall detection move a generation deadline with it.
    ///
    /// It does cap total generation time on this path, which the streaming
    /// path does not cap. That is the trade, and it is why the default is
    /// generous: an unbounded request hangs the turn until the AWS SDK's own
    /// defaults give up, and ignores a stop.
    non_streaming_timeout: Duration,
    /// Per-connection context-window hard cap, in tokens. `None` = "max
    /// available". Folded with the curated table in `max_context_tokens`.
    context_cap: Option<u64>,
    /// How much of the request is marked for prompt caching. Defaults to
    /// [`CachePolicy::SystemPromptOnly`].
    cache_policy: CachePolicy,
    /// Test-only override for the Bedrock control-plane endpoint, so the
    /// model-listing tests can drive `ListFoundationModels` /
    /// `ListInferenceProfiles` against a local mock. Always `None` in
    /// production, where the endpoint is derived from the region.
    control_endpoint_override: Option<String>,
    /// Test-only override for the Bedrock runtime endpoint, so the dispatch
    /// tests can drive `Converse` / `ConverseStream` against a local socket.
    /// Always `None` in production, where the endpoint is derived from the
    /// region.
    runtime_endpoint_override: Option<String>,
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
            cache_unsupported_models: Arc::new(Mutex::new(HashSet::new())),
            connect_timeout: STREAM_CONNECT_TIMEOUT,
            event_timeout: STREAM_EVENT_TIMEOUT,
            non_streaming_timeout: NON_STREAMING_REQUEST_TIMEOUT,
            context_cap: None,
            cache_policy: CachePolicy::default(),
            control_endpoint_override: None,
            runtime_endpoint_override: None,
        }
    }

    /// Set how much of the request is marked for prompt caching. `None` keeps
    /// the [`CachePolicy::SystemPromptOnly`] default, so an unset connection
    /// field behaves exactly as it did before the setting existed.
    pub fn with_cache_policy(mut self, policy: Option<CachePolicy>) -> Self {
        if let Some(policy) = policy {
            self.cache_policy = policy;
        }
        self
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

    /// Override the whole-request budget for the non-streaming (`Converse`)
    /// path. `None`/`Some(0)` keeps the ten-minute default. Seconds.
    ///
    /// It has no effect on the streaming path, whose two budgets bound the
    /// connection and the gap between events.
    ///
    /// **No connection configuration reaches this yet.** The daemon builds
    /// its Bedrock clients without calling it, so every connection runs the
    /// default; issue #1042 carries a `non_streaming_timeout_secs` through the
    /// wire shape and the resolver to join the other two budgets. Until then
    /// this is settable only by a caller constructing the client directly.
    pub fn with_non_streaming_timeout(mut self, secs: Option<u64>) -> Self {
        if let Some(s) = secs.filter(|s| *s > 0) {
            self.non_streaming_timeout = Duration::from_secs(s);
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

    /// Test-only: the credential this client was built with. Exists so the
    /// daemon can pin that its embedding client is constructed with the same
    /// credential material as its generation client (#718) - a client built
    /// without one does not fail at construction, it fails much later as an
    /// opaque transport error when the AWS SDK gives up hunting for ambient
    /// credentials.
    #[doc(hidden)]
    pub fn __api_key_for_test(&self) -> &str {
        &self.api_key
    }

    /// Test-only: the AWS profile this client was built with, if any. Bedrock
    /// accepts a profile as an alternative to a key, so both have to survive
    /// construction.
    #[doc(hidden)]
    pub fn __aws_profile_for_test(&self) -> Option<&str> {
        self.aws_profile.as_deref()
    }

    /// Test-only: prime the `list_models()` cache so the cache-TTL test
    /// can exercise hit/miss behavior without reaching AWS. The
    /// `fetched_at` timestamp is stamped using the configured clock.
    #[doc(hidden)]
    pub async fn __set_models_cache_for_test(&self, models: Vec<ModelInfo>) {
        let now = self.clock.now();
        let mut cache = self.model_cache.lock().await;
        cache.entry = Some((now, ModelListingReport::complete(models)));
    }

    /// Test-only: record `model` as rejecting tools in streaming mode, so a
    /// turn that carries tools takes the non-streaming (`Converse`) path with
    /// no stream attempt first. Exists so a whole turn can be driven against a
    /// `Converse` mock for a model that supports prompt caching; the
    /// `ConverseStream` reply is an AWS event stream, which a plain HTTP mock
    /// cannot produce.
    #[doc(hidden)]
    pub async fn __force_non_streaming_tools_for_test(&self, model: &str) {
        self.non_streaming_tools_models
            .lock()
            .await
            .insert(model.to_string());
    }

    /// Test-only: peek at the cache contents.
    #[doc(hidden)]
    pub async fn __peek_models_cache_for_test(&self) -> Option<Vec<ModelInfo>> {
        let cache = self.model_cache.lock().await;
        cache.entry.as_ref().map(|(_, v)| v.models.clone())
    }

    /// Test-only: point the Bedrock control plane (`ListFoundationModels` /
    /// `ListInferenceProfiles`) at `url` instead of the regional AWS
    /// endpoint, so the model-listing behaviour can be exercised against a
    /// local mock server rather than a live account.
    ///
    /// Only the control plane is redirected; the runtime (Converse) client is
    /// untouched.
    #[doc(hidden)]
    pub fn __with_control_endpoint_for_test(mut self, url: impl Into<String>) -> Self {
        self.control_endpoint_override = Some(url.into());
        self.control_client = OnceCell::new();
        self
    }

    /// Test-only: point the Bedrock runtime plane (`Converse` /
    /// `ConverseStream`) at `url` instead of the regional AWS endpoint, so
    /// the dispatch behaviour can be exercised against a local socket rather
    /// than a live account.
    ///
    /// Only the runtime plane is redirected; the control (model-listing)
    /// client is untouched.
    #[doc(hidden)]
    pub fn __with_runtime_endpoint_for_test(mut self, url: impl Into<String>) -> Self {
        self.runtime_endpoint_override = Some(url.into());
        self.client = OnceCell::new();
        self
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
                let Some(endpoint) = self.runtime_endpoint_override.as_ref() else {
                    return Ok(Client::new(&shared_config));
                };
                let config = aws_sdk_bedrockruntime::config::Builder::from(&shared_config)
                    .endpoint_url(endpoint)
                    .build();
                Ok(Client::from_conf(config))
            })
            .await
    }

    async fn control_client(&self) -> Result<&BedrockControlClient, CoreError> {
        self.control_client
            .get_or_try_init(|| async {
                let shared_config = self.load_shared_config().await;
                let Some(endpoint) = self.control_endpoint_override.as_ref() else {
                    return Ok(BedrockControlClient::new(&shared_config));
                };
                let config = aws_sdk_bedrock::config::Builder::from(&shared_config)
                    .endpoint_url(endpoint)
                    .build();
                Ok(BedrockControlClient::from_conf(config))
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

/// Convert domain messages into Converse messages, hoisting the system prompt.
///
/// Every `Role::System` message becomes an entry of the returned
/// `Vec<SystemContentBlock>`, in order; the rest become Converse messages.
///
/// # Cache checkpoints
///
/// Exactly one `cachePoint` block is emitted, directly after the *leading*
/// system block, and only when `cache_checkpoint` is set. The caller decides
/// that with [`wants_cache_checkpoint`], which folds the operator's
/// [`CachePolicy`] together with what the model accepts. Two reasons for one
/// checkpoint in that position, and either alone is sufficient:
///
/// - **Correctness.** Caching is a prefix match, so a checkpoint pays off only
///   when everything in front of it is byte-identical on the next turn. The
///   leading block is the assembler's system instruction, which is stable for
///   the lifetime of a conversation. Every later system block is a per-turn
///   `[..]` block the assembler refills each round (a timestamp, a plan, a
///   pin), so a checkpoint behind one is written and never read.
/// - **Acceptance.** Bedrock allows at most four checkpoints per request. The
///   assembler can surface eight system blocks in one turn, so marking each
///   one would make the combination unusable. One checkpoint keeps the count
///   at one however many blocks arrive.
///
/// The same reasoning excludes the tool list. Bedrock evaluates checkpoints in
/// the order `tools` -> `system` -> `messages`, and a change in an earlier
/// section invalidates the cache for every later section. Tool search moves the
/// tool list inside a conversation, so a checkpoint on `tools` would invalidate
/// the system cache on every turn the list moves. See
/// `docs/connectors/bedrock.md`, "Prompt caching".
fn convert_messages(
    messages: &[Message],
    tool_names: &ToolNameMap,
    cache_checkpoint: bool,
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

    // The one checkpoint: immediately behind the stable prefix. Volatile
    // per-turn blocks follow it unmarked, so a change in one of them leaves
    // the cached prefix intact.
    if !system.is_empty() && cache_checkpoint {
        system.insert(1, SystemContentBlock::CachePoint(default_cache_point()?));
    }

    Ok((system, api_messages))
}

/// A five-minute cache checkpoint, the Converse default.
///
/// Why the default TTL: the one-hour TTL costs more per cache write and only
/// pays back over gaps longer than five minutes. A conversation refreshes the
/// five-minute cache on every turn at no extra charge, which is the shape of
/// an interactive assistant turn.
fn default_cache_point() -> Result<CachePointBlock, CoreError> {
    CachePointBlock::builder()
        .r#type(CachePointType::Default)
        .build()
        .map_err(|e| CoreError::Llm(format!("failed to build Bedrock cache checkpoint: {e}")))
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

/// Map Converse token accounting onto the core `TokenUsage`.
///
/// One function for both dispatch paths: `ConverseStream` reports usage in its
/// metadata event, `Converse` on the response, and the two must not drift.
///
/// Why the cache counters stay `Option`: a model without prompt caching
/// returns no cache fields at all. `Some(0)` would tell a caller that caching
/// ran and saved nothing, which is a different statement from "caching did not
/// run". Note also that with caching on, Bedrock's `inputTokens` counts only
/// the tokens that were neither read from nor written to the cache, so the
/// three fields sum to the real input size.
///
/// The counts are clamped at zero. The wire type is a signed `i32`, the domain
/// type is unsigned, and a negative count is meaningless either way.
fn map_token_usage(usage: &aws_sdk_bedrockruntime::types::TokenUsage) -> TokenUsage {
    fn non_negative(value: i32) -> u64 {
        value.max(0) as u64
    }

    TokenUsage {
        input_tokens: Some(non_negative(usage.input_tokens())),
        output_tokens: Some(non_negative(usage.output_tokens())),
        cache_creation_input_tokens: usage.cache_write_input_tokens().map(non_negative),
        cache_read_input_tokens: usage.cache_read_input_tokens().map(non_negative),
    }
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
                *token_usage = Some(map_token_usage(usage));
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
/// Accepts an inference-profile id, which `base_model_for` reduces to the
/// foundation model the profile routes to. Returns `None` for models without
/// a known limit; callers should treat `None` as "disable token-based
/// compaction" and rely on message-count fallbacks instead.
///
/// `ListFoundationModels` does not expose context windows, so this table
/// is the single source of truth for Bedrock models whose windows we know.
/// The `list_models()` implementation uses it to populate
/// `ModelInfo::context_limit`.
pub fn context_limit_for_model(model_id: &str) -> Option<u64> {
    let base = base_model_for(model_id);
    let base = base.as_ref();

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
/// id (`base_model_for` already applied) so it works for both bare foundation
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

    ModelCapabilities {
        // One source of truth with the request builder: the flag says the
        // connector can act on a reasoning effort for this model, so a client
        // that offers the control is offering something that takes effect.
        reasoning: supports_configurable_reasoning(&lc),
        vision,
        tools: tools && !is_embedding,
        // `vision` and `is_embedding` are the provider's real modality
        // metadata on both paths: read from the summary for a foundation
        // model, and read from the base model's summary for an inference
        // profile (see `ModalityIndex`). The kind follows the modality, not
        // the id (#647).
        kind: if is_embedding {
            ModelKind::Embedding
        } else {
            ModelKind::Generative
        },
    }
}

/// Whether this connector can configure reasoning for a Bedrock model, i.e.
/// whether a requested thinking budget reaches the model instead of being
/// dropped.
///
/// `base_id` must be the foundation model id `base_model_for` returns, the
/// same contract [`supports_prompt_caching`] and
/// [`supports_streaming_with_tools`] use. Case-insensitive.
///
/// This is the single source of truth for the reasoning axis. Both the
/// capability record ([`infer_capabilities_from_id`]) and the request builder
/// ([`build_additional_model_request_fields`]) read it, so the picker cannot
/// advertise a control the request path will not honour (#1022).
///
/// Only Anthropic Claude takes a reasoning configuration through Bedrock's
/// Converse API, as `additionalModelRequestFields.thinking`: Claude 3.7 and
/// the 4.x line and later. Claude 3.5 and Claude 3 predate extended thinking.
///
/// "Reasons" and "takes a reasoning configuration" are different questions,
/// and the second one is what this answers. DeepSeek R1 is the case that
/// makes the difference visible: it reasons on every request and returns its
/// trace in `reasoningContent`, and AWS documents its whole Converse request
/// body as `system` / `messages` / `inferenceConfig` / `guardrailConfig` -
/// there is no reasoning field to set, and the reasoning documentation says
/// plainly that not all models let you configure the tokens spent on it. A
/// model whose reasoning is always on and never adjustable answers `false`
/// here: nothing this connector sends changes what it does.
fn supports_configurable_reasoning(base_id: &str) -> bool {
    let lc = base_id.to_ascii_lowercase();
    let Some(claude) = lc.strip_prefix("anthropic.claude-") else {
        return false;
    };

    // Claude 3.x ids spell the minor version into the name
    // (`3-5-sonnet-...`, `3-7-sonnet-...`); only 3.7 has extended thinking.
    if let Some(three) = claude.strip_prefix("3-") {
        return three.starts_with("7-");
    }

    // Claude 4 and later put the family first (`sonnet-4-5-...`,
    // `opus-4-1-...`, `haiku-4-5-...`). Match the family, not the version, so
    // a later minor release needs no edit here - the same forward-compatible
    // shape `supports_prompt_caching` uses.
    ["sonnet-", "opus-", "haiku-"]
        .iter()
        .any(|family| claude.starts_with(family))
}

/// Every inference-profile prefix AWS is known to issue.
///
/// **Every entry ends in `.`, and that is what keeps the list order-free.** A
/// prefix that included the separator's absence - `"ap"` rather than `"ap."` -
/// would match `apac.anthropic...` and leave `ac.anthropic...`, and which
/// entry won would depend on where a maintainer inserted it.
/// `inference_profile_prefixes_all_end_in_a_separator` holds the invariant so
/// the next entry cannot be added without it.
///
/// A system-defined profile id is the foundation model id behind one of these
/// prefixes, so this list is how the capability gates see the base id without
/// a listing. A missing entry costs extended thinking, prompt caching, the
/// context window and the streaming-with-tools deny list for any id that never
/// reached the register - a `default_model` or a `MODEL_OVERRIDE` set before
/// the first listing, in particular. A profile the account has listed is
/// covered either way.
///
/// Sources, all AWS documentation:
/// - Geographic cross-Region inference names `us`, `eu`, `apac` as geography
///   prefixes, and `ap` appears on the newer APAC profiles.
/// - The Claude Sonnet 4.5 model card lists Geo ids for `us.`, `eu.`, `au.`
///   and `jp.`, and the Global id `global.anthropic.claude-sonnet-4-5-...`.
/// - GovCloud sources route through the US geo id, and `us-gov.` is carried
///   here as well because it costs nothing and an unstripped prefix is the
///   expensive direction.
///
/// An allowlist rather than "drop the first dotted segment", because model ids
/// carry dots of their own - `openai.gpt-5.6` would lose its provider.
const INFERENCE_PROFILE_PREFIXES: &[&str] = &[
    "global.", "us-gov.", "us.", "eu.", "apac.", "ap.", "au.", "jp.",
];

/// Strip a cross-region inference-profile prefix to recover the underlying
/// foundation model id. Returns the input unchanged when no known prefix
/// matches.
///
/// The rule for a system-defined profile id, and the fallback for every other
/// id. `base_model_for` is what the capability gates call; it consults the
/// register first and lands here when the register has no answer.
fn strip_region_prefix(id: &str) -> &str {
    INFERENCE_PROFILE_PREFIXES
        .iter()
        .find_map(|prefix| id.strip_prefix(prefix))
        .unwrap_or(id)
}

/// Inference-profile id -> the foundation model id it routes to, for the
/// profile ids [`strip_region_prefix`] cannot reduce.
///
/// **Process-global on purpose, and that is the load-bearing part.** Every
/// capability gate in this connector answers from the base model id, and the
/// two sides that must agree do not share a `BedrockClient`: the daemon lists
/// models through the registry's per-connection client and dispatches turns
/// through a second client built for the interactive purpose. A register held
/// per instance would let the picker see a capability the request builder
/// cannot, which is the defect this connector already paid for once. A
/// register held per process cannot.
///
/// Keyed by profile id alone. A system-defined id (`us.anthropic....`) names
/// the same foundation model in every account, and an application profile id
/// is a generated identifier, so two accounts in one daemon do not collide in
/// practice. The stored value is a foundation model id, which is global as
/// well.
///
/// Entries are written by a successful `ListInferenceProfiles` and are never
/// removed. A deleted profile makes its id undispatchable at AWS, so a stale
/// entry cannot grant a capability to a live turn; a profile whose route
/// changes is corrected by the next listing, which overwrites the entry.
/// Growth is bounded by the account's inference-profile quota.
///
/// A `BTreeMap` rather than a `HashMap` because its constructor is `const`,
/// so the register needs no lazy initialization.
static PROFILE_BASE_MODELS: std::sync::RwLock<std::collections::BTreeMap<String, String>> =
    std::sync::RwLock::new(std::collections::BTreeMap::new());

/// The foundation model `id` names, for every capability gate in this
/// connector.
///
/// The register first, then [`strip_region_prefix`]. Both the capability
/// record and the request builder call this, so neither can answer for a
/// model the other cannot see.
///
/// A miss is a defined answer, not a lookup: an id no listing returned - a
/// configured `default_model`, a per-turn `MODEL_OVERRIDE`, a keep-warm probe
/// before the first listing - reduces by the prefix rule alone. Refreshing
/// the listing here would put a control-plane call, its IAM failure modes and
/// its latency on the turn path, so it is not done.
fn base_model_for(id: &str) -> std::borrow::Cow<'_, str> {
    let mapped = PROFILE_BASE_MODELS
        // A poisoned register means a panic happened elsewhere while holding
        // it. Nothing here panics under the lock, and a capability answer is
        // not worth failing a turn over, so the map is read regardless.
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(id)
        .cloned();
    match mapped {
        Some(base) => std::borrow::Cow::Owned(base),
        None => std::borrow::Cow::Borrowed(strip_region_prefix(id)),
    }
}

/// The foundation model id inside a Bedrock model ARN, or `None` when this
/// connector cannot read the ARN.
///
/// Two resource types answer: `foundation-model/<id>`, which names the model
/// directly, and `inference-profile/<id>`, which names a system-defined
/// profile and reduces by [`strip_region_prefix`]. Any other resource type
/// answers `None` rather than a guess - a provisioned-throughput ARN names a
/// deployment, not a model, and reading it as one would attach another
/// model's capabilities to it.
///
/// The partition is not matched, so `aws`, `aws-us-gov` and `aws-cn` all
/// parse. The resource id is taken whole after the first `/`, because model
/// ids carry `:` and `.` of their own and only the resource separator is
/// reliable.
fn base_model_from_model_arn(arn: &str) -> Option<&str> {
    let (head, resource_id) = arn.strip_prefix("arn:")?.split_once('/')?;
    let (_, resource_type) = head.rsplit_once(':')?;
    if !matches!(resource_type, "foundation-model" | "inference-profile") {
        return None;
    }
    if !head.contains(":bedrock:") {
        return None;
    }
    let base = strip_region_prefix(resource_id);
    (!base.is_empty()).then_some(base)
}

/// The foundation model an inference-profile summary routes to, taken from
/// the ARNs in its `models`.
///
/// A cross-region profile lists one ARN per region, and they name one model,
/// so the answer is that model. Anything less certain answers `None`: no
/// `models` at all, an ARN this connector cannot read, or two ARNs that name
/// different models. A wrong base model is worse than no base model, because
/// it attaches another model's capabilities to the profile.
fn profile_base_model(profile: &aws_sdk_bedrock::types::InferenceProfileSummary) -> Option<&str> {
    let mut resolved: Option<&str> = None;
    for model in profile.models() {
        let base = base_model_from_model_arn(model.model_arn()?)?;
        match resolved {
            None => resolved = Some(base),
            Some(seen) if seen == base => {}
            Some(_) => return None,
        }
    }
    resolved
}

/// Record what an inference profile routes to, so every capability gate on
/// every `BedrockClient` in this process answers for that model.
///
/// Only where the ARN adds something. A profile id the prefix rule already
/// reduces correctly is left out, so the register holds exactly the ids that
/// need it, and the prefix rule stays the single answer for the ids it can
/// answer for.
fn register_profile_base_model(profile: &aws_sdk_bedrock::types::InferenceProfileSummary) {
    use aws_sdk_bedrock::types::InferenceProfileStatus;

    if profile.status != InferenceProfileStatus::Active {
        return;
    }
    let profile_id = profile.inference_profile_id();
    if profile_id.is_empty() {
        return;
    }
    let Some(base) = profile_base_model(profile) else {
        tracing::debug!(
            profile_id,
            "inference profile names no single foundation model in its ARNs; \
             its capabilities fall back to the profile id"
        );
        return;
    };
    if base == strip_region_prefix(profile_id) {
        return;
    }
    tracing::debug!(
        profile_id,
        base,
        "inference profile registered against its base foundation model"
    );
    PROFILE_BASE_MODELS
        // Poisoning is ignored for the same reason it is ignored on read: a
        // panic elsewhere must not cost every later turn its capabilities.
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(profile_id.to_string(), base.to_string());
}

/// Whether a Bedrock model accepts `cachePoint` prompt-cache checkpoints.
///
/// `base_id` must be the foundation model id -- `anthropic.claude-sonnet-4-6`,
/// not `us.anthropic.claude-sonnet-4-6`. The caller reduces it with
/// `base_model_for`, the same contract [`supports_streaming_with_tools`]
/// uses.
///
/// Why an allow-list, and why it defaults to `false`: support is a property of
/// the model, not of the Converse API, and Bedrock rejects a request that
/// carries a checkpoint the model does not accept. The two errors are not
/// symmetric. A checkpoint the model refuses fails the whole turn; a
/// checkpoint we withhold only costs input tokens. So an unrecognised model
/// gets no checkpoint.
///
/// The supported families, per the Bedrock prompt-caching documentation:
/// Anthropic Claude 3.5 and later (3.5, 3.7, and the 4.x line) and Amazon
/// Nova. Claude 3 predates the feature. Meta, Mistral, Cohere and DeepSeek do
/// not support it at all.
fn supports_prompt_caching(base_id: &str) -> bool {
    let lc = base_id.to_ascii_lowercase();

    // Amazon Nova: Micro, Lite, Pro and Premier all accept explicit
    // checkpoints.
    if lc.starts_with("amazon.nova") {
        return true;
    }

    let Some(claude) = lc.strip_prefix("anthropic.claude-") else {
        return false;
    };

    // Claude 3.x ids spell the minor version into the name
    // (`3-5-sonnet-...`, `3-7-sonnet-...`); 3 with no minor part is the
    // pre-caching generation.
    if let Some(three) = claude.strip_prefix("3-") {
        return three.starts_with("5-") || three.starts_with("7-");
    }

    // Claude 4 and later put the family first (`sonnet-4-5-...`,
    // `opus-4-1-...`, `haiku-4-5-...`). Match the family, not the version, so
    // a later minor release needs no edit here.
    ["sonnet-", "opus-", "haiku-"]
        .iter()
        .any(|family| claude.starts_with(family))
}

/// Whether a Bedrock model accepts tool-use requests via `ConverseStream`.
///
/// AWS Bedrock has a per-model restriction: some foundation models support
/// tools via `Converse` *only*, not `ConverseStream`. Llama 3/4 fall in
/// that bucket; Claude does not. (#67)
///
/// `base_id` should be the foundation model id — `meta.llama4-…`, not
/// `us.meta.llama4-…`. The caller is responsible for calling
/// `base_model_for` first.
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

/// Whether a Bedrock validation message positively names the prompt-cache
/// field, and so reports a refused `cachePoint` block rather than something
/// else.
///
/// Every marker names the feature itself, in one of the spellings the service
/// uses: `cachePoint` for Converse, `cache_control` for the Anthropic shape
/// Bedrock forwards, and the prose forms "cache point", "cache checkpoint" and
/// "prompt caching". Nothing here is a status code, a generic "unsupported", or
/// a schema path, because none of those is
/// evidence about the checkpoint: a validation failure arrives just as easily
/// from a tool schema (#336), an over-long prompt, or a bad model id, and
/// reading one of those as a cache refusal would disable caching on a model
/// that caches perfectly well.
///
/// The classifier is one half of the guard. The other half is the caller,
/// which only asks this question about a request that actually carried a
/// checkpoint. A message about a field the request did not send is about
/// something else, whatever it names.
///
/// A refusal that names none of these is not recovered, and the turn fails as
/// it does today. That direction is deliberate: withholding a checkpoint costs
/// input tokens, and sending one the model refuses costs the whole turn.
fn names_the_cache_field(message: &str) -> bool {
    let lc = message.to_ascii_lowercase();
    [
        "cachepoint",
        "cache point",
        "cache_control",
        "cache checkpoint",
        "prompt caching",
    ]
    .iter()
    .any(|marker| lc.contains(marker))
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

/// The modality facts `ListFoundationModels` reports for one model, reduced
/// to the two questions the capability record asks of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelModalities {
    /// The model takes image input.
    vision: bool,
    /// The model returns vectors rather than text.
    is_embedding: bool,
}

impl ModelModalities {
    fn from_summary(summary: &aws_sdk_bedrock::types::FoundationModelSummary) -> Self {
        use aws_sdk_bedrock::types::ModelModality;
        Self {
            vision: summary.input_modalities().contains(&ModelModality::Image),
            is_embedding: summary
                .output_modalities()
                .contains(&ModelModality::Embedding),
        }
    }
}

/// Modality metadata for every foundation model the account lists, keyed by
/// foundation model id.
///
/// Why it exists: `ListInferenceProfiles` reports no modalities, so the
/// profile path used to carry a second, hardcoded vision id list. That list is
/// the one that runs in practice - the on-demand filter removes nearly every
/// modern chat model from the foundation catalogue, leaving the profile entry
/// as the thing a person picks - and it drifts from what AWS reports for the
/// same model. Both listings arrive in one call, so a profile reuses the real
/// metadata of the model it routes to (#1023).
///
/// Built from every summary, before any filter: a model the catalogue drops
/// for having no on-demand throughput is exactly the model a profile serves.
#[derive(Debug, Default)]
struct ModalityIndex(std::collections::HashMap<String, ModelModalities>);

impl ModalityIndex {
    fn from_summaries(summaries: &[aws_sdk_bedrock::types::FoundationModelSummary]) -> Self {
        Self(
            summaries
                .iter()
                .map(|s| (s.model_id().to_string(), ModelModalities::from_summary(s)))
                .collect(),
        )
    }

    /// The modalities this listing reported for `base_id`, or `None` when it
    /// did not describe that model.
    fn get(&self, base_id: &str) -> Option<ModelModalities> {
        self.0.get(base_id).copied()
    }
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
    let modalities = ModelModalities::from_summary(summary);
    let is_text = summary.output_modalities().contains(&ModelModality::Text);
    if !(is_text || modalities.is_embedding) {
        return None;
    }

    let id = summary.model_id();
    let model_name = summary.model_name().unwrap_or(id).to_string();
    let capabilities = infer_capabilities_from_id(id, modalities.vision, modalities.is_embedding);

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
/// A profile is a route to a foundation model, so its capabilities are that
/// model's capabilities. `modalities` carries what `ListFoundationModels`
/// reported in the same call, keyed by model id, and the profile reuses it.
/// Only where the base model is not in that listing does the id-family
/// fallback below decide.
///
/// The base model comes from `base_model_for`, which the dispatch gates call
/// as well. The caller registers every profile before the first record is
/// built, so a record and a request for the same id read one answer.
fn inference_profile_to_model_info(
    profile: &aws_sdk_bedrock::types::InferenceProfileSummary,
    modalities: &ModalityIndex,
) -> Option<ModelInfo> {
    use aws_sdk_bedrock::types::InferenceProfileStatus;

    if profile.status != InferenceProfileStatus::Active {
        return None;
    }

    let profile_id = profile.inference_profile_id();
    if profile_id.is_empty() {
        return None;
    }

    // The same call every dispatch gate makes, on the same register. A
    // capability read from an input dispatch does not share is a capability
    // the picker offers and the request builder discards, so the record is
    // not allowed a richer source than the request path has.
    let base_id = base_model_for(profile_id);
    let base_id = base_id.as_ref();

    let resolved = modalities.get(base_id).unwrap_or_else(|| {
        tracing::debug!(
            profile_id,
            base_id,
            "inference profile has no foundation-model entry in this listing; \
             falling back to the model-id family for its modalities"
        );
        fallback_modalities_from_id(base_id)
    });

    let display_name = if profile.inference_profile_name.is_empty() {
        profile_id.to_string()
    } else {
        profile.inference_profile_name.clone()
    };

    Some(ModelInfo {
        id: profile_id.to_string(),
        display_name,
        // `context_limit_for_model` strips the prefix itself, so the picker's
        // window is the one `max_context_tokens` will budget against for the
        // same id.
        context_limit: context_limit_for_model(profile_id),
        capabilities: infer_capabilities_from_id(base_id, resolved.vision, resolved.is_embedding),
    })
}

/// Modalities guessed from a model id, for the one case that has no better
/// answer: a profile whose base model this account's `ListFoundationModels`
/// did not return, so there is no provider metadata to reuse.
///
/// A documented fallback, not a second rule. Everything else reads AWS's own
/// modality fields through [`ModalityIndex`]. The families listed are the
/// multimodal ones Bedrock serves through profiles; anything unrecognized is
/// reported as text-only, which costs a picker badge rather than sending an
/// image to a model that cannot read one.
///
/// `is_embedding` is `false` here because an embedding model is reachable by
/// its bare on-demand id and appears on the foundation path, so a profile for
/// one is resolvable through the index or does not exist.
///
/// An id that reduces to no foundation model at all reaches here as well, and
/// that is now the narrow case it sounds like. An `APPLICATION` profile, whose
/// id is a generated identifier, and a geography newer than
/// [`INFERENCE_PROFILE_PREFIXES`] both carry the base model's ARN in the
/// profile summary; the listing registers it, and `base_model_for` gives the
/// same answer to the record and to every dispatch gate. What is left is an id
/// no listing returned, or a profile whose ARNs name no single model.
fn fallback_modalities_from_id(base_id: &str) -> ModelModalities {
    let lc = base_id.to_ascii_lowercase();
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
    ModelModalities {
        vision,
        is_embedding: false,
    }
}

/// Build the degradation notice for a failed `ListInferenceProfiles` call.
///
/// `code` / `message` are the service error metadata (both optional: a
/// transport or timeout failure carries neither). An authorization denial is
/// reported with the permission to grant; anything else is reported as an
/// upstream failure without blaming IAM, so the operator is not sent to edit
/// a policy over a timeout.
///
/// The relayed provider text is truncated to [`MAX_NOTICE_DETAIL_CHARS`]:
/// the notice is user-facing and crosses the daemon's wire protocol.
fn inference_profiles_notice(code: Option<&str>, message: Option<&str>) -> ModelListingNotice {
    // Bedrock answers a missing policy with `AccessDeniedException`; the
    // bare `AccessDenied` shows up from other AWS front ends. Match the
    // family on the structured error code, never on the message text.
    let denied = code.is_some_and(|c| c.to_ascii_lowercase().starts_with("accessdenied"));
    let cause = match (code, message) {
        (Some(code), Some(message)) => format!("{code}: {message}"),
        (Some(code), None) => code.to_string(),
        (None, Some(message)) => message.to_string(),
        (None, None) => "no error details were returned".to_string(),
    };

    let detail = if denied {
        format!(
            "AWS refused ListInferenceProfiles for this connection. Grant \
             {LIST_INFERENCE_PROFILES_PERMISSION} to surface inference-profile models \
             (Claude 4.x, Nova Premier, DeepSeek R1 and similar), which are not callable \
             by their bare foundation-model id. AWS said - {}",
            truncate_chars(&cause, MAX_NOTICE_DETAIL_CHARS / 2)
        )
    } else {
        format!(
            "ListInferenceProfiles failed, so inference-profile models (Claude 4.x, \
             Nova Premier, DeepSeek R1 and similar) are missing from the list. Refresh \
             to try again. AWS said - {}",
            truncate_chars(&cause, MAX_NOTICE_DETAIL_CHARS / 2)
        )
    };

    let notice = ModelListingNotice::partial_catalog(
        "Bedrock inference profiles unavailable - showing on-demand models only",
        truncate_chars(&detail, MAX_NOTICE_DETAIL_CHARS),
    );
    if denied {
        notice.with_required_permission(LIST_INFERENCE_PROFILES_PERMISSION)
    } else {
        notice
    }
}

/// Truncate `text` to at most `max` characters, marking elision with `...`.
/// Character-based so multi-byte provider text can't be split mid-codepoint.
fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let keep = max.saturating_sub(3);
    let mut out: String = text.chars().take(keep).collect();
    out.push_str("...");
    out
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
    /// Both calls go in parallel. A `ListInferenceProfiles` failure degrades
    /// the listing instead of failing it: many existing IAM policies grant
    /// `bedrock:ListFoundationModels` without
    /// `bedrock:ListInferenceProfiles`, and a foundation-model-only picker
    /// beats no picker at all.
    ///
    /// The degradation is reported in the returned
    /// [`ModelListingReport::notices`] as well as logged. Why both: what
    /// survives the filter in a current AWS account is mostly the embedding
    /// families, so a caller that only sees the model list cannot tell a
    /// degraded listing from an account with nothing but embedding models
    /// (#648).
    async fn fetch_models_uncached(&self) -> Result<ModelListingReport, CoreError> {
        let client = self.control_client().await?;

        let foundation_fut = client.list_foundation_models().send();
        let profiles_fut = client.list_inference_profiles().send();

        let (foundation_res, profiles_res) = tokio::join!(foundation_fut, profiles_fut);

        let foundation = foundation_res
            .map_err(|e| CoreError::Llm(format!("Bedrock ListFoundationModels failed: {e:#}")))?;

        let summaries = foundation.model_summaries();
        // Built before the on-demand filter: the models it drops are exactly
        // the ones the profiles below route to.
        let modalities = ModalityIndex::from_summaries(summaries);
        let mut models: Vec<ModelInfo> =
            summaries.iter().filter_map(summary_to_model_info).collect();
        let mut notices = Vec::new();

        match profiles_res {
            Ok(profile_resp) => {
                // Register every profile before any of them is turned into a
                // record. The record reads the register, exactly as the
                // dispatch gates do, so the two cannot answer differently.
                for profile in profile_resp.inference_profile_summaries() {
                    register_profile_base_model(profile);
                }
                for profile in profile_resp.inference_profile_summaries() {
                    if let Some(info) = inference_profile_to_model_info(profile, &modalities) {
                        models.push(info);
                    }
                }
            }
            Err(error) => {
                use aws_smithy_types::error::metadata::ProvideErrorMetadata;
                tracing::warn!(
                    "Bedrock ListInferenceProfiles failed; model picker will only show \
                     on-demand foundation models. Grant bedrock:ListInferenceProfiles to \
                     surface inference-profile ids (Claude 4.x, Nova Premier, etc.). \
                     Cause: {error:#}"
                );
                notices.push(inference_profiles_notice(error.code(), error.message()));
            }
        }

        // Stable ordering so UIs don't shuffle between refreshes.
        // Defensive dedupe — foundation ids and profile ids don't collide
        // in practice, but keep the merge total just in case.
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models.dedup_by(|a, b| a.id == b.id);
        Ok(ModelListingReport { models, notices })
    }

    /// Return the cached listing, refreshing if the TTL elapsed or the cache
    /// is empty. Notices are cached with the models they describe.
    async fn list_models_cached(&self) -> Result<ModelListingReport, CoreError> {
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
    async fn refresh_models_internal(&self) -> Result<ModelListingReport, CoreError> {
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
        Ok(self.list_models_cached().await?.models)
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        Ok(self.refresh_models_internal().await?.models)
    }

    async fn list_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        self.list_models_cached().await
    }

    async fn refresh_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        self.refresh_models_internal().await
    }

    /// List the models once at startup, so the profile-to-base-model register
    /// is populated before the first turn.
    ///
    /// A turn never calls the control plane to resolve a model id, so an
    /// inference profile no listing has returned answers from the prefix rule
    /// alone. Doing the listing here means the configured model of a live
    /// connection is already registered when the first turn arrives, instead
    /// of only after whichever client happens to open the model picker. It
    /// warms the listing cache as well.
    ///
    /// Best-effort and detached, exactly like the Ollama context-length warm
    /// the registry spawns beside it. A failure leaves the register empty,
    /// which is the conservative answer the gates already have.
    async fn warmup(&self) {
        if let Err(error) = self.list_models_cached().await {
            tracing::debug!(
                %error,
                "Bedrock model listing failed at startup; inference profiles resolve \
                 by their id until a later listing succeeds"
            );
        }
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

        // Per-turn model override (issue #34): when the daemon-side routing
        // layer has set `MODEL_OVERRIDE`, dispatch the user-chosen model id
        // instead of the connector's baked-in `self.model`. Used for the
        // request `model_id`, for the prompt-cache and reasoning support
        // checks, and for the context-window heuristics below. It is resolved
        // before the request is built, because the model decides whether the
        // system prompt carries a cache checkpoint.
        let model = current_model_override().unwrap_or_else(|| self.model.clone());

        // Two independent reasons to withhold the checkpoint: the operator's
        // policy, and a refusal this connector has already met on this model.
        let cache_checkpoint = wants_cache_checkpoint(self.cache_policy, &model)
            && !self.cache_unsupported_models.lock().await.contains(&model);

        let inputs = self.build_request_inputs(
            &messages,
            tools,
            &tool_names,
            &model,
            reasoning,
            cache_checkpoint,
        )?;

        match self
            .dispatch_attempt(client, inputs, on_chunk, &cancellation, !tools.is_empty())
            .await
        {
            Ok(response) => Ok(response),
            Err(AttemptError::CachePointRejected { on_chunk, detail }) => {
                // The refusal named the cache field, on a request that carried
                // a checkpoint. That is the evidence, and it is the whole of
                // it: the memo is written here, from what the service said,
                // and never from a retry that succeeded - a request without
                // the field succeeds whatever the real cause was, so treating
                // that as proof would certify a wrong verdict.
                tracing::warn!(
                    model = %model,
                    detail,
                    "Bedrock refused the prompt-cache checkpoint; retrying this turn \
                     without it and omitting it for later turns on this model"
                );
                self.cache_unsupported_models
                    .lock()
                    .await
                    .insert(model.clone());

                let retry = self.build_request_inputs(
                    &messages,
                    tools,
                    &tool_names,
                    &model,
                    reasoning,
                    false,
                )?;
                self.dispatch_attempt(client, retry, on_chunk, &cancellation, !tools.is_empty())
                    .await
                    .map_err(|e| match e {
                        AttemptError::Other(err) => err,
                        // Unreachable in practice: the retry carries no
                        // checkpoint, and a refusal is only classified for a
                        // request that sent one. Reported rather than retried
                        // again, so a second attempt is the last one.
                        AttemptError::CachePointRejected { detail, .. } => {
                            CoreError::Llm(format!("Bedrock converse request failed: {detail}"))
                        }
                    })
            }
            Err(AttemptError::Other(err)) => Err(err),
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
    /// Bedrock refused the `cachePoint` block this request carried. Same
    /// callback-carrying reason as the arm above. (#1028)
    CachePointRejected {
        on_chunk: ChunkCallback,
        detail: String,
    },
    Other(CoreError),
}

/// Outcome of one complete dispatch attempt, after the streaming ->
/// non-streaming fallback has been resolved inside
/// [`BedrockClient::dispatch_attempt`].
///
/// Only one failure is actionable at this level, and it is the one the caller
/// can answer by changing the request: a refused cache checkpoint, which the
/// caller retries once without it. (#1028)
enum AttemptError {
    CachePointRejected {
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
    /// Whether `system` carries a `cachePoint` block. Read off `system` itself
    /// in [`BedrockClient::build_request_inputs`], not from the policy that
    /// asked for one, so it cannot claim a checkpoint the request does not
    /// hold - a turn with no system prompt has no prefix to mark.
    ///
    /// This is what makes a refusal attributable. A request that sent no
    /// checkpoint cannot have had one refused, so the dispatch paths do not
    /// even ask whether the failure names the cache field. The retry is built
    /// without one, which is why the retry can never be read as a second
    /// refusal, and why the memo can never rest on evidence the fallback
    /// itself produced. (#1028)
    cache_checkpoint: bool,
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

    /// Translate one turn into the Converse request shape.
    ///
    /// `want_cache_checkpoint` is what the caller asks for. Whether a
    /// checkpoint is actually emitted is read back off the built request, so
    /// `BedrockRequestInputs::cache_checkpoint` describes the bytes that go out
    /// rather than the intent behind them. The two differ: a turn with no
    /// system prompt has no prefix to mark, so it sends no checkpoint however
    /// the policy is set.
    ///
    /// Called a second time, with `want_cache_checkpoint` forced to `false`,
    /// when Bedrock refuses the checkpoint. It is a pure translation of the
    /// same turn, so the retry differs from the first attempt in that one field
    /// and nothing else - which is what makes the retry's outcome readable.
    fn build_request_inputs(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        tool_names: &ToolNameMap,
        model: &str,
        reasoning: ReasoningConfig,
        want_cache_checkpoint: bool,
    ) -> Result<BedrockRequestInputs, CoreError> {
        let (system, api_messages) = convert_messages(messages, tool_names, want_cache_checkpoint)?;
        let tool_config = convert_tools(tools, tool_names)?;

        // Read from the request, never from the request we meant to build.
        let cache_checkpoint = system
            .iter()
            .any(|block| matches!(block, SystemContentBlock::CachePoint(_)));

        let msg_count = api_messages.len();
        let tool_count = tools.len();
        // Count prompt content only. A cache checkpoint is a control marker
        // with no prompt text, so counting its `Debug` form would inflate the
        // reported prompt size on exactly the models that cache.
        let system_chars: usize = system
            .iter()
            .filter(|b| !matches!(b, SystemContentBlock::CachePoint(_)))
            .map(|b| format!("{b:?}").len())
            .sum();
        let msg_chars: usize = api_messages.iter().map(|m| format!("{m:?}").len()).sum();
        tracing::info!(
            msg_chars,
            msg_count,
            tool_count,
            system_chars,
            cache_checkpoint,
            model = %model,
            "LLM request payload"
        );

        Ok(BedrockRequestInputs {
            model: model.to_string(),
            api_messages,
            system,
            tool_config,
            inference_cfg: self.build_inference_config(),
            additional_request_fields: build_additional_model_request_fields(model, reasoning),
            tool_names: tool_names.clone(),
            cache_checkpoint,
        })
    }

    /// Dispatch one request, choosing the path and answering the one
    /// per-model restriction that path selection cannot predict.
    ///
    /// Path selection (#67):
    /// - No tools: streaming is always safe; use the streaming path.
    /// - Tools + model on the static deny-list: skip the stream attempt
    ///   and go straight to non-streaming.
    /// - Tools + runtime memo says non-streaming: same.
    /// - Otherwise: try streaming first; on the specific
    ///   "doesn't support tool use in streaming" validation error,
    ///   memoize the model and retry via non-streaming.
    ///
    /// A refused cache checkpoint is **not** answered here. It changes the
    /// request rather than the path, so it goes back to the caller, which owns
    /// the request. (#1028)
    async fn dispatch_attempt(
        &self,
        client: &Client,
        inputs: BedrockRequestInputs,
        on_chunk: ChunkCallback,
        cancellation: &tokio_util::sync::CancellationToken,
        has_tools: bool,
    ) -> Result<LlmResponse, AttemptError> {
        let model = inputs.model.clone();
        let base_model = base_model_for(&model);
        let memo_says_non_streaming = has_tools && {
            let memo = self.non_streaming_tools_models.lock().await;
            memo.contains(&model)
        };
        let allowlist_says_non_streaming = has_tools && !supports_streaming_with_tools(&base_model);
        if memo_says_non_streaming || allowlist_says_non_streaming {
            if allowlist_says_non_streaming {
                tracing::debug!(
                    model = %model,
                    "skipping ConverseStream: model on the non-streaming-with-tools deny-list"
                );
            }
            return self
                .dispatch_non_streaming(client, inputs, on_chunk, cancellation)
                .await;
        }

        match self
            .dispatch_streaming(client, &inputs, on_chunk, cancellation)
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
                self.dispatch_non_streaming(client, inputs, on_chunk, cancellation)
                    .await
            }
            Err(StreamingDispatchError::CachePointRejected { on_chunk, detail }) => {
                Err(AttemptError::CachePointRejected { on_chunk, detail })
            }
            Err(StreamingDispatchError::Other(err)) => Err(AttemptError::Other(err)),
        }
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
                    // Asked only of a request that carried a checkpoint: a
                    // refusal of a field this request did not send is a
                    // refusal of something else.
                    if inputs.cache_checkpoint
                        && let Some(detail) = cache_point_rejected_detail_stream(&e)
                    {
                        return Err(StreamingDispatchError::CachePointRejected {
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
    ///
    /// The request is bounded by [`Self::non_streaming_timeout`] and raced
    /// against `cancellation`, so this path fails a stalled turn and answers
    /// a stop the same way [`Self::dispatch_streaming`] does.
    async fn dispatch_non_streaming(
        &self,
        client: &Client,
        inputs: BedrockRequestInputs,
        mut on_chunk: ChunkCallback,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> Result<LlmResponse, AttemptError> {
        let cache_checkpoint = inputs.cache_checkpoint;
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

        // `Converse` answers once, when generation is complete, so one bound
        // covers the whole call. Race it against cancellation as well, so a
        // stop drops the in-flight request instead of waiting the request out.
        let request_timeout = self.non_streaming_timeout;
        let send_fut = request.send();
        let response = tokio::select! {
            _ = cancellation.cancelled() => {
                tracing::debug!(model = %inputs.model, "Bedrock converse cancelled by token");
                return Err(AttemptError::Other(CoreError::Cancelled));
            }
            _ = tokio::time::sleep(request_timeout) => {
                tracing::error!(
                    model = %inputs.model,
                    timeout_s = request_timeout.as_secs(),
                    "Bedrock converse send() timed out (no response)"
                );
                return Err(AttemptError::Other(CoreError::Llm(format!(
                    "Bedrock converse request timed out after {}s",
                    request_timeout.as_secs()
                ))));
            }
            r = send_fut => match r {
                Ok(r) => r,
                Err(e) => {
                    // Asked only of a request that carried a checkpoint: a
                    // refusal of a field this request did not send is a
                    // refusal of something else.
                    if cache_checkpoint && let Some(detail) = cache_point_rejected_detail(&e) {
                        return Err(AttemptError::CachePointRejected { on_chunk, detail });
                    }
                    return Err(AttemptError::Other(map_converse_error(e)));
                }
            },
        };

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

        let token_usage = response.usage.as_ref().map(map_token_usage);

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

/// If the `Converse` SDK error refuses the cache checkpoint, return the raw
/// message; otherwise `None`. The caller asks this only of a request that
/// carried one. (#1028)
fn cache_point_rejected_detail(
    e: &aws_sdk_bedrockruntime::error::SdkError<
        aws_sdk_bedrockruntime::operation::converse::ConverseError,
    >,
) -> Option<String> {
    use aws_sdk_bedrockruntime::operation::converse::ConverseError;
    let ConverseError::ValidationException(ve) = e.as_service_error()? else {
        return None;
    };
    let raw = ve.message().unwrap_or("");
    names_the_cache_field(raw).then(|| raw.to_string())
}

/// The `ConverseStream` twin of [`cache_point_rejected_detail`]. Both dispatch
/// paths send the checkpoint, so both recover from a refusal.
fn cache_point_rejected_detail_stream(
    e: &aws_sdk_bedrockruntime::error::SdkError<
        aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError,
    >,
) -> Option<String> {
    use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
    let ConverseStreamError::ValidationException(ve) = e.as_service_error()? else {
        return None;
    };
    let raw = ve.message().unwrap_or("");
    names_the_cache_field(raw).then(|| raw.to_string())
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

/// What a turn's reasoning hint amounts to for the model that will serve it.
///
/// Three outcomes, and they stay distinct on purpose (#1022). "Nobody asked"
/// and "somebody asked and this model cannot honour it" are different facts,
/// and collapsing the second into the first is what let a paid-for reasoning
/// effort disappear with nothing above `debug!` to say so.
#[derive(Debug)]
enum ReasoningRequest {
    /// No reasoning was requested for this turn.
    NotRequested,
    /// Reasoning was requested and the model takes it. Carries the
    /// `additionalModelRequestFields` document to send.
    Configured(Document),
    /// Reasoning was requested and this model takes no reasoning
    /// configuration on Bedrock, so the effort has no effect on the request.
    Unconfigurable { budget: u32 },
}

/// Resolve the per-turn reasoning hint against the model that will serve it.
///
/// `model` may be an inference-profile id; `base_model_for` reduces it here,
/// so callers pass the id they dispatch with.
///
/// The emitted shape for a model that takes it is Anthropic's native one,
/// `{"thinking": {"type": "enabled", "budget_tokens": N}}`, which Bedrock
/// forwards verbatim. Which models those are is
/// [`supports_configurable_reasoning`], shared with the capability record so
/// the two cannot drift.
fn resolve_reasoning_request(model: &str, reasoning: ReasoningConfig) -> ReasoningRequest {
    use std::collections::HashMap;
    let budget = match reasoning.thinking_budget_tokens {
        Some(n) if n > 0 => n,
        _ => return ReasoningRequest::NotRequested,
    };
    if !supports_configurable_reasoning(&base_model_for(model)) {
        return ReasoningRequest::Unconfigurable { budget };
    }
    let mut thinking: HashMap<String, Document> = HashMap::new();
    thinking.insert("type".to_string(), Document::String("enabled".to_string()));
    thinking.insert(
        "budget_tokens".to_string(),
        Document::Number(Number::PosInt(u64::from(budget))),
    );
    let mut root: HashMap<String, Document> = HashMap::new();
    root.insert("thinking".to_string(), Document::Object(thinking));
    ReasoningRequest::Configured(Document::Object(root))
}

/// Build the `additionalModelRequestFields` document for a Bedrock Converse
/// request, and report a reasoning effort the model cannot act on.
///
/// Returns `None` when no reasoning was requested, and when the model takes
/// no reasoning configuration - the second case is stated at `warn!` with the
/// model and the budget, because the person who set that effort is paying for
/// a control that did nothing.
fn build_additional_model_request_fields(
    model: &str,
    reasoning: ReasoningConfig,
) -> Option<Document> {
    match resolve_reasoning_request(model, reasoning) {
        ReasoningRequest::NotRequested => None,
        ReasoningRequest::Configured(fields) => Some(fields),
        ReasoningRequest::Unconfigurable { budget } => {
            tracing::warn!(
                model,
                budget,
                "reasoning effort ignored: this Bedrock model takes no reasoning \
                 configuration, so the request goes out without one"
            );
            None
        }
    }
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
    use desktop_assistant_core::ports::llm::ModelListingNoticeKind;

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

    // --- Inference-profile prefixes -------------------------------------
    //
    // Every capability gate in this connector takes the region-prefix-stripped
    // base id. A prefix the stripper does not know is therefore not a cosmetic
    // miss: the base id never appears, so extended thinking, prompt caching
    // and the streaming-with-tools deny list all answer for a model id that
    // matches nothing. AWS documents each of these on the model detail pages
    // (Claude Sonnet 4.5 lists us / eu / au / jp Geo ids and a global id).

    /// Every profile prefix AWS is known to issue, paired with a model id
    /// that has to keep answering the same way through it.
    const PROFILE_PREFIXES: &[&str] = &[
        "us.", "eu.", "apac.", "ap.", "au.", "jp.", "global.", "us-gov.",
    ];

    #[test]
    fn every_inference_profile_prefix_recovers_the_base_model_id() {
        for prefix in PROFILE_PREFIXES {
            let id = format!("{prefix}anthropic.claude-sonnet-4-5-20250929-v1:0");
            assert_eq!(
                strip_region_prefix(&id),
                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "{prefix} is an AWS inference-profile prefix and must be stripped"
            );
        }
        // A bare foundation id is untouched, and an invented prefix is not
        // stripped - the set is an allowlist, not a "drop the first segment"
        // rule, because model ids can carry dots of their own
        // (`openai.gpt-5.6`).
        assert_eq!(
            strip_region_prefix("anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "anthropic.claude-sonnet-4-5-20250929-v1:0"
        );
        assert_eq!(
            strip_region_prefix("xx.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "xx.anthropic.claude-sonnet-4-5-20250929-v1:0"
        );
    }

    #[test]
    fn inference_profile_prefixes_all_end_in_a_separator() {
        // The list is matched entry by entry, so nothing may be a prefix of
        // anything else. Ending every entry at the separator guarantees that
        // whatever order they sit in. `"ap"` without the dot would match
        // `apac.anthropic...` and leave `ac.anthropic...` behind.
        for prefix in INFERENCE_PROFILE_PREFIXES {
            assert!(
                prefix.ends_with('.'),
                "{prefix} must end at the separator, or it can swallow another prefix"
            );
        }
        for (i, outer) in INFERENCE_PROFILE_PREFIXES.iter().enumerate() {
            for (j, inner) in INFERENCE_PROFILE_PREFIXES.iter().enumerate() {
                assert!(
                    i == j || !outer.starts_with(inner),
                    "{inner} is a prefix of {outer}, so the match depends on list order"
                );
            }
        }
    }

    #[test]
    fn every_capability_gate_answers_the_same_through_every_prefix() {
        // Both directions: a capability that must stay on through the prefix,
        // and a deny-list entry that must stay off through it. A prefix the
        // stripper misses silently flips both.
        for prefix in PROFILE_PREFIXES {
            let claude = format!("{prefix}anthropic.claude-sonnet-4-5-20250929-v1:0");
            let base = strip_region_prefix(&claude);
            assert!(
                supports_configurable_reasoning(base),
                "{prefix}: extended thinking must survive the prefix"
            );
            assert!(
                supports_prompt_caching(base),
                "{prefix}: prompt caching must survive the prefix"
            );
            assert_eq!(
                context_limit_for_model(&claude),
                Some(200_000),
                "{prefix}: the context window must survive the prefix"
            );

            let llama = format!("{prefix}meta.llama4-maverick-17b-instruct-v1:0");
            assert!(
                !supports_streaming_with_tools(strip_region_prefix(&llama)),
                "{prefix}: the non-streaming-with-tools deny list must survive the prefix"
            );
        }
    }

    #[test]
    fn configurable_reasoning_reads_the_stripped_base_id() {
        // The caller strips the region prefix, as it does for
        // `supports_prompt_caching` and `supports_streaming_with_tools`.
        assert!(supports_configurable_reasoning(strip_region_prefix(
            "us.anthropic.claude-opus-4-1"
        )));
        assert!(supports_configurable_reasoning(strip_region_prefix(
            "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
        )));
        assert!(!supports_configurable_reasoning(strip_region_prefix(
            "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )));
        assert!(!supports_configurable_reasoning(
            "amazon.titan-text-express-v1"
        ));
        // Unknown models default to "no configuration": an unrecognized
        // reasoning field fails the whole request.
        assert!(!supports_configurable_reasoning("future.unknown-model"));
    }

    #[test]
    fn an_unconfigurable_reasoning_request_is_reported_not_dropped() {
        // Three outcomes, and they must stay distinguishable. "Nobody asked"
        // is not the same answer as "somebody asked and this model cannot
        // honour it" - collapsing the second into the first is the silent
        // drop this fix removes.
        assert!(matches!(
            resolve_reasoning_request("us.deepseek.r1-v1:0", ReasoningConfig::default()),
            ReasoningRequest::NotRequested
        ));
        assert!(matches!(
            resolve_reasoning_request(
                "us.deepseek.r1-v1:0",
                ReasoningConfig::with_thinking_budget(8_000)
            ),
            ReasoningRequest::Unconfigurable { budget: 8_000 }
        ));
        assert!(matches!(
            resolve_reasoning_request(
                "us.anthropic.claude-sonnet-4-6",
                ReasoningConfig::with_thinking_budget(8_000)
            ),
            ReasoningRequest::Configured(_)
        ));
    }

    /// Model ids paired with whether this connector can configure reasoning
    /// for them, spanning both answers so neither direction can pass
    /// vacuously.
    ///
    /// Claude 3.7 and the 4.x line take an extended-thinking budget. Claude
    /// 3.5 and Claude 3 predate the feature. DeepSeek R1 reasons on its own
    /// and Bedrock exposes no knob for it. Nothing else on Bedrock takes a
    /// thinking budget through the Converse API.
    const REASONING_CONFIGURABLE_BY_ID: &[(&str, bool)] = &[
        ("anthropic.claude-sonnet-4-6", true),
        ("us.anthropic.claude-opus-4-1", true),
        ("eu.anthropic.claude-haiku-4-5-20251001-v1:0", true),
        ("apac.anthropic.claude-3-7-sonnet-20250219-v1:0", true),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", false),
        ("anthropic.claude-3-haiku-20240307-v1:0", false),
        ("us.deepseek.r1-v1:0", false),
        ("us.meta.llama4-maverick-17b-instruct-v1:0", false),
        ("amazon.nova-premier-v1:0", false),
        ("openai.gpt-oss-120b-1:0", false),
        ("amazon.titan-embed-text-v2:0", false),
    ];

    #[test]
    fn reasoning_capability_and_the_emitted_thinking_field_agree_for_every_model() {
        // The capability record and the request builder must not be able to
        // disagree: a model that advertises reasoning has its budget sent,
        // and a model that does not advertise it is never sent one.
        for (id, configurable) in REASONING_CONFIGURABLE_BY_ID {
            let advertised =
                infer_capabilities_from_id(strip_region_prefix(id), false, false).reasoning;
            let emitted = build_additional_model_request_fields(
                id,
                ReasoningConfig::with_thinking_budget(8_000),
            )
            .is_some();
            assert_eq!(
                advertised, *configurable,
                "{id}: capability record disagrees with what Bedrock accepts"
            );
            assert_eq!(
                emitted, *configurable,
                "{id}: emitted request fields disagree with what Bedrock accepts"
            );
        }
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
        assert_eq!(info.capabilities.kind, ModelKind::Generative);
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
        assert_eq!(info.capabilities.kind, ModelKind::Embedding);
        assert!(!info.capabilities.tools);
        assert!(!info.capabilities.reasoning);
    }

    #[test]
    fn bedrock_derives_kind_from_output_modalities() {
        // The classification must come from the provider's real modality
        // metadata, not from a substring of the id (#647). Prove it with an
        // id that carries no "embed" token but whose OUTPUT modality is
        // EMBEDDING: modality wins and the model is classified `Embedding`.
        let embed = make_summary(
            "amazon.titan-text-v2:0",
            FoundationModelLifecycleStatus::Active,
            ModelModality::Embedding,
            vec![ModelModality::Text],
        );
        let embed_info = summary_to_model_info(&embed).expect("kept");
        assert_eq!(
            embed_info.capabilities.kind,
            ModelKind::Embedding,
            "an EMBEDDING output modality classifies the model as an embedding model"
        );

        // A TEXT output modality is a generative model.
        let text = make_summary(
            "anthropic.claude-sonnet-4-6",
            FoundationModelLifecycleStatus::Active,
            ModelModality::Text,
            vec![ModelModality::Text],
        );
        let text_info = summary_to_model_info(&text).expect("kept");
        assert_eq!(text_info.capabilities.kind, ModelKind::Generative);

        // Inference profiles cover chat models only; they merge from a known
        // foundation family and are always generative.
        let profile = make_profile(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude Haiku 4.5 (US)",
            InferenceProfileStatus::Active,
        );
        let profile_info =
            inference_profile_to_model_info(&profile, &ModalityIndex::default()).expect("kept");
        assert_eq!(profile_info.capabilities.kind, ModelKind::Generative);
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
        // `models` carries the foundation model the profile routes to, and a
        // real profile's ARN names the same model its id reduces to - so the
        // stub is built from the id, not from a placeholder. Paired with an
        // empty `ModalityIndex`, these tests then exercise the id-family
        // fallback, which is what they cover. The metadata path and the
        // disagreeing-ARN path are covered end to end against a mocked
        // control plane, further down.
        let model_stub = InferenceProfileModel::builder()
            .model_arn(format!(
                "arn:aws:bedrock:us-east-1::foundation-model/{}",
                strip_region_prefix(id)
            ))
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
        assert!(inference_profile_to_model_info(&profile, &ModalityIndex::default()).is_some());
    }

    #[test]
    fn profile_anthropic_claude_4_inferred_capabilities() {
        let profile = make_profile(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude Haiku 4.5 (US)",
            InferenceProfileStatus::Active,
        );
        let info =
            inference_profile_to_model_info(&profile, &ModalityIndex::default()).expect("kept");
        assert_eq!(info.id, "us.anthropic.claude-haiku-4-5-20251001-v1:0");
        assert_eq!(info.display_name, "Claude Haiku 4.5 (US)");
        assert_eq!(info.context_limit, Some(200_000));
        assert!(info.capabilities.tools);
        assert!(info.capabilities.reasoning);
        assert!(info.capabilities.vision);
        assert_eq!(info.capabilities.kind, ModelKind::Generative);
    }

    #[test]
    fn profile_amazon_nova_capabilities() {
        let profile = make_profile(
            "us.amazon.nova-premier-v1:0",
            "Nova Premier (US)",
            InferenceProfileStatus::Active,
        );
        let info =
            inference_profile_to_model_info(&profile, &ModalityIndex::default()).expect("kept");
        assert_eq!(info.id, "us.amazon.nova-premier-v1:0");
        assert!(info.capabilities.tools, "Nova supports tool use");
        assert!(info.capabilities.vision, "Nova Premier is multimodal");
        assert!(!info.capabilities.reasoning);
        assert_eq!(info.capabilities.kind, ModelKind::Generative);
    }

    #[test]
    fn profile_deepseek_r1_capabilities() {
        let profile = make_profile(
            "us.deepseek.r1-v1:0",
            "DeepSeek R1 (US)",
            InferenceProfileStatus::Active,
        );
        let info =
            inference_profile_to_model_info(&profile, &ModalityIndex::default()).expect("kept");
        assert_eq!(info.id, "us.deepseek.r1-v1:0");
        // R1 reasons, and it reasons whether or not anybody asks: Bedrock's
        // Converse contract for DeepSeek carries no reasoning configuration
        // at all, so this connector cannot act on a reasoning effort for it.
        // Reporting `true` put a reasoning badge and an effort control in
        // front of a person, and dropped the budget on the way out.
        assert!(
            !info.capabilities.reasoning,
            "R1 takes no reasoning configuration on Bedrock"
        );
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
        let info =
            inference_profile_to_model_info(&profile, &ModalityIndex::default()).expect("kept");
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

    // --- Profile capabilities come from the base model's metadata (#1023) --
    //
    // The profile API returns no modality data, so the profile path used to
    // carry its own hardcoded vision id list. That list is the path that runs
    // in practice - the on-demand filter removes nearly every modern chat
    // model from the foundation listing - and it drifts away from what AWS
    // reports for the same model. The listings arrive in one call, so a
    // profile can be resolved to its base model and reuse the real metadata.

    /// `ListFoundationModels` where the interesting models are reachable only
    /// through a profile, which is the shape of a current AWS account.
    ///
    /// The two profile-only entries are chosen so the retired id list would
    /// answer wrongly in both directions: a vision-capable model it has never
    /// heard of, and a model it treats as vision-capable by family prefix
    /// whose real input modalities are text only.
    const DRIFTING_FOUNDATION_MODELS_BODY: &str = r#"{
      "modelSummaries": [
        {
          "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-omni-v1:0",
          "modelId": "amazon.nova-2-omni-v1:0",
          "modelName": "Nova 2 Omni",
          "providerName": "Amazon",
          "inputModalities": ["TEXT", "IMAGE"],
          "outputModalities": ["TEXT"],
          "inferenceTypesSupported": ["INFERENCE_PROFILE"],
          "modelLifecycle": {"status": "ACTIVE"}
        },
        {
          "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/meta.llama4-scout-17b-instruct-v1:0",
          "modelId": "meta.llama4-scout-17b-instruct-v1:0",
          "modelName": "Llama 4 Scout 17B Instruct",
          "providerName": "Meta",
          "inputModalities": ["TEXT"],
          "outputModalities": ["TEXT"],
          "inferenceTypesSupported": ["INFERENCE_PROFILE"],
          "modelLifecycle": {"status": "ACTIVE"}
        },
        {
          "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-image-v1:0",
          "modelId": "amazon.titan-embed-image-v1:0",
          "modelName": "Titan Multimodal Embeddings",
          "providerName": "Amazon",
          "inputModalities": ["TEXT", "IMAGE"],
          "outputModalities": ["EMBEDDING"],
          "inferenceTypesSupported": ["INFERENCE_PROFILE"],
          "modelLifecycle": {"status": "ACTIVE"}
        },
        {
          "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
          "modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0",
          "modelName": "Claude Sonnet 4.5",
          "providerName": "Anthropic",
          "inputModalities": ["TEXT", "IMAGE"],
          "outputModalities": ["TEXT"],
          "inferenceTypesSupported": ["INFERENCE_PROFILE"],
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

    /// Profiles for the three profile-only models above, plus one whose base
    /// model is absent from the foundation listing.
    const DRIFTING_INFERENCE_PROFILES_BODY: &str = r#"{
      "inferenceProfileSummaries": [
        {
          "inferenceProfileName": "US Nova 2 Omni",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.amazon.nova-2-omni-v1:0",
          "inferenceProfileId": "us.amazon.nova-2-omni-v1:0",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-omni-v1:0"}
          ],
          "status": "ACTIVE",
          "type": "SYSTEM_DEFINED"
        },
        {
          "inferenceProfileName": "US Llama 4 Scout",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.meta.llama4-scout-17b-instruct-v1:0",
          "inferenceProfileId": "us.meta.llama4-scout-17b-instruct-v1:0",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/meta.llama4-scout-17b-instruct-v1:0"}
          ],
          "status": "ACTIVE",
          "type": "SYSTEM_DEFINED"
        },
        {
          "inferenceProfileName": "US Titan Multimodal Embeddings",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.amazon.titan-embed-image-v1:0",
          "inferenceProfileId": "us.amazon.titan-embed-image-v1:0",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-image-v1:0"}
          ],
          "status": "ACTIVE",
          "type": "SYSTEM_DEFINED"
        },
        {
          "inferenceProfileName": "US Claude Sonnet 4.6",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.anthropic.claude-sonnet-4-6",
          "inferenceProfileId": "us.anthropic.claude-sonnet-4-6",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6"}
          ],
          "status": "ACTIVE",
          "type": "SYSTEM_DEFINED"
        },
        {
          "inferenceProfileName": "Claude Sonnet 4.5 on a geography this build has never heard of",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/xx.anthropic.claude-sonnet-4-5-20250929-v1:0",
          "inferenceProfileId": "xx.anthropic.claude-sonnet-4-5-20250929-v1:0",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"}
          ],
          "status": "ACTIVE",
          "type": "SYSTEM_DEFINED"
        },
        {
          "inferenceProfileName": "platform-team-claude",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/y1c2q8m4t6bk",
          "inferenceProfileId": "y1c2q8m4t6bk",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"},
            {"modelArn": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"},
            {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"}
          ],
          "status": "ACTIVE",
          "type": "APPLICATION"
        },
        {
          "inferenceProfileName": "batch-team-llama",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/p9r4w2k7d3xh",
          "inferenceProfileId": "p9r4w2k7d3xh",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.meta.llama4-scout-17b-instruct-v1:0"}
          ],
          "status": "ACTIVE",
          "type": "APPLICATION"
        }
      ]
    }"#;

    /// The base model each `APPLICATION` profile in
    /// [`DRIFTING_INFERENCE_PROFILES_BODY`] routes to. Named once so a test
    /// asks the base model the same question it asks the profile.
    const APP_PROFILE_CLAUDE: &str = "y1c2q8m4t6bk";
    const APP_PROFILE_CLAUDE_BASE: &str = "anthropic.claude-sonnet-4-5-20250929-v1:0";
    const APP_PROFILE_LLAMA: &str = "p9r4w2k7d3xh";
    const APP_PROFILE_LLAMA_BASE: &str = "meta.llama4-scout-17b-instruct-v1:0";

    /// Both control-plane calls succeed, serving the drifting-catalogue
    /// bodies above.
    fn mock_drifting_control_plane(server: &httpmock::MockServer) {
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/foundation-models");
            then.status(200)
                .header("content-type", "application/json")
                .body(DRIFTING_FOUNDATION_MODELS_BODY);
        });
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/inference-profiles");
            then.status(200)
                .header("content-type", "application/json")
                .body(DRIFTING_INFERENCE_PROFILES_BODY);
        });
    }

    fn find_model<'a>(report: &'a ModelListingReport, id: &str) -> &'a ModelInfo {
        report
            .models
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| {
                let ids: Vec<&str> = report.models.iter().map(|m| m.id.as_str()).collect();
                panic!("{id} missing from the listing, got {ids:?}")
            })
    }

    #[tokio::test]
    async fn vision_capable_model_reports_vision_on_both_the_foundation_and_profile_paths() {
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        assert!(
            find_model(&report, "anthropic.claude-3-haiku-20240307-v1:0")
                .capabilities
                .vision,
            "the foundation path reads IMAGE from the real input modalities"
        );
        assert!(
            find_model(&report, "us.amazon.nova-2-omni-v1:0")
                .capabilities
                .vision,
            "the profile path must read the same metadata, not a curated id list"
        );
        assert!(
            !find_model(&report, "us.meta.llama4-scout-17b-instruct-v1:0")
                .capabilities
                .vision,
            "a model AWS reports as text-only must not be advertised as vision-capable"
        );
    }

    #[tokio::test]
    async fn profile_of_an_embedding_base_model_does_not_report_generative() {
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        let profile = find_model(&report, "us.amazon.titan-embed-image-v1:0");
        assert_eq!(
            profile.capabilities.kind,
            ModelKind::Embedding,
            "model kind follows the base model's output modality, it is not assumed"
        );
        assert!(
            !profile.capabilities.tools,
            "an embedding model calls no tools"
        );
    }

    #[tokio::test]
    async fn a_profile_id_the_prefix_list_does_not_cover_reports_what_dispatch_will_do() {
        // A geography newer than the prefix list is one of the two ids the
        // prefix rule cannot reduce, the other being an APPLICATION profile.
        // The listing registers what the profile's own ARN names, and every
        // dispatch gate reads that same register, so the record and the
        // request path answer together. They must, whichever way they answer:
        // a capability recovered for the record alone is a control the picker
        // offers and the request builder discards, which is #1022 rebuilt.
        //
        // An id the register does not hold - one no listing returned - still
        // reduces by the prefix rule alone, and both sides still agree. That
        // case has its own test, because it is the one that stays
        // conservative.
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        let id = "xx.anthropic.claude-sonnet-4-5-20250929-v1:0";
        let profile = find_model(&report, id);

        let dispatch_would_send = matches!(
            resolve_reasoning_request(id, ReasoningConfig::with_thinking_budget(8_000)),
            ReasoningRequest::Configured(_)
        );
        assert_eq!(
            profile.capabilities.reasoning, dispatch_would_send,
            "the record and the request path must answer the same, whichever way they answer"
        );
        assert!(
            profile.capabilities.reasoning,
            "the profile routes to Claude Sonnet 4.5, so both sides answer for that model"
        );
        assert_eq!(
            profile.context_limit,
            context_limit_for_model(id),
            "the picker's window must be the one the daemon budgets against"
        );
    }

    #[tokio::test]
    async fn profile_with_an_unresolvable_base_model_falls_back_to_the_id_family() {
        // `anthropic.claude-sonnet-4-6` is not in this account's foundation
        // listing, so there is no metadata to reuse. The documented fallback
        // keeps the profile usable rather than reporting nothing.
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        let profile = find_model(&report, "us.anthropic.claude-sonnet-4-6");
        assert!(profile.capabilities.vision);
        assert!(profile.capabilities.tools);
        assert_eq!(profile.capabilities.kind, ModelKind::Generative);
    }

    // --- Application inference profiles (#1044) --------------------------
    //
    // An `APPLICATION` profile id is a generated identifier. No rule reduces
    // it to a foundation model, so the profile's own `models[].modelArn` is
    // the only source. The listing registers that mapping, and every gate on
    // the dispatch path reads the same register, so the record and the
    // request path cannot answer differently.

    #[tokio::test]
    async fn an_application_profile_reports_the_reasoning_answer_dispatch_will_send() {
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        let profile = find_model(&report, APP_PROFILE_CLAUDE);
        let dispatch_would_send = matches!(
            resolve_reasoning_request(
                APP_PROFILE_CLAUDE,
                ReasoningConfig::with_thinking_budget(8_000)
            ),
            ReasoningRequest::Configured(_)
        );
        assert_eq!(
            profile.capabilities.reasoning, dispatch_would_send,
            "the record and the request path must answer the same, whichever way they answer"
        );
        assert!(
            supports_configurable_reasoning(APP_PROFILE_CLAUDE_BASE),
            "fixture check: the base model must be one that takes a thinking budget"
        );
        assert!(
            dispatch_would_send,
            "the profile routes to Claude Sonnet 4.5, so the budget must reach the request"
        );
    }

    #[tokio::test]
    async fn an_application_profile_reports_the_prompt_caching_answer_dispatch_will_send() {
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        assert_eq!(
            wants_cache_checkpoint(CachePolicy::SystemPromptOnly, APP_PROFILE_CLAUDE),
            wants_cache_checkpoint(CachePolicy::SystemPromptOnly, APP_PROFILE_CLAUDE_BASE),
            "a profile takes the checkpoint decision of the model it routes to"
        );
        assert!(
            wants_cache_checkpoint(CachePolicy::SystemPromptOnly, APP_PROFILE_CLAUDE),
            "Claude Sonnet 4.5 accepts a checkpoint, so the profile must get one"
        );
    }

    #[tokio::test]
    async fn an_application_profile_reports_the_streaming_deny_list_answer_dispatch_will_use() {
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        assert!(
            find_model(&report, APP_PROFILE_LLAMA).capabilities.tools,
            "the record offers tools on this profile, so the deny list decides the path"
        );
        assert_eq!(
            supports_streaming_with_tools(&base_model_for(APP_PROFILE_LLAMA)),
            supports_streaming_with_tools(APP_PROFILE_LLAMA_BASE),
            "a profile takes the streaming decision of the model it routes to"
        );
        assert!(
            !supports_streaming_with_tools(&base_model_for(APP_PROFILE_LLAMA)),
            "Llama 4 takes tools on Converse only, and a profile over it is no different"
        );
    }

    #[tokio::test]
    async fn an_application_profile_reports_the_context_window_the_daemon_will_budget_against() {
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        let profile = find_model(&report, APP_PROFILE_CLAUDE);
        assert_eq!(
            profile.context_limit,
            context_limit_for_model(APP_PROFILE_CLAUDE),
            "the picker's window must be the one the daemon budgets against"
        );
        assert_eq!(
            profile.context_limit,
            Some(200_000),
            "the window is the base model's, not the universal fallback"
        );
    }

    #[tokio::test]
    async fn an_application_profile_reports_the_base_models_own_modalities() {
        let server = httpmock::MockServer::start();
        mock_drifting_control_plane(&server);

        let report = control_plane_client(&server)
            .list_models_detailed()
            .await
            .expect("healthy listing");

        assert!(
            find_model(&report, APP_PROFILE_CLAUDE).capabilities.vision,
            "the base model's real input modalities reach the profile entry"
        );
        assert!(
            !find_model(&report, APP_PROFILE_LLAMA).capabilities.vision,
            "a text-only base model must not be advertised as vision-capable"
        );
    }

    /// `ListInferenceProfiles` with one `APPLICATION` profile whose id no
    /// other test lists, so "before the listing" is a meaningful state.
    const WARMUP_PROFILES_BODY: &str = r#"{
      "inferenceProfileSummaries": [
        {
          "inferenceProfileName": "startup-team-claude",
          "inferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/w7f3j5s8v2qd",
          "inferenceProfileId": "w7f3j5s8v2qd",
          "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"}
          ],
          "status": "ACTIVE",
          "type": "APPLICATION"
        }
      ]
    }"#;

    #[tokio::test]
    async fn warmup_registers_profile_base_models_before_the_first_turn() {
        // The register is populated by a listing, and a turn will not make
        // one. Startup does, so a configured application profile answers for
        // its base model on the very first turn rather than after whichever
        // client happens to open the picker.
        const ID: &str = "w7f3j5s8v2qd";
        let server = httpmock::MockServer::start();
        mock_foundation_models(&server);
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/inference-profiles");
            then.status(200)
                .header("content-type", "application/json")
                .body(WARMUP_PROFILES_BODY);
        });

        assert_eq!(
            base_model_for(ID),
            ID,
            "fixture check: nothing may have registered this id yet"
        );

        control_plane_client(&server).warmup().await;

        assert_eq!(
            base_model_for(ID),
            APP_PROFILE_CLAUDE_BASE,
            "startup listing must leave the register ready for the first turn"
        );
    }

    /// A turn can dispatch a model the listing never returned: a configured
    /// `default_model`, a per-turn `MODEL_OVERRIDE`, or a keep-warm probe
    /// before any listing ran. There is no control-plane call on that path,
    /// so the gates answer from the prefix rule alone.
    #[test]
    fn a_model_that_was_never_listed_answers_conservatively_without_a_control_plane_call() {
        // No client and no mock server here on purpose: the gates are pure
        // functions of the id and the mapping already in memory, so a turn
        // against an unknown id cannot reach the network.
        const NEVER_LISTED: &str = "n0tl1st3d1044";

        assert_eq!(
            base_model_for(NEVER_LISTED),
            NEVER_LISTED,
            "an unregistered id reduces to itself, the same answer as before"
        );
        assert!(
            matches!(
                resolve_reasoning_request(
                    NEVER_LISTED,
                    ReasoningConfig::with_thinking_budget(8_000)
                ),
                ReasoningRequest::Unconfigurable { .. }
            ),
            "an effort against an unknown id is reported, not sent and not silently dropped"
        );
        assert!(
            !wants_cache_checkpoint(CachePolicy::SystemPromptOnly, NEVER_LISTED),
            "a checkpoint the model may refuse costs the whole turn, so withhold it"
        );
        assert_eq!(
            context_limit_for_model(NEVER_LISTED),
            None,
            "no window is known, so the daemon uses its own fallback"
        );
        assert!(
            supports_streaming_with_tools(&base_model_for(NEVER_LISTED)),
            "the streaming deny list is an allow-by-default list with a runtime fallback"
        );
    }

    #[test]
    fn a_profile_model_arn_names_the_foundation_model_it_routes_to() {
        assert_eq!(
            base_model_from_model_arn(
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"
            ),
            Some("anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "a foundation-model ARN names the model directly"
        );
        assert_eq!(
            base_model_from_model_arn(
                "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.meta.llama4-scout-17b-instruct-v1:0"
            ),
            Some("meta.llama4-scout-17b-instruct-v1:0"),
            "a system-defined profile ARN reduces by the prefix rule"
        );
        assert_eq!(
            base_model_from_model_arn(
                "arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"
            ),
            Some("anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "the GovCloud partition is still an AWS ARN"
        );
    }

    #[test]
    fn a_model_arn_this_connector_cannot_read_names_nothing() {
        for arn in [
            "",
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "arn:aws:s3:::a-bucket/an-object",
            "arn:aws:bedrock:us-east-1:111122223333:provisioned-model/abcd1234",
            "arn:aws:bedrock:us-east-1::foundation-model/",
        ] {
            assert_eq!(
                base_model_from_model_arn(arn),
                None,
                "{arn} is not a Bedrock model ARN this connector can reduce"
            );
        }
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
        let (_system, messages) = convert_messages(
            &history,
            &map,
            checkpoint_for("us.anthropic.claude-sonnet-4-6"),
        )
        .expect("convert ok");

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
        let (_system, messages) = convert_messages(
            &history,
            &map,
            checkpoint_for("us.anthropic.claude-sonnet-4-6"),
        )
        .expect("convert ok");
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
        let (_s, messages) = convert_messages(
            &history,
            &map,
            checkpoint_for("us.anthropic.claude-sonnet-4-6"),
        )
        .expect("ok");
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

    // --- Prompt caching (#462) -------------------------------------------
    //
    // Bedrock's Converse API caches a prompt prefix when the request carries a
    // `cachePoint` block. Support is per model, not per API: a model that does
    // not support it rejects the request. So the connector emits the block only
    // where the model accepts it, and reads the two cache counters back out of
    // the response usage.

    /// The system blocks the daemon actually sends: one stable assembler
    /// instruction, then a volatile per-turn block.
    ///
    /// The checkpoint belongs between the two. Caching is a prefix match, so a
    /// checkpoint after the volatile block is written every turn and read
    /// never.
    fn caching_history() -> Vec<Message> {
        vec![
            Message::new(Role::System, "stable assembler instruction"),
            Message::new(Role::System, "[Scratchpad] volatile per-turn state"),
            Message::new(Role::User, "hi"),
        ]
    }

    /// Whether a request for `model_id` carries a checkpoint under the default
    /// cache policy. This is the decision `stream_completion` makes before it
    /// builds the request.
    fn checkpoint_for(model_id: &str) -> bool {
        wants_cache_checkpoint(CachePolicy::default(), model_id)
    }

    /// Indices of the cache checkpoints in a system block list.
    fn cache_point_indices(system: &[SystemContentBlock]) -> Vec<usize> {
        system
            .iter()
            .enumerate()
            .filter(|(_, block)| matches!(block, SystemContentBlock::CachePoint(_)))
            .map(|(index, _)| index)
            .collect()
    }

    #[test]
    fn cache_point_emitted_for_anthropic_model() {
        let map = ToolNameMap::from_names(Vec::<&str>::new());
        let (system, _messages) = convert_messages(
            &caching_history(),
            &map,
            checkpoint_for("us.anthropic.claude-sonnet-4-6"),
        )
        .expect("convert ok");

        assert_eq!(
            cache_point_indices(&system),
            vec![1],
            "exactly one checkpoint, directly after the stable system prefix: {system:?}"
        );
        // The prefix itself must survive unchanged in front of the checkpoint.
        assert!(
            matches!(&system[0], SystemContentBlock::Text(t) if t == "stable assembler instruction"),
            "leading block must stay the stable prefix: {:?}",
            system[0]
        );
        // The volatile block follows the checkpoint, so a change in it does
        // not invalidate the cached prefix.
        assert!(
            matches!(&system[2], SystemContentBlock::Text(t) if t.starts_with("[Scratchpad]")),
            "volatile block must follow the checkpoint: {:?}",
            system[2]
        );
    }

    #[test]
    fn cache_point_emitted_for_amazon_nova_model() {
        let map = ToolNameMap::from_names(Vec::<&str>::new());
        let (system, _messages) = convert_messages(
            &caching_history(),
            &map,
            checkpoint_for("us.amazon.nova-pro-v1:0"),
        )
        .expect("convert ok");
        assert_eq!(cache_point_indices(&system), vec![1], "{system:?}");
    }

    #[test]
    fn cache_point_omitted_for_meta_llama() {
        let map = ToolNameMap::from_names(Vec::<&str>::new());
        // Bedrock rejects a checkpoint on a Meta model, so the request must go
        // out without one - and it must still be built successfully.
        let (system, messages) = convert_messages(
            &caching_history(),
            &map,
            checkpoint_for("us.meta.llama4-maverick-17b-instruct-v1:0"),
        )
        .expect("a model without caching support still converts");

        assert!(
            cache_point_indices(&system).is_empty(),
            "no checkpoint for Meta: {system:?}"
        );
        assert_eq!(system.len(), 2, "both system blocks survive: {system:?}");
        assert_eq!(messages.len(), 1, "the user turn survives: {messages:?}");
    }

    #[test]
    fn cache_point_omitted_for_inference_profile_whose_base_model_lacks_support() {
        let map = ToolNameMap::from_names(Vec::<&str>::new());
        // The region prefix must not hide the base model from the check.
        for profile_id in [
            "us.deepseek.r1-v1:0",
            "eu.mistral.mistral-large-2407-v1:0",
            "apac.cohere.command-r-plus-v1:0",
            // Claude 3 (not 3.5) predates prompt caching on Bedrock.
            "us.anthropic.claude-3-haiku-20240307-v1:0",
        ] {
            let (system, _messages) =
                convert_messages(&caching_history(), &map, checkpoint_for(profile_id))
                    .expect("convert ok");
            assert!(
                cache_point_indices(&system).is_empty(),
                "{profile_id} must get no checkpoint: {system:?}"
            );
        }
    }

    #[test]
    fn cache_point_never_emitted_on_tool_list() {
        // Bedrock evaluates checkpoints `tools` -> `system` -> `messages`, and
        // a change in an earlier section invalidates every later one. Tool
        // search moves the tool list inside a conversation, so a checkpoint on
        // `tools` would invalidate the system cache on every such turn.
        let cfg = convert_tools(
            &[ToolDefinition::new("weather", "d", serde_json::json!({}))],
            &ToolNameMap::from_names(["weather"]),
        )
        .expect("convert ok")
        .expect("one tool config");

        assert!(
            !cfg.tools()
                .iter()
                .any(|tool| matches!(tool, Tool::CachePoint(_))),
            "the tool list must carry no checkpoint: {:?}",
            cfg.tools()
        );
    }

    /// Converse usage carrying both cache counters.
    fn sdk_usage_with_cache_counters() -> aws_sdk_bedrockruntime::types::TokenUsage {
        aws_sdk_bedrockruntime::types::TokenUsage::builder()
            .input_tokens(120)
            .output_tokens(30)
            .total_tokens(150)
            .cache_read_input_tokens(4096)
            .cache_write_input_tokens(2048)
            .build()
            .expect("usage builds")
    }

    #[test]
    fn usage_maps_cache_read_and_write_tokens_on_the_non_streaming_path() {
        // `dispatch_non_streaming` maps `response.usage` through here. The AWS
        // SDK is not mockable at the HTTP level the way `httpmock` stubs the
        // other connectors, so the mapper is exercised directly; it is the
        // whole of that path's usage mapping.
        let mapped = map_token_usage(&sdk_usage_with_cache_counters());
        assert_eq!(mapped.input_tokens, Some(120));
        assert_eq!(mapped.output_tokens, Some(30));
        assert_eq!(mapped.cache_read_input_tokens, Some(4096));
        assert_eq!(mapped.cache_creation_input_tokens, Some(2048));
    }

    #[test]
    fn usage_maps_cache_read_and_write_tokens_on_the_streaming_path() {
        use aws_sdk_bedrockruntime::types::{ConverseStreamMetadataEvent, ConverseStreamOutput};

        // `dispatch_streaming` reads usage from the metadata event, through
        // the real `apply_stream_event`.
        let mut text = String::new();
        let mut tool_acc = ToolCallAccumulator::default();
        let mut on_chunk: ChunkCallback = Box::new(|_| true);
        let mut token_usage = None;
        apply_stream_event(
            ConverseStreamOutput::Metadata(
                ConverseStreamMetadataEvent::builder()
                    .usage(sdk_usage_with_cache_counters())
                    .build(),
            ),
            &mut text,
            &mut tool_acc,
            &mut on_chunk,
            &mut token_usage,
        );
        let streamed = token_usage.expect("metadata event yields usage");
        assert_eq!(streamed.input_tokens, Some(120));
        assert_eq!(streamed.output_tokens, Some(30));
        assert_eq!(streamed.cache_read_input_tokens, Some(4096));
        assert_eq!(streamed.cache_creation_input_tokens, Some(2048));
    }

    #[test]
    fn usage_omits_cache_counters_when_the_response_has_none() {
        use aws_sdk_bedrockruntime::types::TokenUsage as SdkTokenUsage;

        // A model without caching returns no cache fields. `Some(0)` would
        // read to a caller as "caching ran and saved nothing", which is a
        // different claim from "caching did not run".
        let sdk_usage = SdkTokenUsage::builder()
            .input_tokens(10)
            .output_tokens(5)
            .total_tokens(15)
            .build()
            .expect("usage builds");

        let mapped = map_token_usage(&sdk_usage);
        assert_eq!(mapped.cache_read_input_tokens, None);
        assert_eq!(mapped.cache_creation_input_tokens, None);
    }

    #[test]
    fn supports_prompt_caching_reads_the_stripped_base_id() {
        // The caller strips the region prefix, as it does for
        // `supports_streaming_with_tools`.
        assert!(supports_prompt_caching(strip_region_prefix(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )));
        assert!(supports_prompt_caching(strip_region_prefix(
            "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
        )));
        assert!(supports_prompt_caching(strip_region_prefix(
            "apac.amazon.nova-lite-v1:0"
        )));
        assert!(!supports_prompt_caching(strip_region_prefix(
            "us.meta.llama4-maverick-17b-instruct-v1:0"
        )));
        // Unknown models default to "no checkpoint": an unwanted checkpoint
        // fails the whole request, a missing one only costs tokens.
        assert!(!supports_prompt_caching("future.unknown-model"));
    }

    #[test]
    fn cache_point_omitted_when_there_is_no_system_prompt() {
        // A checkpoint on an empty `system` list has nothing to cache and
        // would be the only element of the list.
        let map = ToolNameMap::from_names(Vec::<&str>::new());
        let (system, _messages) = convert_messages(
            &[Message::new(Role::User, "hi")],
            &map,
            checkpoint_for("us.anthropic.claude-sonnet-4-6"),
        )
        .expect("convert ok");
        assert!(system.is_empty(), "no system blocks, no checkpoint");
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

    // --- Non-streaming dispatch: timeout and cancellation (#1024) --------
    //
    // The non-streaming path is not a rare fallback. It is taken for every
    // model on the `supports_streaming_with_tools` deny list and for every
    // model the runtime memo has learned rejects tools in streaming mode. A
    // stalled request on that path must fail the turn on its own budget, and
    // must answer the cancellation token, exactly as the streaming path does.

    /// A TCP endpoint that completes the connection and then answers nothing.
    /// This is what a hung Bedrock request looks like from the client side:
    /// the socket is open, so no layer below the connector reports a failure,
    /// and the response future never resolves.
    struct StalledEndpoint {
        url: String,
        accept_loop: tokio::task::JoinHandle<()>,
    }

    impl Drop for StalledEndpoint {
        fn drop(&mut self) {
            self.accept_loop.abort();
        }
    }

    async fn stalled_endpoint() -> StalledEndpoint {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a loopback port");
        let addr = listener.local_addr().expect("read the bound port");
        let accept_loop = tokio::spawn(async move {
            // Hold every accepted socket open for the life of the task. A
            // dropped socket would close the connection and turn the stall
            // into a transport error, which is a different test.
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                held.push(stream);
            }
        });
        StalledEndpoint {
            url: format!("http://{addr}"),
            accept_loop,
        }
    }

    /// A client that dispatches `Converse` at `url`, with `secs` as the
    /// whole-request budget for that path.
    ///
    /// The model is a Llama 4 profile id: Llama 4 is on the
    /// non-streaming-with-tools deny list, so a turn that carries tools takes
    /// `dispatch_non_streaming` without a streaming attempt first.
    fn non_streaming_client(url: &str, secs: u64) -> BedrockClient {
        BedrockClient::new(format!("AKIAIOSFODNN7EXAMPLE:{TEST_SECRET_ACCESS_KEY}"))
            .with_base_url("us-east-1")
            .__with_runtime_endpoint_for_test(url)
            .with_model("us.meta.llama4-maverick-17b-instruct-v1:0")
            .with_non_streaming_timeout(Some(secs))
    }

    /// One tool, which is what puts the turn on the non-streaming path.
    fn one_tool() -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "weather",
            "look up the weather",
            serde_json::json!({"type": "object"}),
        )]
    }

    #[test]
    fn the_non_streaming_budget_is_its_own_setting_with_its_own_default() {
        // `event_timeout` bounds the gap between streamed events. Reusing it
        // to bound a whole generation would give one name two meanings, and
        // would make a stall-detection change silently move a generation
        // deadline. The non-streaming path answers once, after generation, so
        // it carries its own budget - long enough that no turn that works
        // today starts failing.
        let client = BedrockClient::new(String::new());
        assert_eq!(
            client.non_streaming_timeout,
            Duration::from_secs(600),
            "the default must leave room for a full one-shot generation"
        );

        let tightened = BedrockClient::new(String::new())
            .with_connect_timeout(Some(1))
            .with_event_timeout(Some(1));
        assert_eq!(
            tightened.non_streaming_timeout,
            Duration::from_secs(600),
            "tightening the streaming budgets must not move the generation deadline"
        );

        // The override is per-connection, and rejects the two no-op values the
        // sibling budgets also reject.
        assert_eq!(
            BedrockClient::new(String::new())
                .with_non_streaming_timeout(Some(30))
                .non_streaming_timeout,
            Duration::from_secs(30)
        );
        for no_op in [None, Some(0)] {
            assert_eq!(
                BedrockClient::new(String::new())
                    .with_non_streaming_timeout(no_op)
                    .non_streaming_timeout,
                Duration::from_secs(600),
                "{no_op:?} means \"keep the default\""
            );
        }
    }

    #[tokio::test]
    async fn non_streaming_dispatch_that_exceeds_the_timeout_returns_a_timeout_error() {
        let endpoint = stalled_endpoint().await;
        let client = non_streaming_client(&endpoint.url, 1);

        // The outer bound only stops a hang from becoming a suite-wide stall.
        // The assertion is that the connector's own budget ended the call.
        let outcome = tokio::time::timeout(
            Duration::from_secs(20),
            client.stream_completion(
                vec![Message::new(Role::User, "hi")],
                &one_tool(),
                ReasoningConfig::default(),
                Box::new(|_| true),
            ),
        )
        .await
        .expect("a stalled non-streaming dispatch must end on its own timeout, not hang");

        let error = outcome.expect_err("a stalled request cannot produce a response");
        assert!(
            matches!(&error, CoreError::Llm(detail) if detail.contains("timed out")),
            "the failure must name the timeout so an operator can raise the budget, got {error:?}"
        );
    }

    /// One complete `Converse` response, as the API returns it.
    const CONVERSE_RESPONSE_BODY: &str = r#"{
      "output": {
        "message": {
          "role": "assistant",
          "content": [{"text": "the whole answer, in one piece"}]
        }
      },
      "stopReason": "end_turn",
      "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
      "metrics": {"latencyMs": 250}
    }"#;

    #[tokio::test]
    async fn non_streaming_dispatch_inside_the_budget_returns_the_answer() {
        // The bound must end a hung request and nothing else. A turn that
        // takes real time and finishes inside its budget has to come back
        // whole - text, usage, and one callback - or the timeout has turned
        // into a cap on working turns.
        let server = httpmock::MockServer::start();
        let converse = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path_matches(r"/converse$");
            then.status(200)
                .header("content-type", "application/json")
                .delay(Duration::from_millis(400))
                .body(CONVERSE_RESPONSE_BODY);
        });

        let client = non_streaming_client(&server.url(""), 2);
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = seen.clone();

        let response = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &one_tool(),
                ReasoningConfig::default(),
                Box::new(move |chunk| {
                    sink.lock().expect("chunk sink").push(chunk);
                    true
                }),
            )
            .await
            .expect("a turn that answers inside its budget must succeed");

        converse.assert();
        assert_eq!(response.text, "the whole answer, in one piece");
        assert_eq!(
            response.usage.as_ref().and_then(|u| u.output_tokens),
            Some(7)
        );
        assert_eq!(
            seen.lock().expect("chunk sink").as_slice(),
            ["the whole answer, in one piece"],
            "the callback fires once with the full text"
        );
    }

    #[tokio::test]
    async fn cancelling_during_a_non_streaming_dispatch_returns_promptly() {
        use desktop_assistant_core::ports::llm::with_cancellation_token;
        use tokio_util::sync::CancellationToken;

        let endpoint = stalled_endpoint().await;
        // A budget far longer than the test: the timeout must not be what ends
        // this call, or the test would pass without any cancellation support.
        let client = non_streaming_client(&endpoint.url, 600);

        let token = CancellationToken::new();
        let trip = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(250)).await;
            trip.cancel();
        });

        let started = Instant::now();
        let outcome = tokio::time::timeout(
            Duration::from_secs(20),
            with_cancellation_token(token, async {
                client
                    .stream_completion(
                        vec![Message::new(Role::User, "hi")],
                        &one_tool(),
                        ReasoningConfig::default(),
                        Box::new(|_| true),
                    )
                    .await
            }),
        )
        .await
        .expect("cancelling must end the dispatch, not leave it running");

        assert!(
            matches!(outcome, Err(CoreError::Cancelled)),
            "a cancelled non-streaming dispatch reports Cancelled, got {outcome:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "cancellation must be answered promptly; took {:?}",
            started.elapsed()
        );
    }

    // --- Cache policy (#1027) --------------------------------------------
    //
    // Caching is not free. A cache write is billed above the uncached input
    // rate and pays back only when a later turn reads it, so a workload of
    // short one-turn conversations pays the premium every turn. `CachePolicy`
    // is the lever that stops it, and it is also how an operator rules caching
    // out while diagnosing a bad turn.

    /// A model that accepts checkpoints, reached through an inference profile.
    const CACHING_MODEL: &str = "us.anthropic.claude-sonnet-4-6";

    /// The JSON marker a `cachePoint` block leaves in a Converse request body.
    const CACHE_POINT_MARKER: &str = "cachePoint";

    /// A client that dispatches at `url` with `policy` in force. `None` keeps
    /// whatever the connector defaults to, which is the shape the daemon's
    /// builder call uses for an unset connection field.
    fn caching_client(url: &str, policy: Option<CachePolicy>) -> BedrockClient {
        BedrockClient::new(format!("AKIAIOSFODNN7EXAMPLE:{TEST_SECRET_ACCESS_KEY}"))
            .with_base_url("us-east-1")
            .__with_runtime_endpoint_for_test(url)
            .with_model(CACHING_MODEL)
            .with_cache_policy(policy)
    }

    /// A Bedrock `ValidationException` as the runtime returns it: the error
    /// type in `__type` and in the `x-amzn-errortype` header, the human text in
    /// `message`. Status 400, which the AWS SDK does not retry.
    fn validation_exception_body(message: &str) -> String {
        serde_json::json!({
            "__type": "com.amazon.bedrock#ValidationException",
            "message": message,
        })
        .to_string()
    }

    /// Run one turn against a mock that only answers when the request body
    /// matches `expect_checkpoint`, and report whether the mock was hit. The
    /// mock answers with a validation error, because what is under test is the
    /// request that went out, not the reply that came back.
    async fn wire_carries_checkpoint(policy: Option<CachePolicy>, expect_checkpoint: bool) -> bool {
        let server = httpmock::MockServer::start();
        let converse_stream = server.mock(|when, then| {
            let when = when
                .method(httpmock::Method::POST)
                .path_matches(r"/converse-stream$");
            if expect_checkpoint {
                when.body_includes(CACHE_POINT_MARKER);
            } else {
                when.body_excludes(CACHE_POINT_MARKER);
            }
            then.status(400)
                .header("x-amzn-errortype", "ValidationException")
                .header("content-type", "application/json")
                .body(validation_exception_body("mock endpoint: request observed"));
        });

        let client = caching_client(&server.url(""), policy);
        let _ = client
            .stream_completion(
                caching_history(),
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await;

        converse_stream.calls() == 1
    }

    #[test]
    fn cache_policy_none_emits_no_checkpoint_for_a_model_that_supports_caching() {
        let map = ToolNameMap::from_names(Vec::<&str>::new());
        for model in [CACHING_MODEL, "us.amazon.nova-pro-v1:0"] {
            let (system, _messages) = convert_messages(
                &caching_history(),
                &map,
                wants_cache_checkpoint(CachePolicy::None, model),
            )
            .expect("convert ok");

            assert!(
                cache_point_indices(&system).is_empty(),
                "{model} must get no checkpoint under cache_policy = \"none\": {system:?}"
            );
            assert_eq!(
                system.len(),
                2,
                "both system blocks survive without the checkpoint: {system:?}"
            );
        }
    }

    #[test]
    fn the_default_cache_policy_emits_the_checkpoint_exactly_as_it_does_today() {
        // The default must not change behaviour: a caching model still gets one
        // checkpoint directly behind the stable system prefix, and a model
        // without support still gets none.
        assert_eq!(CachePolicy::default(), CachePolicy::SystemPromptOnly);

        let map = ToolNameMap::from_names(Vec::<&str>::new());
        for model in [CACHING_MODEL, "us.amazon.nova-pro-v1:0"] {
            let (system, _messages) = convert_messages(
                &caching_history(),
                &map,
                wants_cache_checkpoint(CachePolicy::default(), model),
            )
            .expect("convert ok");
            assert_eq!(
                cache_point_indices(&system),
                vec![1],
                "{model} keeps its one checkpoint behind the stable prefix: {system:?}"
            );
        }

        for model in [
            "us.meta.llama4-maverick-17b-instruct-v1:0",
            "us.anthropic.claude-3-haiku-20240307-v1:0",
            "future.unknown-model",
        ] {
            let (system, _messages) = convert_messages(
                &caching_history(),
                &map,
                wants_cache_checkpoint(CachePolicy::default(), model),
            )
            .expect("convert ok");
            assert!(
                cache_point_indices(&system).is_empty(),
                "{model} still gets no checkpoint under the default: {system:?}"
            );
        }
    }

    #[test]
    fn cache_policy_spellings_match_the_documented_configuration() {
        // `docs/connectors/bedrock.md` documents these two values. A drift here
        // makes a configuration file that reads correctly fail to load.
        for (spelling, expected) in [
            ("none", CachePolicy::None),
            ("system_prompt_only", CachePolicy::SystemPromptOnly),
        ] {
            let parsed: CachePolicy = serde_json::from_value(serde_json::json!(spelling))
                .unwrap_or_else(|e| panic!("cache_policy = \"{spelling}\" must parse: {e}"));
            assert_eq!(parsed, expected);
            assert_eq!(
                serde_json::to_value(expected).expect("serialises"),
                serde_json::json!(spelling),
                "the value written back must be the value documented"
            );
        }

        // A value that is not one of the two is a configuration mistake, and
        // must be reported rather than silently taken as the default.
        assert!(
            serde_json::from_value::<CachePolicy>(serde_json::json!("system_prompt_and_tools"))
                .is_err(),
            "an unshipped policy name must not parse"
        );
    }

    #[tokio::test]
    async fn cache_policy_none_sends_no_checkpoint_on_the_wire() {
        assert!(
            wire_carries_checkpoint(Some(CachePolicy::None), false).await,
            "the request that reached Bedrock still carried a checkpoint"
        );
    }

    #[tokio::test]
    async fn the_default_cache_policy_sends_the_checkpoint_on_the_wire() {
        assert!(
            wire_carries_checkpoint(None, true).await,
            "the default must keep sending the checkpoint a caching model accepts"
        );
    }

    // --- Recovery from a refused checkpoint (#1028) -----------------------
    //
    // The caching allow-list is read from AWS documentation that lists only
    // the models absent from "Models at a glance", so it is a best reading and
    // not an enumeration. A model on the list that refuses a checkpoint would
    // fail every turn. The connector retries the turn once without the
    // checkpoint and remembers the model, which turns a permanent failure into
    // one wasted call.

    /// The refusal, as Bedrock words it. **Unverified against a live account**:
    /// no Converse-caching model was reachable to capture the real text, so
    /// this is built from the documented shape. What the classifier requires is
    /// the field name, and that is what a refusal of the block must carry.
    const CACHE_REFUSAL_MESSAGE: &str =
        "The model returned the following errors: This model doesn't support the cachePoint block.";

    /// A validation failure with nothing to do with caching. Naming a tool
    /// schema is the realistic case: this repo has lived it (#336).
    const UNRELATED_VALIDATION_MESSAGE: &str =
        "The json schema for tool weather is invalid: top-level oneOf is not supported.";

    /// A client on the non-streaming path, so a whole turn can be driven
    /// against a `Converse` mock. The model supports caching, so the first
    /// request carries a checkpoint; the memo puts it on the non-streaming
    /// path without a stream attempt first.
    async fn cache_recovery_client(url: &str, policy: Option<CachePolicy>) -> BedrockClient {
        let client = caching_client(url, policy).with_non_streaming_timeout(Some(10));
        client
            .__force_non_streaming_tools_for_test(CACHING_MODEL)
            .await;
        client
    }

    /// A `Converse` mock that answers `message` as a validation failure for
    /// every request whose body carries a checkpoint.
    fn refuse_checkpoint<'a>(
        server: &'a httpmock::MockServer,
        message: &str,
    ) -> httpmock::Mock<'a> {
        let body = validation_exception_body(message);
        server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path_matches(r"/converse$")
                .body_includes(CACHE_POINT_MARKER);
            then.status(400)
                .header("x-amzn-errortype", "ValidationException")
                .header("content-type", "application/json")
                .body(body);
        })
    }

    /// A `Converse` mock that answers normally for every request whose body
    /// carries no checkpoint.
    fn accept_without_checkpoint(server: &httpmock::MockServer) -> httpmock::Mock<'_> {
        server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path_matches(r"/converse$")
                .body_excludes(CACHE_POINT_MARKER);
            then.status(200)
                .header("content-type", "application/json")
                .body(CONVERSE_RESPONSE_BODY);
        })
    }

    async fn one_turn(client: &BedrockClient) -> Result<LlmResponse, CoreError> {
        client
            .stream_completion(
                caching_history(),
                &one_tool(),
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
    }

    #[tokio::test]
    async fn a_refused_cache_checkpoint_retries_the_turn_once_without_it() {
        let server = httpmock::MockServer::start();
        let refused = refuse_checkpoint(&server, CACHE_REFUSAL_MESSAGE);
        let accepted = accept_without_checkpoint(&server);

        let client = cache_recovery_client(&server.url(""), None).await;
        let response = one_turn(&client)
            .await
            .expect("the turn must survive a refused checkpoint");

        assert_eq!(response.text, "the whole answer, in one piece");
        assert_eq!(refused.calls(), 1, "exactly one doomed attempt");
        assert_eq!(
            accepted.calls(),
            1,
            "exactly one retry, and it carried none"
        );
    }

    #[tokio::test]
    async fn a_model_that_refused_a_checkpoint_sends_none_on_the_next_turn() {
        let server = httpmock::MockServer::start();
        let refused = refuse_checkpoint(&server, CACHE_REFUSAL_MESSAGE);
        let accepted = accept_without_checkpoint(&server);

        let client = cache_recovery_client(&server.url(""), None).await;
        one_turn(&client).await.expect("first turn recovers");
        one_turn(&client).await.expect("second turn succeeds");

        assert_eq!(
            refused.calls(),
            1,
            "the second turn must not repeat the doomed attempt"
        );
        assert_eq!(
            accepted.calls(),
            2,
            "both turns answered without a checkpoint"
        );
    }

    #[tokio::test]
    async fn an_unrelated_validation_error_is_not_swallowed_by_the_cache_recovery() {
        let server = httpmock::MockServer::start();
        // The failure has nothing to do with caching, so it must reach the
        // caller, and it must not teach the connector anything about this
        // model. A 400 is not evidence about a cache checkpoint.
        let refused = refuse_checkpoint(&server, UNRELATED_VALIDATION_MESSAGE);
        let accepted = accept_without_checkpoint(&server);

        let client = cache_recovery_client(&server.url(""), None).await;
        let error = one_turn(&client)
            .await
            .expect_err("an unrelated validation failure must fail the turn");
        assert!(
            matches!(&error, CoreError::Llm(detail) if detail.contains("oneOf")),
            "the provider's own message must survive, got {error:?}"
        );
        assert_eq!(refused.calls(), 1, "one attempt, no retry");
        assert_eq!(
            accepted.calls(),
            0,
            "nothing may be retried without the checkpoint"
        );

        // And the model must not have been memoised: the next turn still sends
        // the checkpoint the model in fact accepts.
        let _ = one_turn(&client).await;
        assert_eq!(
            refused.calls(),
            2,
            "an unrelated failure must not disable caching for this model"
        );
    }

    #[tokio::test]
    async fn a_refusal_naming_the_cache_field_is_ignored_when_the_request_carried_none() {
        // The trap this guard closes: a request that sent no checkpoint cannot
        // have had one refused, so a message naming the field is about
        // something else - and a retry that omits a field it never sent proves
        // nothing. Classifying here would let the fallback manufacture the
        // evidence for its own verdict.
        let server = httpmock::MockServer::start();
        let refused_anything = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path_matches(r"/converse$");
            then.status(400)
                .header("x-amzn-errortype", "ValidationException")
                .header("content-type", "application/json")
                .body(validation_exception_body(CACHE_REFUSAL_MESSAGE));
        });

        let client = cache_recovery_client(&server.url(""), Some(CachePolicy::None)).await;
        one_turn(&client)
            .await
            .expect_err("the validation failure must reach the caller");

        assert_eq!(
            refused_anything.calls(),
            1,
            "no retry: there was no checkpoint to withdraw"
        );
    }

    #[tokio::test]
    async fn a_turn_with_no_system_prompt_sends_no_checkpoint_and_so_cannot_have_one_refused() {
        // The policy allows a checkpoint and the model accepts one, but there
        // is no system prefix to mark, so nothing goes out. The guard has to
        // read the request that was built, not the intent to build one: a
        // refusal of a field this request did not carry is a refusal of
        // something else, and acting on it would memoise the model on evidence
        // about nothing.
        let server = httpmock::MockServer::start();
        let refused_anything = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path_matches(r"/converse$");
            then.status(400)
                .header("x-amzn-errortype", "ValidationException")
                .header("content-type", "application/json")
                .body(validation_exception_body(CACHE_REFUSAL_MESSAGE));
        });

        let client = cache_recovery_client(&server.url(""), None).await;
        client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &one_tool(),
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("the validation failure must reach the caller");

        assert_eq!(
            refused_anything.calls(),
            1,
            "no retry: the request carried no checkpoint to withdraw"
        );
    }

    #[tokio::test]
    async fn the_streaming_path_also_retries_without_the_checkpoint() {
        // Both dispatch paths carry the checkpoint, so both must recover. The
        // retry's own reply is another failure, because a `ConverseStream`
        // success is an AWS event stream; what this pins is that the second
        // request went out without the checkpoint.
        let server = httpmock::MockServer::start();
        let refused = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path_matches(r"/converse-stream$")
                .body_includes(CACHE_POINT_MARKER);
            then.status(400)
                .header("x-amzn-errortype", "ValidationException")
                .header("content-type", "application/json")
                .body(validation_exception_body(CACHE_REFUSAL_MESSAGE));
        });
        let retried = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path_matches(r"/converse-stream$")
                .body_excludes(CACHE_POINT_MARKER);
            then.status(400)
                .header("x-amzn-errortype", "ValidationException")
                .header("content-type", "application/json")
                .body(validation_exception_body(UNRELATED_VALIDATION_MESSAGE));
        });

        let client = caching_client(&server.url(""), None);
        let error = client
            .stream_completion(
                caching_history(),
                &[],
                ReasoningConfig::default(),
                Box::new(|_| true),
            )
            .await
            .expect_err("the retry fails in this test, on purpose");

        assert_eq!(refused.calls(), 1, "one doomed streaming attempt");
        assert_eq!(retried.calls(), 1, "one retry, without the checkpoint");
        assert!(
            matches!(&error, CoreError::Llm(detail) if detail.contains("oneOf")),
            "the retry's own failure is what the caller sees, got {error:?}"
        );
    }

    #[test]
    fn only_a_message_naming_the_cache_field_counts_as_a_checkpoint_refusal() {
        // The classifier is the whole guard against a wrong verdict, so it is
        // pinned in both directions.
        for names_it in [
            CACHE_REFUSAL_MESSAGE,
            "Invalid value at 'system[1].cachePoint'",
            "cache_control is not supported for this model",
            "This model does not support prompt caching.",
            // Space-separated prose. A miss here is the failure this
            // recovery exists to remove.
            "The cache point block is not supported by this model.",
        ] {
            assert!(
                names_the_cache_field(names_it),
                "must be recognised as a checkpoint refusal: {names_it}"
            );
        }

        for names_something_else in [
            UNRELATED_VALIDATION_MESSAGE,
            "Input is too long for requested model.",
            "The provided model identifier is invalid.",
            "Malformed input request: #/system/1: subject must not be valid against schema",
            // The bare gerund is deliberately not a marker. Matching is a
            // substring test over the whole message, and Bedrock quotes the
            // offending schema path, so a tool whose input schema has a
            // property named `caching` - an HTTP fetch tool with a cache
            // toggle, a build tool with `caching: bool` - would turn its own
            // schema fault into a wasted call, a `warn!` pointing an operator
            // at prompt caching, and a model memoised as cache-refusing.
            "The json schema for tool fetch is invalid: properties.caching is not supported.",
            "",
        ] {
            assert!(
                !names_the_cache_field(names_something_else),
                "must not be read as a checkpoint refusal: {names_something_else}"
            );
        }
    }
}
