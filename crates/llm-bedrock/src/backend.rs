//! One Bedrock API surface, behind one trait.
//!
//! Bedrock serves its models through several different APIs, and no single API
//! reaches all of them. A backend is one of those APIs. Backends exist for
//! **reach**, not for capability: each one reaches models the others cannot,
//! and the capability differences between them decide which backend serves a
//! request, not whether the backend exists at all.
//!
//! The connector holds the backends and hides the choice from the user. A
//! person configures one Bedrock connection and picks a model. They never pick
//! an API. `docs/connectors/bedrock.md` records the design.

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Message, ToolDefinition};
use desktop_assistant_core::ports::llm::{
    ChunkCallback, LlmResponse, ModelListingReport, ReasoningConfig,
};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::ToolNameMap;

pub(crate) mod converse;
pub(crate) mod invoke;

/// One Bedrock API surface.
///
/// An implementor translates the connector's request into its own API shape
/// and translates the response back.
///
/// What a backend does **not** do:
///
/// - **It does not retry a failed call.** `RetryingLlmClient` in `core` wraps
///   the whole connector from outside and already applies exponential backoff
///   to retryable errors. A second loop inside a backend nests the backoffs
///   and multiplies the attempts. Changing the *request* and sending it again
///   is a different thing, and is allowed: see `cache_checkpoint` on
///   [`BedrockRequest`].
/// - **It does not decide which models the user sees.** It reports what it
///   reaches; the connector merges, de-duplicates and caches.
/// - **It does not read task-locals.** Everything a request depends on arrives
///   on [`BedrockRequest`], so a backend is testable without a task context.
///
/// The trait uses `#[async_trait]` and not `-> impl Future` in return
/// position, because the connector holds `Vec<Arc<dyn BedrockBackend>>` and
/// return-position `impl Trait` is not dyn-compatible.
///
/// It does not extend `LlmClient`. The connector implements `LlmClient` and
/// the backends sit behind it, which keeps `core` unaware of Bedrock
/// internals.
#[async_trait]
pub(crate) trait BedrockBackend: Send + Sync {
    /// Short API name, for logs, notices and model annotation.
    fn api_name(&self) -> &'static str;

    /// Whether this backend can serve a **completion** for `model_id`.
    ///
    /// This is the routing primitive, and completion is the whole of the
    /// question: backend selection asks it once, about a turn. It is not "does
    /// this backend list the model", and the two answers come apart. A surface
    /// serving a modality that cannot hold a conversation answers `false` for
    /// every model it serves, and still contributes those models to the
    /// catalogue, because a person picks them for that other purpose.
    ///
    /// So a backend that answered `true` here for such a model would take a
    /// chat turn and fail at the service, and a catalogue filtered by this
    /// answer would lose those models from the picker.
    ///
    /// Answer permissively for a model this backend knows nothing about, where
    /// it can serve completions at all. A model no listing described is a
    /// model the connector has no reason to refuse, and refusing it would turn
    /// missing metadata into a failed turn.
    fn can_serve(&self, model_id: &str) -> bool;

    /// The models this backend reaches, with any listing notices.
    ///
    /// Returns a [`ModelListingReport`] and not a plain list because a partial
    /// answer must stay distinguishable from a small account: a failure that
    /// degrades the catalogue contributes a notice beside the models it could
    /// still read. Caching the answer is the connector's job.
    async fn list_models(&self) -> Result<ModelListingReport, CoreError>;

    /// What this API surface supports for one model.
    ///
    /// Why the model id: support varies per model *inside* a single API.
    /// Converse accepts a cache checkpoint for Anthropic and Amazon Nova
    /// models, and rejects it for Meta, Mistral and Cohere models. A per-API
    /// constant cannot express that.
    fn capabilities(&self, model_id: &str) -> BackendApiCapabilities;

    /// Stream a completion.
    ///
    /// `on_chunk` fires at least once with the model's prose output, even
    /// where the underlying API answers in one piece, so the caller's contract
    /// does not change with the path taken.
    async fn stream_completion(
        &self,
        request: BedrockRequest,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError>;
}

/// One turn, in terms every backend understands.
///
/// It carries domain types rather than one API's wire types. The Converse
/// request shape is `aws_sdk_bedrockruntime` messages and content blocks,
/// which the Responses and Invoke surfaces do not accept, so translation
/// belongs inside each backend and this type stays neutral.
pub(crate) struct BedrockRequest {
    /// The id to dispatch against. It may be an inference-profile id, so a
    /// backend reduces it with `base_model_for` before consulting any
    /// per-model gate.
    pub model: String,
    /// The conversation, oldest first, system messages included.
    pub messages: Vec<Message>,
    /// The tools offered this turn, with their schemas as the caller supplied
    /// them. A backend applies whatever its own API needs.
    pub tools: Vec<ToolDefinition>,
    /// The sanitized-to-original tool-name bijection for this turn.
    ///
    /// The connector builds it once so every backend spells a tool the same
    /// way, in the request and in the message history, and reverses it the
    /// same way on the response.
    pub tool_names: ToolNameMap,
    /// The reasoning effort asked for this turn. A backend that cannot act on
    /// it reports that rather than dropping it silently.
    pub reasoning: ReasoningConfig,
    /// Temperature, nucleus sampling and output cap, from the connection.
    pub sampling: SamplingParams,
    /// Whether this turn may carry a prompt-cache checkpoint.
    ///
    /// The connector answers the operator's half: the cache policy allows one,
    /// and the model accepts one. A backend answers its own half - whether the
    /// model has already refused a checkpoint on *this* surface - and may
    /// withhold the checkpoint on that ground. It may also send the turn
    /// again without the checkpoint when the service refuses it, because that
    /// changes the request rather than retrying a transport failure.
    ///
    /// The two halves stay apart because a refusal is per (surface, model): a
    /// model can accept Anthropic `cache_control` through Invoke and reject
    /// `cachePoint` through Converse.
    pub cache_checkpoint: bool,
    /// Cancellation for this turn. A backend races every network wait against
    /// it, so a stop ends the turn rather than waiting a timeout out.
    pub cancellation: CancellationToken,
}

/// One thing an API surface either does or does not do for a model.
///
/// Named rather than a bare field access, so a refusal can say which
/// capability was missing in words a person reads.
/// It carries only what a turn can genuinely demand. A capability whose
/// absence merely costs money or plainness is not one of these, and adding it
/// here is how a degraded turn becomes a refused turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Feature {
    /// The turn offers tools, so a surface that cannot carry them cannot serve
    /// it. A surface that drops them answers a different question from the one
    /// the caller asked, which is why `CoreError::ToolsUnsupported` exists and
    /// says a caller must switch model or disable tools rather than retry.
    Tools,
    /// The turn carries image input, so a surface that cannot send one cannot
    /// serve it.
    Vision,
}

impl Feature {
    /// Whether `capabilities` provides this feature.
    pub(crate) fn provided_by(self, capabilities: &BackendApiCapabilities) -> bool {
        match self {
            Feature::Tools => capabilities.tools,
            Feature::Vision => capabilities.vision,
        }
    }

    /// The name a refusal uses.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Feature::Tools => "tools",
            Feature::Vision => "vision",
        }
    }
}

/// Whether this turn carries image input.
///
/// Always `false`, and not a placeholder: `Message` is text with tool fields,
/// so no image can reach a connector at all. #1059 covers giving the Converse
/// surface an image content block, and this predicate is what turns that into
/// a routing requirement when it lands.
fn turn_carries_an_image(_request: &BedrockRequest) -> bool {
    false
}

/// What this turn cannot be served without.
///
/// The test is whether the absence makes the answer **wrong**, not whether it
/// makes the answer dearer or plainer. Every field of
/// [`BackendApiCapabilities`] is weighed against it:
///
/// - **Tools** - hard, and demanded here. A surface that drops the tool list
///   answers a different question from the one the caller asked, and the turn
///   cannot recover: `CoreError::ToolsUnsupported` states exactly that, and
///   says the caller must switch model or disable tools.
/// - **Vision** - hard, and not yet demandable. No domain message can carry an
///   image, so no turn can need one. #1059 makes it real, and
///   `turn_carries_an_image` is where it enters.
/// - **Cache control** - soft. A withheld checkpoint costs input tokens and
///   nothing else.
/// - **Reasoning** - soft, and deliberately so. A budget the model cannot take
///   is reported at `warn!` with the model and the budget, and the turn goes
///   out without it. Demanding it here would refuse every model that reasons
///   without taking a setting, DeepSeek R1 among them.
/// - **Streaming** - soft. The trait guarantees `on_chunk` fires at least once
///   whatever path a surface takes, so a surface that cannot stream still
///   honours the contract.
/// - **Embeddings** and **hosted tool search** - neither is a property a
///   completion turn can ask for. Reach answers the first, and nothing in this
///   connector offers the second.
///
/// The derivation lives in one function so a future requirement is added once,
/// and so a test can hold the soft ones out of it: promoting one turns a
/// degraded turn into a refused turn across the whole fleet.
pub(crate) fn required_features(request: &BedrockRequest) -> Vec<Feature> {
    let mut required = Vec::new();
    if !request.tools.is_empty() {
        required.push(Feature::Tools);
    }
    if turn_carries_an_image(request) {
        required.push(Feature::Vision);
    }
    required
}

/// The sampling settings a connection carries, in provider-neutral form.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
}

/// The network budgets a connection applies to every backend.
///
/// Timeouts are connector-level because they are network concerns rather than
/// API concerns, and every backend gets the same three.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BackendTimeouts {
    /// Time to establish the connection and receive the first response.
    pub connect: Duration,
    /// Time between streamed events before the stream counts as stalled.
    pub event: Duration,
    /// Whole-request budget for a path that answers once, after generation is
    /// complete, and so has no intermediate event to time.
    pub non_streaming: Duration,
}

/// What one API surface supports for one model.
///
/// Bedrock-local, and deliberately so: only Bedrock has several API surfaces
/// with differing capabilities. Azure's surfaces differ in URL shape with
/// identical capabilities, and Google's differ in host and credential, so
/// eight other connectors should not carry a layer that describes one.
/// `docs/design/connector-capabilities.md` records that boundary.
///
/// **No `Default` implementation, deliberately.** Every field is stated at
/// every construction site, so the compiler stops a new capability from
/// arriving as a silent `false` on a backend nobody edited - the direction
/// that invents or erases a capability without anyone reading a diff. A
/// backend is never uncertain about its own API surface, so the three-state
/// `Unknown` the shared capability model uses has no meaning at this layer;
/// uncertainty enters when these answers are composed with the model's and the
/// connector's into the per-(connection, model) answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackendApiCapabilities {
    /// The surface can stream a completion token by token.
    pub streaming: bool,
    /// The surface accepts tool definitions and returns tool calls.
    pub tools: bool,
    /// The surface can carry image input **as this backend builds a request**,
    /// which is not the same as what the API documents. A backend reports what
    /// it actually sends.
    pub vision: bool,
    /// The surface accepts a prompt-cache checkpoint for this model.
    pub cache_control: bool,
    /// The surface takes a reasoning configuration for this model that changes
    /// what the model does. A model that always reasons and takes no setting
    /// answers `false`: nothing the connector sends changes its behaviour.
    pub reasoning: bool,
    /// The surface runs tool search on the server.
    pub hosted_tool_search: bool,
    /// The surface returns vectors for this model.
    pub embeddings: bool,
}
