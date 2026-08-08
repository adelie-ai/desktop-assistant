use std::sync::{Arc, Mutex};
use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};
use tokio_util::sync::CancellationToken;

use crate::CoreError;
use crate::domain::{Message, ToolCall, ToolDefinition, ToolNamespace};

/// Callback invoked for each chunk of a streaming LLM response.
/// Return `true` to continue, `false` to abort the stream.
pub type ChunkCallback = Box<dyn FnMut(String) -> bool + Send>;

/// Callback invoked to report progress while the assistant is working
/// (e.g. "Searching knowledge base...", "Querying timeclock sessions...").
pub type StatusCallback = Box<dyn FnMut(String) + Send>;

/// Per-turn context-window fill snapshot reported by the dispatch loop after
/// each LLM call (issue #341). Carries token COUNTS only — never message
/// content — so a client can render a "used / budget (%)" indicator and shift
/// colour as the proactive-compaction line is approached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextUsage {
    /// Prompt/input tokens the provider reported for this turn.
    pub used_tokens: u64,
    /// Resolved max input-token budget for this turn.
    pub budget_tokens: u64,
    /// `true` once the effective window was shrunk and the dropped range
    /// compacted on this turn (proactive compaction ran).
    pub compaction_active: bool,
}

/// Sink the dispatch loop calls with each turn's [`ContextUsage`]. Installed
/// as a task-local by the daemon/application transport layer (which owns the
/// event channel) via [`with_context_usage_sink`]; read in `send_prompt` via
/// [`emit_context_usage`]. `Arc<dyn Fn>` so it is `Clone` for the task-local
/// slot and may fan to the event sink from any concurrent turn. Unset outside
/// the scope (tests, dreaming jobs), in which case emission is a no-op.
pub type ContextUsageSink = std::sync::Arc<dyn Fn(ContextUsage) + Send + Sync>;

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

    /// Per-turn context-usage sink (issue #341). Installed by the transport
    /// layer (which owns the client event channel) via
    /// [`with_context_usage_sink`]; the dispatch loop reports each turn's fill
    /// via [`emit_context_usage`]. Lives in core so the read site in
    /// `service::ConversationHandler` need not know the transport's event
    /// plumbing — same rationale as [`CONTEXT_BUDGET`]. Unset for callers that
    /// don't route through the transport layer (tests, dreaming jobs), where
    /// emission is a silent no-op.
    static CONTEXT_USAGE_SINK: ContextUsageSink;

    /// Per-turn cancellation token for `send_prompt` (issue #109).
    ///
    /// Lifecycle: installed by `ConversationService::send_prompt_with_override`
    /// at the top of the dispatch via [`with_cancellation_token`], read at the
    /// cooperative cancellation checkpoints inside `send_prompt` (between
    /// agentic turns, before each tool-round dispatch, inside the chunk
    /// callback) and inside each LLM adapter's streaming loop via
    /// [`current_cancellation_token`]. Unset outside the scope, which
    /// `current_cancellation_token` returns as `None` so legacy callers
    /// (tests, dreaming jobs) get the pre-#109 "never cancel" behaviour.
    ///
    /// Why a task-local: matches the existing pattern used by
    /// [`MODEL_OVERRIDE`], [`REASONING_CONFIG`], and [`CONTEXT_BUDGET`] —
    /// threading the value through every connector trait method would mean
    /// touching dozens of call sites; the task-local keeps the
    /// `LlmClient::stream_completion` signature unchanged so adapters opt in
    /// at the boundary that actually needs cancellation (the streaming
    /// loop) instead of every wrapper / decorator on the chain.
    static CANCELLATION_TOKEN: CancellationToken;

    /// Per-turn tool allowlist (issues #112 / #113).
    ///
    /// When set, only tool names in the list may be exposed to the LLM
    /// for this turn — every other tool is hidden from the dispatch
    /// path. When unset, all available tools are exposed (the pre-#112
    /// behaviour). An empty allowlist means "no tools allowed for this
    /// turn" — useful for safety-critical agent runs that should never
    /// take tool actions.
    ///
    /// Both `spawn_subagent` (#112) and `SpawnStandaloneAgent` (#113)
    /// install this task-local from their respective `tools` field, so
    /// the actual gating implementation can live in a single place
    /// (tool-selection in the dispatch loop) and serve both call sites.
    ///
    /// Why a task-local: mirrors the other per-turn task-locals in this
    /// module so the dispatch loop can read it without a signature
    /// change on `ConversationService::send_prompt_with_override` or
    /// the connector traits.
    static TOOL_ALLOWLIST: Vec<String>;

    /// Reentrancy guard for the backend-error classifier (epic #178, tier 3).
    ///
    /// The classifier's LLM tier calls the cheap task LLM to classify an
    /// opaque error. That LLM client is itself wrapped by the classifying
    /// decorator, so if *its* call errors the error would be classified —
    /// calling the LLM again, and so on. The decorator installs this
    /// task-local around the classification call; any decorator that sees it
    /// set skips the learned-cache and LLM tiers (tier 1 still runs), which
    /// breaks the recursion regardless of how the LLM handles are wired.
    static CLASSIFICATION_IN_PROGRESS: ();

    /// Per-request system-prompt refinement.
    ///
    /// An optional client-supplied addition to the system prompt that applies
    /// to a single `send_prompt` request only. Installed by the daemon's
    /// dispatch wrapper (`send_prompt_with_override`) via
    /// [`with_system_refinement`] from the request's `system_refinement`
    /// field, and read by the context assembler via
    /// [`current_system_refinement`] when it builds the system message. The
    /// text is appended *after* the conversation's normal system prompt for
    /// that turn.
    ///
    /// Crucially this is **request-scoped, not conversation-scoped**: it is
    /// never stored as a message and never written to the conversation, so it
    /// does not appear in chat history and does not affect later turns. A
    /// voice client can therefore attach "respond briefly, by voice" to one
    /// turn dictated into an existing chat without permanently changing how
    /// that conversation behaves for subsequent typed turns.
    ///
    /// Unset (or empty) outside the daemon dispatch path — which
    /// [`current_system_refinement`] returns as an empty string — so tests,
    /// dreaming jobs, and any caller that doesn't route through the dispatch
    /// wrapper get the unchanged "no refinement" behaviour.
    ///
    /// Why a task-local: mirrors the other per-turn task-locals in this
    /// module (e.g. [`REASONING_CONFIG`], [`CONTEXT_BUDGET`]) so the value
    /// threads to the assembler without changing the `LlmClient` connector
    /// trait signatures. The *request* value still travels as an explicit
    /// argument through the application layer (so it survives the
    /// `tokio::spawn` background-task boundary, which task-locals do not
    /// cross); the task-local is installed only at the final dispatch hop,
    /// inside the spawned body.
    static SYSTEM_REFINEMENT: String;

    /// The active assistant personality for this turn (issue #226, Phase 1:
    /// global). Installed by the daemon's dispatch wrapper via
    /// [`with_personality`] from the resolved global config, and read by the
    /// context assembler via [`current_personality`] when it builds the system
    /// message. The rendered disposition blurb is injected as a system-prompt
    /// section before the tool note and the per-turn system refinement.
    ///
    /// Unlike [`SYSTEM_REFINEMENT`] this is *configuration*, not per-request
    /// client input: it carries the standing personality the user configured,
    /// not a one-turn instruction. It still rides a task-local for the same
    /// reason the others do — it threads to the assembler without changing the
    /// `LlmClient` trait or the `send_prompt` signature.
    ///
    /// Phase 1 resolves it from the global config on every send. Phase 2 will
    /// add per-conversation resolution; that future seam belongs at the install
    /// site (the dispatch wrapper picks *which* personality to install), so the
    /// read side here stays unchanged.
    ///
    /// Unset outside the dispatch path — which [`current_personality`] returns
    /// as [`crate::prompts::Personality::default`], so tests, dreaming jobs, and
    /// any caller that doesn't route through the dispatch wrapper still get the
    /// default Expressive-7 disposition.
    static PERSONALITY: crate::prompts::Personality;

    /// Pre-rendered ambient "now" line for this turn — e.g.
    /// `Sunday, 2026-06-28, 2:32 PM EDT`. Installed by the daemon's dispatch
    /// wrapper via [`with_now_context`] from a single [`crate::clock::NowSnapshot`]
    /// captured at turn entry, and read by the context assembler via
    /// [`current_now_context`], which surfaces it as a `[Now]` system message so
    /// the assistant has a standing sense of the current date/time without
    /// spending a `builtin_sys_props` tool round to find out.
    ///
    /// Captured once per turn (not per tool round) so every assembly pass within
    /// the turn — including the budget shrink loop's repeat passes — sees a
    /// stable value. The string is rendered from the same snapshot logic that
    /// backs the `builtin_sys_props` tool, so the ambient block and the tool can
    /// never disagree about the clock.
    ///
    /// Unset (or empty) outside the daemon dispatch path — which
    /// [`current_now_context`] returns as an empty string — so tests, dreaming
    /// jobs, and any caller that doesn't route through the dispatch wrapper get
    /// the unchanged "no `[Now]` block" behaviour. Mirrors [`SYSTEM_REFINEMENT`]:
    /// request-scoped, never persisted, threaded via a task-local so the
    /// `send_prompt` and `LlmClient` signatures stay unchanged.
    static NOW_CONTEXT: String;

    /// The client-supplied idempotency key for the FOREGROUND send in progress
    /// (#570 Phase 1b).
    ///
    /// Installed by the application layer's foreground dispatch wrapper via
    /// [`with_idempotency_key`] from the request's echoed key, and read by
    /// `send_prompt` via [`current_idempotency_key`] at the single
    /// user-message persist site so the key is stamped onto the USER row. It is
    /// then surfaced back on load so a reconnecting client dedups an echoed
    /// `UserMessageAdded` by exact match rather than a content compare.
    ///
    /// Deliberately installed on the FOREGROUND path only: agent runs
    /// (standalone / subagent) dispatch through `send_prompt_with_override`
    /// without this wrap, so their user rows persist `None` — a background agent
    /// turn is never a client-retryable send. Unset outside the wrap — which
    /// [`current_idempotency_key`] returns as `None` — so tests, dreaming jobs,
    /// and agent runs persist no key. The stored value is itself an `Option` so
    /// a keyless foreground send (installed as `None`) is distinguishable in
    /// contract from "no wrap at all", though both read back as `None`.
    ///
    /// Why a task-local: mirrors the other per-turn task-locals in this module
    /// so the value threads to `send_prompt` without changing the
    /// `ConversationService`/`LlmClient` signatures.
    static IDEMPOTENCY_KEY: Option<String>;

    /// The per-conversation tool-provenance gate override for this turn
    /// (issue #1007). Installed by the daemon's dispatch wrapper via
    /// [`with_tool_gate_disabled`] from the conversation's stored override,
    /// resolved fresh on every send; read by [`current_tool_gate_disabled`]
    /// at the point `TurnProvenance` is constructed
    /// (`ConversationHandler::send_prompt`), which passes the value to
    /// [`crate::tool_provenance::TurnProvenance::new_with_gate_disabled`].
    ///
    /// `true` means the tool-provenance gate never refuses for this turn,
    /// whatever it ingests - a deliberate, per-conversation safety-off
    /// switch. Fail-closed by construction: unset outside the scope (tests,
    /// dreaming jobs, any caller that doesn't route through the daemon
    /// dispatch wrapper), which [`current_tool_gate_disabled`] returns as
    /// `false` — the gate stays enforced. The daemon-side resolver
    /// (`RoutingConversationHandler::resolve_tool_gate_disabled`) applies the
    /// same fail-closed rule one layer up: a missing row, a cross-user row,
    /// or a store error all resolve to `false` before this scope is even
    /// installed.
    ///
    /// Why a task-local: mirrors [`PERSONALITY`] and the other per-turn
    /// task-locals in this module so the value threads to
    /// `ConversationHandler::send_prompt` without changing the
    /// `ConversationService`/`LlmClient` signatures.
    static TOOL_GATE_DISABLED: bool;
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
    REASONING_CONFIG.try_with(|c| *c).unwrap_or_default()
}

/// Run `fut` with `refinement` installed as the current request's
/// system-prompt refinement. The context assembler reads it via
/// [`current_system_refinement`] and appends it to the system message for
/// the turn. See `SYSTEM_REFINEMENT`.
pub async fn with_system_refinement<F, T>(refinement: String, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    SYSTEM_REFINEMENT.scope(refinement, fut).await
}

/// Current task-local system-prompt refinement, or an empty string when not
/// set. An empty result means "no refinement" — the assembler appends
/// nothing and the system prompt is unchanged. Safe to call from any async
/// context.
pub fn current_system_refinement() -> String {
    SYSTEM_REFINEMENT
        .try_with(|r| r.clone())
        .unwrap_or_default()
}

/// Run `fut` with `personality` installed as the active personality for this
/// turn. The context assembler reads it via [`current_personality`] and injects
/// the rendered disposition blurb into the system message. See `PERSONALITY`.
pub async fn with_personality<F, T>(personality: crate::prompts::Personality, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    PERSONALITY.scope(personality, fut).await
}

/// Current task-local personality, or [`crate::prompts::Personality::default`]
/// when not set. The default means "the Expressive-7 disposition" — callers
/// that don't route through the daemon dispatch wrapper (tests, dreaming jobs)
/// still get the standard personality blurb. Safe to call from any async
/// context.
pub fn current_personality() -> crate::prompts::Personality {
    PERSONALITY.try_with(|p| *p).unwrap_or_default()
}

/// Run `fut` with `now_line` installed as this turn's ambient "now" context.
/// The context assembler reads it via [`current_now_context`] and surfaces it
/// as a `[Now]` system message for the turn. Pass the rendered output of
/// [`crate::clock::NowSnapshot::ambient_line`]. See `NOW_CONTEXT`.
pub async fn with_now_context<F, T>(now_line: String, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    NOW_CONTEXT.scope(now_line, fut).await
}

/// Current task-local ambient "now" line, or an empty string when not set. An
/// empty result means "no `[Now]` block" — the assembler surfaces nothing and
/// the message list is unchanged. Safe to call from any async context.
pub fn current_now_context() -> String {
    NOW_CONTEXT.try_with(|n| n.clone()).unwrap_or_default()
}

/// Run `fut` with `key` installed as the current foreground send's client
/// idempotency key. `send_prompt` reads it via [`current_idempotency_key`] and
/// stamps it onto the USER message row it persists. See `IDEMPOTENCY_KEY`.
///
/// The application layer wraps ONLY the foreground dispatch with this; agent
/// runs deliberately do not, so their user rows persist `None`.
pub async fn with_idempotency_key<F, T>(key: Option<String>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    IDEMPOTENCY_KEY.scope(key, fut).await
}

/// The current foreground send's client idempotency key, or `None` when no
/// [`with_idempotency_key`] scope is installed (agent runs, dreaming jobs,
/// tests) or when the installed key is itself `None` (a keyless foreground
/// send). Safe to call from any async context.
pub fn current_idempotency_key() -> Option<String> {
    IDEMPOTENCY_KEY.try_with(|k| k.clone()).ok().flatten()
}

/// Run `fut` with `disabled` installed as the current turn's tool-provenance
/// gate override. `ConversationHandler::send_prompt` reads it via
/// [`current_tool_gate_disabled`] when constructing `TurnProvenance`. See
/// `TOOL_GATE_DISABLED`.
pub async fn with_tool_gate_disabled<F, T>(disabled: bool, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    TOOL_GATE_DISABLED.scope(disabled, fut).await
}

/// The current turn's tool-provenance gate override, or `false` when no
/// [`with_tool_gate_disabled`] scope is installed. `false` means the gate
/// stays enforced — the fail-closed default for callers that don't route
/// through the daemon dispatch wrapper (tests, dreaming jobs, agent runs).
/// Safe to call from any async context.
pub fn current_tool_gate_disabled() -> bool {
    TOOL_GATE_DISABLED.try_with(|d| *d).unwrap_or(false)
}

/// Run `fut` with `model` installed as the current turn's model override.
/// Connectors read it via [`current_model_override`] in `stream_completion`
/// and use it in place of `self.model`. See `MODEL_OVERRIDE`.
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
    /// A learned observed-overflow window (issue #343) capped the budget
    /// DOWN below the value the tiers above resolved. Only set when the
    /// learned cap actually applied; identifies a budget driven by the
    /// adaptive safety net rather than config/connector.
    LearnedCap,
}

/// Run `fut` with `budget` installed as the resolved per-turn context
/// budget. The dispatch loop in [`crate::service::ConversationHandler`]
/// reads this via [`current_context_budget`] to drive token-pressure
/// compaction.
///
/// Why a task-local: see the doc on `CONTEXT_BUDGET`.
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

/// Run `fut` with `sink` installed as the per-turn context-usage sink.
/// The dispatch loop reports each turn's [`ContextUsage`] via
/// [`emit_context_usage`], which the transport layer forwards as an
/// `Event::ContextUsage` on the client stream (issue #341).
pub async fn with_context_usage_sink<F, T>(sink: ContextUsageSink, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CONTEXT_USAGE_SINK.scope(sink, fut).await
}

/// Report this turn's context fill to the installed sink, if any. A silent
/// no-op when no sink is installed (tests, dreaming jobs) — context usage is
/// best-effort telemetry, never load-bearing for the turn's outcome.
pub fn emit_context_usage(usage: ContextUsage) {
    let _ = CONTEXT_USAGE_SINK.try_with(|sink| sink(usage));
}

/// Run `fut` with `token` installed as the per-turn cancellation token.
///
/// The dispatch loop in [`crate::service::ConversationHandler`] and every
/// LLM adapter's streaming loop read this via
/// [`current_cancellation_token`] to cooperatively bail out of the
/// agentic loop when the token is tripped. The token is cheap to clone
/// (it's an `Arc<…>` internally), so the dispatch wrapper hands a clone
/// to the inner future and keeps a clone for its own monitoring.
///
/// Why a task-local: see the doc on `CANCELLATION_TOKEN`.
pub async fn with_cancellation_token<F, T>(token: CancellationToken, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CANCELLATION_TOKEN.scope(token, fut).await
}

/// Returns the per-turn cancellation token, or `None` if no token has
/// been installed for this task. Callers should treat `None` as "never
/// cancelled" — matching the documented contract that legacy call sites
/// (tests, background jobs) that don't route through the dispatch
/// wrapper retain the pre-#109 behaviour.
pub fn current_cancellation_token() -> Option<CancellationToken> {
    CANCELLATION_TOKEN.try_with(|t| t.clone()).ok()
}

/// Run `fut` with `tools` installed as the current task-local tool
/// allowlist. Used by the `SpawnStandaloneAgent` (#113) handler and the
/// `spawn_subagent` builtin tool (#112) so the spawned task body can
/// restrict the LLM's tool surface for the duration of that run.
///
/// See `TOOL_ALLOWLIST` for the read side and the semantic contract.
pub async fn with_tool_allowlist<F, T>(tools: Vec<String>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    TOOL_ALLOWLIST.scope(tools, fut).await
}

/// Returns the current task-local tool allowlist, or `None` when no
/// allowlist has been installed.
///
/// Resolution rules:
/// - `None` — no restriction; expose every available tool. Pre-#112
///   behaviour for callers that don't spawn through the helpers.
/// - `Some(vec)` — only tool names in `vec` may be exposed to the LLM
///   for this turn. An empty vec means "no tools at all", which is
///   distinct from `None` and the dispatch path must honour it.
pub fn current_tool_allowlist() -> Option<Vec<String>> {
    TOOL_ALLOWLIST.try_with(|t| t.clone()).ok()
}

/// Run `fut` with the classification reentrancy guard installed. See
/// `CLASSIFICATION_IN_PROGRESS`. Used by the classifier's LLM tier to wrap
/// its own LLM call so the result can't be recursively classified.
pub async fn with_classification_in_progress<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CLASSIFICATION_IN_PROGRESS.scope((), fut).await
}

/// Whether the current task is already inside a classification LLM call.
/// Decorators consult this and skip the learned-cache/LLM tiers when set,
/// so a classification call's own errors never trigger another round.
pub fn is_classification_in_progress() -> bool {
    CLASSIFICATION_IN_PROGRESS
        .try_with(|_| true)
        .unwrap_or(false)
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

/// The span field a connector reports the provider's own request id on.
///
/// Named once, here, because the connectors record onto a span this crate
/// builds and `tracing` drops a `record` for a field the span never declared.
/// A silent drop is exactly what a drifting string literal produces.
pub const PROVIDER_REQUEST_ID_FIELD: &str = "provider_request_id";

/// Put the provider's own request identifier on the open provider-call span.
///
/// A trace stops at a boundary we do not own: no LLM provider continues our
/// trace, and none ever will. The useful move there is capture rather than
/// propagation. This is the value quoted when a support ticket is opened with
/// a provider - Bedrock's request id, an `x-request-id` header - and it is the
/// closest thing to end-to-end that this boundary allows.
///
/// A connector calls this while its call is in flight, where the `llm.call`
/// span is the open one. Outside such a span the call does nothing, which is
/// what a model-catalog or embedding round trip made outside a turn should do.
///
/// An identifier is an id, not content, so it belongs at INFO under D10. The
/// value still arrives from a remote host, so it goes through
/// `adelie_telemetry::Safe::name`, which bounds it and strips the control,
/// bidi and line-breaking characters that would otherwise let a header end the
/// log line and start one that reads as genuine.
pub fn record_provider_request_id(id: &str) {
    let id = id.trim();
    if id.is_empty() {
        return;
    }
    tracing::Span::current().record(
        PROVIDER_REQUEST_ID_FIELD,
        tracing::field::display(adelie_telemetry::Safe::name(id)),
    );
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

/// What kind of model this is -- the axis that decides which purposes may bind
/// it. Distinct from the `reasoning` / `vision` / `tools` feature flags, which
/// describe a *generative* model's abilities; `kind` answers the prior question
/// of whether the model is a chat/completion model at all or an embedding model.
///
/// Three states, not two, on purpose (#647): a connector pointed at an arbitrary
/// OpenAI-compatible endpoint may genuinely not know what a model is, and an
/// `Unknown` must not lock an operator out of a working configuration. The
/// daemon enforces `Generative` / `Embedding` on the write path but always
/// allows `Unknown` (with a warning) -- the capability-degradation posture in
/// `AGENTS.md`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// A chat/completion model that generates text (and may call tools, take
    /// images, or expose reasoning traces -- see the feature flags).
    Generative,
    /// A vector-embedding model. Usable only for the embedding purpose.
    Embedding,
    /// The connector could not positively classify this model. Allowed with a
    /// warning rather than blocked, so an unrecognized custom id or a listing
    /// that failed transiently never blocks a config edit.
    #[default]
    Unknown,
}

/// Capability flags describing what an LLM model supports.
///
/// `kind` is the single source of truth for whether a model is generative or an
/// embedding model; there is deliberately no separate `embedding: bool` field
/// that could drift from it. Read it through [`Self::is_embedding`]. The
/// `reasoning` / `vision` / `tools` flags are a different axis: abilities of a
/// *generative* model, meaningful only when `kind` is [`ModelKind::Generative`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCapabilities {
    /// The connector can configure reasoning for this model: a
    /// [`ReasoningConfig`] carrying an effort or a budget reaches the provider
    /// and changes what the model does.
    ///
    /// Not "the model reasons". The two questions have different answers, and
    /// this is the one every consumer acts on - a client offers a reasoning
    /// control, and a connector decides whether to send the field. DeepSeek R1
    /// on Bedrock is the case that separates them: it reasons on every request
    /// and returns the trace, and Bedrock's request contract for it carries no
    /// reasoning field at all, so it reports `false`. A model that reasons but
    /// takes no configuration would otherwise show a control that does
    /// nothing, and a budget the connector drops on the way out.
    ///
    /// Populate it from what the request path will honour, and read the same
    /// answer there - one function, not two. A capability record that can
    /// disagree with the request builder is the defect this field exists to
    /// prevent.
    #[serde(default)]
    pub reasoning: bool,
    /// Model accepts image input.
    #[serde(default)]
    pub vision: bool,
    /// Model supports tool/function calling.
    #[serde(default)]
    pub tools: bool,
    /// What kind of model this is. Defaults to [`ModelKind::Unknown`] so an old
    /// serialized payload (or a connector that hasn't classified) deserializes
    /// to the degrade-with-a-warning state rather than a wrong guess.
    #[serde(default)]
    pub kind: ModelKind,
}

impl ModelCapabilities {
    /// Whether this is an embedding model. The one and only reading of the
    /// embedding/generative distinction -- derived from [`Self::kind`] so it can
    /// never disagree with it.
    pub fn is_embedding(&self) -> bool {
        matches!(self.kind, ModelKind::Embedding)
    }
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

/// Why a model listing came back incomplete.
///
/// Why an enum for a single case: the value is machine-readable and travels
/// to clients, which branch on it to decide how to render the notice. A
/// free-form string would make that branch a substring match, and adding the
/// discriminator later would be a wire change rather than a new variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelListingNoticeKind {
    /// Part of the catalog could not be enumerated, so the returned models
    /// are a subset of what the account can actually reach. The listing is
    /// still usable: this is a degradation, not a failure.
    PartialCatalog,
}

/// A non-fatal problem a connector hit while enumerating models.
///
/// Why this exists: a connector that degrades silently is indistinguishable
/// from one that has nothing to offer. Bedrock is the motivating case: when
/// `ListInferenceProfiles` is denied, what survives is the on-demand
/// foundation models, which in a current AWS account is mostly the embedding
/// families. Carrying the reason as data lets a client say "inference
/// profiles unavailable, showing on-demand models only" instead of leaving
/// the operator to read daemon logs (#648).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelListingNotice {
    /// Machine-readable classification of the problem.
    pub kind: ModelListingNoticeKind,
    /// One-line, user-facing summary of what is missing.
    pub summary: String,
    /// User-facing cause and remedy. Must be actionable on its own, since a
    /// client may render it without the summary.
    pub detail: String,
    /// The provider permission that most likely needs granting, when the
    /// failure was an authorization denial. `None` for other causes, so a
    /// client never blames IAM for a timeout or a malformed request.
    pub required_permission: Option<String>,
}

impl ModelListingNotice {
    /// Build a [`ModelListingNoticeKind::PartialCatalog`] notice.
    pub fn partial_catalog(summary: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            kind: ModelListingNoticeKind::PartialCatalog,
            summary: summary.into(),
            detail: detail.into(),
            required_permission: None,
        }
    }

    /// Name the provider permission whose absence explains this notice.
    pub fn with_required_permission(mut self, permission: impl Into<String>) -> Self {
        self.required_permission = Some(permission.into());
        self
    }
}

/// The result of enumerating a connector's models: the models themselves,
/// plus any non-fatal problems hit while collecting them.
///
/// Why a report instead of a bare `Vec<ModelInfo>`: connectors assemble the
/// list from several provider calls, and losing one of them changes what the
/// list *means* without changing its type. The extra channel keeps partial
/// failures visible to callers while preserving the degradation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelListingReport {
    /// Models the caller can select, in the connector's stable order.
    pub models: Vec<ModelInfo>,
    /// Non-fatal problems encountered while listing. Empty on a clean run:
    /// connectors must not manufacture a notice for the happy path.
    pub notices: Vec<ModelListingNotice>,
}

impl ModelListingReport {
    /// A report for a listing that completed with nothing to report.
    pub fn complete(models: Vec<ModelInfo>) -> Self {
        Self {
            models,
            notices: Vec::new(),
        }
    }

    /// Whether the listing is known to be incomplete.
    pub fn is_degraded(&self) -> bool {
        !self.notices.is_empty()
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

/// Server-side hosted tool search: the connector sends namespaces to the
/// provider with deferred loading and lets the provider's own search entry
/// pull individual tools in, instead of putting every tool in the request.
///
/// This is a separate trait, not a pair of methods on [`LlmClient`], because
/// the capability answer and the implementation have to travel together. A
/// client offers hosted tool search exactly when
/// [`LlmClient::hosted_tool_search`] hands back one of these, so the answer
/// cannot be a claim about code sitting beside it.
///
/// What that buys, precisely: a connector's only candidate object is `self`,
/// and `Some(self)` does not compile without `impl HostedToolSearch for Self`.
/// So a connector cannot report the capability and inherit a flattening body
/// by omission - the combination that produced a turn carrying the whole tool
/// fleet inline *and* no discovery tool, because the service layer strips
/// `builtin_tool_search` whenever hosted search is active (#1033).
///
/// What it does not buy: the return type is `Option<&dyn HostedToolSearch>`,
/// so any reference reachable from `&self` type-checks, including some other
/// object that only flattens. That latitude is load-bearing - a decorator
/// needs it - and it makes pointing elsewhere a deliberate act someone has to
/// write and a reviewer can see, rather than a default nobody chose. What the
/// object then puts on the wire is checked by the cross-connector sweep in
/// the daemon's `registry.rs`, which the type system cannot reach.
///
/// Decorators implement this trait for themselves rather than handing back
/// their inner client's object; see [`LlmClient::hosted_tool_search`].
#[async_trait::async_trait]
pub trait HostedToolSearch: Send + Sync {
    /// Stream a completion with namespaced tool definitions.
    ///
    /// The implementation is expected to serialize `namespaces` in the
    /// provider's deferred-loading shape and append the provider's
    /// tool-search entry. An implementation that simply flattens the
    /// namespaces into an ordinary tool list is a deliberate, reviewable
    /// choice - see [`flatten_namespaces`] - not something a connector can
    /// fall into by not writing the method.
    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError>;
}

/// Collapse core tools and namespace tools into one ordinary tool list.
///
/// This is what a turn sends to a connector without hosted tool search: every
/// tool inline, no discovery entry. It used to be the default body of a trait
/// method, where a connector inherited it silently; it is now a named helper
/// a caller reaches for on purpose.
pub fn flatten_namespaces(
    core_tools: &[ToolDefinition],
    namespaces: &[ToolNamespace],
) -> Vec<ToolDefinition> {
    let mut all: Vec<ToolDefinition> = core_tools.to_vec();
    for ns in namespaces {
        all.extend(ns.tools.iter().cloned());
    }
    all
}

/// Send one namespaced turn to `client`.
///
/// Routes through the client's [`HostedToolSearch`] implementation when it has
/// one, and flattens every namespace into a plain [`LlmClient::stream_completion`]
/// call when it does not. Every caller of a namespaced turn goes through here,
/// so the choice between the two paths is made in exactly one place from
/// exactly one fact.
///
/// A decorator calls this on its *inner* client, which is what keeps a
/// decorator chain intact for a namespaced turn: each link decorates, then
/// hands down to the next.
pub async fn dispatch_namespaced(
    client: &(impl LlmClient + ?Sized),
    messages: Vec<Message>,
    core_tools: &[ToolDefinition],
    namespaces: &[ToolNamespace],
    reasoning: ReasoningConfig,
    on_chunk: ChunkCallback,
) -> Result<LlmResponse, CoreError> {
    match client.hosted_tool_search() {
        Some(hosted) => {
            hosted
                .stream_completion_with_namespaces(
                    messages, core_tools, namespaces, reasoning, on_chunk,
                )
                .await
        }
        None => {
            let all = flatten_namespaces(core_tools, namespaces);
            client
                .stream_completion(messages, &all, reasoning, on_chunk)
                .await
        }
    }
}

/// Outbound port for LLM completion requests.
///
/// Uses [`async_trait::async_trait`] so the trait is dyn-compatible
/// — required because the daemon registry stores clients as
/// `Arc<dyn LlmClient>` (#44). The per-call heap allocation that
/// async-trait introduces is negligible next to an LLM round-trip.
#[async_trait::async_trait]
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

    /// Approximate the prompt-token cost of a string. Used by the core
    /// service for pre-flight budget checks; does not need to be exact —
    /// a consistent over-estimate is preferable to an under-estimate.
    ///
    /// Why a default of `chars/4` (rounded up): a well-known, dependency-free
    /// approximation for English BPE tokenisation. Connectors that have a
    /// more accurate option (their own tokeniser, a known per-model factor)
    /// can override this without touching callers.
    fn estimate_tokens(&self, text: &str) -> u64 {
        (text.chars().count() as u64).div_ceil(4)
    }

    /// Stream a completion from the LLM given a message history.
    /// Calls `on_chunk` for each text token/chunk received.
    /// Optionally accepts tool definitions to enable tool calling.
    /// `reasoning` carries optional extended-thinking / reasoning-effort
    /// hints; connectors may ignore it (Ollama) or translate it into a
    /// per-API request field (Anthropic `thinking`, OpenAI
    /// `reasoning_effort`, Bedrock `additionalModelRequestFields`).
    /// Returns an `LlmResponse` which may include tool calls.
    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError>;

    /// This client's server-side hosted tool search, if it has one.
    ///
    /// `Some` is both the capability answer and the dispatch object, so a
    /// connector cannot answer one way and behave another: `Some(self)`
    /// requires `Self` to implement [`HostedToolSearch`], and implementing
    /// that trait means writing the namespaced request. A client with no
    /// hosted search answers `None` by leaving this method alone, and its
    /// namespaced turns flatten (see [`dispatch_namespaced`]).
    ///
    /// **A decorator returns `Some(self)`, never its inner client's object.**
    /// Handing back the inner object drops the decorator from the call path
    /// for exactly the turns that carry the most tools - losing retry,
    /// classification, reasoning substitution or per-turn routing.
    /// Nothing in the type system stops that, which is why each decorator has
    /// a named test asserting its own effect on a namespaced turn. The shape
    /// to copy:
    ///
    /// ```ignore
    /// fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
    ///     self.inner.hosted_tool_search().is_some().then_some(self as &dyn HostedToolSearch)
    /// }
    /// ```
    ///
    /// with a [`HostedToolSearch`] implementation that decorates and then
    /// calls [`dispatch_namespaced`] on `self.inner`.
    ///
    /// A *transparent forwarder* is the exception and hands back its inner
    /// object directly, because it has no per-call work to lose: the `Arc<T>`
    /// blanket impl below adds no hop of its own, so forwarding there could
    /// only be neutral.
    fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
        None
    }

    /// Enumerate the models this connector can serve.
    ///
    /// Connectors should return every model the caller could reasonably
    /// select (chat and embedding). The default implementation returns an
    /// empty list so test mocks and decorators that delegate can opt out;
    /// production connectors override this.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        Ok(Vec::new())
    }

    /// Force a fresh fetch of `list_models()`, bypassing any per-connector
    /// cache. Connectors without a cache can delegate to `list_models`.
    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.list_models().await
    }

    /// Enumerate models *and* report any non-fatal problems hit on the way.
    ///
    /// Why a second method rather than changing [`list_models`]: only
    /// connectors that assemble the catalog from several provider calls can
    /// degrade, and every other caller and connector keeps the simpler
    /// contract. The default wraps [`list_models`] in a clean report, so a
    /// connector that cannot degrade never has to think about notices.
    ///
    /// [`list_models`]: LlmClient::list_models
    async fn list_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        Ok(ModelListingReport::complete(self.list_models().await?))
    }

    /// Cache-bypassing counterpart of [`list_models_detailed`].
    ///
    /// A refresh always reports an outcome, even when the list is byte-for-byte
    /// what it was before: a reload that returns nothing is indistinguishable
    /// from one that failed.
    ///
    /// [`list_models_detailed`]: LlmClient::list_models_detailed
    async fn refresh_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        Ok(ModelListingReport::complete(self.refresh_models().await?))
    }

    /// Optional one-shot warmup hook called once after registry construction.
    /// Default no-op; Ollama uses it to populate the GGUF context-length
    /// cache so [`LlmClient::max_context_tokens`] returns a real value on first use.
    /// Errors are intentionally swallowed by the registry — warmup is
    /// best-effort and a failure just falls back to the universal default.
    async fn warmup(&self) {}
}

// Blanket impl so generic wrappers — `RetryingLlmClient<L>`,
// `RoutingLlmClient` — accept `Arc<dyn LlmClient>`
// as their inner type. Without this the daemon registry's
// `Arc<dyn LlmClient>` couldn't be wrapped by the same chain that
// existed when the inner type was a concrete enum (#44).
#[async_trait::async_trait]
impl<T: LlmClient + ?Sized> LlmClient for Arc<T> {
    fn get_default_model(&self) -> Option<&str> {
        (**self).get_default_model()
    }

    fn get_default_base_url(&self) -> Option<&str> {
        (**self).get_default_base_url()
    }

    fn max_context_tokens(&self) -> Option<u64> {
        (**self).max_context_tokens()
    }

    fn estimate_tokens(&self, text: &str) -> u64 {
        (**self).estimate_tokens(text)
    }

    fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
        (**self).hosted_tool_search()
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        (**self)
            .stream_completion(messages, tools, reasoning, on_chunk)
            .await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        (**self).list_models().await
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        (**self).refresh_models().await
    }

    async fn list_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        (**self).list_models_detailed().await
    }

    async fn refresh_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        (**self).refresh_models_detailed().await
    }

    async fn warmup(&self) {
        (**self).warmup().await
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

// --- Tool-call accumulator -------------------------------------------------

/// Stream-time accumulator for assembling [`ToolCall`]s from a sequence
/// of provider-specific events.
///
/// The shape is the same across every connector: each tool call has a
/// stable per-stream index, an `id`, a `name`, and an `arguments` JSON
/// string that may arrive in pieces (Anthropic / Bedrock / OpenAI all
/// stream `arguments` as concatenated partial JSON deltas, with OpenAI
/// also emitting a final `done` event carrying the full string).
///
/// Generic over the index type — Anthropic uses `usize` (zero-based
/// content-block index), Bedrock uses `i32` (signed by the SDK),
/// OpenAI uses `usize` (`output_index`). All three need `Ord` so
/// [`Self::into_tool_calls`] can return calls in stable, ascending
/// order regardless of arrival order.
#[derive(Debug, Clone, Default)]
pub struct ToolCallAccumulator<K> {
    entries: std::collections::BTreeMap<K, ToolCallEntry>,
}

#[derive(Debug, Clone, Default)]
struct ToolCallEntry {
    id: String,
    name: String,
    arguments: String,
}

impl<K: Ord + Copy> ToolCallAccumulator<K> {
    pub fn new() -> Self {
        Self {
            entries: std::collections::BTreeMap::new(),
        }
    }

    /// Register a new tool call at `key`. If a call already exists at
    /// `key`, its `id` and `name` are overwritten and any
    /// already-accumulated arguments are preserved — providers that
    /// emit `start` for the same index twice (none seen in practice)
    /// will get last-write-wins on the metadata without losing
    /// streamed argument bytes.
    pub fn start(&mut self, key: K, id: impl Into<String>, name: impl Into<String>) {
        let entry = self.entries.entry(key).or_default();
        entry.id = id.into();
        entry.name = name.into();
    }

    /// Append a partial-JSON chunk to the arguments string at `key`.
    /// Silently dropped when no `start` was emitted for `key` first
    /// (defensive against malformed event sequences). Use this for
    /// providers that stream `arguments` deltas (Anthropic
    /// `input_json_delta`, Bedrock `ContentBlockDelta::ToolUse`,
    /// OpenAI `response.function_call_arguments.delta`).
    pub fn append(&mut self, key: K, partial_json: &str) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.arguments.push_str(partial_json);
        }
    }

    /// Replace the arguments string at `key` with the full final
    /// payload. Used by OpenAI's `response.function_call_arguments.done`
    /// event, which carries the canonical full JSON; the deltas are a
    /// preview the SDK doesn't promise are byte-equivalent. No-op for
    /// connectors that don't emit a finalize event — they just keep
    /// the accumulated deltas as-is.
    pub fn finalize(&mut self, key: K, arguments: impl Into<String>) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.arguments = arguments.into();
        }
    }

    /// Drain into [`ToolCall`]s in ascending key order. Entries with
    /// empty `id` *and* empty `name` are filtered out — they're zombies
    /// from a stream that emitted `append` without a matching `start`
    /// (Bedrock's older shape did this; the new `append` guards against
    /// it but the filter is kept as belt-and-braces).
    pub fn into_tool_calls(self) -> Vec<ToolCall> {
        self.entries
            .into_values()
            .filter(|e| !e.id.is_empty() || !e.name.is_empty())
            .map(|e| ToolCall::new(e.id, e.name, e.arguments))
            .collect()
    }

    /// Number of registered tool calls. Test-only.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no tool calls have been registered. Test-only.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Decorator that wraps any `LlmClient` and retries on transient rate-limit errors
/// with exponential backoff (1s, 2s, 4s, ...) provided by the `backon` crate.
pub struct RetryingLlmClient<L> {
    inner: L,
    max_retries: u32,
}

impl<L> RetryingLlmClient<L> {
    pub fn new(inner: L, max_retries: u32) -> Self {
        Self { inner, max_retries }
    }

    fn backoff(&self) -> ExponentialBuilder {
        ExponentialBuilder::default()
            .with_min_delay(Duration::from_secs(1))
            .with_factor(2.0)
            .with_max_times(self.max_retries as usize)
    }
}

/// Build a fresh per-attempt callback that forwards into the shared real callback.
/// The real callback is consumed only once across all retries. `forwarded` is
/// flipped the moment any chunk reaches the real callback, so the retry
/// predicate can refuse to replay a stream the consumer already saw (DA-10).
fn proxy_callback(
    shared: &Arc<Mutex<Option<ChunkCallback>>>,
    forwarded: &Arc<std::sync::atomic::AtomicBool>,
) -> ChunkCallback {
    let cb_ref = Arc::clone(shared);
    let forwarded = Arc::clone(forwarded);
    Box::new(move |chunk: String| -> bool {
        let mut guard = cb_ref.lock().unwrap();
        if let Some(ref mut cb) = *guard {
            forwarded.store(true, std::sync::atomic::Ordering::Relaxed);
            cb(chunk)
        } else {
            false
        }
    })
}

/// Retry predicate shared by both streaming entry points: a transient error
/// is retryable only while nothing has been emitted downstream. Once chunks
/// were delivered, a replay would duplicate the already-rendered prefix (the
/// chunk consumer is append-only), so the error must surface instead (DA-10).
fn retry_unless_stream_started(
    forwarded: &Arc<std::sync::atomic::AtomicBool>,
) -> impl Fn(&CoreError) -> bool {
    let forwarded = Arc::clone(forwarded);
    move |e: &CoreError| {
        if !is_retryable_error(e) {
            return false;
        }
        if forwarded.load(std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(
                "retryable LLM error after chunks were already streamed — \
                 surfacing the error instead of replaying the stream: {e}"
            );
            return false;
        }
        true
    }
}

fn log_retry(err: &CoreError, dur: Duration) {
    tracing::warn!("retryable LLM error, retrying in {:?}: {err}", dur);
}

#[async_trait::async_trait]
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

    fn estimate_tokens(&self, text: &str) -> u64 {
        self.inner.estimate_tokens(text)
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.list_models().await
    }

    async fn refresh_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
        self.inner.refresh_models().await
    }

    async fn list_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        self.inner.list_models_detailed().await
    }

    async fn refresh_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
        self.inner.refresh_models_detailed().await
    }

    async fn stream_completion(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let shared_cb: Arc<Mutex<Option<ChunkCallback>>> = Arc::new(Mutex::new(Some(on_chunk)));
        let forwarded = Arc::new(std::sync::atomic::AtomicBool::new(false));

        (|| async {
            self.inner
                .stream_completion(
                    messages.clone(),
                    tools,
                    reasoning,
                    proxy_callback(&shared_cb, &forwarded),
                )
                .await
        })
        .retry(self.backoff())
        .when(retry_unless_stream_started(&forwarded))
        .notify(log_retry)
        .await
    }

    /// Hands back `self`, never the inner client's object, so this
    /// decorator stays in the call path for a namespaced turn. See
    /// [`LlmClient::hosted_tool_search`].
    fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
        self.inner
            .hosted_tool_search()
            .is_some()
            .then_some(self as &dyn HostedToolSearch)
    }
}

#[async_trait::async_trait]
impl<L: LlmClient> HostedToolSearch for RetryingLlmClient<L> {
    async fn stream_completion_with_namespaces(
        &self,
        messages: Vec<Message>,
        core_tools: &[ToolDefinition],
        namespaces: &[ToolNamespace],
        reasoning: ReasoningConfig,
        on_chunk: ChunkCallback,
    ) -> Result<LlmResponse, CoreError> {
        let shared_cb: Arc<Mutex<Option<ChunkCallback>>> = Arc::new(Mutex::new(Some(on_chunk)));
        let forwarded = Arc::new(std::sync::atomic::AtomicBool::new(false));

        (|| async {
            dispatch_namespaced(
                &self.inner,
                messages.clone(),
                core_tools,
                namespaces,
                reasoning,
                proxy_callback(&shared_cb, &forwarded),
            )
            .await
        })
        .retry(self.backoff())
        .when(retry_unless_stream_started(&forwarded))
        .notify(log_retry)
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Role;

    struct MockLlm {
        chunks: Vec<String>,
    }

    #[async_trait::async_trait]
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

    #[async_trait::async_trait]
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
    async fn mid_stream_retry_does_not_duplicate_emitted_text() {
        // DA-10: a retryable error AFTER chunks were already delivered must
        // not replay the stream — the consumer has already rendered (and the
        // turn accumulator already holds) the emitted prefix, so a replay
        // appends it twice. Once anything was emitted, the only safe move is
        // to surface the error instead of retrying.
        tokio::time::pause();

        struct FailMidStreamLlm {
            attempts: Mutex<u32>,
        }

        #[async_trait::async_trait]
        impl LlmClient for FailMidStreamLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                mut on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                let mut attempts = self.attempts.lock().unwrap();
                *attempts += 1;
                if *attempts == 1 {
                    // First attempt: emit a prefix, then die retryably
                    // (Anthropic mid-SSE `overloaded` shape).
                    on_chunk("Hello, ".into());
                    return Err(CoreError::RateLimited {
                        retry_after: None,
                        detail: "overloaded mid-stream".into(),
                    });
                }
                on_chunk("Hello, ".into());
                on_chunk("world".into());
                Ok(LlmResponse::text("Hello, world"))
            }
        }

        let client = RetryingLlmClient::new(
            FailMidStreamLlm {
                attempts: Mutex::new(0),
            },
            3,
        );

        let received = Arc::new(Mutex::new(String::new()));
        let received_clone = Arc::clone(&received);
        let result = client
            .stream_completion(
                vec![Message::new(Role::User, "hi")],
                &[],
                ReasoningConfig::default(),
                Box::new(move |chunk| {
                    received_clone.lock().unwrap().push_str(&chunk);
                    true
                }),
            )
            .await;

        let received = received.lock().unwrap();
        assert_eq!(
            *received, "Hello, ",
            "already-emitted text must never be replayed to the consumer"
        );
        assert!(
            result.is_err(),
            "a mid-stream failure after emitted chunks must surface the error, \
             not silently return a spliced response"
        );
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
        #[async_trait::async_trait]
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
            kind: ModelKind::Generative,
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
        assert!(!info.capabilities.is_embedding());
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
                kind: ModelKind::Generative,
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
            kind: ModelKind::Embedding,
            ..Default::default()
        };
        assert!(caps.is_embedding());
        assert!(!caps.reasoning);
        assert!(!caps.tools);
        assert!(!caps.vision);
    }

    #[test]
    fn model_kind_is_the_single_source_of_truth_for_embedding() {
        // `is_embedding()` is derived from `kind` and cannot drift from it.
        assert!(
            ModelCapabilities {
                kind: ModelKind::Embedding,
                ..Default::default()
            }
            .is_embedding()
        );
        assert!(
            !ModelCapabilities {
                kind: ModelKind::Generative,
                ..Default::default()
            }
            .is_embedding()
        );
        // An unclassified model is not treated as an embedding model.
        assert!(!ModelCapabilities::default().is_embedding());
    }

    #[test]
    fn model_capabilities_default_kind_is_unknown() {
        // The degrade-with-a-warning state, so an old payload or an
        // unclassified connector never presents a wrong guess.
        assert_eq!(ModelCapabilities::default().kind, ModelKind::Unknown);
    }

    #[test]
    fn model_kind_missing_from_json_deserializes_as_unknown() {
        // Additive wire change: a payload written before `kind` existed must
        // still deserialize, landing on `Unknown` rather than failing.
        let caps: ModelCapabilities =
            serde_json::from_str(r#"{"reasoning":true,"vision":false,"tools":true}"#).unwrap();
        assert_eq!(caps.kind, ModelKind::Unknown);
        assert!(caps.reasoning);
    }

    #[tokio::test]
    async fn default_list_models_is_empty() {
        struct NoopLlm;
        #[async_trait::async_trait]
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

    /// A connector that only knows how to list models still answers the
    /// detailed API, reporting "nothing went wrong" rather than nothing at
    /// all — otherwise every non-Bedrock connector would look degraded (#648).
    #[tokio::test]
    async fn detailed_listing_defaults_to_the_plain_model_list() {
        struct PlainLlm;
        #[async_trait::async_trait]
        impl LlmClient for PlainLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                Ok(LlmResponse::text(""))
            }

            async fn list_models(&self) -> Result<Vec<ModelInfo>, CoreError> {
                Ok(vec![ModelInfo::new("only-model")])
            }
        }

        let llm = PlainLlm;
        let listed = llm.list_models_detailed().await.expect("detailed listing");
        assert_eq!(listed.models, vec![ModelInfo::new("only-model")]);
        assert!(listed.notices.is_empty());
        assert!(!listed.is_degraded());

        let refreshed = llm
            .refresh_models_detailed()
            .await
            .expect("detailed refresh");
        assert_eq!(refreshed.models, listed.models);
        assert!(refreshed.notices.is_empty());
    }

    /// The wrappers every client is built through (`Arc`, retry) must not
    /// swallow notices on the way out — that would re-hide the degradation
    /// the connector went to the trouble of reporting (#648).
    #[tokio::test]
    async fn wrappers_forward_listing_notices() {
        struct DegradedLlm;
        #[async_trait::async_trait]
        impl LlmClient for DegradedLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                Ok(LlmResponse::text(""))
            }

            async fn list_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
                Ok(ModelListingReport {
                    models: vec![ModelInfo::new("m")],
                    notices: vec![ModelListingNotice::partial_catalog("partial", "detail")],
                })
            }

            async fn refresh_models_detailed(&self) -> Result<ModelListingReport, CoreError> {
                self.list_models_detailed().await
            }
        }

        let arced: Arc<dyn LlmClient> = Arc::new(DegradedLlm);
        assert_eq!(
            arced
                .list_models_detailed()
                .await
                .expect("arc forwards")
                .notices
                .len(),
            1
        );

        let retrying = RetryingLlmClient::new(arced, 0);
        assert_eq!(
            retrying
                .refresh_models_detailed()
                .await
                .expect("retry wrapper forwards")
                .notices
                .len(),
            1
        );
    }

    #[test]
    fn default_estimate_tokens_uses_chars_div_4() {
        struct NoopLlm;
        #[async_trait::async_trait]
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
        // Empty input → 0 tokens.
        assert_eq!(llm.estimate_tokens(""), 0);
        // 4 ASCII chars rounds to 1 token.
        assert_eq!(llm.estimate_tokens("abcd"), 1);
        // 5 ASCII chars rounds up to 2 tokens (over-estimate).
        assert_eq!(llm.estimate_tokens("abcde"), 2);
        // 16 ASCII chars → 4 tokens exactly.
        assert_eq!(llm.estimate_tokens("0123456789abcdef"), 4);
        // Multi-byte chars count as one each — emoji-heavy or CJK input
        // produces a much smaller token count than `bytes/4` would.
        let four_emoji = "\u{1F600}\u{1F601}\u{1F602}\u{1F603}";
        assert_eq!(four_emoji.chars().count(), 4);
        assert_eq!(llm.estimate_tokens(four_emoji), 1);
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

    // --- PERSONALITY tests (issue #226) ---

    #[tokio::test]
    async fn current_personality_is_default_outside_scope() {
        // Callers that never install a scope (tests, dreaming jobs, any path
        // not routed through the daemon dispatch wrapper) observe the default
        // disposition rather than an empty one.
        assert_eq!(
            current_personality(),
            crate::prompts::Personality::default()
        );
    }

    #[tokio::test]
    async fn current_personality_observes_installed_scope() {
        let custom = crate::prompts::Personality {
            humor: crate::prompts::PersonalityLevel::Never,
            ..crate::prompts::Personality::default()
        };
        let observed = with_personality(custom, async { current_personality() }).await;
        assert_eq!(observed, custom);
        // After the scope exits the task-local is unset again (back to default).
        assert_eq!(
            current_personality(),
            crate::prompts::Personality::default()
        );
    }

    #[tokio::test]
    async fn nested_personality_shadows_outer() {
        let outer = crate::prompts::Personality::default();
        let inner = crate::prompts::Personality {
            sarcasm: crate::prompts::PersonalityLevel::Always,
            ..crate::prompts::Personality::default()
        };
        let observed = with_personality(outer, async {
            with_personality(inner, async { current_personality() }).await
        })
        .await;
        assert_eq!(observed, inner);
    }

    // --- TOOL_GATE_DISABLED tests (issue #1007) ---

    #[tokio::test]
    async fn current_tool_gate_disabled_is_false_outside_scope() {
        // Callers that never install the scope (tests, dreaming jobs, any
        // path not routed through the daemon dispatch wrapper) must see the
        // gate enforced — the fail-closed default.
        assert!(!current_tool_gate_disabled());
    }

    #[tokio::test]
    async fn current_tool_gate_disabled_observes_installed_scope() {
        let observed = with_tool_gate_disabled(true, async { current_tool_gate_disabled() }).await;
        assert!(observed);
        // After the scope exits the task-local is unset again (back to false).
        assert!(!current_tool_gate_disabled());
    }

    #[tokio::test]
    async fn nested_tool_gate_disabled_shadows_outer() {
        let observed = with_tool_gate_disabled(true, async {
            with_tool_gate_disabled(false, async { current_tool_gate_disabled() }).await
        })
        .await;
        assert!(!observed);
    }

    // --- MODEL_OVERRIDE tests (issue #34) ---

    #[tokio::test]
    async fn current_model_override_is_none_outside_scope() {
        assert_eq!(current_model_override(), None);
    }

    #[tokio::test]
    async fn current_model_override_observes_scope() {
        let observed =
            with_model_override("gpt-5-mini".to_string(), async { current_model_override() }).await;
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

    // --- CONTEXT_USAGE_SINK tests (issue #341) --------------------------

    #[tokio::test]
    async fn emit_context_usage_is_noop_when_no_sink_installed() {
        // No `with_context_usage_sink` wrapper — the foreground send path,
        // a dreaming job, or a unit test. Emission must be a silent no-op,
        // never a panic: context usage is best-effort telemetry.
        emit_context_usage(ContextUsage {
            used_tokens: 100,
            budget_tokens: 1_000,
            compaction_active: false,
        });
    }

    #[tokio::test]
    async fn emit_context_usage_invokes_installed_sink() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_for_sink = std::sync::Arc::clone(&captured);
        let sink: ContextUsageSink = std::sync::Arc::new(move |u: ContextUsage| {
            captured_for_sink.lock().unwrap().push(u);
        });

        with_context_usage_sink(sink, async {
            emit_context_usage(ContextUsage {
                used_tokens: 12_000,
                budget_tokens: 32_000,
                compaction_active: false,
            });
            emit_context_usage(ContextUsage {
                used_tokens: 30_000,
                budget_tokens: 32_000,
                compaction_active: true,
            });
        })
        .await;

        let got = captured.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                ContextUsage {
                    used_tokens: 12_000,
                    budget_tokens: 32_000,
                    compaction_active: false,
                },
                ContextUsage {
                    used_tokens: 30_000,
                    budget_tokens: 32_000,
                    compaction_active: true,
                },
            ]
        );

        // After the scope exits the slot is unset again — emission no-ops.
        emit_context_usage(ContextUsage {
            used_tokens: 1,
            budget_tokens: 2,
            compaction_active: false,
        });
        assert_eq!(captured.lock().unwrap().len(), 2);
    }

    // --- TOOL_ALLOWLIST tests (issues #112 / #113) ----------------------

    #[tokio::test]
    async fn current_tool_allowlist_is_none_outside_scope() {
        // Callers that don't install a scope (legacy paths, tests, the
        // foreground send path) observe `None`, meaning "no
        // restriction" — exposes every tool, matching pre-#112
        // behaviour.
        assert_eq!(current_tool_allowlist(), None);
    }

    #[tokio::test]
    async fn current_tool_allowlist_observes_installed_scope() {
        let observed = with_tool_allowlist(vec!["search".into(), "fetch".into()], async {
            current_tool_allowlist()
        })
        .await;
        assert_eq!(
            observed,
            Some(vec!["search".to_string(), "fetch".to_string()])
        );
        // After the scope exits the task-local is unset again.
        assert_eq!(current_tool_allowlist(), None);
    }

    #[tokio::test]
    async fn empty_tool_allowlist_is_distinct_from_none() {
        // An explicit empty allowlist means "no tools at all". The
        // dispatch path must NOT collapse it to "expose everything";
        // this test pins the distinction down so a future refactor
        // can't silently merge the two.
        let observed = with_tool_allowlist(vec![], async { current_tool_allowlist() }).await;
        assert_eq!(observed, Some(vec![]));
        assert_ne!(observed, None);
    }

    #[tokio::test]
    async fn nested_tool_allowlist_shadows_outer() {
        let observed = with_tool_allowlist(vec!["outer".into()], async {
            with_tool_allowlist(vec!["inner".into()], async { current_tool_allowlist() }).await
        })
        .await;
        assert_eq!(observed, Some(vec!["inner".to_string()]));
    }

    // --- ToolCallAccumulator (#45) ----------------------------------------

    #[test]
    fn tool_call_accumulator_assembles_streamed_deltas() {
        let mut acc = ToolCallAccumulator::<usize>::new();
        acc.start(0, "call_a", "search");
        acc.append(0, "{\"q\":");
        acc.append(0, "\"hello\"}");
        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_a");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, "{\"q\":\"hello\"}");
    }

    #[test]
    fn tool_call_accumulator_orders_by_key_ascending() {
        let mut acc = ToolCallAccumulator::<i32>::new();
        // Insert out of order — output must still be sorted by key.
        acc.start(2, "call_z", "zebra");
        acc.append(2, "{}");
        acc.start(0, "call_a", "alpha");
        acc.append(0, "{}");
        acc.start(1, "call_m", "middle");
        acc.append(1, "{}");
        let calls = acc.into_tool_calls();
        let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "middle", "zebra"]);
    }

    #[test]
    fn tool_call_accumulator_finalize_replaces_partial_arguments() {
        let mut acc = ToolCallAccumulator::<usize>::new();
        acc.start(0, "call_a", "search");
        acc.append(0, "{\"part");
        // OpenAI emits a `done` event with the full canonical JSON;
        // finalize must replace whatever deltas have accumulated.
        acc.finalize(0, "{\"q\":\"final\"}");
        let calls = acc.into_tool_calls();
        assert_eq!(calls[0].arguments, "{\"q\":\"final\"}");
    }

    #[test]
    fn tool_call_accumulator_append_without_start_is_dropped() {
        let mut acc = ToolCallAccumulator::<usize>::new();
        acc.append(0, "lost data");
        assert!(acc.into_tool_calls().is_empty());
    }

    #[test]
    fn tool_call_accumulator_filters_zombie_entries() {
        // An entry whose id and name are both empty (from a hypothetical
        // mis-sequenced provider) is filtered out — see method doc for the
        // belt-and-braces rationale.
        let mut acc = ToolCallAccumulator::<usize>::new();
        acc.entries.insert(
            0,
            ToolCallEntry {
                id: String::new(),
                name: String::new(),
                arguments: "garbage".into(),
            },
        );
        acc.start(1, "real", "real");
        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "real");
    }

    /// #570 Phase 1b: outside a `with_idempotency_key` scope the current key is
    /// `None`, so callers that don't route through the foreground dispatch
    /// wrapper (agent runs, dreaming jobs, tests) persist no key.
    #[tokio::test]
    async fn current_idempotency_key_is_none_outside_scope() {
        assert_eq!(
            current_idempotency_key(),
            None,
            "no key is installed outside the dispatch scope"
        );
    }

    /// Inside the scope the installed key is observable; a `None` scope reads
    /// back as `None` (a keyless send wrapped by the dispatcher), and the value
    /// does not leak past the scope boundary.
    #[tokio::test]
    async fn current_idempotency_key_observes_installed_scope() {
        let observed =
            with_idempotency_key(Some("k1".to_string()), async { current_idempotency_key() }).await;
        assert_eq!(
            observed,
            Some("k1".to_string()),
            "installed key is observed"
        );

        let keyless = with_idempotency_key(None, async { current_idempotency_key() }).await;
        assert_eq!(keyless, None, "a None scope reads back as None");

        assert_eq!(
            current_idempotency_key(),
            None,
            "the key must not leak past the scope boundary"
        );
    }
}

/// Test doubles and tests for the hosted-tool-search seam (#1033).
///
/// The headline property of that seam - a connector cannot report hosted
/// tool search without implementing the dispatch, because `Some(self)` needs
/// the impl - is a compile-time property and has no runtime test. What is
/// tested here is the runtime half: that [`dispatch_namespaced`] picks the
/// right path, and that every decorator with per-call work of its own stays
/// in the call path for a namespaced turn.
#[cfg(test)]
pub(crate) mod hosted_search_test_support {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records which entry point a turn arrived through, and how many times.
    #[derive(Default)]
    pub(crate) struct Probe {
        pub plain: AtomicUsize,
        pub namespaced: AtomicUsize,
        /// Tool names seen by the plain path, joined with `,`.
        pub flattened: Mutex<String>,
    }

    impl Probe {
        pub fn plain_calls(&self) -> usize {
            self.plain.load(Ordering::SeqCst)
        }

        pub fn namespaced_calls(&self) -> usize {
            self.namespaced.load(Ordering::SeqCst)
        }
    }

    /// Leaf client whose hosted-search support is chosen at construction.
    ///
    /// `hosted = false` leaves [`LlmClient::hosted_tool_search`] answering
    /// `None`, which is how the great majority of clients (and every test
    /// double that does not care) behave.
    pub(crate) struct ProbeLlm {
        pub probe: Arc<Probe>,
        pub hosted: bool,
        /// Errors this many times before succeeding. Lets a retry decorator
        /// be observed doing its job on the namespaced path.
        pub fail_times: AtomicUsize,
    }

    impl ProbeLlm {
        pub fn new(hosted: bool) -> Self {
            Self {
                probe: Arc::new(Probe::default()),
                hosted,
                fail_times: AtomicUsize::new(0),
            }
        }

        /// A probe that returns a retryable error `times` times before it
        /// succeeds, on whichever path the turn arrives through.
        pub fn failing(hosted: bool, times: usize) -> Self {
            let client = Self::new(hosted);
            client.fail_times.store(times, Ordering::SeqCst);
            client
        }

        fn maybe_fail(&self) -> Option<CoreError> {
            let remaining = self.fail_times.load(Ordering::SeqCst);
            if remaining == 0 {
                return None;
            }
            self.fail_times.store(remaining - 1, Ordering::SeqCst);
            Some(CoreError::RateLimited {
                retry_after: None,
                detail: "probe: transient".into(),
            })
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for ProbeLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.probe.plain.fetch_add(1, Ordering::SeqCst);
            *self.probe.flattened.lock().unwrap() = tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(",");
            if let Some(e) = self.maybe_fail() {
                return Err(e);
            }
            Ok(LlmResponse::text("plain"))
        }

        fn hosted_tool_search(&self) -> Option<&dyn HostedToolSearch> {
            self.hosted.then_some(self as &dyn HostedToolSearch)
        }
    }

    #[async_trait::async_trait]
    impl HostedToolSearch for ProbeLlm {
        async fn stream_completion_with_namespaces(
            &self,
            _messages: Vec<Message>,
            _core_tools: &[ToolDefinition],
            _namespaces: &[ToolNamespace],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.probe.namespaced.fetch_add(1, Ordering::SeqCst);
            if let Some(e) = self.maybe_fail() {
                return Err(e);
            }
            Ok(LlmResponse::text("namespaced"))
        }
    }

    pub(crate) fn tool(name: &str) -> ToolDefinition {
        ToolDefinition::new(name, "probe tool", serde_json::json!({"type": "object"}))
    }

    pub(crate) fn namespace(name: &str, tools: Vec<ToolDefinition>) -> ToolNamespace {
        ToolNamespace::new(name, "probe namespace", tools)
    }

    pub(crate) fn noop_chunk() -> ChunkCallback {
        Box::new(|_| true)
    }
}

#[cfg(test)]
mod hosted_search_dispatch_tests {
    use super::hosted_search_test_support::*;
    use super::*;

    #[tokio::test]
    async fn dispatch_namespaced_uses_hosted_search_when_the_client_has_it() {
        let client = ProbeLlm::new(true);
        let probe = Arc::clone(&client.probe);

        dispatch_namespaced(
            &client,
            vec![],
            &[tool("core")],
            &[namespace("ns", vec![tool("deferred")])],
            ReasoningConfig::default(),
            noop_chunk(),
        )
        .await
        .expect("probe turn");

        assert_eq!(probe.namespaced_calls(), 1, "hosted dispatch was used");
        assert_eq!(probe.plain_calls(), 0, "the flattening path was not used");
    }

    #[tokio::test]
    async fn dispatch_namespaced_flattens_when_the_client_has_no_hosted_search() {
        let client = ProbeLlm::new(false);
        let probe = Arc::clone(&client.probe);

        dispatch_namespaced(
            &client,
            vec![],
            &[tool("core")],
            &[namespace("ns", vec![tool("deferred")])],
            ReasoningConfig::default(),
            noop_chunk(),
        )
        .await
        .expect("probe turn");

        assert_eq!(probe.namespaced_calls(), 0, "no hosted dispatch exists");
        assert_eq!(probe.plain_calls(), 1, "the turn went out flattened");
        assert_eq!(
            *probe.flattened.lock().unwrap(),
            "core,deferred",
            "core tools first, then every namespace tool inline"
        );
    }

    #[test]
    fn flatten_namespaces_appends_namespace_tools_after_core_tools() {
        let flat = flatten_namespaces(
            &[tool("core_a"), tool("core_b")],
            &[
                namespace("one", vec![tool("n1")]),
                namespace("two", vec![tool("n2"), tool("n3")]),
            ],
        );
        let names: Vec<&str> = flat.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["core_a", "core_b", "n1", "n2", "n3"]);
    }

    /// Decorator-in-the-path criterion for [`RetryingLlmClient`].
    ///
    /// The inner client fails once with a retryable error, then succeeds. If
    /// the retry decorator is in the namespaced path the turn succeeds after
    /// two hosted-search dispatches. If the decorator handed back the inner
    /// client's hosted-search object, the first error reaches the caller and
    /// a turn carrying the whole tool fleet loses its retry.
    #[tokio::test]
    async fn retrying_decorator_stays_in_the_namespaced_path() {
        let inner = ProbeLlm::failing(true, 1);
        let probe = Arc::clone(&inner.probe);
        let client = RetryingLlmClient::new(inner, 3);

        let result = dispatch_namespaced(
            &client,
            vec![],
            &[],
            &[namespace("ns", vec![tool("deferred")])],
            ReasoningConfig::default(),
            noop_chunk(),
        )
        .await;

        assert!(
            result.is_ok(),
            "the retry decorator must retry a namespaced turn: {result:?}"
        );
        assert_eq!(
            probe.namespaced_calls(),
            2,
            "one failed hosted dispatch and one retry"
        );
        assert_eq!(probe.plain_calls(), 0, "the turn never flattened");
    }

    #[tokio::test]
    async fn an_arc_forwards_the_hosted_search_object() {
        let client: Arc<dyn LlmClient> = Arc::new(ProbeLlm::new(true));
        assert!(
            client.hosted_tool_search().is_some(),
            "Arc is a transparent forwarder, not a decorator: it must not \
             hide the inner client's hosted search"
        );
    }
}
