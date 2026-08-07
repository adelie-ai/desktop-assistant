//! Context-window management for the conversation handler.
//!
//! This module groups three related concerns that conspire to keep the
//! prompt under the model's input-token budget:
//!
//! - **Assembly** (`assemble_turn_within_budget`, `assemble_turn`):
//!   Builds the per-turn `Vec<Message>` from conversation history,
//!   summaries, tool definitions, and the active-task anchor — applying
//!   pre-flight token-budget checks and shrinking the window when the
//!   estimated cost exceeds the threshold. A shrink narrows the window past
//!   what the caller asked for, and past what turn-entry compaction covered,
//!   so `compact_preflight_shrink` folds the difference in before the call —
//!   otherwise those messages are in neither the prompt nor the summary.
//! - **Recovery** (`recover_from_overflow`): When the provider rejects a
//!   turn with [`crate::CoreError::ContextOverflow`], runs a structured
//!   recovery ladder (truncate the largest tool result → compact the oldest
//!   tool results → summarise-and-shrink) before the dispatch loop retries.
//!   The rungs that replace message content write to the round's
//!   `ContextProjection`, so the stored transcript keeps every message and
//!   every byte.
//! - **Projection** (`ContextProjection`): what the round reads where that
//!   differs from what is stored. Seeded at turn entry from the eviction
//!   decisions earlier turns recorded, so a distilled result costs a pointer
//!   rather than its payload on every later turn too.
//! - **Summarisation** (`generate_context_summary`, `compact_into_summary`):
//!   Asks the LLM for a bullet-point summary of dropped messages and merges it
//!   with any existing rolling summary, so windowed-out history is not lost.
//!   The compaction marker moves only when a summary was actually produced.
//!
//! Constants exposed here are tuning knobs read by the dispatch loop in
//! `service.rs` to mirror this module's defaults (e.g., the floor on
//! window size, the compaction-token-pressure threshold).

mod projection;

pub(crate) use projection::ContextProjection;

use crate::domain::{
    Conversation, Message, MessageSummary, Role, ToolDefinition, ToolLocality, ToolNamespace,
    TransportKind,
};
use crate::planning;
use crate::ports::llm::{ContextBudget, LlmClient, ReasoningConfig};

/// Default maximum number of conversation messages sent to the LLM per turn.
/// When the conversation exceeds this limit, only the most recent messages
/// are included, with the cut point snapped forward to a genuine `Role::User`
/// message to avoid splitting tool-call/result pairs.
pub(crate) const MAX_CONTEXT_MESSAGES: usize = 40;

/// Lower bound applied when the window is shrunk in response to token pressure.
/// Keeps enough room for at least the current user prompt plus a tool round.
pub(crate) const MIN_CONTEXT_MESSAGES: usize = 8;

/// Minimum number of newly-dropped messages before re-compacting the summary.
pub(crate) const COMPACTION_INTERVAL: usize = 20;

/// Fraction of the model's prompt-token budget at which proactive compaction
/// triggers. Checked against `LlmResponse.usage.input_tokens` after each
/// successful LLM call.
pub(crate) const COMPACTION_TOKEN_RATIO: f64 = 0.85;

/// How far past the provider's reported gap the recovery ladder must free
/// before it skips the window-shrinking step.
///
/// The ladder measures what it freed with `LlmClient::estimate_tokens`, and the
/// gap comes from the provider's own tokenizer. The two disagree, and the
/// direction that costs a retry is the estimator reading high. Doubling is a
/// coarse guard, not a calibration.
pub(crate) const ESTIMATE_SAFETY_MARGIN: u64 = 2;

/// Maximum number of `CoreError::ContextOverflow` recoveries allowed within
/// a single `send_prompt` call. Each recovery applies one step of the
/// context-recovery ladder; if successive calls still overflow we surface
/// the error rather than loop.
pub(crate) const MAX_OVERFLOW_RETRIES: u32 = 3;

/// Floor below which a tool result isn't worth truncating in response to a
/// `ContextOverflow`. Measured in estimated tokens (via
/// `LlmClient::estimate_tokens`) so non-ASCII payloads are weighed by the
/// cost the model actually pays. Below this size the resulting truncation
/// notice may be larger than the original payload, so the savings are
/// negligible and step 1 of the recovery ladder hands off to step 2.
///
/// Why 1024: roughly equivalent to 4 KB of ASCII at the chars/4 default
/// estimate, but the choice is intentionally coarse — the goal is just to
/// avoid the "notice larger than payload" pathology, not to be precise.
pub(crate) const MIN_TRUNCATION_TOKENS: u64 = 1024;

/// Maximum byte length a single tool result may occupy before it is
/// truncated at ingestion (issue #174). A misbehaving tool can return a
/// multi-megabyte payload (observed: 124 MB across 8 messages); stored
/// verbatim it wedges the conversation against the model's context window
/// on *every* subsequent turn and stalls the `messages` INSERT. Capping at
/// ingestion bounds the blast radius of any single tool call.
///
/// Why a byte cap rather than a token cap: it's deterministic, O(1) to
/// check, requires no estimator pass over a huge string, and directly
/// bounds what is written to the database. 256 KiB is ~64K tokens at the
/// chars/4 default — far above any legitimate tool result, so honest tools
/// are never touched.
pub(crate) const DEFAULT_MAX_TOOL_RESULT_BYTES: usize = 256 * 1024;

/// Replacement tail appended when a tool result is truncated at ingestion.
/// Addressed to the model so it learns to re-run the tool with a narrower
/// request instead of assuming the output was complete.
pub(crate) fn tool_result_truncation_notice(original_bytes: usize) -> String {
    format!(
        "\n\n<tool output truncated: {original_bytes} bytes exceeded the per-result \
         storage cap; only the beginning is shown. Re-run the tool with a narrower \
         request — e.g. a smaller byte/line range, a filtered listing, or only the \
         fields you need — to see the rest.>"
    )
}

/// Cap a tool result to `max_bytes` before it is stored as a message.
///
/// Returns `None` when `content` already fits (the common case — no
/// allocation, caller stores the original). Returns `Some(truncated)` when
/// it is over the cap: the longest UTF-8 prefix that, together with
/// [`tool_result_truncation_notice`], stays within `max_bytes`. Truncation
/// always lands on a `char` boundary so the result is valid UTF-8.
pub(crate) fn cap_tool_result(content: &str, max_bytes: usize) -> Option<String> {
    if content.len() <= max_bytes {
        return None;
    }

    let notice = tool_result_truncation_notice(content.len());
    // Reserve room for the notice. If the cap is so small the notice alone
    // would not fit, keep no prefix — the notice still tells the model what
    // happened (a pathological case; real caps dwarf the notice).
    let body_budget = max_bytes.saturating_sub(notice.len());

    // Largest char boundary at or below the body budget. `is_char_boundary`
    // is O(1) and at most three steps back from any byte index, so this is
    // cheap even for a multi-megabyte payload.
    let mut cut = body_budget.min(content.len());
    while cut > 0 && !content.is_char_boundary(cut) {
        cut -= 1;
    }

    let mut truncated = String::with_capacity(cut + notice.len());
    truncated.push_str(&content[..cut]);
    truncated.push_str(&notice);
    Some(truncated)
}

/// Fraction of the prompt-token budget the system instruction (static
/// prompt + tool availability listing) is allowed to consume before the
/// listing is demoted to a namespace-only summary.
///
/// Why 0.20: the system block is always re-included in every turn, so
/// any space it claims is permanently displaced from conversation
/// history. 20% is a soft cap that comfortably accommodates ~50–100
/// tools at the chars/4 estimate; beyond that, demotion preserves
/// recovery headroom.
const SYSTEM_BLOCK_BUDGET_RATIO: f64 = 0.20;

/// Number of consecutive tool rounds within a single `send_prompt` call after
/// which the active-task anchor must be re-injected even if it is still in
/// the windowed message list. Why: long agentic loops drift away from the
/// goal; surfacing it again every few rounds keeps the model on-task.
const ACTIVE_TASK_ROUND_THRESHOLD: u32 = 5;

/// Maximum number of pre-flight shrink iterations attempted by
/// [`assemble_turn_within_budget`] when the assembled prompt exceeds the budget.
/// Why bounded: each iteration halves the message window, so 5 iterations
/// already drop the count by 32x — enough to reach [`MIN_CONTEXT_MESSAGES`]
/// from any plausible starting point. The bound also guarantees termination
/// regardless of estimator behaviour.
const MAX_PREFLIGHT_SHRINK_ITERATIONS: u32 = 5;

/// Build the replacement content used when a tool result is truncated in
/// response to a `ContextOverflow` error. The text is addressed to the
/// model so it learns to chunk subsequent requests more narrowly.
pub(crate) fn overflow_truncation_notice(
    original_bytes: usize,
    prompt_tokens: Option<u64>,
    max_tokens: Option<u64>,
) -> String {
    let measured = match (prompt_tokens, max_tokens) {
        (Some(p), Some(m)) => format!(" (prompt was {p} tokens vs {m} max)"),
        _ => String::new(),
    };
    format!(
        "<tool output omitted: {original_bytes} bytes exceeded the model's \
         context window{measured}. Re-run the tool with a narrower request — \
         for example read the file in smaller byte/line ranges, list a single \
         directory level with filters, or query for only the fields you need.>"
    )
}

/// The conversation material assembly draws on this turn: the live message
/// log, any message summaries eligible for collapse, and the rolling context
/// summary of already-dropped history.
#[derive(Clone, Copy, Default)]
pub(crate) struct ConversationView<'a> {
    pub messages: &'a [Message],
    pub summaries: &'a [MessageSummary],
    pub context_summary: &'a str,
}

/// The tools exposed this turn and where they run — drives the
/// tool-availability section of the system prompt.
#[derive(Clone, Copy, Default)]
pub(crate) struct ToolContext<'a> {
    pub tool_defs: &'a [ToolDefinition],
    pub deferred_namespaces: &'a [ToolNamespace],
    pub locality: Option<&'a ToolLocalityContext>,
}

/// Per-turn anchors re-surfaced as `[..]` system messages so the model stays
/// on-task across windowing/compaction, plus the round counter that gates
/// whether they re-surface.
#[derive(Clone, Copy, Default)]
pub(crate) struct TurnAnchors<'a> {
    pub active_task: Option<&'a str>,
    pub plan: Option<&'a str>,
    pub scratchpad_index: Option<&'a str>,
    /// Rendered `[Pinned]` block (#597): the full content of the notes the
    /// model pinned, and the live content of the knowledge entries they attach
    /// (#1104). Ungated — unlike `scratchpad_index`, it renders every turn
    /// whenever anything is pinned.
    pub pinned: Option<&'a str>,
    /// Candidate memory for the `[Recall]` block (#1100, #1101): what the
    /// user's prompt may be about, found by a semantic lookup the model did not
    /// have to ask for.
    ///
    /// Candidates rather than a rendered block, because two of the rules that
    /// decide what may appear in it are settled here and not by the producer:
    /// the block renders on the **first round of a turn only**, and it must not
    /// repeat a scratchpad key the `[Scratchpad]` index has just listed - and
    /// whether that index speaks depends on the window this assembly chose.
    /// The lookup still runs once per turn; only the rendering is here.
    pub recall: Option<crate::recall::RecallSurface<'a>>,
    /// Counts behind the always-on `[Working state]` nudge (#598), carried as
    /// counts rather than a rendered line so the block can drop whichever half
    /// a fuller block covers on this particular turn.
    pub working_state: crate::planning::WorkingState,
    pub tool_rounds_since_anchor: u32,
}

/// The per-turn "ambient" context: the standing personality, the ambient
/// `[Now]` line, and any one-turn system-prompt refinement. These previously
/// arrived three different ways (two task-locals read deep inside assembly,
/// one threaded parameter). Grouped here and read once at the wrapper boundary
/// via [`AmbientContext::current`] so [`assemble_turn`] is a pure
/// function of its inputs.
#[derive(Clone, Default)]
pub(crate) struct AmbientContext {
    /// Active assistant personality; rendered as the disposition section.
    pub personality: crate::prompts::Personality,
    /// Pre-rendered ambient "now" line, or empty for no `[Now]` block.
    pub now_line: String,
    /// One-turn system-prompt refinement, or empty for none.
    pub system_refinement: String,
    /// Self-reported client context (#549): the user + their device. `None` when
    /// the client sent none — the assembled prompt then carries no client-context
    /// block (fail-closed; the daemon never substitutes its own host values).
    pub client_context: Option<crate::prompts::ClientContext>,
}

impl AmbientContext {
    /// Read the per-turn ambient context from the task-locals the daemon
    /// dispatch wrapper installs. The single place these task-locals are read;
    /// unset (tests, background jobs) yields the defaults — the standard
    /// personality, no `[Now]` block, no refinement, no client context — so
    /// those callers behave exactly as before.
    pub fn current() -> Self {
        Self {
            personality: crate::ports::llm::current_personality(),
            now_line: crate::ports::llm::current_now_context(),
            system_refinement: crate::ports::llm::current_system_refinement(),
            client_context: crate::ports::transport::current_client_context(),
        }
    }
}

/// One turn's assembled prompt, and where in the conversation it starts.
pub(crate) struct AssembledTurn {
    /// The prompt, system block first.
    pub messages: Vec<Message>,
    /// Index into the conversation's message list where the assembled window
    /// begins. Overflow recovery works from here: a message before it is not in
    /// this prompt, so replacing it frees the provider nothing.
    ///
    /// Not the same as the caller's window size. The pre-flight budget check
    /// can shrink the window further than the caller asked for, and the whole
    /// point of reporting this is that the caller cannot know that.
    pub window_from: usize,
    /// The knowledge entries the `[Recall]` block put in front of the model,
    /// in the order it rendered them. Empty on any round but a turn's first,
    /// where the block does not render at all.
    ///
    /// Reported rather than re-derived, because only the renderer knows what it
    /// showed: it applies the floor, the width, and every "already in view"
    /// drop. The use log (#698) records these as offered.
    pub recalled_entry_ids: Vec<String>,
}

/// Build the message list for a single turn, optionally enforcing a
/// pre-flight token budget by shrinking the window before any LLM call.
///
/// Why a separate wrapper around [`assemble_turn`]: assembly is
/// pure — given the same inputs it returns the same `Vec<Message>` — but
/// budget enforcement is iterative (try, measure, halve, retry). Splitting
/// keeps the inner builder simple and lets the test suite call it directly
/// without exercising the loop.
///
/// When `budget` is `Some(b)`, the assembled token estimate (system
/// instruction plus every assembled message body, summed via `estimate`)
/// must come in below `COMPACTION_TOKEN_RATIO * b.max_input_tokens`. If
/// not, `max_messages` is halved (clamped to `MIN_CONTEXT_MESSAGES`) and
/// assembly is repeated, up to [`MAX_PREFLIGHT_SHRINK_ITERATIONS`] times.
/// Once `max_messages` reaches the floor, further iterations would have no
/// effect and the loop returns the current assembly.
///
/// When `budget` is `None`, the wrapper performs a single assembly pass —
/// preserving pre-#65 behaviour for tests and background jobs that don't
/// route through the daemon's dispatch wrapper.
pub(crate) fn assemble_turn_within_budget(
    conversation: &ConversationView,
    tools: &ToolContext,
    anchors: &TurnAnchors,
    projection: &ContextProjection,
    max_messages: usize,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> AssembledTurn {
    // Read the per-turn ambient context (personality, `[Now]` line, refinement)
    // once at this boundary. Passing it in keeps `assemble_turn` a
    // pure function of its inputs, so the shrink loop's repeat passes stay
    // deterministic.
    let ambient = AmbientContext::current();

    // One assembly pass at a given window size. The only thing that varies
    // across the shrink loop is `max_messages`, so everything else is captured
    // once here and the two call sites collapse to `assemble(current_max)`.
    let assemble = |max: usize| {
        assemble_turn(
            conversation,
            tools,
            anchors,
            &ambient,
            projection,
            max,
            budget,
            estimate,
        )
    };
    let finish = |pass: TurnMessages, max: usize| AssembledTurn {
        messages: pass.messages,
        window_from: window_start(conversation.messages, max),
        recalled_entry_ids: pass.recalled_entry_ids,
    };

    let mut current_max = max_messages;
    let mut assembled = assemble(current_max);

    let Some(active_budget) = budget else {
        return finish(assembled, current_max);
    };

    // Pre-flight token estimate: sum the cost of every assembled message's
    // body, plus the active tool schemas. The threshold mirrors
    // `COMPACTION_TOKEN_RATIO` used by the post-call token-pressure path so
    // the two checks agree on what counts as "near the limit".
    //
    // Tool schemas are sent to the model out-of-band (the `tools` array, not
    // a message body), so summing message bodies alone undercounts: namespace
    // activation can inject tens of KB of JSON Schema the budget never sees
    // (issue #305 item 7). Account for it explicitly. The cost is constant
    // across shrink iterations (shrinking only drops *messages*), so it is
    // computed once here.
    let max_input_tokens = active_budget.max_input_tokens;
    let threshold = (max_input_tokens as f64 * COMPACTION_TOKEN_RATIO) as u64;
    let tool_schema_tokens =
        tool_schema_estimate(tools.tool_defs, tools.deferred_namespaces, estimate);

    for _ in 0..MAX_PREFLIGHT_SHRINK_ITERATIONS {
        let message_tokens: u64 = assembled
            .messages
            .iter()
            .map(|m| estimate(&m.content))
            .sum();
        let assembled_tokens = message_tokens + tool_schema_tokens;
        if assembled_tokens <= threshold {
            return finish(assembled, current_max);
        }
        // Already at the floor — further halving has no effect, so stop.
        if current_max <= MIN_CONTEXT_MESSAGES {
            return finish(assembled, current_max);
        }
        let new_max = (current_max / 2).max(MIN_CONTEXT_MESSAGES);
        if new_max == current_max {
            return finish(assembled, current_max);
        }
        tracing::debug!(
            assembled_tokens,
            budget = max_input_tokens,
            prev_max_messages = current_max,
            new_max_messages = new_max,
            "assembly over budget, shrinking"
        );
        current_max = new_max;
        assembled = assemble(current_max);
    }

    finish(assembled, current_max)
}

/// Estimate the prompt-token cost of the tool schemas sent alongside the
/// messages on each turn.
///
/// The model is billed for the `tools` array — every active tool's name,
/// description, and JSON Schema parameters — which never appears in a message
/// body, so the preflight's message-body sum would otherwise miss it entirely
/// (issue #305 item 7). A single namespace activation can add tens of KB.
///
/// Deferred namespaces (sent with `defer_loading` so the model fetches them on
/// demand) are *not* counted: their schemas are not in the active context
/// window until activated, and once activated they arrive as `tool_defs`. We
/// count only their lightweight namespace name/description stubs, which are
/// what the provider keeps resident.
///
/// Estimation reuses the same `estimate` closure as message bodies so the
/// units agree. We serialize each tool's parameters once and weigh name +
/// description + schema together.
fn tool_schema_estimate(
    tool_defs: &[ToolDefinition],
    deferred_namespaces: &[ToolNamespace],
    estimate: &dyn Fn(&str) -> u64,
) -> u64 {
    let tool_cost = |t: &ToolDefinition| -> u64 {
        // Name and description are short; the schema dominates. Serialize the
        // parameters compactly — the absolute count only needs to track the
        // real payload's order of magnitude for the budget check.
        let schema = t.parameters.to_string();
        estimate(&t.name) + estimate(&t.description) + estimate(&schema)
    };

    let active: u64 = tool_defs.iter().map(tool_cost).sum();

    // Deferred namespaces contribute only their stub (name + description); the
    // per-tool schemas are off-context until the model activates them.
    let deferred: u64 = deferred_namespaces
        .iter()
        .map(|ns| estimate(&ns.name) + estimate(&ns.description))
        .sum();

    active + deferred
}

/// Per-turn tool execution-locality context (issue #243, refined in #248).
///
/// Bundles the inputs the tool-note builder needs to tag each tool with where
/// it runs: the per-machine **system-id co-location** result (#248) with the
/// connection's [`TransportKind`] as a fallback, the daemon's self-identity
/// `host` label, and the names of the tools registered as client-local for this
/// turn. Cheap to build — the dispatch loop assembles it once per turn from the
/// transport + co-location task-locals, the handler's host label, and the
/// client-tool port's definitions.
#[derive(Debug, Clone)]
pub(crate) struct ToolLocalityContext {
    /// Authoritative co-location result from the per-machine system-id
    /// handshake (#248): `Some(true)` when the client's reported id equals the
    /// daemon's own id (same machine — even over WebSocket), `Some(false)` when
    /// the ids differ, and `None` when the client reported no id (an older
    /// client). When `None`, co-location falls back to [`Self::transport`],
    /// preserving the Phase-1 (#243) behaviour exactly.
    pub co_located: Option<bool>,
    /// How the turn's connection reaches the daemon. The **fallback**
    /// co-location signal (#243) used only when [`Self::co_located`] is `None`:
    /// local transports collapse the server/client distinction.
    pub transport: TransportKind,
    /// The daemon's self-identity label used for `Server { host }` (the
    /// hostname).
    pub host: String,
    /// Whether the daemon runs on a person's own workstation, rather than in a
    /// container or on a server (#534). Decides whether the topology section
    /// may describe daemon-side tools as acting on the user's own machine.
    pub daemon_on_workstation: bool,
    /// Hostname the client reported in the handshake (#248), for the remote
    /// tool note and the topology section. Empty when the client reported none,
    /// which each renderer phrases for itself.
    pub client_label: String,
    /// Names of the tools that run server-side (MCP / built-in) on the daemon
    /// host. A name in BOTH this set and [`Self::client_tool_names`] is a
    /// capability duplicated across machines (the routing case).
    pub server_tool_names: Vec<String>,
    /// Names of the tools registered as client-local for this turn (run on the
    /// registering client's machine).
    pub client_tool_names: Vec<String>,
}

impl ToolLocalityContext {
    /// Whether the connection is co-located with the daemon (same machine).
    ///
    /// Prefers the authoritative system-id match (#248) when the client
    /// reported an id ([`Self::co_located`] is `Some`); otherwise falls back to
    /// the transport heuristic (#243) for older clients that send no id.
    pub(crate) fn is_co_located(&self) -> bool {
        self.co_located
            .unwrap_or_else(|| self.transport.is_co_located())
    }

    fn is_server(&self, name: &str) -> bool {
        self.server_tool_names.iter().any(|n| n == name)
    }

    fn is_client(&self, name: &str) -> bool {
        self.client_tool_names.iter().any(|n| n == name)
    }

    /// Client-registered tools this turn cannot call, because a daemon-side
    /// tool already holds the name (#1083).
    ///
    /// The turn loop's tool-set merge drops a client definition whose name a
    /// server-side tool already holds, so the model is never offered it and
    /// dispatch routes the name to the server executor. Names are returned in
    /// registration order, so a report of them is stable.
    pub(crate) fn shadowed_client_tools(&self) -> Vec<&str> {
        self.client_tool_names
            .iter()
            .map(String::as_str)
            .filter(|name| self.is_server(name))
            .collect()
    }

    /// The topology this turn's connection describes (#534), for the
    /// "where things run" prompt section.
    fn topology(&self) -> crate::prompts::Topology {
        crate::prompts::Topology {
            daemon_host: self.host.clone(),
            daemon_on_workstation: self.daemon_on_workstation,
            client_label: self.client_label.clone(),
            same_machine: self.is_co_located(),
            client_has_tools: !self.client_tool_names.is_empty(),
        }
    }
}

/// One entry in the resolved per-turn locality plan (issue #243).
///
/// `resolve_tool_localities` turns the flat tool set plus the locality context
/// into a plan the note renders. Each entry records the tool's `name`, its
/// [`ToolLocality`], and whether it is the **primary** for its capability —
/// the tool the service nudges the LLM toward when the same capability exists
/// on both the server and a (remote) client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolLocalityEntry {
    pub name: String,
    pub locality: ToolLocality,
    /// True when this entry is the primary for a capability that is duplicated
    /// across localities. For a non-duplicated capability every entry is
    /// trivially primary. Only meaningful in the remote case (the local case
    /// collapses duplicates to the single server tool).
    pub primary: bool,
}

/// Resolve the flat tool set into a locality plan (issue #243).
///
/// A tool registered on both machines resolves to the **server-side** entry
/// alone, on every transport. Only that one can be called: the turn loop's
/// tool-set merge drops a client definition whose name a server-side tool
/// already holds, so the model never receives the client one and dispatch
/// routes the name to the server executor. Naming the client twin as an
/// alternative would advertise a tool that does not exist for this turn.
/// A co-located connection has nothing to choose between anyway - the two are
/// one machine - and a remote one has lost the client capability, which the
/// turn loop reports once when it resolves the collision.
///
/// Everything else keeps its own locality: a client-only tool is tagged to the
/// client, and anything not registered client-side is server-side.
///
/// Tool order is preserved: each entry keeps the position of its name in
/// `tool_names`, one entry per name.
pub(crate) fn resolve_tool_localities(
    tool_names: &[&str],
    ctx: &ToolLocalityContext,
) -> Vec<ToolLocalityEntry> {
    let mut entries: Vec<ToolLocalityEntry> = Vec::with_capacity(tool_names.len());

    for &name in tool_names {
        let is_server = ctx.is_server(name);
        let is_client = ctx.is_client(name);
        let duplicated = is_server && is_client;

        if duplicated {
            // The same name on both machines. Only the server-side tool is
            // reachable either way: the turn loop's tool-set merge drops a
            // client definition whose name a server-side tool already holds, so
            // the model is never offered the client one and dispatch routes the
            // name to the server executor. Naming a "your device (alternative)"
            // entry here would advertise a tool that cannot be called - true of
            // a remote connection as much as a co-located one, where the two
            // are the same machine anyway. So emit the server entry alone.
            // The loss of the client capability is reported once per turn,
            // where the merge happens, rather than on every note build.
            entries.push(ToolLocalityEntry {
                name: name.to_string(),
                locality: ToolLocality::server(&ctx.host),
                primary: true,
            });
        } else if is_client {
            // Client-only capability: a plain local tool when co-located, a
            // labelled remote tool otherwise.
            entries.push(ToolLocalityEntry {
                name: name.to_string(),
                locality: ToolLocality::client(name, &ctx.client_label),
                primary: true,
            });
        } else {
            // Server-side (MCP / built-in), the default for anything not
            // registered as client-local.
            entries.push(ToolLocalityEntry {
                name: name.to_string(),
                locality: ToolLocality::server(&ctx.host),
                primary: true,
            });
        }
    }
    entries
}

/// Render a locality plan into the human-readable tool list used in the note.
///
/// - **Co-located**: a plain comma-joined name list — everything is on this
///   machine, so no per-tool label is added.
/// - **Remote**: each tool is labelled with its locality, e.g.
///   `terminal — server 'daemon-host'` / `terminal — your device 'laptop'`,
///   and a duplicated capability's non-primary alternative is noted.
fn render_locality_list(entries: &[ToolLocalityEntry], co_located: bool) -> String {
    if co_located {
        return entries
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
    }
    entries
        .iter()
        .map(|e| match &e.locality {
            ToolLocality::Server { host } => format!("{} — server '{host}'", e.name),
            ToolLocality::Client { label, .. } => {
                let alt = if e.primary { "" } else { " (alternative)" };
                // A client that reported no hostname leaves the label empty;
                // "your device ''" would read as a name the model could quote.
                if label.trim().is_empty() {
                    format!("{} — your device{alt}", e.name)
                } else {
                    format!("{} — your device '{label}'{alt}", e.name)
                }
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Concise test entry point for assembly: takes the grouped inputs and fills
/// in the window size every test holds constant (`MAX_CONTEXT_MESSAGES`).
/// Ambient context (personality, `[Now]`, refinement) is read from task-locals
/// by the wrapper, defaulting to empty when unset. Adding a per-turn field to
/// any input struct leaves this helper and its callers untouched — the whole
/// point of the grouping.
#[cfg(test)]
fn assemble_for_test(
    conversation: &ConversationView,
    tools: &ToolContext,
    anchors: &TurnAnchors,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> Vec<Message> {
    assemble_turn_within_budget(
        conversation,
        tools,
        anchors,
        &ContextProjection::default(),
        MAX_CONTEXT_MESSAGES,
        budget,
        estimate,
    )
    .messages
}

/// Build the full tool-availability note enumerating every tool name and
/// the deferred-namespace index. Returned by default; demoted to a
/// namespace-only summary by [`build_demoted_tool_note`] when the
/// assembled system block exceeds [`SYSTEM_BLOCK_BUDGET_RATIO`].
///
/// When `locality` is `Some`, tools are tagged with where they run (issue
/// #243): a co-located connection gets a plain list because everything is on
/// this machine, while a connection to another machine gets per-tool locality
/// labels. How to choose between the machines is stated once, in the
/// `Where things run` prompt section, rather than repeated here. When `None`
/// (callers that don't thread a transport context) the listing is the plain
/// name list, byte-identical to the pre-#243 behaviour.
fn build_full_tool_note(
    tool_defs: &[ToolDefinition],
    deferred_namespaces: &[ToolNamespace],
    locality: Option<&ToolLocalityContext>,
) -> String {
    if tool_defs.is_empty() && deferred_namespaces.is_empty() {
        return "No tools are available in this turn.".to_string();
    }

    let has_tool_search = tool_defs.iter().any(|t| t.name == "builtin_tool_search");
    let mut note = String::new();

    if !tool_defs.is_empty() {
        // Resolve locality and render the tool list. The co-located common
        // case (and the no-context fallback) produce the plain comma-joined
        // list; a remote connection produces per-tool locality labels.
        let names = match locality {
            Some(ctx) => {
                let tool_names: Vec<&str> = tool_defs.iter().map(|t| t.name.as_str()).collect();
                let entries = resolve_tool_localities(&tool_names, ctx);
                render_locality_list(&entries, ctx.is_co_located())
            }
            None => tool_defs
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        };
        if has_tool_search {
            note = format!(
                "Available tools in this turn: {names}. \
                 Additional tools may be available — use builtin_tool_search to discover \
                 tools for tasks not covered by the tools listed above."
            );
        } else {
            note = format!("Available tools in this turn: {names}.");
        }
    }

    // When deferred namespaces exist (hosted or not), append a compact
    // name-only index so the model knows what tools are reachable.
    if !deferred_namespaces.is_empty() {
        if !note.is_empty() {
            note.push('\n');
        }
        for ns in deferred_namespaces {
            let tool_names: Vec<&str> = ns.tools.iter().map(|t| t.name.as_str()).collect();
            note.push_str(&format!("{}=[{}]\n", ns.name, tool_names.join(", ")));
        }
        note.push_str(
            "These tools are available via search or deferred loading. \
             Use builtin_tool_search if you cannot call one directly.",
        );
    }

    note
}

/// Build a namespace-only summary used when the full tool listing would
/// push the system block past the budget. Why: the static prompt is
/// always re-included on every turn, so an oversized listing permanently
/// displaces conversation history. The model still has
/// `builtin_tool_search` as a real tool definition (in `tool_defs`); the
/// listing demotion only collapses what the system prompt enumerates.
///
/// The collapsed listing is where the per-tool machine labels lived, so when
/// the connection reaches a second machine this note names the tools that run
/// on the user's own. They are few, and they are the part the model cannot
/// recover for itself: what it collapsed is the daemon-side listing, and
/// daemon-side tools are exactly what `builtin_tool_search` returns. A
/// co-located connection keeps the plain summary - one machine draws no
/// per-machine distinction.
fn build_demoted_tool_note(
    tool_defs: &[ToolDefinition],
    deferred_namespaces: &[ToolNamespace],
    locality: Option<&ToolLocalityContext>,
) -> String {
    let total_tools: usize = tool_defs.len()
        + deferred_namespaces
            .iter()
            .map(|ns| ns.tools.len())
            .sum::<usize>();
    let namespace_count = deferred_namespaces.len();
    let mut note = format!(
        "There are {total_tools} tools across {namespace_count} namespaces. \
         Use builtin_tool_search to discover a tool for any task you need."
    );

    if let Some(ctx) = locality
        && !ctx.is_co_located()
        && !ctx.client_tool_names.is_empty()
    {
        let device_tools = ctx.client_tool_names.join(", ");
        note.push_str(&format!(
            " These run on the user's own machine: {device_tools}. Every other tool \
             runs on the daemon's machine."
        ));
    }
    note
}

/// Render the assembled system instruction containing `tool_note` as the
/// tool-availability section, optionally followed by a per-request
/// `system_refinement` section. Centralised so the demotion path rebuilds
/// the same shape as the default path.
///
/// `system_refinement` is a client-supplied, request-scoped addition to the
/// system prompt (see `crate::ports::llm::SYSTEM_REFINEMENT`). When empty
/// (the common case), no section is appended and the output is byte-for-byte
/// identical to the pre-refinement prompt. When present, it is appended
/// last — after every static section and the tool note — so it can refine or
/// override the standing guidance for this turn only.
fn assemble_system_instruction(
    tool_note: String,
    topology: Option<String>,
    ambient: &AmbientContext,
) -> String {
    use crate::prompts::{self, PromptSection, PromptSectionKind};
    let mut sections = prompts::static_sections();

    // Personality disposition (#226): injected *before* the tool note and the
    // per-turn refinement so the standing disposition is established up front
    // while a one-turn refinement can still adjust tone last. Always rendered —
    // the blurb at minimum carries the adaptation clause — so every turn carries
    // a personality, defaulting to the standard disposition when no personality
    // scope was installed.
    let personality_blurb = crate::prompts::render_blurb(&ambient.personality);
    if !personality_blurb.trim().is_empty() {
        sections.push(PromptSection::new(
            PromptSectionKind::Personality,
            personality_blurb,
        ));
    }

    // Client context (#549): a stable, per-connection grounding block ("about the
    // user & their device") rendered from the self-reported context. Injected
    // here — into the cached system instruction, unlike the volatile `[Now]`
    // line — because it is stable for the connection, so it can be cached. A
    // DYNAMIC section (not a `static_sections()` member), so it never perturbs
    // the golden static-prompt snapshot. Omitted entirely when no field is
    // present (fail-closed): the daemon never substitutes its own host values.
    if let Some(ctx) = &ambient.client_context
        && let Some(section) = prompts::render_client_context(ctx)
    {
        sections.push(PromptSection::new(
            PromptSectionKind::ClientContext,
            section,
        ));
    }

    // Topology (#534): which machines exist, and what each one's tools reach.
    // Injected immediately before the tool listing, so the model reads the
    // machines before it reads the tools that are labelled by machine. A
    // DYNAMIC section — the answer changes with the connection — so it never
    // perturbs the golden static-prompt snapshot. Absent for callers that
    // thread no locality context, which leaves their prompt unchanged.
    if let Some(section) = topology {
        sections.push(PromptSection::new(PromptSectionKind::Topology, section));
    }

    sections.push(PromptSection::new(
        PromptSectionKind::ToolAvailability,
        tool_note,
    ));
    let trimmed = ambient.system_refinement.trim();
    if !trimmed.is_empty() {
        sections.push(PromptSection::new(
            PromptSectionKind::SystemRefinement,
            trimmed.to_string(),
        ));
    }
    prompts::assemble(&sections)
}

#[allow(clippy::too_many_arguments)]
fn assemble_turn(
    conversation: &ConversationView,
    tools: &ToolContext,
    anchors: &TurnAnchors,
    ambient: &AmbientContext,
    projection: &ContextProjection,
    max_messages: usize,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> TurnMessages {
    let system_instruction = system_block(tools, ambient, budget, estimate);

    // Apply context windowing: if the conversation exceeds the limit, keep
    // only the most recent messages, snapping the cut point forward to a
    // genuine User message so we never split tool-call/result pairs.
    let start = window_start(conversation.messages, max_messages);
    let windowed = &conversation.messages[start..];
    let is_windowed = start > 0;

    // Summary IDs still active this turn. Their tagged messages collapse to a
    // single marker in `expand_history`, and a message about to be replaced by
    // summary text doesn't count as a "visible" anchor in `surfaced_blocks`.
    let active_summary_ids: std::collections::HashSet<&str> = conversation
        .summaries
        .iter()
        .map(|s| s.id.as_str())
        .collect();

    // Assemble as a pipeline: the cached system instruction, then the per-turn
    // `[..]` re-surfaced context blocks, then the windowed history (with
    // collapsed runs replaced by summary markers).
    let surfaced = surfaced_blocks(
        anchors,
        ambient,
        conversation.context_summary,
        is_windowed,
        windowed,
        &active_summary_ids,
    );
    let mut messages = Vec::with_capacity(windowed.len() + 2);
    messages.push(Message::new(Role::System, system_instruction));
    messages.extend(surfaced.blocks);
    messages.extend(expand_history(
        windowed,
        start,
        conversation.summaries,
        &active_summary_ids,
        projection,
    ));
    TurnMessages {
        messages,
        recalled_entry_ids: surfaced.recalled_entry_ids,
    }
}

/// One assembly pass: the prompt, and what its `[Recall]` block offered.
///
/// The shrink loop repeats the pass at a smaller window, so the ids are a
/// per-pass value rather than something accumulated: the pass that is finally
/// returned is the prompt that ships, and its ids are the ones recorded.
struct TurnMessages {
    messages: Vec<Message>,
    /// See [`AssembledTurn::recalled_entry_ids`].
    recalled_entry_ids: Vec<String>,
}

/// Build the turn's system-instruction string: the assembled prompt sections
/// plus the tool-availability note. When a budget is installed and the block
/// exceeds `SYSTEM_BLOCK_BUDGET_RATIO` of it, the tool listing is demoted to a
/// namespace-only summary — the system instruction is re-included verbatim on
/// every turn, so any space it claims is permanently displaced from history.
fn system_block(
    tools: &ToolContext,
    ambient: &AmbientContext,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> String {
    let tool_note =
        build_full_tool_note(tools.tool_defs, tools.deferred_namespaces, tools.locality);
    // Rendered once and reused by both the full and the demoted assembly: the
    // topology is a fact about the connection, not about the tool listing, so
    // demoting the listing must not change it.
    let topology = tools
        .locality
        .map(|ctx| crate::prompts::render_topology(&ctx.topology()));
    let system_instruction = assemble_system_instruction(tool_note, topology.clone(), ambient);

    // Without a budget there's nothing to measure against — emit as-is.
    let Some(b) = budget else {
        return system_instruction;
    };

    let system_tokens_before = estimate(&system_instruction);
    tracing::info!(
        system_tokens = system_tokens_before,
        budget = b.max_input_tokens,
        ratio = (system_tokens_before as f64 / b.max_input_tokens as f64),
        "system block size"
    );
    let threshold = (b.max_input_tokens as f64 * SYSTEM_BLOCK_BUDGET_RATIO) as u64;
    if system_tokens_before <= threshold {
        return system_instruction;
    }

    let demoted_note =
        build_demoted_tool_note(tools.tool_defs, tools.deferred_namespaces, tools.locality);
    let demoted_system = assemble_system_instruction(demoted_note, topology, ambient);
    let system_tokens_after = estimate(&demoted_system);
    tracing::warn!(
        original_tokens = system_tokens_before,
        demoted_tokens = system_tokens_after,
        budget = b.max_input_tokens,
        "system block exceeded budget threshold; demoted tool listing"
    );
    demoted_system
}

/// The turn's re-surfaced context blocks, and what the `[Recall]` block among
/// them offered.
struct SurfacedBlocks {
    blocks: Vec<Message>,
    /// See [`AssembledTurn::recalled_entry_ids`].
    recalled_entry_ids: Vec<String>,
}

/// Build the per-turn `[..]` system messages that re-surface durable context so
/// the model stays oriented across windowing and compaction. Returned in
/// display order; each block is gated independently:
///
/// - `[Now]` — the ambient date/time line, whenever one is installed.
/// - `[Summary of earlier conversation]` — the rolling summary, once windowing
///   has begun.
/// - `[Current task]` — the anchor prompt, re-injected when it has drifted out
///   of view (windowed out, or collapsed behind an active summary) or after a
///   long agentic loop (`> ACTIVE_TASK_ROUND_THRESHOLD` rounds).
/// - `[Working state]` — a one-line count of notes and open to-dos, rendered
///   every turn either count is non-zero, minus whichever half a fuller block
///   below already covers.
/// - `[Plan]` — the open todo tree, whenever one exists.
/// - `[Pinned]` — the full content of the model's pinned notes, plus the live
///   content of any knowledge entry those notes attach (#1104), whenever
///   anything is pinned. Deliberately ungated: the point of a pin is that the
///   fact stays in view without the model having to notice context is under
///   pressure.
/// - `[Scratchpad]` — the free-form note-key index, gated on the same
///   "context is dropping" signal as `[Current task]`.
/// - `[Recall]` — candidate memory for the user's prompt, on the first round of
///   a turn only. Every block above re-renders each round because each answers
///   "is this still in view?". This one answers "what might this prompt be
///   about?", which the user prompt asks once, so repeating it across twenty
///   tool rounds would spend thousands of tokens on an answer the model has
///   already taken or ignored. It also yields to the two blocks above: a memory
///   `[Pinned]` or `[Scratchpad]` already shows is dropped from it rather than
///   paid for twice (#1101).
fn surfaced_blocks(
    anchors: &TurnAnchors,
    ambient: &AmbientContext,
    context_summary: &str,
    is_windowed: bool,
    windowed: &[Message],
    active_summary_ids: &std::collections::HashSet<&str>,
) -> SurfacedBlocks {
    let mut blocks = Vec::new();
    let mut recalled_entry_ids = Vec::new();

    // Ambient "now": a tiny, always-present line giving the assistant a sense of
    // the current date/time without spending a `builtin_sys_props` tool round.
    // Pushed as a per-turn system message — deliberately NOT folded into the
    // cached system instruction — so the volatile timestamp never busts the
    // prompt-prefix cache.
    if !ambient.now_line.is_empty() {
        blocks.push(Message::new(
            Role::System,
            format!("[Now] {}", ambient.now_line),
        ));
    }

    // Rolling context summary, once windowing has dropped earlier history.
    if is_windowed && !context_summary.is_empty() {
        blocks.push(Message::new(
            Role::System,
            format!("[Summary of earlier conversation]\n{context_summary}"),
        ));
    }

    // Shared "context is starting to drop" signal, used by both `[Current task]`
    // and `[Scratchpad]` (#340): once a long agentic loop has run past the round
    // threshold, surfacing durable anchors again keeps the model on-task even if
    // they're nominally still visible.
    let many_tool_rounds = anchors.tool_rounds_since_anchor > ACTIVE_TASK_ROUND_THRESHOLD;

    // Re-inject the active-task anchor when the original prompt has drifted out
    // of view: windowed out, or still present but collapsed behind an active
    // summary (so the model only sees summary text) — or unconditionally once a
    // long tool-calling session risks burying the goal under tool results.
    if let Some(task) = anchors.active_task.filter(|t| !t.is_empty()) {
        let anchor_visible = windowed.iter().any(|m| {
            m.role == Role::User
                && m.content == task
                && !m
                    .summary_id
                    .as_deref()
                    .is_some_and(|sid| active_summary_ids.contains(sid))
        });
        if !anchor_visible || many_tool_rounds {
            blocks.push(Message::new(Role::System, format!("[Current task] {task}")));
        }
    }

    // Open plan (#240): the dispatch loop renders the conversation's `todo`
    // notes into a compact tree each round, so the plan stays in view cheaply
    // while the verbose work that produced it is evicted from the message log.
    let plan = anchors.plan.filter(|p| !p.is_empty());

    // Free-form scratchpad note keys (#340): durable in storage but otherwise
    // invisible once the writing message is windowed/compacted away. The index
    // lists the keys (recognition over recall), gated on the same "context is
    // dropping" condition as `[Current task]` so it doesn't burn tokens while
    // the note content is still live in the window.
    let scratchpad_index = anchors
        .scratchpad_index
        .filter(|s| !s.is_empty() && (is_windowed || many_tool_rounds));

    // Working-state nudge (#598): the always-on floor beneath the two blocks
    // above. Neither of them is guaranteed to speak when it matters most - the
    // index is gated on context dropping, so before that trigger fires a note
    // stashed earlier is durable but invisible - and one line of counts is
    // cheap enough to send from turn one regardless. It yields rather than
    // duplicating: each half drops when the fuller block covering it renders,
    // and with both present the line disappears entirely.
    let mut working_state = anchors.working_state;
    if plan.is_some() {
        working_state.open_todos = 0;
    }
    if scratchpad_index.is_some() {
        working_state.notes = 0;
    }
    if let Some(counts) = working_state.render() {
        blocks.push(Message::new(
            Role::System,
            format!("[Working state] {counts}"),
        ));
    }

    if let Some(plan) = plan {
        blocks.push(Message::new(Role::System, format!("[Plan]\n{plan}")));
    }

    // Pinned note content (#597). No gate: `[Scratchpad]` is deliberately quiet
    // until context starts dropping because it is a recall aid, but a pin exists
    // precisely so a load-bearing fact is never one forgotten search away. The
    // cap and byte budget - not a visibility gate - are what bound its cost.
    if let Some(pinned) = anchors.pinned.filter(|p| !p.is_empty()) {
        blocks.push(Message::new(Role::System, format!("[Pinned]\n{pinned}")));
    }

    if let Some(index) = scratchpad_index {
        blocks.push(Message::new(Role::System, format!("[Scratchpad] {index}")));
    }

    // Pre-prompt recall (#1100). Last, and closest to the user prompt that
    // follows: it is the least authoritative block here - a hint about what the
    // prompt may be about, which the model is told to ignore where it does not
    // fit. The first-round gate is what bounds its cost.
    //
    // The two decisions just taken above feed the render: recall drops a note
    // `[Scratchpad]` has listed and a step or finding `[Plan]` has named (#1101),
    // and when either block is silent it drops nothing for it - the silent index
    // is the case the scratchpad arm exists for.
    if anchors.tool_rounds_since_anchor == 0
        && let Some(surface) = anchors.recall
    {
        let surface = crate::recall::RecallSurface {
            indexed_keys: if scratchpad_index.is_some() {
                surface.indexed_keys
            } else {
                &[]
            },
            planned_keys: if plan.is_some() {
                surface.planned_keys
            } else {
                &[]
            },
            ..surface
        };
        if let Some(recall) = crate::recall::render_recall(&surface) {
            blocks.push(Message::new(
                Role::System,
                format!("[Recall] {}", recall.text),
            ));
            recalled_entry_ids = recall.entry_ids;
        }
    }

    SurfacedBlocks {
        blocks,
        recalled_entry_ids,
    }
}

/// Expand the windowed message slice into final history: live messages pass
/// through with the round's projected content, while each run of
/// summary-collapsed messages is replaced by a single `[Summary of messages
/// X–Y]` marker injected at the first collapsed message of the run. The
/// absolute ordinal range is recovered from the tagged positions (offset by
/// `start`), falling back to a range-less label when no tagged message is
/// visible in the window.
///
/// This is the one place assembly reads the projection. Every message here is
/// cloned on its way into the prompt, so reading a replacement costs nothing
/// that the clone did not already cost, and the stored message is never
/// touched.
fn expand_history(
    windowed: &[Message],
    start: usize,
    summaries: &[MessageSummary],
    active_summary_ids: &std::collections::HashSet<&str>,
    projection: &ContextProjection,
) -> Vec<Message> {
    let mut out = Vec::with_capacity(windowed.len());
    let mut injected_summaries: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for msg in windowed.iter() {
        if let Some(sid) = &msg.summary_id
            && active_summary_ids.contains(sid.as_str())
        {
            // This message is collapsed. Inject the summary at the first
            // collapsed message we encounter for this summary.
            if !injected_summaries.contains(sid.as_str()) {
                injected_summaries.insert(sid);
                if let Some(s) = summaries.iter().find(|s| s.id == *sid) {
                    let mut first: Option<usize> = None;
                    let mut last: Option<usize> = None;
                    for (i, m) in windowed.iter().enumerate() {
                        if m.summary_id.as_deref() == Some(s.id.as_str()) {
                            let abs = start + i;
                            if first.is_none() {
                                first = Some(abs);
                            }
                            last = Some(abs);
                        }
                    }
                    let body = match (first, last) {
                        (Some(f), Some(l)) => {
                            format!("[Summary of messages {}\u{2013}{}] {}", f, l, s.summary)
                        }
                        _ => format!("[Summary of earlier messages] {}", s.summary),
                    };
                    out.push(Message::new(Role::System, body));
                }
            }
            continue;
        }

        let mut projected = msg.clone();
        if projection.is_replaced(msg) {
            projected.content = projection.content(msg).to_string();
        }
        out.push(projected);
    }

    out
}

/// Locate the largest `Role::Tool` message the round still reads in full,
/// measured at or above `min_tokens`. Returns `None` when no tool message
/// clears the threshold — a small result is not worth truncating, because the
/// truncation notice may be larger than the original.
///
/// The size measured is what the round reads, not what is stored, so a result
/// the projection already replaced can never be picked a second time.
///
/// Why estimated tokens (not bytes): non-ASCII payloads (CJK, emoji,
/// JSON-with-deep-escapes, base64) have wildly different byte-vs-token
/// ratios. Sorting by bytes mis-targets those cases. Step 1 of
/// [`recover_from_overflow`] aims to free the most prompt-token budget,
/// not the most filesystem bytes, so we measure with the same currency
/// the LLM pays in.
fn find_largest_tool_result_above(
    messages: &[Message],
    projection: &ContextProjection,
    min_tokens: u64,
    estimate: &dyn Fn(&str) -> u64,
) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            if m.role != Role::Tool {
                return None;
            }
            let tokens = estimate(projection.content(m));
            if tokens >= min_tokens {
                Some((i, tokens))
            } else {
                None
            }
        })
        .max_by_key(|(_, tokens)| *tokens)
        .map(|(i, _)| i)
}

/// Text the turn reads in place of a tool result that overflow recovery
/// compacted.
///
/// The prefix is deliberately distinct from
/// [`planning::COMPACTION_POINTER_PREFIX`]: that one names a scratchpad note
/// the model can search. This one names none, so a re-run is the model's only
/// way back to the output. The output itself is not lost - the conversation's
/// stored transcript still holds it - but nothing carries it back into the
/// turn on its own.
fn overflow_compaction_notice(original_bytes: usize) -> String {
    format!(
        "<earlier tool output omitted: {original_bytes} bytes are out of view to fit the \
         model's context window. The call above and its arguments are unchanged; \
         re-run the tool if you need this output again.>"
    )
}

/// Outcome of [`compact_oldest_tool_groups`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct CompactionResult {
    /// Tool results replaced by a notice.
    compacted: usize,
    /// Message-content bytes the replacement freed.
    freed_bytes: usize,
    /// Estimated prompt tokens the replacement freed. What the ladder measures
    /// progress in, because tokens are the currency the provider refused the
    /// prompt in.
    freed_tokens: u64,
}

/// Shrink the oldest assistant(tool_calls)+tool_result groups by reading their
/// RESULTS as [`overflow_compaction_notice`] for the rest of the turn.
///
/// The replacement goes in the round's projection, so the prompt loses the
/// bulk and the stored transcript keeps it. The message count is unchanged, so
/// the caller's `compacted_through` boundary still points at the same message
/// and needs no adjustment.
///
/// Only results at or above [`planning::COMPACTION_MIN_EVICT_BYTES`] are worth
/// replacing; below that the notice can be larger than what it replaces. The
/// most recent group is never touched, and a group whose results the round
/// already reads as a notice contributes nothing — so repeated recoveries walk
/// backwards through history and then hand off to the next rung instead of
/// spinning on the same messages.
fn compact_oldest_tool_groups(
    messages: &[Message],
    projection: &mut ContextProjection,
    estimate: &dyn Fn(&str) -> u64,
) -> CompactionResult {
    // Ranges of (assistant-with-tool-calls, tool_result, ..., tool_result)
    // that still hold something worth compacting, oldest first.
    let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        if messages[i].role == Role::Assistant && !messages[i].tool_calls.is_empty() {
            let start = i;
            i += 1;
            while i < messages.len() && messages[i].role == Role::Tool {
                i += 1;
            }
            if messages[start..i]
                .iter()
                .any(|m| is_worth_compacting(m, projection))
            {
                groups.push(start..i);
            }
        } else {
            i += 1;
        }
    }

    if groups.len() <= 1 {
        // The most recent tool interaction stays intact: it is the one the
        // model is still working from.
        return CompactionResult::default();
    }

    // The oldest half, so each recovery leaves the newer material alone.
    let compact_count = groups.len() / 2;
    let mut result = CompactionResult::default();
    for range in groups.into_iter().take(compact_count) {
        for msg in &messages[range] {
            if !is_worth_compacting(msg, projection) {
                continue;
            }
            let current = projection.content(msg).len();
            let current_tokens = estimate(projection.content(msg));
            let notice = overflow_compaction_notice(current);
            // Never trade a result for something bigger, whatever the floor
            // above happens to be set to.
            let Some(freed) = current.checked_sub(notice.len()).filter(|f| *f > 0) else {
                continue;
            };
            result.freed_bytes += freed;
            result.freed_tokens += current_tokens.saturating_sub(estimate(&notice));
            result.compacted += 1;
            projection.replace(msg, notice);
        }
    }

    result
}

/// A tool result the round still reads in full, and big enough that reading a
/// notice instead frees space.
fn is_worth_compacting(msg: &Message, projection: &ContextProjection) -> bool {
    msg.role == Role::Tool && projection.content(msg).len() >= planning::COMPACTION_MIN_EVICT_BYTES
}

/// Compute the window-start index, snapped forward to a `Role::User` boundary.
/// Returns 0 when the conversation fits within `MAX_CONTEXT_MESSAGES`.
/// Find the start index for the context window.
///
/// The returned index must never land on a `Role::Tool` message, because that
/// would orphan tool results from their preceding assistant `tool_calls`
/// message — which the OpenAI API rejects with HTTP 400.  We prefer snapping
/// to a `Role::User` boundary; when none exists (common in long agentic
/// tool-calling loops) we skip past any leading Tool messages instead.
pub(crate) fn window_start(messages: &[Message], max_messages: usize) -> usize {
    let max = max_messages.max(MIN_CONTEXT_MESSAGES);
    if messages.len() <= max {
        return 0;
    }
    let tentative = messages.len() - max;
    let search = &messages[tentative..];
    // Prefer starting on a User message to keep tool groups intact.
    if let Some(offset) = search.iter().position(|m| m.role == Role::User) {
        return tentative + offset;
    }
    // No User message found; at minimum skip past any Tool messages so we
    // never start with orphaned tool results.
    if let Some(offset) = search.iter().position(|m| m.role != Role::Tool) {
        return tentative + offset;
    }
    // The entire window is Tool messages (one assistant message fanned out
    // more tool calls than the window holds). Walk back to the owning
    // assistant `tool_calls` message so the invariant above still holds
    // (DA-12); a slightly larger window beats a guaranteed provider 400.
    messages[..tentative]
        .iter()
        .rposition(|m| m.role != Role::Tool)
        .unwrap_or(0)
}

/// Determine which message range (if any) should be compacted into the
/// rolling context summary. Returns `Some((from, to))` when there are
/// enough newly-dropped messages, or `None` otherwise.
pub(crate) fn compaction_range(conv: &Conversation, max_messages: usize) -> Option<(usize, usize)> {
    let start = window_start(&conv.messages, max_messages);
    if start == 0 {
        return None;
    }
    // First compaction: trigger immediately when crossing the threshold.
    if conv.compacted_through == 0 {
        return Some((0, start));
    }
    // Subsequent compactions: require COMPACTION_INTERVAL new messages,
    // OR any forward progress when the window has been shrunk below the
    // default (so token-pressure triggers don't stall waiting for 20 more
    // messages to accumulate).
    if start >= conv.compacted_through + COMPACTION_INTERVAL
        || (max_messages < MAX_CONTEXT_MESSAGES && start > conv.compacted_through)
    {
        return Some((conv.compacted_through, start));
    }
    None
}

/// Maximum bytes of one user or assistant message carried into the
/// summariser's transcript.
const SUMMARY_PROSE_BYTES: usize = 2000;

/// Maximum number of messages folded into the rolling summary in one call.
///
/// The marker only advances when a summary was produced, so a summariser that
/// keeps failing leaves the range in place and the next turn offers a wider
/// one. Without a cap that range grows for as long as the failure lasts, and a
/// prompt too large for the task model can never succeed - the marker would
/// then be stuck for good. Capping the span makes the fold walk forward in
/// bounded steps instead: each success advances the marker by at most this
/// many messages, and the rest is offered on the next turn.
///
/// Why 100: five times [`COMPACTION_INTERVAL`], so a healthy conversation
/// always folds its whole range in one call, and a conversation catching up
/// after a failure or a bulk import still does it in a few.
const MAX_COMPACTION_SPAN: usize = 100;

/// Maximum bytes of one tool result carried into the summariser's transcript.
///
/// Tool output is the bulk of a long agentic range. A summary records what a
/// call produced, so the head of the payload is enough; carrying results whole
/// would cost more prompt than the range they came from.
const SUMMARY_TOOL_RESULT_BYTES: usize = 600;

/// Maximum bytes of one tool call's arguments carried into the transcript.
const SUMMARY_TOOL_ARGS_BYTES: usize = 200;

/// What [`generate_context_summary`] made of a range of dropped messages.
///
/// The caller advances `compacted_through` only for [`SummaryOutcome::Summarised`].
/// A range that produced no summary stays in front of [`compaction_range`], so a
/// later turn tries it again instead of dropping it from both the window and the
/// rolling summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SummaryOutcome {
    /// A new rolling summary that folds the range in.
    Summarised(String),
    /// The range held no user prose, no assistant prose, no tool call and no
    /// tool result, so there is nothing a summary can carry.
    NothingToSummarise,
    /// The summariser call failed, or it returned no text.
    Failed,
}

/// Append `text` to `out`, cut to `max_bytes` on a char boundary, and mark the
/// cut so the summariser knows it read a head and not the whole value.
fn push_truncated(out: &mut String, text: &str, max_bytes: usize) {
    if text.len() > max_bytes {
        // Char-boundary-safe cut: a naive byte slice panics when the cut lands
        // inside a multibyte character (DA-2).
        out.push_str(&planning::truncate_on_char_boundary(text, max_bytes));
        out.push_str("...[truncated]");
    } else {
        out.push_str(text);
    }
}

/// Render a message range as the transcript the summariser reads.
///
/// Every role that carries work is represented. A long agentic range is mostly
/// tool calls and tool results, and a transcript of prose alone leaves such a
/// range empty - the summariser then has nothing to fold in, and the range
/// drops out of the model's view with no record of what it did. Tool payloads
/// are cut to a head, because the summary records what a call produced rather
/// than reproducing it. `Role::System` messages are skipped: they are the
/// assembler's own re-surfaced blocks, not conversation.
fn summary_transcript(messages: &[Message]) -> String {
    let mut transcript = String::new();
    for msg in messages {
        match msg.role {
            Role::User => {
                transcript.push_str("User: ");
                push_truncated(&mut transcript, &msg.content, SUMMARY_PROSE_BYTES);
                transcript.push('\n');
            }
            Role::Assistant => {
                if !msg.content.is_empty() {
                    transcript.push_str("Assistant: ");
                    push_truncated(&mut transcript, &msg.content, SUMMARY_PROSE_BYTES);
                    transcript.push('\n');
                }
                if !msg.tool_calls.is_empty() {
                    transcript.push_str("Assistant called: ");
                    for (i, call) in msg.tool_calls.iter().enumerate() {
                        if i > 0 {
                            transcript.push_str(", ");
                        }
                        transcript.push_str(&call.name);
                        transcript.push('(');
                        push_truncated(&mut transcript, &call.arguments, SUMMARY_TOOL_ARGS_BYTES);
                        transcript.push(')');
                    }
                    transcript.push('\n');
                }
            }
            Role::Tool if !msg.content.is_empty() => {
                transcript.push_str("Tool result: ");
                push_truncated(&mut transcript, &msg.content, SUMMARY_TOOL_RESULT_BYTES);
                transcript.push('\n');
            }
            _ => {}
        }
    }
    transcript
}

/// Fold a message range into the rolling summary, and report whether the fold
/// succeeded.
///
/// The caller uses the outcome to decide whether the range is safe to leave
/// behind. Returning the existing summary for a failed call would read as
/// success and let the caller mark the range compacted, which drops it from
/// both the prompt window and the summary for good.
pub(crate) async fn generate_context_summary<L: LlmClient>(
    existing_summary: &str,
    messages: &[Message],
    llm: &L,
) -> SummaryOutcome {
    let transcript = summary_transcript(messages);
    if transcript.is_empty() {
        return SummaryOutcome::NothingToSummarise;
    }

    let mut prompt = String::new();
    if !existing_summary.is_empty() {
        prompt.push_str("Existing summary of earlier messages:\n");
        prompt.push_str(existing_summary);
        prompt.push_str("\n\nNew messages to incorporate:\n");
    } else {
        prompt.push_str("Messages to summarize:\n");
    }
    prompt.push_str(&transcript);

    let llm_messages = vec![
        Message::new(
            Role::System,
            "You are a conversation summarizer. The summary MUST begin with a single \
             line \"Active task: <one sentence describing what the user is currently \
             trying to accomplish>\". After that line, produce a concise bullet-point \
             summary of key decisions, user preferences, and established facts. Merge \
             with any existing summary provided. Keep the total summary under 500 \
             words. Output ONLY the formatted summary, no preamble.",
        ),
        Message::new(Role::User, prompt),
    ];

    match llm
        .stream_completion(
            llm_messages,
            &[],
            ReasoningConfig::default(),
            Box::new(|_| true),
        )
        .await
    {
        Ok(response) if !response.text.trim().is_empty() => {
            SummaryOutcome::Summarised(response.text.trim().to_string())
        }
        Ok(_) => {
            tracing::warn!("context summary generation returned empty");
            SummaryOutcome::Failed
        }
        Err(e) => {
            tracing::warn!("context summary generation failed: {e}");
            SummaryOutcome::Failed
        }
    }
}

/// Fold the conversation's next compaction range into its rolling summary, and
/// advance `compacted_through` only when the fold succeeded. Reports whether
/// the marker moved.
///
/// `compaction_range` only ever returns ranges that start at
/// `compacted_through`, so a range the marker steps over is never revisited.
/// Holding the marker back when no summary was produced keeps that range in
/// front of the next turn, which retries it over a range that has since grown.
/// The cost of holding back is one wider range later; the cost of advancing is
/// history that no prompt and no summary carries.
pub(crate) async fn compact_into_summary<L: LlmClient>(
    conv: &mut Conversation,
    max_messages: usize,
    llm: &L,
) -> bool {
    let Some((from, to)) = compaction_range(conv, max_messages) else {
        return false;
    };
    compact_range_into_summary(conv, from, to, llm).await == FoldResult::Moved
}

/// What one fold of a message range did.
///
/// [`FoldResult::Nothing`] is not a failure and costs no LLM call: the range
/// held nothing a summary could describe, so nothing was lost by leaving it.
/// Callers that ration fold attempts must not spend one on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FoldResult {
    /// A summary was produced and the marker moved over the range.
    Moved,
    /// The range rendered to an empty transcript. No call was made.
    Nothing,
    /// A call was made and produced no summary. The marker stays put.
    Failed,
}

/// Fold `conv.messages[from..to]` into the rolling summary, and advance
/// `compacted_through` only when the fold succeeded. Reports whether the marker
/// moved.
///
/// `from` is always the current marker, so a range the marker steps over is
/// never revisited.
async fn compact_range_into_summary<L: LlmClient>(
    conv: &mut Conversation,
    from: usize,
    to: usize,
    llm: &L,
) -> FoldResult {
    // Bounded so one long-running summariser failure cannot grow the fold past
    // what the task model can read. The marker lands on the capped end, so
    // nothing is skipped - the rest is offered again next turn.
    let to = to.min(from + MAX_COMPACTION_SPAN).min(conv.messages.len());
    if from >= to {
        return FoldResult::Nothing;
    }
    match generate_context_summary(&conv.context_summary, &conv.messages[from..to], llm).await {
        SummaryOutcome::Summarised(summary) => {
            conv.context_summary = summary;
            conv.compacted_through = to;
            FoldResult::Moved
        }
        SummaryOutcome::NothingToSummarise => {
            tracing::debug!(from, to, "compaction range held nothing to summarise");
            FoldResult::Nothing
        }
        SummaryOutcome::Failed => {
            tracing::warn!(
                from,
                to,
                "context summary failed; the range stays uncompacted and is retried"
            );
            FoldResult::Failed
        }
    }
}

/// What one call to [`compact_preflight_shrink`] did.
///
/// Three outcomes rather than a `bool`, because the caller has to tell "there
/// was nothing to fold" from "there was, and the summariser declined". The
/// first must not use up the turn's one attempt; the second must, or a
/// summariser that is down costs one call per round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreflightFold {
    /// Nothing to do, and no call was made: either the budget check did not
    /// narrow the window past the marker, or the range it dropped held nothing
    /// a summary could describe. Neither spends the caller's one attempt.
    NotNeeded,
    /// The dropped range was folded in and the marker moved over it.
    Folded,
    /// The range needed folding and the summariser produced no summary. The
    /// marker stays put, so the range is offered again on a later turn.
    Declined,
}

/// Fold what the assembler's pre-flight budget check dropped past the
/// compaction marker, so no message lands in neither the prompt nor the rolling
/// summary.
///
/// [`assemble_turn_within_budget`] answers an oversized prompt by halving the
/// message window, down to [`MIN_CONTEXT_MESSAGES`]. Turn-entry compaction ran
/// against the window the caller asked for, so the messages between the two
/// window starts are in neither place: the prompt does not carry them and the
/// summary does not describe them. Recent context is dropped with nothing
/// standing in for it, and nothing later detects it.
///
/// `window_from` is where the assembler said the prompt starts
/// ([`AssembledTurn::window_from`]); `requested_window` is the window the caller
/// asked for. This answers [`PreflightFold::NotNeeded`] when the two agree,
/// which is the normal case - the deliberate lag between the marker and the
/// window start is the compaction cadence ([`COMPACTION_INTERVAL`]) doing its
/// job, and folding it early would spend a summariser call every turn.
///
/// The fold starts at the marker, not at what the shrink added, because the
/// marker may only move over history a summary describes. It ends at
/// `window_from` or one `MAX_COMPACTION_SPAN` on from the marker, whichever
/// comes first - so a very wide range still costs one bounded call. A capped
/// fold still answers [`PreflightFold::Folded`], and leaves the marker short of
/// `window_from`; the rest is offered again on a later turn, exactly as the
/// cadence path leaves it.
///
/// **Call this at most once per turn.** Every round appends messages, so a
/// shrunk window keeps sliding forward and both guards keep passing; called per
/// round it would spend a summariser call per round on the turns that are
/// already the most expensive. The rounds' own drift needs no fold: the next
/// turn assembles at the full window again and carries those messages itself.
/// The caller assembles again after a [`PreflightFold::Folded`], so the prompt
/// this turn sends carries the summary of what it dropped.
///
/// The turn's closing wind-down assembles without this. It is one message with
/// no tools, at the point the turn is already ending, so a summariser call
/// there would cost the user a wait for context the next turn recovers on its
/// own.
pub(crate) async fn compact_preflight_shrink<L: LlmClient>(
    conv: &mut Conversation,
    window_from: usize,
    requested_window: usize,
    llm: &L,
) -> PreflightFold {
    if window_from <= window_start(&conv.messages, requested_window) {
        return PreflightFold::NotNeeded;
    }
    if window_from <= conv.compacted_through {
        return PreflightFold::NotNeeded;
    }
    let from = conv.compacted_through;
    let folded = compact_range_into_summary(conv, from, window_from, llm).await;
    tracing::info!(
        from,
        window_from,
        requested_window,
        ?folded,
        "the pre-flight budget check narrowed the window past the compaction \
         marker; folding what it dropped into the rolling summary"
    );
    match folded {
        FoldResult::Moved => PreflightFold::Folded,
        // No call was made and the range held nothing to lose, so this does not
        // spend the turn's one attempt.
        FoldResult::Nothing => PreflightFold::NotNeeded,
        FoldResult::Failed => PreflightFold::Declined,
    }
}

/// What one run of the recovery ladder achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryOutcome {
    /// The next prompt is smaller than the one the provider refused: the round
    /// reads less, the window is narrower, or both. A retry is worth making.
    Progressed,
    /// Nothing is left to free and the window is already at its floor. A retry
    /// would send the provider the prompt it has just refused.
    Exhausted,
}

/// Recover from a `ContextOverflow` error by reducing prompt size.
///
/// The ladder runs three steps:
///   1. Read the largest tool result as a chunking notice (the tool_call and
///      result pair stays, so the model still sees what it tried).
///   2. When step 1 frees nothing, read the oldest tool groups' results as
///      notices via [`compact_oldest_tool_groups`].
///   3. Shrink the active window and fold what that drops into the rolling
///      summary, unless steps 1 and 2 provably freed enough on their own.
///
/// Why this order: step 1 costs the least history; step 3 costs the most.
///
/// Steps 1 and 2 work on `window_from..`, the slice the assembler reported the
/// prompt was built from. A tool result before it costs the prompt nothing, so
/// replacing one frees nothing the provider measured - it looked like recovery
/// and sent the same prompt on the retry. The caller's window size is not the
/// same answer, because the pre-flight budget check can narrow the window
/// further than the caller asked for.
///
/// Step 3 is not an else-branch. Steps 1 and 2 free a bounded amount each time,
/// so a conversation with many tool groups could free a little on every attempt
/// and use up every retry with the window untouched. The step runs unless the
/// freed estimate clears the gap the provider reported between the prompt and
/// its limit, by [`ESTIMATE_SAFETY_MARGIN`], which is the one case where a
/// retry has room to spare.
///
/// Steps 1 and 2 write to the round's projection, not to `conv.messages`. The
/// caller writes that list to storage at the end of the turn, so a rewrite here
/// would be a rewrite of the user's stored transcript. The retry counter in
/// `send_prompt` bounds total attempts so a persistently-oversized request
/// cannot loop indefinitely.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn recover_from_overflow<L: LlmClient>(
    conv: &mut Conversation,
    projection: &mut ContextProjection,
    prompt_tokens: Option<u64>,
    max_tokens: Option<u64>,
    window_from: usize,
    target_window: &mut usize,
    task_llm: &L,
    estimate: &(dyn Fn(&str) -> u64 + Send + Sync),
) -> RecoveryOutcome {
    // The slice the prompt was built from, as the assembler reported it.
    let window_from = window_from.min(conv.messages.len());
    let window = &conv.messages[window_from..];
    let mut freed_tokens = 0u64;

    // Step 1: largest in-window tool result, if it's >= MIN_TRUNCATION_TOKENS.
    if let Some(idx) =
        find_largest_tool_result_above(window, projection, MIN_TRUNCATION_TOKENS, estimate)
    {
        let msg = &window[idx];
        let original_bytes = projection.content(msg).len();
        let before = estimate(projection.content(msg));
        let notice = overflow_truncation_notice(original_bytes, prompt_tokens, max_tokens);
        let freed = before.saturating_sub(estimate(&notice));
        if freed > 0 {
            projection.replace(msg, notice);
            freed_tokens = freed;
            tracing::warn!(
                tool_result_index = window_from + idx,
                original_bytes,
                freed_tokens,
                prompt_tokens = ?prompt_tokens,
                max_tokens = ?max_tokens,
                "context overflow — truncating tool result (step 1)"
            );
        }
    }

    // Step 2: read the oldest in-window tool results as notices.
    if freed_tokens == 0 {
        let compacted = compact_oldest_tool_groups(window, projection, estimate);
        freed_tokens = compacted.freed_tokens;
        if compacted.compacted > 0 {
            tracing::warn!(
                compacted = compacted.compacted,
                freed_bytes = compacted.freed_bytes,
                freed_tokens,
                "context overflow — compacted oldest tool results (step 2)"
            );
        }
    }

    // How far over its limit the provider said the prompt was. Absent for a
    // provider that reports neither number, which leaves step 3 to run.
    //
    // The two numbers are in different units: the gap is counted by the
    // provider's tokenizer, and `freed_tokens` by the local estimator, which
    // over-counts compressible payloads such as a padded table or a log with a
    // repeated banner. The margin is what keeps an over-count from skipping
    // step 3 and spending a retry on an unchanged prompt.
    let deficit = match (prompt_tokens, max_tokens) {
        (Some(prompt), Some(max)) if prompt > max => Some(prompt - max),
        _ => None,
    };
    if deficit.is_some_and(|gap| freed_tokens >= gap.saturating_mul(ESTIMATE_SAFETY_MARGIN)) {
        tracing::warn!(
            freed_tokens,
            deficit = ?deficit,
            "context overflow — freed enough to retry without shrinking the window"
        );
        return RecoveryOutcome::Progressed;
    }

    // Step 3: shrink the active window and summarise what that drops. Mirrors
    // the proactive token-pressure path on the success branch.
    let new_window = (*target_window / 2).max(MIN_CONTEXT_MESSAGES);
    let shrank = new_window < *target_window;
    if shrank {
        *target_window = new_window;
    }
    let summarised = compact_into_summary(conv, *target_window, task_llm).await;
    if shrank || summarised {
        tracing::warn!(
            new_window = *target_window,
            shrank,
            summarised,
            "context overflow — summarised and shrank window (step 3)"
        );
    }

    // A new summary alone is not a smaller prompt. It replaces the summary
    // block rather than dropping messages, and the replacement can be longer
    // than what it replaced, so it does not on its own earn a retry.
    if freed_tokens > 0 || shrank {
        RecoveryOutcome::Progressed
    } else {
        tracing::warn!(
            window = *target_window,
            "context overflow — no recovery action available"
        );
        RecoveryOutcome::Exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CoreError;
    use crate::domain::{Conversation, ToolCall, ToolDefinition};
    use crate::ports::llm::{ChunkCallback, LlmResponse};

    /// Token estimator used by the existing assembly tests. Mirrors the
    /// `LlmClient::estimate_tokens` default so tests don't depend on any
    /// connector and behave identically to the real default-impl path.
    fn default_estimate(s: &str) -> u64 {
        (s.chars().count() as u64).div_ceil(4)
    }

    // --- Topology in the system block (#534) -------------------------------

    /// Assemble a system block for a turn whose connection has the given
    /// locality, at the given budget. `None` budget means no demotion check.
    fn system_block_for(
        locality: Option<&ToolLocalityContext>,
        budget: Option<ContextBudget>,
    ) -> String {
        let tool_defs: Vec<ToolDefinition> = (0..40)
            .map(|i| {
                ToolDefinition::new(
                    format!("tool_{i}"),
                    "a tool with a description long enough to cost real tokens",
                    serde_json::json!({}),
                )
            })
            .collect();
        system_block(
            &ToolContext {
                tool_defs: &tool_defs,
                deferred_namespaces: &[],
                locality,
            },
            &AmbientContext::default(),
            budget,
            &default_estimate,
        )
    }

    #[test]
    fn a_client_that_reported_no_label_is_not_quoted_in_the_tool_note() {
        // The label is empty for a client that sent no hostname. The note must
        // not render "your device ''", which reads as a name.
        let mut ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &[],
            &["device_terminal"],
        );
        ctx.client_label = String::new();
        let entries = resolve_tool_localities(&["device_terminal"], &ctx);
        let rendered = render_locality_list(&entries, false);
        assert_eq!(rendered, "device_terminal — your device");
    }

    #[test]
    fn shadowing_a_client_tool_is_reported_by_name() {
        // The names the turn loop reports once, where it resolves the
        // collision. Registration order, so the report is stable, and only the
        // names a daemon-side tool actually holds.
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal", "read_file"],
            &["terminal", "device_only", "read_file"],
        );
        assert_eq!(ctx.shadowed_client_tools(), vec!["terminal", "read_file"]);

        // Nothing to report when every client tool has its own name - which is
        // what a namespaced client registration produces.
        let clean = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal"],
            &["device__terminal"],
        );
        assert!(clean.shadowed_client_tools().is_empty());
    }

    #[test]
    fn the_demoted_note_names_the_tools_on_the_users_machine() {
        // Demotion collapses the tool listing, which is where the per-tool
        // machine labels live. What the model cannot recover by searching is
        // which tools reach the user's own machine, because tool search returns
        // the daemon-side set the listing collapsed.
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal"],
            &["device_terminal", "device_read_file"],
        );
        let note = build_demoted_tool_note(&[], &[], Some(&ctx));
        assert!(
            note.contains("device_terminal") && note.contains("device_read_file"),
            "the demoted note must name the user's own tools: {note}"
        );
        assert!(
            note.contains("Every other tool runs on the daemon's machine"),
            "and say where the rest run: {note}"
        );
    }

    #[test]
    fn the_demoted_note_is_unchanged_when_the_daemon_and_client_are_one_machine() {
        // One machine draws no per-machine distinction, so the summary stays
        // exactly what it was, and costs no extra tokens on the turns that are
        // already over budget.
        let plain = build_demoted_tool_note(&[], &[], None);
        let co_located = locality_ctx(
            TransportKind::Uds,
            "daemon-host",
            &["terminal"],
            &["device_terminal"],
        );
        assert_eq!(build_demoted_tool_note(&[], &[], Some(&co_located)), plain);

        // A remote connection that registered no client tools has nothing extra
        // to say either.
        let no_client_tools =
            locality_ctx(TransportKind::WebSocket, "daemon-host", &["terminal"], &[]);
        assert_eq!(
            build_demoted_tool_note(&[], &[], Some(&no_client_tools)),
            plain
        );
    }

    #[test]
    fn the_system_block_states_where_things_run() {
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal"],
            &["device_terminal"],
        );
        let block = system_block_for(Some(&ctx), None);
        assert!(
            block.contains("== Where things run =="),
            "the assembled block must carry the topology section: {block}"
        );
        assert!(
            block.contains("Two different machines"),
            "and describe a remote connection as two machines: {block}"
        );
    }

    #[test]
    fn the_demoted_block_still_states_where_things_run() {
        // Demotion collapses the tool listing, which is where the per-tool
        // location labels live. The topology is a fact about the connection,
        // not about the listing, so it must survive.
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal"],
            &["device_terminal"],
        );
        let block = system_block_for(
            Some(&ctx),
            Some(ContextBudget {
                max_input_tokens: 1_000,
                source: crate::ports::llm::BudgetSource::UniversalFallback,
            }),
        );
        assert!(
            block.contains("Use builtin_tool_search to discover a tool"),
            "this budget must actually demote the listing: {block}"
        );
        assert!(
            block.contains("== Where things run =="),
            "and the topology must survive the demotion: {block}"
        );
    }

    #[test]
    fn a_block_without_locality_carries_no_topology() {
        // Callers that thread no locality context (background jobs, tests) keep
        // the prompt they had before the section existed.
        let block = system_block_for(None, None);
        assert!(
            !block.contains("== Where things run =="),
            "no locality context means no topology claim: {block}"
        );
    }

    #[test]
    fn assemble_system_instruction_appends_refinement_last() {
        let base =
            assemble_system_instruction("TOOLNOTE".to_string(), None, &AmbientContext::default());
        let refined = assemble_system_instruction(
            "TOOLNOTE".to_string(),
            None,
            &AmbientContext {
                system_refinement: "Respond briefly, by voice.".to_string(),
                ..Default::default()
            },
        );

        // Empty refinement is byte-identical to no refinement.
        assert_eq!(
            base,
            assemble_system_instruction(
                "TOOLNOTE".to_string(),
                None,
                &AmbientContext {
                    system_refinement: "   ".to_string(),
                    ..Default::default()
                },
            ),
            "whitespace-only refinement must be treated as empty"
        );

        // The refined form is a strict superset: the base prompt is preserved
        // verbatim as a prefix, and the refinement is appended at the end.
        assert!(
            refined.starts_with(&base),
            "refined prompt must keep the entire base prompt as a prefix"
        );
        assert!(
            refined.ends_with("Respond briefly, by voice."),
            "refinement must be the final section, got: {refined:?}"
        );
        assert!(
            refined.contains("TOOLNOTE"),
            "tool note must still be present"
        );
    }

    #[test]
    fn assemble_system_instruction_injects_personality_before_tools_and_refinement() {
        use crate::prompts::{Personality, PersonalityLevel};

        // A personality with a recognizable trait so we can locate the
        // injected section in the assembled output.
        let personality = Personality {
            sarcasm: PersonalityLevel::Always,
            ..Personality::default()
        };
        let assembled = assemble_system_instruction(
            "TOOLNOTE".to_string(),
            None,
            &AmbientContext {
                personality,
                system_refinement: "REFINEMENT".to_string(),
                ..Default::default()
            },
        );

        // The personality blurb is present.
        let blurb = crate::prompts::render_blurb(&personality);
        assert!(
            assembled.contains(&blurb),
            "assembled prompt must contain the personality blurb:\n{assembled}"
        );
        // Ordering: personality blurb appears before the tool note, which
        // appears before the per-turn refinement.
        let p_idx = assembled.find(&blurb).unwrap();
        let t_idx = assembled.find("TOOLNOTE").unwrap();
        let r_idx = assembled.find("REFINEMENT").unwrap();
        assert!(p_idx < t_idx, "personality must precede the tool note");
        assert!(t_idx < r_idx, "tool note must precede the refinement");
    }

    #[test]
    fn assemble_system_instruction_default_personality_present_without_scope() {
        // A default `AmbientContext` still injects the default disposition
        // (the global personality applies to every turn).
        let assembled =
            assemble_system_instruction("TOOLNOTE".to_string(), None, &AmbientContext::default());
        let default_blurb = crate::prompts::render_blurb(&crate::prompts::Personality::default());
        assert!(
            assembled.contains(&default_blurb),
            "default personality must be injected even without a scope:\n{assembled}"
        );
    }

    // --- Client context in the system instruction (#549) -------------------

    fn full_client_context() -> crate::prompts::ClientContext {
        crate::prompts::ClientContext {
            real_name: Some("Ada Lovelace".into()),
            username: Some("ada".into()),
            home_dir: Some("/home/ada".into()),
            hostname: Some("analytical-engine".into()),
            timezone: Some("Europe/London".into()),
            os: Some("Ubuntu 24.04".into()),
        }
    }

    #[test]
    fn assemble_system_instruction_includes_client_context_block() {
        // Acceptance (a): a full injected ClientContext renders an "about the
        // user" block inside the cached system instruction, before the tool note.
        let assembled = assemble_system_instruction(
            "TOOLNOTE".to_string(),
            None,
            &AmbientContext {
                client_context: Some(full_client_context()),
                ..Default::default()
            },
        );
        assert!(
            assembled.contains("== About the user & their device =="),
            "{assembled}"
        );
        assert!(assembled.contains("Ada Lovelace"), "{assembled}");
        assert!(assembled.contains("Europe/London"), "{assembled}");
        assert!(assembled.contains("/home/ada"), "{assembled}");
        // Dynamic section is part of the cached system block, ahead of the tools.
        let c_idx = assembled.find("== About the user").unwrap();
        let t_idx = assembled.find("TOOLNOTE").unwrap();
        assert!(c_idx < t_idx, "client context must precede the tool note");
    }

    #[test]
    fn assemble_system_instruction_omits_absent_home_dir_line() {
        // Acceptance (b): an absent field drops only its clause — no home line.
        let ctx = crate::prompts::ClientContext {
            home_dir: None,
            ..full_client_context()
        };
        let assembled = assemble_system_instruction(
            "TOOLNOTE".to_string(),
            None,
            &AmbientContext {
                client_context: Some(ctx),
                ..Default::default()
            },
        );
        assert!(assembled.contains("== About the user & their device =="));
        assert!(!assembled.contains("home directory"), "{assembled}");
        assert!(!assembled.contains("/home/ada"), "{assembled}");
    }

    #[test]
    fn assemble_system_instruction_no_client_context_is_identical_to_none() {
        // Acceptance (c): an all-absent context emits no header at all, and the
        // output is byte-identical to having no client context installed.
        let baseline =
            assemble_system_instruction("TOOLNOTE".to_string(), None, &AmbientContext::default());
        let all_absent = assemble_system_instruction(
            "TOOLNOTE".to_string(),
            None,
            &AmbientContext {
                client_context: Some(crate::prompts::ClientContext::default()),
                ..Default::default()
            },
        );
        assert!(!baseline.contains("== About the user"));
        assert_eq!(
            baseline, all_absent,
            "all-absent context must add nothing to the prompt"
        );
    }

    #[test]
    fn assemble_system_instruction_never_substitutes_daemon_host_values() {
        // Acceptance (d): fail-closed. With every field `None`, the daemon must
        // NOT fall back to its own process HOME / hostname — that substitution
        // would leak the daemon host into a multi-tenant prompt. We read the
        // ambient env (never set it) and assert it does not appear.
        let assembled = assemble_system_instruction(
            "TOOLNOTE".to_string(),
            None,
            &AmbientContext {
                client_context: Some(crate::prompts::ClientContext::default()),
                ..Default::default()
            },
        );
        assert!(!assembled.contains("== About the user"), "{assembled}");
        if let Ok(home) = std::env::var("HOME") {
            assert!(
                !assembled.contains(&home),
                "daemon HOME must never leak into the prompt as a fallback"
            );
        }
        if let Ok(host) = std::env::var("HOSTNAME")
            && !host.is_empty()
        {
            assert!(
                !assembled.contains(&host),
                "daemon hostname must never leak into the prompt as a fallback"
            );
        }
    }

    #[tokio::test]
    async fn now_block_surfaced_right_after_system_instruction_when_scope_installed() {
        use crate::ports::llm::with_now_context;

        let now_line = "Sunday, 2026-06-28, 2:32 PM EDT";
        let msgs = vec![Message::new(Role::User, "what's the date?")];
        let assembled = with_now_context(now_line.to_string(), async {
            assemble_for_test(
                &ConversationView {
                    messages: &msgs,
                    ..Default::default()
                },
                &ToolContext::default(),
                &TurnAnchors::default(),
                None,
                &default_estimate,
            )
        })
        .await;

        // [0] is always the system instruction; the ambient [Now] block is
        // surfaced immediately after it as its own system message.
        assert_eq!(assembled[1].role, Role::System);
        assert_eq!(assembled[1].content, format!("[Now] {now_line}"));
    }

    #[test]
    fn no_now_block_without_scope() {
        // No `with_now_context` scope installed (the common test / dreaming-job
        // path) → no [Now] message and the list is unchanged.
        let msgs = vec![Message::new(Role::User, "hi")];
        let assembled = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );
        assert!(
            !assembled.iter().any(|m| m.content.starts_with("[Now]")),
            "no [Now] block should appear without an installed scope"
        );
    }

    /// Mock LLM that returns canned chunks. Used by summary-generation
    /// tests that exercise [`generate_context_summary`] directly.
    struct MockLlm {
        response_chunks: Vec<String>,
    }

    impl MockLlm {
        fn new(chunks: Vec<&str>) -> Self {
            Self {
                response_chunks: chunks.into_iter().map(String::from).collect(),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for MockLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let mut full = String::new();
            for chunk in &self.response_chunks {
                full.push_str(chunk);
                if !on_chunk(chunk.clone()) {
                    return Ok(LlmResponse::text(full));
                }
            }
            Ok(LlmResponse::text(full))
        }
    }

    /// Mock summariser that always succeeds and records the largest prompt it
    /// was sent, so a test can assert the transcript stays bounded.
    #[derive(Default)]
    struct CountingSummariser {
        longest: std::sync::Mutex<usize>,
    }

    impl CountingSummariser {
        fn longest_prompt(&self) -> usize {
            *self.longest.lock().unwrap()
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for CountingSummariser {
        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let bytes: usize = messages.iter().map(|m| m.content.len()).sum();
            let mut longest = self.longest.lock().unwrap();
            *longest = (*longest).max(bytes);
            Ok(LlmResponse::text("- a summary"))
        }
    }

    /// Mock LLM that returns an error on every call. Used to drive the
    /// fallback branches in [`generate_context_summary`].
    struct FailingLlm;

    #[async_trait::async_trait]
    impl LlmClient for FailingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            Err(CoreError::Llm("fail".into()))
        }
    }

    // --- Overflow recovery: shrink the round, never the record (#733, #798) ---

    /// A tool result big enough to be worth replacing with a notice, but below
    /// the rung-1 truncation floor (`MIN_TRUNCATION_TOKENS`) so rung 1 declines
    /// and recovery reaches rung 2.
    fn mid_sized_result(marker: char) -> String {
        marker.to_string().repeat(2048)
    }

    /// Conversation with `groups` (assistant-with-tool_calls, tool_result)
    /// groups, each result mid-sized. Group `n` uses call id `cN`.
    fn conv_with_tool_groups(groups: usize) -> Conversation {
        let mut conv = Conversation::new("c1", "t");
        conv.messages.push(Message::new(Role::User, "hello"));
        for n in 1..=groups {
            conv.messages
                .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                    format!("c{n}"),
                    "read_file",
                    format!(r#"{{"path":"/tmp/{n}"}}"#),
                )]));
            conv.messages
                .push(Message::tool_result(format!("c{n}"), mid_sized_result('r')));
        }
        conv
    }

    async fn run_recovery(
        conv: &mut Conversation,
        projection: &mut ContextProjection,
        target_window: &mut usize,
    ) {
        recover_from_overflow(
            conv,
            projection,
            Some(100_000),
            Some(8_000),
            window_start(&conv.messages, *target_window),
            target_window,
            &FailingLlm,
            &default_estimate,
        )
        .await;
    }

    /// One recovery attempt on a fresh projection.
    async fn recover_once(conv: &mut Conversation, target_window: &mut usize) -> ContextProjection {
        let mut projection = ContextProjection::default();
        run_recovery(conv, &mut projection, target_window).await;
        projection
    }

    /// What the round reads for the result of `call_id`.
    fn projected_result<'a>(
        conv: &'a Conversation,
        projection: &'a ContextProjection,
        call_id: &str,
    ) -> &'a str {
        let msg = conv
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some(call_id))
            .expect("the result row");
        projection.content(msg)
    }

    /// #798: the stored transcript is the observation layer. Recovery shrinks
    /// the round, and the record it shrinks from is left as it was.
    #[tokio::test]
    async fn overflow_recovery_leaves_the_stored_transcript_byte_for_byte() {
        let mut conv = conv_with_tool_groups(4);
        let before = conv.messages.clone();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(
            conv.messages, before,
            "recovery must not write to the stored transcript"
        );
        assert!(
            projection.replaced_count() > 0,
            "recovery must still have freed something from the round"
        );
    }

    /// #798: a truncation notice is what the model reads, not what is stored.
    #[tokio::test]
    async fn overflow_recovery_rung1_keeps_the_stored_result_whole() {
        // One result over the rung-1 truncation floor, so rung 1 takes it.
        let mut conv = conv_with_tool_groups(2);
        let big = "b".repeat(8192);
        conv.messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c9",
                "read_file",
                "{}",
            )]));
        conv.messages.push(Message::tool_result("c9", &big));
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert!(
            projected_result(&conv, &projection, "c9").contains("exceeded the model's"),
            "the round must read the truncation notice"
        );
        let stored = conv
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("c9"))
            .expect("the result row");
        assert_eq!(stored.content, big, "the stored result must be untouched");
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_keeps_every_message_row() {
        let mut conv = conv_with_tool_groups(4);
        let before = conv.messages.len();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let _ = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(
            conv.messages.len(),
            before,
            "rung 2 must not delete rows from the stored transcript"
        );
        for n in 1..=4 {
            let id = format!("c{n}");
            assert!(
                conv.messages
                    .iter()
                    .any(|m| m.tool_call_id.as_deref() == Some(id.as_str())),
                "the result row for {id} must still exist"
            );
        }
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_leaves_a_notice_in_place_of_the_dropped_output() {
        let mut conv = conv_with_tool_groups(4);
        let original_bytes = mid_sized_result('r').len();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        let oldest = projected_result(&conv, &projection, "c1");
        assert!(
            oldest.contains(&format!("{original_bytes} bytes")),
            "the notice must say how much left the round, got {oldest:?}"
        );
        assert!(
            !oldest.contains(&mid_sized_result('r')),
            "the bulk must actually leave the round"
        );
        assert!(
            oldest.len() < original_bytes,
            "the notice must be smaller than what it stands for"
        );
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_preserves_the_tool_call_audit_trail() {
        let mut conv = conv_with_tool_groups(4);
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let _ = recover_once(&mut conv, &mut target_window).await;

        let call = conv
            .messages
            .iter()
            .find(|m| m.tool_calls.iter().any(|c| c.id == "c1"))
            .expect("the oldest assistant tool-call message must survive");
        let tc = &call.tool_calls[0];
        assert_eq!(tc.name, "read_file", "the tool name must survive");
        assert_eq!(
            tc.arguments, r#"{"path":"/tmp/1"}"#,
            "the arguments must survive so the user can see what ran"
        );
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_keeps_the_most_recent_tool_result_intact() {
        let mut conv = conv_with_tool_groups(4);
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(
            projected_result(&conv, &projection, "c4"),
            mid_sized_result('r'),
            "the most recent tool interaction must stay in view"
        );
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_does_not_move_the_compaction_marker() {
        // Nothing is removed, so the summary boundary still points at the same
        // logical message — the DA-11 / #298 hazard cannot arise.
        let mut conv = conv_with_tool_groups(4);
        conv.compacted_through = 4;
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let _ = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(
            conv.compacted_through, 4,
            "an in-place compaction must leave the summary marker alone"
        );
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_does_not_recompact_an_already_compacted_result() {
        let mut conv = conv_with_tool_groups(4);
        let mut projection = ContextProjection::default();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        run_recovery(&mut conv, &mut projection, &mut target_window).await;
        let after_first = projected_result(&conv, &projection, "c1").to_string();

        run_recovery(&mut conv, &mut projection, &mut target_window).await;
        let after_second = projected_result(&conv, &projection, "c1").to_string();
        assert_eq!(
            after_first, after_second,
            "a notice must never be compacted a second time"
        );
        assert!(
            after_second.contains(&format!("{} bytes", mid_sized_result('r').len())),
            "the notice must keep naming the ORIGINAL size, not its own, got {after_second:?}"
        );
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_skips_results_too_small_to_be_worth_a_notice() {
        // Every result is tiny: replacing one would GROW the prompt, so rung 2
        // must decline and recovery must escalate to rung 3.
        let mut conv = Conversation::new("c1", "t");
        conv.messages.push(Message::new(Role::User, "hello"));
        for n in 1..=4 {
            conv.messages
                .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                    format!("c{n}"),
                    "t",
                    "{}",
                )]));
            conv.messages
                .push(Message::tool_result(format!("c{n}"), "ok"));
        }
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(
            projection.replaced_count(),
            0,
            "tiny results must be left alone"
        );
        assert!(
            target_window < MAX_CONTEXT_MESSAGES,
            "recovery must escalate to rung 3 when rung 2 declines"
        );
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_keeps_a_lone_tool_group() {
        // One group is the most recent interaction; there is nothing older to
        // compact, so rung 2 declines and rung 3 takes over.
        let mut conv = conv_with_tool_groups(1);
        let before = conv.messages.clone();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(projection.replaced_count(), 0);
        assert_eq!(
            conv.messages, before,
            "the only tool group must survive untouched"
        );
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_handles_a_group_whose_call_has_no_result() {
        // Malformed shape: an assistant tool-call message with no result after
        // it (a turn abandoned mid-dispatch). Recovery must not panic and must
        // not fabricate a result.
        let mut conv = conv_with_tool_groups(2);
        conv.messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c3", "t", "{}",
            )]));
        let before = conv.messages.len();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let _ = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(conv.messages.len(), before);
        assert!(
            !conv
                .messages
                .iter()
                .any(|m| m.tool_call_id.as_deref() == Some("c3")),
            "recovery must not invent a result for an unanswered call"
        );
    }

    #[tokio::test]
    async fn overflow_recovery_rung2_is_skipped_when_there_are_no_tool_groups() {
        let mut conv = Conversation::new("c1", "t");
        conv.messages = vec![
            Message::new(Role::User, "hello"),
            Message::new(Role::Assistant, "hi there"),
        ];
        let before = conv.messages.clone();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(projection.replaced_count(), 0);
        assert_eq!(conv.messages, before);
    }

    // --- The ladder targets the assembled window (#754) --------------------

    /// A conversation long enough to be windowed, whose only large tool group
    /// sits before the window start. Group `cN` ids the out-of-window call.
    fn conv_with_out_of_window_tool_result(result: &str) -> Conversation {
        let mut conv = Conversation::new("c1", "t");
        conv.messages.push(Message::new(Role::User, "start"));
        conv.messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "old",
                "read_file",
                "{}",
            )]));
        conv.messages.push(Message::tool_result("old", result));
        for n in 0..MAX_CONTEXT_MESSAGES * 2 {
            conv.messages
                .push(Message::new(Role::User, format!("prompt {n}")));
            conv.messages
                .push(Message::new(Role::Assistant, format!("reply {n}")));
        }
        conv
    }

    /// #754: the prompt is built from the last `MAX_CONTEXT_MESSAGES` messages,
    /// so replacing a tool result before that boundary frees nothing the
    /// provider measured. The ladder must leave it alone and escalate.
    #[tokio::test]
    async fn overflow_recovery_ignores_a_tool_result_outside_the_prompt_window() {
        let huge = "z".repeat(64_000);
        let mut conv = conv_with_out_of_window_tool_result(&huge);
        let out_of_window = conv.messages[2].clone();
        assert!(
            window_start(&conv.messages, MAX_CONTEXT_MESSAGES) > 2,
            "the fixture must put the tool result outside the window"
        );

        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert!(
            !projection.is_replaced(&out_of_window),
            "a result the prompt does not carry must not be touched"
        );
        assert!(
            target_window < MAX_CONTEXT_MESSAGES,
            "with nothing to free in the window, recovery must shrink it"
        );
    }

    /// #754: step 2 used to return as soon as it freed anything, so a
    /// conversation with two or more tool groups never reached the step that
    /// shrinks the prompt actually sent. Freeing less than the provider's
    /// reported gap must shrink the window on the same attempt.
    #[tokio::test]
    async fn overflow_recovery_shrinks_the_window_when_it_frees_less_than_the_gap() {
        let mut conv = conv_with_tool_groups(4);
        let mut target_window = MAX_CONTEXT_MESSAGES;
        // run_recovery reports a 92k-token gap; four mid-sized results are
        // worth about 1k, so the ladder cannot close it on its own.
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert!(
            projection.replaced_count() > 0,
            "step 2 must still compact what it can"
        );
        assert!(
            target_window < MAX_CONTEXT_MESSAGES,
            "step 3 must run too, so the retry sends a smaller window"
        );
    }

    /// #754: the window is not shrunk for its own sake. When the ladder frees
    /// more than the provider's reported gap, the retry has room and the
    /// conversation keeps its context.
    #[tokio::test]
    async fn overflow_recovery_keeps_the_window_when_it_freed_enough() {
        let mut conv = Conversation::new("c1", "t");
        conv.messages.push(Message::new(Role::User, "go"));
        conv.messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1",
                "read_file",
                "{}",
            )]));
        conv.messages
            .push(Message::tool_result("c1", "x".repeat(32_768)));

        let mut projection = ContextProjection::default();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        // 32768 chars is about 8192 estimated tokens; the gap is 1000, so the
        // freeing clears it several times over.
        let outcome = recover_from_overflow(
            &mut conv,
            &mut projection,
            Some(10_000),
            Some(9_000),
            0,
            &mut target_window,
            &FailingLlm,
            &default_estimate,
        )
        .await;

        assert_eq!(outcome, RecoveryOutcome::Progressed);
        assert_eq!(
            target_window, MAX_CONTEXT_MESSAGES,
            "freeing well past the gap must not also cost the window"
        );
        assert!(projection.is_replaced(&conv.messages[2]));
    }

    /// A provider that reports no token counts leaves the gap unknown, so the
    /// ladder cannot prove the retry has room and must shrink the window too.
    #[tokio::test]
    async fn overflow_recovery_shrinks_the_window_when_the_provider_reports_no_counts() {
        let mut conv = Conversation::new("c1", "t");
        conv.messages.push(Message::new(Role::User, "go"));
        conv.messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1",
                "read_file",
                "{}",
            )]));
        conv.messages
            .push(Message::tool_result("c1", "x".repeat(32_768)));

        let mut projection = ContextProjection::default();
        let mut target_window = MAX_CONTEXT_MESSAGES;
        let outcome = recover_from_overflow(
            &mut conv,
            &mut projection,
            None,
            None,
            0,
            &mut target_window,
            &FailingLlm,
            &default_estimate,
        )
        .await;

        assert_eq!(outcome, RecoveryOutcome::Progressed);
        assert!(projection.is_replaced(&conv.messages[2]));
        assert!(
            target_window < MAX_CONTEXT_MESSAGES,
            "with no gap to measure against, the window must shrink as well"
        );
    }

    /// Exactly at the window boundary nothing is out of view, so the ladder
    /// works on the whole conversation and there is no range to summarise.
    #[tokio::test]
    async fn overflow_recovery_at_the_window_boundary_still_compacts_in_view() {
        let mut conv = Conversation::new("c1", "t");
        while conv.messages.len() < MAX_CONTEXT_MESSAGES - 4 {
            let n = conv.messages.len();
            conv.messages
                .push(Message::new(Role::User, format!("prompt {n}")));
        }
        for n in 1..=2 {
            conv.messages
                .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                    format!("c{n}"),
                    "read_file",
                    "{}",
                )]));
            conv.messages
                .push(Message::tool_result(format!("c{n}"), mid_sized_result('r')));
        }
        assert_eq!(conv.messages.len(), MAX_CONTEXT_MESSAGES);
        assert_eq!(
            window_start(&conv.messages, MAX_CONTEXT_MESSAGES),
            0,
            "the fixture must sit exactly on the boundary"
        );

        let mut target_window = MAX_CONTEXT_MESSAGES;
        let projection = recover_once(&mut conv, &mut target_window).await;

        assert_eq!(
            projection.replaced_count(),
            1,
            "the oldest of the two groups is compacted, the newest is kept"
        );
        assert_eq!(
            conv.compacted_through, 0,
            "nothing was out of view, so there is no range to fold in"
        );
    }

    /// #754: a recovery that can free nothing and cannot shrink further says
    /// so, rather than reporting success and letting the caller spend another
    /// attempt on the prompt the provider has just refused.
    #[tokio::test]
    async fn overflow_recovery_reports_exhausted_when_nothing_is_left() {
        let mut conv = Conversation::new("c1", "t");
        conv.messages = vec![
            Message::new(Role::User, "hello"),
            Message::new(Role::Assistant, "hi there"),
        ];
        let mut projection = ContextProjection::default();
        let mut target_window = MIN_CONTEXT_MESSAGES;
        let outcome = recover_from_overflow(
            &mut conv,
            &mut projection,
            Some(100_000),
            Some(8_000),
            0,
            &mut target_window,
            &FailingLlm,
            &default_estimate,
        )
        .await;

        assert_eq!(outcome, RecoveryOutcome::Exhausted);
        assert_eq!(target_window, MIN_CONTEXT_MESSAGES);
        assert_eq!(projection.replaced_count(), 0);
    }

    /// #798: the projection is what the prompt is built from, so a replacement
    /// must reach the assembled messages and nothing else.
    #[test]
    fn the_assembled_prompt_carries_the_projected_content() {
        let conv = conv_with_tool_groups(2);
        let mut projection = ContextProjection::default();
        let target = conv
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("c1"))
            .expect("the result row");
        projection.replace(target, "<a short notice>".to_string());

        let assembled = assemble_turn_within_budget(
            &ConversationView {
                messages: &conv.messages,
                summaries: &[],
                context_summary: "",
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            &projection,
            MAX_CONTEXT_MESSAGES,
            None,
            &default_estimate,
        );

        let tool_bodies: Vec<&str> = assembled
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.content.as_str())
            .collect();
        let untouched = mid_sized_result('r');
        assert_eq!(
            tool_bodies,
            vec!["<a short notice>", untouched.as_str()],
            "the prompt carries the projected content for c1 and nothing else changes"
        );
        assert_eq!(
            conv.messages
                .iter()
                .find(|m| m.tool_call_id.as_deref() == Some("c1"))
                .expect("the result row")
                .content,
            mid_sized_result('r'),
            "assembly must not write back to the transcript"
        );
    }

    // --- Window/compaction tests ---

    #[test]
    fn window_start_skips_orphaned_tool_messages() {
        // When the tentative cut point lands on a Tool message and there are
        // no User messages after it, the window must skip past Tool messages
        // to avoid orphaning tool results from their assistant tool_calls.
        let mut msgs = Vec::new();
        msgs.push(Message::new(Role::User, "initial"));
        // Fill with tool-call groups (assistant + tool_result each = 2 msgs)
        // so the entire tail is tool groups with no User messages.
        let num_groups = MAX_CONTEXT_MESSAGES + 2;
        for i in 0..num_groups {
            msgs.push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                format!("c{i}"),
                "tool_a",
                "{}",
            )]));
            msgs.push(Message::tool_result(format!("c{i}"), format!("result-{i}")));
        }
        // Total = 1 + num_groups*2.  tentative = total - MAX_CONTEXT_MESSAGES.
        // The tentative index lands inside the tool groups.  If it happens to
        // land on a tool_result, the old code would start there (orphaned).
        let start = window_start(&msgs, MAX_CONTEXT_MESSAGES);
        assert_ne!(
            msgs[start].role,
            Role::Tool,
            "window must not start on a Tool message"
        );
    }

    #[test]
    fn window_start_honors_minimum_messages() {
        // A pathologically small max should be clamped to MIN_CONTEXT_MESSAGES
        // so we never serve fewer messages than the minimum.
        let msgs: Vec<Message> = (0..30)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("u-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("a-{i}"))
                }
            })
            .collect();
        let start = window_start(&msgs, 2);
        // With effective floor of MIN_CONTEXT_MESSAGES (=8), start should be
        // around 30 - 8 = 22, snapped forward to a User boundary.
        assert!(start >= 30 - MIN_CONTEXT_MESSAGES);
        assert!(matches!(msgs[start].role, Role::User | Role::Assistant));
    }

    #[test]
    fn window_start_all_tool_window_never_returns_tool_index() {
        // DA-12: when the entire candidate window consists of Tool messages
        // (one assistant message fanning out many tool calls), window_start
        // must still honour its documented invariant and never return a Tool
        // index — otherwise every retry sends orphaned tool results and the
        // provider rejects the request with HTTP 400.
        let mut msgs = vec![Message::new(Role::User, "initial")];
        let calls: Vec<ToolCall> = (0..12)
            .map(|i| ToolCall::new(format!("c{i}"), "tool_a", "{}"))
            .collect();
        msgs.push(Message::assistant_with_tool_calls(calls));
        for i in 0..12 {
            msgs.push(Message::tool_result(format!("c{i}"), format!("r{i}")));
        }
        // max clamps to MIN_CONTEXT_MESSAGES (8); the tail window of 8 is
        // entirely Tool messages, so both fallback searches find nothing.
        let start = window_start(&msgs, 2);
        assert_ne!(
            msgs[start].role,
            Role::Tool,
            "window must never start on a Tool message, got index {start}"
        );
    }

    #[test]
    fn compaction_range_returns_none_under_limit() {
        let mut conv = Conversation::new("c1", "Test");
        for i in 0..10 {
            conv.messages
                .push(Message::new(Role::User, format!("msg-{i}")));
        }
        assert!(compaction_range(&conv, MAX_CONTEXT_MESSAGES).is_none());
    }

    #[test]
    fn compaction_range_returns_some_on_first_overflow() {
        let mut conv = Conversation::new("c1", "Test");
        let count = MAX_CONTEXT_MESSAGES + 10;
        for i in 0..count {
            if i % 2 == 0 {
                conv.messages
                    .push(Message::new(Role::User, format!("user-{i}")));
            } else {
                conv.messages
                    .push(Message::new(Role::Assistant, format!("asst-{i}")));
            }
        }
        let range = compaction_range(&conv, MAX_CONTEXT_MESSAGES);
        assert!(range.is_some());
        let (from, to) = range.unwrap();
        assert_eq!(from, 0);
        assert!(to > 0);
        assert!(to <= count);
    }

    #[test]
    fn compaction_range_respects_interval() {
        let mut conv = Conversation::new("c1", "Test");
        let count = MAX_CONTEXT_MESSAGES + 10;
        for i in 0..count {
            if i % 2 == 0 {
                conv.messages
                    .push(Message::new(Role::User, format!("user-{i}")));
            } else {
                conv.messages
                    .push(Message::new(Role::Assistant, format!("asst-{i}")));
            }
        }
        // Simulate first compaction already done
        let start = window_start(&conv.messages, MAX_CONTEXT_MESSAGES);
        conv.compacted_through = start;

        // No new messages dropped beyond compacted_through → None
        assert!(compaction_range(&conv, MAX_CONTEXT_MESSAGES).is_none());

        // Add COMPACTION_INTERVAL more messages so window slides
        for i in 0..COMPACTION_INTERVAL {
            conv.messages
                .push(Message::new(Role::User, format!("extra-user-{i}")));
            conv.messages
                .push(Message::new(Role::Assistant, format!("extra-asst-{i}")));
        }
        let range = compaction_range(&conv, MAX_CONTEXT_MESSAGES);
        assert!(range.is_some());
        let (from, to) = range.unwrap();
        assert_eq!(from, start);
        assert!(to > start);
    }

    #[test]
    fn compaction_range_advances_on_shrunk_window_without_interval() {
        // When the window has been shrunk below MAX_CONTEXT_MESSAGES (e.g.
        // because the provider reported token pressure), any forward
        // progress past `compacted_through` should re-trigger compaction —
        // the interval guard only applies at the default window size.
        let mut conv = Conversation::new("c1", "Test");
        let count = MAX_CONTEXT_MESSAGES + 4;
        for i in 0..count {
            if i % 2 == 0 {
                conv.messages
                    .push(Message::new(Role::User, format!("user-{i}")));
            } else {
                conv.messages
                    .push(Message::new(Role::Assistant, format!("asst-{i}")));
            }
        }
        // Simulate the default-window compaction already ran.
        conv.compacted_through = window_start(&conv.messages, MAX_CONTEXT_MESSAGES);

        // Shrinking the window pushes `start` past `compacted_through` but
        // less than COMPACTION_INTERVAL messages of new progress — should
        // still trigger because the window has been shrunk.
        let shrunk = MAX_CONTEXT_MESSAGES / 2;
        let range = compaction_range(&conv, shrunk);
        assert!(
            range.is_some(),
            "shrunken window should trigger compaction on any forward progress"
        );
        let (from, to) = range.unwrap();
        assert_eq!(from, conv.compacted_through);
        assert!(to > from);
    }

    // --- Overflow-truncation notice tests ---

    #[test]
    fn overflow_truncation_notice_includes_byte_count_and_hint() {
        let notice = overflow_truncation_notice(12_345, Some(203_524), Some(200_000));
        assert!(notice.contains("12345 bytes"));
        assert!(notice.contains("203524"));
        assert!(notice.contains("200000"));
        assert!(
            notice.contains("narrower") || notice.contains("chunk") || notice.contains("smaller")
        );
    }

    #[test]
    fn overflow_truncation_notice_omits_counts_when_unknown() {
        let notice = overflow_truncation_notice(500, None, None);
        assert!(notice.contains("500 bytes"));
        assert!(!notice.contains("prompt was"));
    }

    // --- Tool-result ingestion cap (issue #174) ---

    #[test]
    fn cap_tool_result_returns_none_when_under_cap() {
        assert_eq!(cap_tool_result("small output", 1024), None);
    }

    #[test]
    fn cap_tool_result_empty_is_unchanged() {
        assert_eq!(cap_tool_result("", 1024), None);
    }

    #[test]
    fn cap_tool_result_exactly_at_cap_is_unchanged() {
        let content = "x".repeat(1024);
        assert_eq!(cap_tool_result(&content, 1024), None);
    }

    #[test]
    fn cap_tool_result_truncates_when_over_cap_with_notice() {
        let content = "x".repeat(10_000);
        let out = cap_tool_result(&content, 1024).expect("over-cap result must truncate");
        assert!(
            out.len() <= 1024,
            "truncated result {} > cap 1024",
            out.len()
        );
        assert!(out.contains("truncated"), "notice must explain truncation");
        assert!(
            out.contains("10000 bytes"),
            "notice must cite the original size"
        );
        // The kept prefix is from the original content.
        assert!(out.starts_with("xxxx"));
    }

    #[test]
    fn cap_tool_result_stays_within_byte_cap_across_sizes() {
        for cap in [512usize, 1024, 4096, 50_000] {
            let content = "y".repeat(cap * 4);
            let out = cap_tool_result(&content, cap).expect("over-cap must truncate");
            assert!(
                out.len() <= cap,
                "cap {cap}: result {} exceeds cap",
                out.len()
            );
        }
    }

    #[test]
    fn cap_tool_result_truncates_on_char_boundary_no_panic() {
        // Dense multi-byte content: every char is 4 bytes. A naive byte cut
        // would land mid-codepoint and panic; the cap must snap to a
        // boundary and always yield valid UTF-8.
        let content = "🚀".repeat(2_000); // 8_000 bytes
        let out = cap_tool_result(&content, 1024).expect("over-cap must truncate");
        assert!(out.len() <= 1024);
        // Valid UTF-8 by construction (String), and the kept prefix is whole rockets.
        assert!(out.starts_with('🚀'));
        assert!(out.contains("truncated"));
    }

    /// PINS CURRENT BEHAVIOUR (possible defect — see PR #445 design triage).
    /// When `max_bytes` is smaller than the truncation notice itself,
    /// `body_budget` saturates to 0 and the function returns the notice alone —
    /// a string LONGER than `max_bytes`. The doc comment acknowledges this as a
    /// "pathological case (real caps dwarf the notice)", so this test documents
    /// the overflow rather than asserting the cap is honoured. If the caller
    /// contract is ever tightened to guarantee `<= max_bytes`, this test is the
    /// canary that must change.
    #[test]
    fn cap_tool_result_smaller_than_notice_is_pinned() {
        let content = "z".repeat(50);
        let max_bytes = 10; // far smaller than the notice
        let out = cap_tool_result(&content, max_bytes).expect("over-cap must truncate");

        // No body prefix survives — the result is exactly the notice.
        let notice = tool_result_truncation_notice(content.len());
        assert_eq!(out, notice);
        // ...and that notice is LONGER than the requested cap. This is the
        // pinned overflow: the output does NOT stay within `max_bytes`.
        assert!(
            out.len() > max_bytes,
            "documented overflow: notice-only result ({} bytes) exceeds cap ({max_bytes})",
            out.len()
        );
    }

    // --- Pure assembly tests (issue #65 + earlier) ---

    #[test]
    fn assemble_turn_returns_all_when_under_limit() {
        let msgs: Vec<Message> = (0..10)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("user-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("assistant-{i}"))
                }
            })
            .collect();

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );
        // System message + all 10 conversation messages
        assert_eq!(result.len(), 11);
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].content, "user-0");
        assert_eq!(result[10].content, "assistant-9");
    }

    #[test]
    fn assemble_turn_windows_when_over_limit() {
        // Build a conversation larger than MAX_CONTEXT_MESSAGES, using
        // simple User/Assistant alternation so the cut lands exactly.
        let count = MAX_CONTEXT_MESSAGES + 20;
        let msgs: Vec<Message> = (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("user-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("assistant-{i}"))
                }
            })
            .collect();

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );
        // The tentative start is count - MAX_CONTEXT_MESSAGES = 20, which is
        // a User message (even index), so the window starts exactly there.
        // Result: 1 system + MAX_CONTEXT_MESSAGES conversation messages.
        assert_eq!(result.len(), MAX_CONTEXT_MESSAGES + 1);
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].role, Role::User);
        assert_eq!(result[1].content, format!("user-20"));
    }

    #[test]
    fn assemble_turn_snaps_to_user_boundary() {
        // Simulate a conversation where the naive cut point would land in
        // the middle of a tool-call/result group.
        let mut msgs = Vec::new();
        // Pad with enough User/Assistant pairs so the total exceeds the limit.
        // We need the cut point to land on a non-User message.
        let padding = MAX_CONTEXT_MESSAGES + 4;
        for i in 0..padding {
            if i % 2 == 0 {
                msgs.push(Message::new(Role::User, format!("user-{i}")));
            } else {
                msgs.push(Message::new(Role::Assistant, format!("asst-{i}")));
            }
        }
        // Now append a tool-call group at the end: assistant(tool_calls) + tool result + user
        msgs.push(Message::assistant_with_tool_calls(vec![ToolCall::new(
            "c1", "tool_a", "{}",
        )]));
        msgs.push(Message::tool_result("c1", "result"));
        msgs.push(Message::new(Role::User, "final-user"));
        msgs.push(Message::new(Role::Assistant, "final-reply"));

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        // The first conversation message (after System) must be a User message.
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].role, Role::User);

        // The tail must be preserved intact.
        let last = result.last().unwrap();
        assert_eq!(last.content, "final-reply");
    }

    #[test]
    fn assembly_skips_pre_flight_when_no_budget() {
        // With `budget = None` the wrapper does not iterate. The output
        // matches the existing message-count windowing exactly — same as
        // the pre-#65 behaviour.
        let count = MAX_CONTEXT_MESSAGES + 20;
        let msgs: Vec<Message> = (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("user-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("asst-{i}"))
                }
            })
            .collect();

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        // 1 system message + MAX_CONTEXT_MESSAGES conversation messages.
        assert_eq!(result.len(), MAX_CONTEXT_MESSAGES + 1);
    }

    #[test]
    fn assembly_shrinks_when_over_token_budget() {
        use crate::ports::llm::BudgetSource;
        // Budget that the assembled prompt cannot fit at the default
        // window. Estimator counts every char as one token so we can
        // tune the math precisely. With the threshold at 85% of 1000,
        // every byte over 850 forces shrinking.
        let big_chunk = "x".repeat(200);
        let count = MAX_CONTEXT_MESSAGES + 20;
        let msgs: Vec<Message> = (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, big_chunk.clone())
                } else {
                    Message::new(Role::Assistant, big_chunk.clone())
                }
            })
            .collect();

        let budget = ContextBudget {
            max_input_tokens: 1_000,
            source: BudgetSource::ConnectorTable,
        };
        // Use a 1-char-per-token estimator so the size math is direct.
        let one_per_char = |s: &str| s.chars().count() as u64;
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            Some(budget),
            &one_per_char,
        );

        // Without shrinking we'd return MAX_CONTEXT_MESSAGES + 1; with
        // shrinking the count must be strictly smaller.
        assert!(
            result.len() < MAX_CONTEXT_MESSAGES + 1,
            "expected pre-flight shrink, got {} messages",
            result.len()
        );
    }

    #[test]
    fn tool_schema_estimate_counts_active_schema_and_deferred_stubs() {
        let one_per_char = |s: &str| s.chars().count() as u64;

        let schema = serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}});
        let schema_cost = schema.to_string().chars().count() as u64;
        let tool = ToolDefinition::new("search", "Find things", schema);
        let active_expected = "search".len() as u64 + "Find things".len() as u64 + schema_cost;

        // A deferred namespace contributes only its name + description stub,
        // never its per-tool schemas (those are off-context until activated).
        let ns = ToolNamespace::new(
            "calendar",
            "Calendar tools",
            vec![ToolDefinition::new(
                "list",
                "List events",
                serde_json::json!({"type": "object"}),
            )],
        );
        let deferred_expected = "calendar".len() as u64 + "Calendar tools".len() as u64;

        let got = tool_schema_estimate(&[tool], &[ns], &one_per_char);
        assert_eq!(got, active_expected + deferred_expected);
    }

    #[test]
    fn assembly_shrinks_when_tool_schemas_push_over_budget() {
        use crate::ports::llm::BudgetSource;
        // Isolate the schema cost: both calls carry a tool with the SAME name
        // and description (so the rendered tool note in the system instruction
        // is byte-identical), differing only in the size of the JSON Schema
        // parameters — which never appear in a message body. The old preflight
        // (message bodies only) would shrink both windows identically; the new
        // one charges for the fat schema and shrinks it harder (issue #305
        // item 7).
        let chunk = "x".repeat(30);
        let count = MAX_CONTEXT_MESSAGES + 20;
        let msgs: Vec<Message> = (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, chunk.clone())
                } else {
                    Message::new(Role::Assistant, chunk.clone())
                }
            })
            .collect();

        let tiny = ToolDefinition::new("tool", "A tool", serde_json::json!({"type": "object"}));
        // Fat schema large enough that, added to the (otherwise identical)
        // turn, it crosses the threshold the tiny turn sits just under.
        let fat = ToolDefinition::new(
            "tool",
            "A tool",
            serde_json::json!({"type": "object", "description": "z".repeat(20_000)}),
        );

        let one_per_char = |s: &str| s.chars().count() as u64;
        let assemble = |tool: &ToolDefinition, budget: ContextBudget| {
            assemble_for_test(
                &ConversationView {
                    messages: &msgs,
                    ..Default::default()
                },
                &ToolContext {
                    tool_defs: std::slice::from_ref(tool),
                    ..Default::default()
                },
                &TurnAnchors::default(),
                Some(budget),
                &one_per_char,
            )
        };

        // Self-calibrating budget so this stays robust to base-prompt drift (a
        // new prompt section must not silently break it): measure the natural,
        // unshrunk assembly of the tiny turn, then size the budget so its
        // threshold (0.85 * budget) sits ~10k chars above that -- clearing the
        // full tiny-schema turn but well under the fat turn's +20k schema, so
        // only the fat schema forces a message-window shrink.
        let huge = ContextBudget {
            max_input_tokens: 10_000_000,
            source: BudgetSource::ConnectorTable,
        };
        // `assemble` returns the message window, so measure the CHAR size of the
        // natural (unshrunk) tiny turn -- the base system message plus its full
        // message window -- and size the budget so its 0.85 threshold sits ~5k
        // chars above it: comfortably clear of the tiny turn (whose schema cost
        // is a handful of chars) yet far below the fat turn's +20k schema, so
        // only the fat schema forces a message-window shrink.
        let natural: u64 = assemble(&tiny, huge)
            .iter()
            .map(|m| m.content.chars().count() as u64)
            .sum();
        let budget = ContextBudget {
            max_input_tokens: (natural + 5_000) * 100 / 85 + 1,
            source: BudgetSource::ConnectorTable,
        };

        let with_tiny = assemble(&tiny, budget);
        let with_fat = assemble(&fat, budget);

        assert!(
            with_fat.len() < with_tiny.len(),
            "fat tool schema should force a stronger shrink: fat={}, tiny={}",
            with_fat.len(),
            with_tiny.len()
        );
    }

    #[test]
    fn assembly_does_not_shrink_below_min_context_messages() {
        use crate::ports::llm::BudgetSource;
        // Even an extreme budget cannot drive the message count below
        // MIN_CONTEXT_MESSAGES — the floor exists to keep enough room
        // for the user's current prompt plus a tool round.
        let big_chunk = "y".repeat(500);
        let count = MAX_CONTEXT_MESSAGES + 20;
        let msgs: Vec<Message> = (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, big_chunk.clone())
                } else {
                    Message::new(Role::Assistant, big_chunk.clone())
                }
            })
            .collect();

        let budget = ContextBudget {
            max_input_tokens: 100,
            source: BudgetSource::ConnectorTable,
        };
        let one_per_char = |s: &str| s.chars().count() as u64;
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            Some(budget),
            &one_per_char,
        );

        // Result includes the system instruction message plus at least
        // MIN_CONTEXT_MESSAGES windowed conversation messages — the
        // floor is enforced even when the budget cannot be satisfied.
        let conversation_count = result
            .iter()
            .filter(|m| !matches!(m.role, Role::System))
            .count();
        assert!(
            conversation_count >= MIN_CONTEXT_MESSAGES,
            "expected at least {} conversation messages, got {}",
            MIN_CONTEXT_MESSAGES,
            conversation_count
        );
    }

    #[test]
    fn assemble_turn_injects_summary_when_windowing() {
        let count = MAX_CONTEXT_MESSAGES + 20;
        let msgs: Vec<Message> = (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("user-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("assistant-{i}"))
                }
            })
            .collect();

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                context_summary: "- User prefers dark mode",
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        // System prompt, then summary system message, then windowed messages
        assert_eq!(result[0].role, Role::System);
        assert!(result[0].content.contains("Adele"));

        assert_eq!(result[1].role, Role::System);
        assert!(
            result[1]
                .content
                .contains("[Summary of earlier conversation]")
        );
        assert!(result[1].content.contains("User prefers dark mode"));

        assert_eq!(result[2].role, Role::User);
    }

    #[test]
    fn assemble_turn_omits_summary_when_under_limit() {
        let msgs: Vec<Message> = (0..10)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("user-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("asst-{i}"))
                }
            })
            .collect();

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                context_summary: "- Some summary",
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        // No summary injected when under limit
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].role, Role::User);
        assert!(
            !result[0]
                .content
                .contains("Summary of earlier conversation")
        );
    }

    #[test]
    fn assemble_turn_omits_empty_summary_when_windowing() {
        let count = MAX_CONTEXT_MESSAGES + 20;
        let msgs: Vec<Message> = (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("user-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("asst-{i}"))
                }
            })
            .collect();

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        // System prompt directly followed by windowed messages — no summary
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].role, Role::User);
    }

    // --- Active-task anchor tests ---

    #[test]
    fn active_task_reinjected_when_user_msg_windowed_out() {
        let task = "build a new feature";
        // Conversation with MAX_CONTEXT_MESSAGES + 5 messages; the original
        // user prompt sits at index 0 and the window slides past it so
        // the anchor must be re-injected.
        let total = MAX_CONTEXT_MESSAGES + 5;
        let mut msgs: Vec<Message> = Vec::with_capacity(total);
        msgs.push(Message::new(Role::User, task));
        for i in 1..total {
            if i % 2 == 0 {
                msgs.push(Message::new(Role::User, format!("noise-user-{i}")));
            } else {
                msgs.push(Message::new(Role::Assistant, format!("noise-asst-{i}")));
            }
        }

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some(task),
                ..Default::default()
            },
            None,
            &default_estimate,
        );

        let injected = result
            .iter()
            .find(|m| m.role == Role::System && m.content.starts_with("[Current task]"))
            .expect("[Current task] system message should be injected when windowed out");
        assert!(
            injected.content.contains(task),
            "injected content {:?} must include the active-task text",
            injected.content
        );
    }

    #[test]
    fn active_task_not_injected_when_user_msg_in_window() {
        let task = "write some unit tests";
        let msgs = vec![
            Message::new(Role::User, task),
            Message::new(Role::Assistant, "ok, let's start"),
        ];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some(task),
                ..Default::default()
            },
            None,
            &default_estimate,
        );

        let any_anchor = result
            .iter()
            .any(|m| m.role == Role::System && m.content.starts_with("[Current task]"));
        assert!(
            !any_anchor,
            "no [Current task] message should be injected when the original prompt is still visible"
        );
    }

    #[test]
    fn active_task_reinjected_after_many_tool_rounds() {
        let task = "trace a flaky test";
        // Anchor message is still in the window — under normal conditions
        // we wouldn't inject, but a high tool-rounds counter forces it.
        let msgs = vec![
            Message::new(Role::User, task),
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "tool_a", "{}")]),
            Message::tool_result("c1", "result"),
        ];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some(task),
                tool_rounds_since_anchor: 6,
                ..Default::default()
            },
            None,
            &default_estimate,
        );

        let any_anchor = result
            .iter()
            .any(|m| m.role == Role::System && m.content == format!("[Current task] {task}"));
        assert!(
            any_anchor,
            "high tool-rounds count should force [Current task] re-injection \
             even when the anchor is still in the window"
        );
    }

    #[test]
    fn active_task_not_injected_when_none() {
        let msgs = vec![Message::new(Role::User, "hello")];
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        let any_anchor = result
            .iter()
            .any(|m| m.role == Role::System && m.content.starts_with("[Current task]"));
        assert!(
            !any_anchor,
            "no anchor should be injected when active_task is None"
        );
    }

    #[test]
    fn active_task_not_injected_when_empty_string() {
        let msgs = vec![Message::new(Role::User, "hello")];
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some(""),
                tool_rounds_since_anchor: 99,
                ..Default::default()
            },
            None,
            &default_estimate,
        );

        let any_anchor = result
            .iter()
            .any(|m| m.role == Role::System && m.content.starts_with("[Current task]"));
        assert!(
            !any_anchor,
            "no anchor should be injected when active_task is an empty string"
        );
    }

    #[test]
    fn active_task_placement_after_summary_before_windowed_messages() {
        let task = "ship the release";
        let count = MAX_CONTEXT_MESSAGES + 10;
        let mut msgs: Vec<Message> = Vec::new();
        msgs.push(Message::new(Role::User, task));
        for i in 0..count {
            if i % 2 == 0 {
                msgs.push(Message::new(Role::User, format!("user-{i}")));
            } else {
                msgs.push(Message::new(Role::Assistant, format!("asst-{i}")));
            }
        }

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                context_summary: "- earlier conversation summary",
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some(task),
                ..Default::default()
            },
            None,
            &default_estimate,
        );

        // Order: system instruction (0) -> rolling-summary system (1)
        // -> [Current task] system (2) -> windowed messages start (3..)
        assert_eq!(result[0].role, Role::System);
        assert!(result[1].role == Role::System);
        assert!(
            result[1]
                .content
                .contains("[Summary of earlier conversation]")
        );
        assert_eq!(result[2].role, Role::System);
        assert!(result[2].content.starts_with("[Current task]"));
        assert!(result[2].content.contains(task));
        // Whatever comes next must not be a System message.
        assert_ne!(result[3].role, Role::System);
    }

    // --- Scratchpad index (#340) ---

    fn scratchpad_index_text(result: &[Message]) -> Option<&str> {
        result
            .iter()
            .find(|m| m.role == Role::System && m.content.starts_with("[Scratchpad]"))
            .map(|m| m.content.as_str())
    }

    #[test]
    fn scratchpad_index_not_shown_on_short_turn() {
        // Anchor still visible, few tool rounds → context isn't dropping yet,
        // so the live notes are still in view. The index would just burn tokens.
        let msgs = vec![
            Message::new(Role::User, "do a thing"),
            Message::new(Role::Assistant, "on it"),
        ];
        let index = "Notes you've stashed (read with builtin_scratchpad_search): foo, bar.";
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("do a thing"),
                scratchpad_index: Some(index),
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        assert!(
            scratchpad_index_text(&result).is_none(),
            "scratchpad index must not appear on a short, fully-visible turn"
        );
    }

    #[test]
    fn scratchpad_index_shown_when_windowed() {
        let total = MAX_CONTEXT_MESSAGES + 5;
        let mut msgs: Vec<Message> = Vec::with_capacity(total);
        msgs.push(Message::new(Role::User, "original task"));
        for i in 1..total {
            if i % 2 == 0 {
                msgs.push(Message::new(Role::User, format!("u-{i}")));
            } else {
                msgs.push(Message::new(Role::Assistant, format!("a-{i}")));
            }
        }
        let index = "Notes you've stashed (read with builtin_scratchpad_search): foo, bar.";
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                scratchpad_index: Some(index),
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        let text = scratchpad_index_text(&result)
            .expect("scratchpad index must appear once windowing has dropped context");
        assert!(text.contains(index));
    }

    #[test]
    fn scratchpad_index_shown_after_many_tool_rounds() {
        let msgs = vec![
            Message::new(Role::User, "trace it"),
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "tool_a", "{}")]),
            Message::tool_result("c1", "result"),
        ];
        let index = "Notes you've stashed (read with builtin_scratchpad_search): foo.";
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("trace it"),
                scratchpad_index: Some(index),
                tool_rounds_since_anchor: ACTIVE_TASK_ROUND_THRESHOLD + 1,
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        assert!(
            scratchpad_index_text(&result).is_some(),
            "scratchpad index must appear after many tool rounds even when anchor is visible"
        );
    }

    #[test]
    fn scratchpad_index_omitted_when_empty() {
        let total = MAX_CONTEXT_MESSAGES + 5;
        let mut msgs: Vec<Message> = Vec::with_capacity(total);
        for i in 0..total {
            msgs.push(Message::new(Role::User, format!("m-{i}")));
        }
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );
        assert!(
            scratchpad_index_text(&result).is_none(),
            "no scratchpad index when there are no free-form notes"
        );
    }

    // --- Working state nudge (#598) ---

    fn working_state_text(result: &[Message]) -> Option<&str> {
        result
            .iter()
            .find(|m| m.role == Role::System && m.content.starts_with("[Working state]"))
            .map(|m| m.content.as_str())
    }

    fn pinned_text(result: &[Message]) -> Option<&str> {
        result
            .iter()
            .find(|m| m.role == Role::System && m.content.starts_with("[Pinned]"))
            .map(|m| m.content.as_str())
    }

    fn recall_text(result: &[Message]) -> Option<&str> {
        result
            .iter()
            .find(|m| m.role == Role::System && m.content.starts_with("[Recall]"))
            .map(|m| m.content.as_str())
    }

    // --- #1100 [Recall] block ------------------------------------------------

    /// One knowledge candidate, near enough to render.
    fn recall_entry(id: &str, summary: &str) -> crate::ports::recall::RecallEntry {
        let mut entry = crate::domain::KnowledgeEntry::new(id, "body", vec![]);
        entry.summary = Some(summary.to_string());
        crate::ports::recall::RecallEntry {
            entry,
            relevance: crate::ports::recall::RecallRelevance::Distance(0.10),
        use_record: None,
        }
    }

    /// One scratchpad candidate, near enough to render.
    fn recall_note(key: &str, content: &str) -> crate::ports::recall::RecallNote {
        crate::ports::recall::RecallNote {
            key: key.to_string(),
            content: content.to_string(),
            pinned: false,
            relevance: crate::ports::recall::RecallRelevance::Distance(0.10),
        }
    }

    /// The turn's recall input. `indexed_keys` is what the `[Scratchpad]` index
    /// lists *when it speaks*; whether it speaks is this builder's call, so the
    /// keys travel in either way.
    fn recall_surface<'a>(
        candidates: &'a crate::ports::recall::RecallCandidates,
        indexed_keys: &'a [String],
    ) -> crate::recall::RecallSurface<'a> {
        planned_recall_surface(candidates, indexed_keys, &[])
    }

    /// The same, plus the steps and findings `[Plan]` names when it renders.
    fn planned_recall_surface<'a>(
        candidates: &'a crate::ports::recall::RecallCandidates,
        indexed_keys: &'a [String],
        planned_keys: &'a [String],
    ) -> crate::recall::RecallSurface<'a> {
        crate::recall::RecallSurface::new(
            candidates,
            crate::recall::RECALL_ENTRY_SCAN_LIMIT,
            crate::recall::RECALL_NOTE_SCAN_LIMIT,
            chrono::Utc::now(),
        )
        .already_in_view(indexed_keys, planned_keys, &[])
    }

    /// A conversation long enough that assembly windows it, which is the signal
    /// `[Scratchpad]` opens on.
    fn windowed_messages() -> Vec<Message> {
        let total = MAX_CONTEXT_MESSAGES + 5;
        let mut msgs: Vec<Message> = Vec::with_capacity(total);
        msgs.push(Message::new(Role::User, "original task"));
        for i in 1..total {
            if i % 2 == 0 {
                msgs.push(Message::new(Role::User, format!("u-{i}")));
            } else {
                msgs.push(Message::new(Role::Assistant, format!("a-{i}")));
            }
        }
        msgs
    }

    #[test]
    fn recall_block_renders_only_on_the_first_round_of_a_turn() {
        // Every other block re-renders each round because each answers "is
        // this still in view?". This one answers "what might this prompt be
        // about?", and the prompt asks that once.
        let msgs = vec![Message::new(Role::User, "where does the registry live?")];
        let candidates = crate::ports::recall::RecallCandidates {
            entries: vec![recall_entry("kb-1", "The registry is on the storage host")],
            ..Default::default()
        };

        let first = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                recall: Some(recall_surface(&candidates, &[])),
                tool_rounds_since_anchor: 0,
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        assert!(
            recall_text(&first).is_some_and(|t| t.contains("the storage host")),
            "the first round must carry the block"
        );

        for round in [1u32, 2, 7, u32::MAX] {
            let later = assemble_for_test(
                &ConversationView {
                    messages: &msgs,
                    ..Default::default()
                },
                &ToolContext::default(),
                &TurnAnchors {
                    recall: Some(recall_surface(&candidates, &[])),
                    tool_rounds_since_anchor: round,
                    ..Default::default()
                },
                None,
                &default_estimate,
            );
            assert!(
                recall_text(&later).is_none(),
                "round {round} must not repeat the block"
            );
        }
    }

    #[test]
    fn pinned_and_recall_both_render_and_recall_sits_last() {
        // Two blocks that arrived from different work and answer different
        // questions. `[Pinned]` says "this is current, do not re-read it";
        // `[Recall]` says "this may not fit, ignore it if not". Order matters:
        // the least authoritative block sits closest to the user prompt that
        // follows, so a hint is never read as a standing fact.
        let msgs = vec![Message::new(Role::User, "where does the registry live?")];
        let candidates = crate::ports::recall::RecallCandidates {
            entries: vec![recall_entry("kb-1", "The registry host")],
            ..Default::default()
        };
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("where does the registry live?"),
                pinned: Some("- api-quirk: /login is form-encoded, not JSON"),
                recall: Some(recall_surface(&candidates, &[])),
                tool_rounds_since_anchor: 0,
                ..Default::default()
            },
            None,
            &default_estimate,
        );

        let pinned_at = result
            .iter()
            .position(|m| m.content.starts_with("[Pinned]"))
            .expect("[Pinned] must still render alongside [Recall]");
        let recall_at = result
            .iter()
            .position(|m| m.content.starts_with("[Recall]"))
            .expect("[Recall] must still render alongside [Pinned]");
        assert!(
            pinned_at < recall_at,
            "the hint goes last, nearest the prompt it is a hint about"
        );
    }

    #[test]
    fn recall_block_is_absent_when_the_lookup_produced_nothing() {
        // No candidate cleared a floor, so the block renders to nothing and the
        // seam emits no empty message.
        let msgs = vec![Message::new(Role::User, "thanks")];
        let empty = crate::ports::recall::RecallCandidates::default();
        for recall in [None, Some(recall_surface(&empty, &[]))] {
            let result = assemble_for_test(
                &ConversationView {
                    messages: &msgs,
                    ..Default::default()
                },
                &ToolContext::default(),
                &TurnAnchors {
                    recall,
                    ..Default::default()
                },
                None,
                &default_estimate,
            );
            assert!(
                recall_text(&result).is_none(),
                "no [Recall] block when there is nothing to recall"
            );
        }
    }

    // --- #1101 the scratchpad arm -------------------------------------------

    /// Acceptance (#1101): the case the arm exists for. A short, fully-visible
    /// turn keeps `[Scratchpad]` silent, so a note stashed earlier is durable
    /// and invisible - and a prompt that is exactly about it must still find it.
    #[test]
    fn recall_block_surfaces_a_note_the_scratchpad_index_is_still_gating() {
        let msgs = vec![
            Message::new(Role::User, "when can we deploy?"),
            Message::new(Role::Assistant, "checking"),
        ];
        let index = "Notes you've stashed (read with builtin_scratchpad_search): deploy-window.";
        let candidates = crate::ports::recall::RecallCandidates {
            notes: vec![recall_note("deploy-window", "Fridays after 18:00")],
            ..Default::default()
        };
        let indexed = vec!["deploy-window".to_string()];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("when can we deploy?"),
                scratchpad_index: Some(index),
                recall: Some(recall_surface(&candidates, &indexed)),
                tool_rounds_since_anchor: 0,
                ..Default::default()
            },
            None,
            &default_estimate,
        );

        assert!(
            scratchpad_index_text(&result).is_none(),
            "precondition: [Scratchpad] is gated silent on a short, visible turn"
        );
        let text = recall_text(&result).expect("[Recall] must carry the note nothing else shows");
        assert!(text.contains("deploy-window"), "{text}");
        assert!(text.contains("Fridays after 18:00"), "{text}");
    }

    /// Acceptance (#1101): once the index has spoken, the key is in view and
    /// recall must not pay for it a second time.
    #[test]
    fn recall_block_omits_a_note_already_listed_in_the_scratchpad_index() {
        let msgs = windowed_messages();
        let index = "Notes you've stashed (read with builtin_scratchpad_search): deploy-window.";
        let candidates = crate::ports::recall::RecallCandidates {
            entries: vec![recall_entry("kb-1", "The registry is on the storage host")],
            notes: vec![recall_note("deploy-window", "Fridays after 18:00")],
            ..Default::default()
        };
        let indexed = vec!["deploy-window".to_string()];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                scratchpad_index: Some(index),
                recall: Some(recall_surface(&candidates, &indexed)),
                tool_rounds_since_anchor: 0,
                ..Default::default()
            },
            None,
            &default_estimate,
        );

        assert!(
            scratchpad_index_text(&result).is_some(),
            "precondition: windowing has opened [Scratchpad]"
        );
        let text = recall_text(&result).expect("the knowledge arm still renders");
        assert!(
            !text.contains("Fridays after 18:00"),
            "the index already named this key: {text}"
        );
        assert!(
            text.contains("the storage host"),
            "dropping one note must not silence the rest of the block: {text}"
        );
    }

    // --- #597 [Pinned] block -------------------------------------------------

    #[test]
    fn pinned_block_not_gated_on_context_pressure() {
        // The defining difference from [Scratchpad]: a pin must be in view
        // without the model having to notice context is dropping first.
        let msgs = vec![
            Message::new(Role::User, "do a thing"),
            Message::new(Role::Assistant, "on it"),
        ];
        let index = "Notes you've stashed (read with builtin_scratchpad_search): foo.";
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("do a thing"),
                scratchpad_index: Some(index),
                pinned: Some("- deploy-target: the managed k3s cluster"),
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        assert!(
            scratchpad_index_text(&result).is_none(),
            "precondition: [Scratchpad] is gated silent on a short, visible turn"
        );
        let text = pinned_text(&result).expect("[Pinned] must render ungated, from turn one");
        assert!(text.contains("the managed k3s cluster"), "{text}");
    }

    #[test]
    fn pin_surfaces_note_content_every_turn() {
        // Same pin, two very different turns: still live, and windowed out.
        let pinned = "- api-quirk: /login is form-encoded, not JSON";
        let short = vec![
            Message::new(Role::User, "trace it"),
            Message::new(Role::Assistant, "ok"),
        ];
        let visible = assemble_for_test(
            &ConversationView {
                messages: &short,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("trace it"),
                pinned: Some(pinned),
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        assert!(
            pinned_text(&visible).is_some_and(|t| t.contains("form-encoded")),
            "pinned content must be present while the writing message is still live"
        );

        let long: Vec<Message> = (0..80)
            .map(|i| Message::new(Role::Assistant, format!("filler {i}")))
            .collect();
        let windowed = assemble_for_test(
            &ConversationView {
                messages: &long,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("trace it"),
                pinned: Some(pinned),
                tool_rounds_since_anchor: ACTIVE_TASK_ROUND_THRESHOLD + 1,
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        assert!(
            pinned_text(&windowed).is_some_and(|t| t.contains("form-encoded")),
            "pinned content must survive into a long, windowed turn — that is the point"
        );
    }

    #[test]
    fn unpin_removes_note_from_context() {
        // Nothing pinned ⇒ no block at all, not an empty one.
        let msgs = vec![Message::new(Role::User, "go")];
        for pinned in [None, Some("")] {
            let result = assemble_for_test(
                &ConversationView {
                    messages: &msgs,
                    ..Default::default()
                },
                &ToolContext::default(),
                &TurnAnchors {
                    active_task: Some("go"),
                    pinned,
                    ..Default::default()
                },
                None,
                &default_estimate,
            );
            assert!(
                pinned_text(&result).is_none(),
                "no [Pinned] block when nothing is pinned (pinned = {pinned:?})"
            );
        }
    }

    #[test]
    fn working_state_renders_before_windowing() {
        // The gap the nudge exists to close: a short, unwindowed turn with zero
        // tool rounds. [Scratchpad] is gated silent here, so a note stashed a
        // few messages ago is durable but invisible - the counts are all the
        // model gets, and it must get them.
        let msgs = vec![
            Message::new(Role::User, "do a thing"),
            Message::new(Role::Assistant, "on it"),
        ];
        let index = "Notes you've stashed (read with builtin_scratchpad_search): foo, bar.";
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("do a thing"),
                scratchpad_index: Some(index),
                working_state: crate::planning::WorkingState {
                    notes: 2,
                    open_todos: 0,
                },
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        assert!(
            scratchpad_index_text(&result).is_none(),
            "precondition: [Scratchpad] is silent on a short, fully-visible turn"
        );
        let text = working_state_text(&result)
            .expect("[Working state] must render from turn one, ungated");
        assert_eq!(text, "[Working state] 2 scratchpad notes.");
    }

    #[test]
    fn working_state_omitted_when_empty() {
        let msgs = vec![Message::new(Role::User, "hi")];
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );
        assert!(
            working_state_text(&result).is_none(),
            "an empty pad must not burn tokens on a zero-count line"
        );
    }

    #[test]
    fn working_state_yields_to_fuller_blocks() {
        let msgs = vec![
            Message::new(Role::User, "trace it"),
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "tool_a", "{}")]),
            Message::tool_result("c1", "result"),
        ];
        let index = "Notes you've stashed (read with builtin_scratchpad_search): foo.";
        let working_state = crate::planning::WorkingState {
            notes: 2,
            open_todos: 3,
        };

        // [Plan] renders (it is ungated), [Scratchpad] does not: the to-do half
        // is redundant and drops, the note count survives.
        let with_plan = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("trace it"),
                plan: Some("- [ ] 1 do the thing"),
                scratchpad_index: Some(index),
                working_state,
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        assert_eq!(
            working_state_text(&with_plan),
            Some("[Working state] 2 scratchpad notes."),
            "the to-do half must drop when [Plan] shows the tree"
        );

        // Both fuller blocks render (many tool rounds ungates [Scratchpad]) -
        // nothing is left for the nudge to say.
        let with_both = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                active_task: Some("trace it"),
                plan: Some("- [ ] 1 do the thing"),
                scratchpad_index: Some(index),
                working_state,
                pinned: None,
                recall: None,
                tool_rounds_since_anchor: ACTIVE_TASK_ROUND_THRESHOLD + 1,
            },
            None,
            &default_estimate,
        );
        assert!(
            scratchpad_index_text(&with_both).is_some(),
            "precondition: [Scratchpad] renders after many tool rounds"
        );
        assert!(
            working_state_text(&with_both).is_none(),
            "the nudge must disappear entirely when both fuller blocks are present"
        );
    }

    #[test]
    fn working_state_precedes_the_plan_block() {
        let msgs = vec![Message::new(Role::User, "go")];
        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors {
                plan: Some("- [ ] 1 do the thing"),
                working_state: crate::planning::WorkingState {
                    notes: 1,
                    open_todos: 1,
                },
                ..Default::default()
            },
            None,
            &default_estimate,
        );
        let pos = |prefix: &str| {
            result
                .iter()
                .position(|m| m.role == Role::System && m.content.starts_with(prefix))
        };
        assert!(pos("[Working state]") < pos("[Plan]"));
    }

    // --- Message summary (collapsing) tests ---

    #[test]
    fn assemble_turn_collapses_summarized_range() {
        let mut msgs = vec![
            Message::new(Role::User, "start"),
            Message::new(Role::Assistant, "step 1"),
            Message::new(Role::Assistant, "step 2"),
            Message::new(Role::Assistant, "step 3"),
            Message::new(Role::User, "follow up"),
            Message::new(Role::Assistant, "final"),
        ];
        // Mark messages 1..=3 as collapsed behind summary "s1"
        msgs[1].summary_id = Some("s1".to_string());
        msgs[2].summary_id = Some("s1".to_string());
        msgs[3].summary_id = Some("s1".to_string());

        let summaries = vec![MessageSummary {
            id: "s1".to_string(),
            summary: "Assistant performed steps 1-3.".to_string(),
        }];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                summaries: &summaries,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        // System + "start" + summary injection + "follow up" + "final" = 5
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].content, "start");
        assert_eq!(result[2].role, Role::System);
        assert!(result[2].content.contains("Summary of messages 1\u{2013}3"));
        assert!(result[2].content.contains("Assistant performed steps 1-3."));
        assert_eq!(result[3].content, "follow up");
        assert_eq!(result[4].content, "final");
    }

    #[test]
    fn assemble_turn_no_summaries_passes_through() {
        let msgs = vec![
            Message::new(Role::User, "hi"),
            Message::new(Role::Assistant, "hello"),
        ];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );
        // System + 2 messages
        assert_eq!(result.len(), 3);
        assert_eq!(result[1].content, "hi");
        assert_eq!(result[2].content, "hello");
    }

    #[test]
    fn assemble_turn_multiple_summaries() {
        let mut msgs = vec![
            Message::new(Role::User, "start"),
            Message::new(Role::Assistant, "a1"),
            Message::new(Role::Assistant, "a2"),
            Message::new(Role::User, "middle"),
            Message::new(Role::Assistant, "b1"),
            Message::new(Role::Assistant, "b2"),
            Message::new(Role::User, "end"),
        ];
        msgs[1].summary_id = Some("s1".to_string());
        msgs[2].summary_id = Some("s1".to_string());
        msgs[4].summary_id = Some("s2".to_string());
        msgs[5].summary_id = Some("s2".to_string());

        let summaries = vec![
            MessageSummary {
                id: "s1".to_string(),
                summary: "First batch.".to_string(),
            },
            MessageSummary {
                id: "s2".to_string(),
                summary: "Second batch.".to_string(),
            },
        ];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                summaries: &summaries,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );
        // System + "start" + summary1 + "middle" + summary2 + "end" = 6
        assert_eq!(result.len(), 6);
        assert!(result[2].content.contains("Summary of messages 1\u{2013}2"));
        assert!(result[2].content.contains("First batch."));
        assert_eq!(result[3].content, "middle");
        assert!(result[4].content.contains("Summary of messages 4\u{2013}5"));
        assert!(result[4].content.contains("Second batch."));
        assert_eq!(result[5].content, "end");
    }

    #[test]
    fn assemble_turn_renders_absolute_ordinals_when_windowed() {
        // Build a long conversation so windowing kicks in. Messages
        // alternate User/Assistant so window_start can land on a User.
        // We tag a contiguous run that survives the window; the rendered
        // range must be the absolute ordinals (offset by the window
        // start), not the windowed-slice positions.
        let total = MAX_CONTEXT_MESSAGES + 20;
        let mut msgs: Vec<Message> = (0..total)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("user-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("asst-{i}"))
                }
            })
            .collect();

        // Tag the last three messages with the summary so they're inside
        // the window regardless of where it starts.
        let first_tagged = total - 3;
        let last_tagged = total - 1;
        for m in &mut msgs[first_tagged..=last_tagged] {
            m.summary_id = Some("s1".to_string());
        }

        let summaries = vec![MessageSummary {
            id: "s1".to_string(),
            summary: "Tail collapsed.".to_string(),
        }];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                summaries: &summaries,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        let injected = result
            .iter()
            .find(|m| m.content.contains("Tail collapsed."))
            .expect("summary must be injected when its messages are in window");
        let expected = format!("Summary of messages {first_tagged}\u{2013}{last_tagged}");
        assert!(
            injected.content.contains(&expected),
            "expected {expected:?} in {:?}",
            injected.content
        );
    }

    #[test]
    fn assemble_turn_skips_summary_when_all_tagged_messages_outside_window() {
        // Tag only messages that the window will exclude. With no tagged
        // message in the window, there's no anchor at which to inject the
        // summary, so it must not appear at all.
        let total = MAX_CONTEXT_MESSAGES + 20;
        let mut msgs: Vec<Message> = (0..total)
            .map(|i| {
                if i % 2 == 0 {
                    Message::new(Role::User, format!("user-{i}"))
                } else {
                    Message::new(Role::Assistant, format!("asst-{i}"))
                }
            })
            .collect();

        // Tag messages 0..=2 — guaranteed to fall outside a window that
        // keeps only the most recent MAX_CONTEXT_MESSAGES.
        for m in msgs.iter_mut().take(3) {
            m.summary_id = Some("s_outside".to_string());
        }

        let summaries = vec![MessageSummary {
            id: "s_outside".to_string(),
            summary: "Old context.".to_string(),
        }];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                summaries: &summaries,
                ..Default::default()
            },
            &ToolContext::default(),
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        assert!(
            result.iter().all(|m| !m.content.contains("Old context.")),
            "summary whose tagged messages fall outside the window must not be injected"
        );
    }

    // --- generate_context_summary tests ---

    #[tokio::test]
    async fn generate_context_summary_produces_summary() {
        let messages = vec![
            Message::new(Role::User, "What is Rust?"),
            Message::new(Role::Assistant, "Rust is a systems programming language."),
            Message::new(Role::User, "What about lifetimes?"),
            Message::new(Role::Assistant, "Lifetimes ensure references are valid."),
        ];
        let llm = MockLlm::new(vec!["- Discussed Rust and lifetimes"]);
        let result = generate_context_summary("", &messages, &llm).await;
        assert_eq!(
            result,
            SummaryOutcome::Summarised("- Discussed Rust and lifetimes".to_string())
        );
    }

    #[tokio::test]
    async fn generate_context_summary_reports_failure_rather_than_the_old_summary() {
        let messages = vec![
            Message::new(Role::User, "Hello"),
            Message::new(Role::Assistant, "Hi"),
        ];
        let llm = FailingLlm;
        let result = generate_context_summary("existing summary", &messages, &llm).await;
        assert_eq!(
            result,
            SummaryOutcome::Failed,
            "a failed summariser must not look like a successful one"
        );
    }

    #[tokio::test]
    async fn generate_context_summary_reports_failure_on_empty_llm_text() {
        let messages = vec![Message::new(Role::User, "Hello")];
        let llm = MockLlm::new(vec!["   "]);
        let result = generate_context_summary("existing summary", &messages, &llm).await;
        assert_eq!(result, SummaryOutcome::Failed);
    }

    #[tokio::test]
    async fn generate_context_summary_reports_nothing_to_summarise_for_an_empty_range() {
        let llm = MockLlm::new(vec!["should not be called"]);
        let result = generate_context_summary("old summary", &[], &llm).await;
        assert_eq!(result, SummaryOutcome::NothingToSummarise);
    }

    #[tokio::test]
    async fn generate_context_summary_truncates_multibyte_content_on_char_boundary() {
        // DA-2: an assistant message longer than 2000 bytes whose byte 2000
        // falls in the middle of a multibyte character must not panic the
        // summariser. 1999 ASCII bytes followed by 2-byte 'é's puts byte
        // 2000 mid-character.
        let mut content = "a".repeat(1999);
        content.push_str(&"é".repeat(20));
        assert!(content.len() > 2000);
        assert!(!content.is_char_boundary(2000));

        let messages = vec![Message::new(Role::Assistant, content)];
        let llm = MockLlm::new(vec!["summary of long message"]);
        let result = generate_context_summary("", &messages, &llm).await;
        assert_eq!(
            result,
            SummaryOutcome::Summarised("summary of long message".to_string())
        );
    }

    /// #751: a range of tool work is the normal shape of a long agentic
    /// stretch. It must reach the summariser, not fall through as a no-op.
    #[tokio::test]
    async fn a_tool_only_range_is_summarised_rather_than_skipped() {
        let messages = vec![
            Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1",
                "read_file",
                r#"{"path":"/tmp/notes"}"#,
            )]),
            Message::tool_result("c1", "the file said something important"),
        ];
        let llm = MockLlm::new(vec!["- read /tmp/notes"]);
        let result = generate_context_summary("old summary", &messages, &llm).await;
        assert_eq!(
            result,
            SummaryOutcome::Summarised("- read /tmp/notes".to_string())
        );
    }

    /// #751: the summariser must be able to say what ran, so the transcript
    /// carries the tool names, the arguments and the results.
    #[tokio::test]
    async fn the_summariser_transcript_carries_tool_names_arguments_and_results() {
        use std::sync::{Arc, Mutex};
        struct CapturingLlm {
            seen: Arc<Mutex<Option<Vec<Message>>>>,
        }
        #[async_trait::async_trait]
        impl LlmClient for CapturingLlm {
            async fn stream_completion(
                &self,
                messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                *self.seen.lock().unwrap() = Some(messages);
                Ok(LlmResponse::text("Active task: stub.\n- a"))
            }
        }
        let seen = Arc::new(Mutex::new(None));
        let llm = CapturingLlm {
            seen: Arc::clone(&seen),
        };
        let messages = vec![
            Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1",
                "read_file",
                r#"{"path":"/tmp/notes"}"#,
            )]),
            Message::tool_result("c1", "the file said something important"),
        ];
        let _ = generate_context_summary("", &messages, &llm).await;

        let captured = seen.lock().unwrap().clone().expect("summariser was called");
        let user = captured
            .iter()
            .find(|m| m.role == Role::User)
            .expect("summariser sends a user message");
        assert!(
            user.content.contains("read_file"),
            "the transcript must name the tool, got {:?}",
            user.content
        );
        assert!(
            user.content.contains("/tmp/notes"),
            "the transcript must carry the call arguments, got {:?}",
            user.content
        );
        assert!(
            user.content.contains("the file said something important"),
            "the transcript must carry the tool result, got {:?}",
            user.content
        );
    }

    /// A single tool result can be hundreds of kilobytes. The transcript takes
    /// a head of it, so one result cannot outweigh the range it belongs to.
    #[tokio::test]
    async fn the_summariser_transcript_cuts_an_oversized_tool_result() {
        let huge = "z".repeat(50_000);
        let messages = vec![Message::tool_result("c1", huge.clone())];
        let transcript = summary_transcript(&messages);
        assert!(
            transcript.len() < SUMMARY_TOOL_RESULT_BYTES + 100,
            "the transcript must carry a head, not the whole result, got {} bytes",
            transcript.len()
        );
        assert!(transcript.contains("...[truncated]"));
    }

    /// A range of only system blocks carries no conversation.
    #[test]
    fn a_system_only_range_has_nothing_to_summarise() {
        let messages = vec![Message::new(Role::System, "[Now] Tuesday")];
        assert!(summary_transcript(&messages).is_empty());
    }

    // --- compact_into_summary: the marker moves only on success (#751) ---

    /// A conversation long enough for `compaction_range` to return a range,
    /// whose dropped head is ordinary user/assistant prose.
    fn conv_ready_for_compaction() -> Conversation {
        let mut conv = Conversation::new("c1", "t");
        for i in 0..(MAX_CONTEXT_MESSAGES * 2) {
            conv.messages
                .push(Message::new(Role::User, format!("prompt {i}")));
            conv.messages
                .push(Message::new(Role::Assistant, format!("reply {i}")));
        }
        conv
    }

    #[tokio::test]
    async fn a_failed_summariser_does_not_advance_the_compaction_marker() {
        let mut conv = conv_ready_for_compaction();
        let before = conv.compacted_through;
        let compacted = compact_into_summary(&mut conv, MAX_CONTEXT_MESSAGES, &FailingLlm).await;

        assert!(!compacted, "a failed summariser did not compact anything");
        assert_eq!(
            conv.compacted_through, before,
            "the marker must stay put so the range is summarised on a later turn"
        );
        assert_eq!(
            conv.context_summary, "",
            "a failed summariser must not rewrite the rolling summary"
        );
    }

    #[tokio::test]
    async fn an_empty_summariser_response_does_not_advance_the_compaction_marker() {
        let mut conv = conv_ready_for_compaction();
        conv.context_summary = "earlier summary".to_string();
        let llm = MockLlm::new(vec!["   "]);
        let compacted = compact_into_summary(&mut conv, MAX_CONTEXT_MESSAGES, &llm).await;

        assert!(!compacted);
        assert_eq!(conv.compacted_through, 0);
        assert_eq!(conv.context_summary, "earlier summary");
    }

    /// Where the marker lands when the whole due range is folded in one call.
    fn expected_marker(conv: &Conversation) -> usize {
        let (from, to) = compaction_range(conv, MAX_CONTEXT_MESSAGES).expect("a range to compact");
        to.min(from + MAX_COMPACTION_SPAN)
    }

    #[tokio::test]
    async fn a_range_the_summariser_declined_is_offered_again_on_the_next_turn() {
        // The failure is transient. The second attempt must still see the
        // range the first one could not summarise.
        let mut conv = conv_ready_for_compaction();
        let expected = compaction_range(&conv, MAX_CONTEXT_MESSAGES).expect("a range to compact");
        let marker = expected_marker(&conv);
        assert!(!compact_into_summary(&mut conv, MAX_CONTEXT_MESSAGES, &FailingLlm).await);

        let retry = compaction_range(&conv, MAX_CONTEXT_MESSAGES).expect("the range is still due");
        assert_eq!(
            retry, expected,
            "the same range must be offered again after a failed summary"
        );

        let llm = MockLlm::new(vec!["- the recovered summary"]);
        assert!(compact_into_summary(&mut conv, MAX_CONTEXT_MESSAGES, &llm).await);
        assert_eq!(conv.context_summary, "- the recovered summary");
        assert_eq!(conv.compacted_through, marker);
    }

    #[tokio::test]
    async fn a_successful_summary_advances_the_compaction_marker() {
        let mut conv = conv_ready_for_compaction();
        let marker = expected_marker(&conv);
        let llm = MockLlm::new(vec!["- what happened earlier"]);

        assert!(compact_into_summary(&mut conv, MAX_CONTEXT_MESSAGES, &llm).await);
        assert_eq!(conv.compacted_through, marker);
        assert_eq!(conv.context_summary, "- what happened earlier");
    }

    /// Holding the marker back on failure must not let the fold grow without
    /// limit. A summariser that keeps failing would otherwise be offered a
    /// wider range every turn until no task model could read it, and the
    /// marker would be stuck for good.
    #[tokio::test]
    async fn the_fold_is_bounded_however_long_the_summariser_stays_down() {
        let mut conv = Conversation::new("c1", "t");
        for i in 0..600 {
            conv.messages
                .push(Message::new(Role::User, format!("prompt {i}")));
            conv.messages
                .push(Message::new(Role::Assistant, format!("reply {i}")));
        }
        let (from, to) = compaction_range(&conv, MAX_CONTEXT_MESSAGES).expect("a range to compact");
        assert!(
            to - from > MAX_COMPACTION_SPAN,
            "the fixture must offer more than one span"
        );

        let llm = CountingSummariser::default();
        assert!(compact_into_summary(&mut conv, MAX_CONTEXT_MESSAGES, &llm).await);
        assert_eq!(
            conv.compacted_through,
            from + MAX_COMPACTION_SPAN,
            "one call folds at most one span, and the marker lands on its end"
        );
        assert!(
            llm.longest_prompt() < MAX_COMPACTION_SPAN * (SUMMARY_PROSE_BYTES + 64),
            "the transcript must stay bounded by the span and the per-message caps"
        );

        // The rest is not skipped: the next call takes the following span.
        assert!(compact_into_summary(&mut conv, MAX_CONTEXT_MESSAGES, &llm).await);
        assert_eq!(conv.compacted_through, from + 2 * MAX_COMPACTION_SPAN);
    }

    // --- The pre-flight shrink and the compaction marker (#1144) -----------
    //
    // The assembler answers an oversized prompt by halving the message window.
    // Turn-entry compaction ran against the window the caller asked for, so
    // whatever sits between the two window starts is in neither the prompt nor
    // the rolling summary unless the turn folds it in.

    /// A conversation of `pairs` user/assistant pairs, long enough to window.
    fn conv_of_pairs(pairs: usize) -> Conversation {
        let mut conv = Conversation::new("c1", "t");
        for i in 0..pairs {
            conv.messages
                .push(Message::new(Role::User, format!("prompt {i}")));
            conv.messages
                .push(Message::new(Role::Assistant, format!("reply {i}")));
        }
        conv
    }

    /// #1144 acceptance: a turn whose pre-flight budget check narrows the
    /// window past the compaction marker folds the newly-dropped range into the
    /// rolling summary.
    #[tokio::test]
    async fn a_preflight_shrink_past_the_compaction_marker_folds_the_dropped_range() {
        let mut conv = conv_of_pairs(40);
        // Turn-entry compaction ran against the window the loop asked for.
        let requested_start = window_start(&conv.messages, MAX_CONTEXT_MESSAGES);
        conv.compacted_through = requested_start;
        // The pre-flight check then halved the window down to the floor.
        let shrunk_start = window_start(&conv.messages, MIN_CONTEXT_MESSAGES);
        assert!(
            shrunk_start > requested_start,
            "the fixture must actually shrink the window"
        );

        let llm = MockLlm::new(vec!["- what the shrink dropped"]);
        let folded =
            compact_preflight_shrink(&mut conv, shrunk_start, MAX_CONTEXT_MESSAGES, &llm).await;

        assert_eq!(
            folded,
            PreflightFold::Folded,
            "the newly-dropped range must be folded in"
        );
        assert_eq!(
            conv.compacted_through, shrunk_start,
            "the marker must cover everything the shrunk prompt no longer carries"
        );
        assert_eq!(conv.context_summary, "- what the shrink dropped");
    }

    /// The fold starts at the marker, not at the window the caller asked for,
    /// so the cadence lag is summarised rather than stepped over.
    #[tokio::test]
    async fn the_preflight_fold_starts_at_the_marker_so_no_range_is_stepped_over() {
        let mut conv = conv_of_pairs(40);
        let requested_start = window_start(&conv.messages, MAX_CONTEXT_MESSAGES);
        // The marker lags the requested window - the normal cadence state.
        conv.compacted_through = requested_start - 10;
        let shrunk_start = window_start(&conv.messages, MIN_CONTEXT_MESSAGES);

        let llm = CountingSummariser::default();
        assert_eq!(
            compact_preflight_shrink(&mut conv, shrunk_start, MAX_CONTEXT_MESSAGES, &llm).await,
            PreflightFold::Folded
        );
        assert_eq!(
            conv.compacted_through, shrunk_start,
            "the marker may only step over a range the summary describes"
        );
        assert!(
            llm.longest_prompt() > 0,
            "the summariser must have been given the whole range from the marker"
        );
    }

    /// A turn the pre-flight check did not shrink pays nothing. The gap between
    /// the marker and the window start is the compaction cadence
    /// (`COMPACTION_INTERVAL`) doing its job, and folding it here would spend a
    /// summariser call on every turn of every long conversation.
    #[tokio::test]
    async fn the_compaction_cadence_lag_is_not_mistaken_for_a_preflight_shrink() {
        let mut conv = conv_of_pairs(40);
        let requested_start = window_start(&conv.messages, MAX_CONTEXT_MESSAGES);
        conv.compacted_through = requested_start - 10;
        let before = conv.compacted_through;

        let llm = CountingSummariser::default();
        let folded =
            compact_preflight_shrink(&mut conv, requested_start, MAX_CONTEXT_MESSAGES, &llm).await;

        assert_eq!(
            folded,
            PreflightFold::NotNeeded,
            "no shrink happened, so there is nothing extra to fold"
        );
        assert_eq!(conv.compacted_through, before);
        assert_eq!(
            llm.longest_prompt(),
            0,
            "the summariser must not be called on an unshrunk turn"
        );
    }

    /// A short conversation assembles from index 0 and can drop nothing.
    #[tokio::test]
    async fn a_conversation_that_fits_the_window_never_folds() {
        let mut conv = conv_of_pairs(3);
        let llm = CountingSummariser::default();
        assert_eq!(
            compact_preflight_shrink(&mut conv, 0, MAX_CONTEXT_MESSAGES, &llm).await,
            PreflightFold::NotNeeded
        );
        assert_eq!(conv.compacted_through, 0);
        assert_eq!(llm.longest_prompt(), 0);
    }

    /// A range that renders to nothing is not a declined fold. No call is made
    /// and nothing is lost by leaving it, so it must not spend the caller's one
    /// fold attempt - the outcome enum's whole reason for having three arms.
    #[tokio::test]
    async fn a_dropped_range_that_holds_nothing_to_summarise_costs_no_attempt() {
        // `System` messages and empty assistant turns render to nothing in the
        // summariser transcript, so the whole dropped range is empty.
        let mut conv = Conversation::new("c1", "t");
        for i in 0..40 {
            conv.messages
                .push(Message::new(Role::System, format!("sys {i}")));
            conv.messages.push(Message::new(Role::Assistant, ""));
        }
        let requested_start = window_start(&conv.messages, MAX_CONTEXT_MESSAGES);
        conv.compacted_through = requested_start;
        let shrunk_start = window_start(&conv.messages, MIN_CONTEXT_MESSAGES);
        assert!(
            shrunk_start > requested_start,
            "the fixture must actually shrink the window"
        );

        let llm = CountingSummariser::default();
        let folded =
            compact_preflight_shrink(&mut conv, shrunk_start, MAX_CONTEXT_MESSAGES, &llm).await;

        assert_eq!(
            folded,
            PreflightFold::NotNeeded,
            "an empty range is nothing to fold, not a fold that failed"
        );
        assert_eq!(
            conv.compacted_through, requested_start,
            "the marker may not step over a range no summary describes"
        );
        assert_eq!(
            llm.longest_prompt(),
            0,
            "no summariser call may be made for an empty range"
        );
    }

    /// A summariser that declines leaves the marker where it was, exactly as
    /// the cadence path does, so the range is offered again rather than lost.
    #[tokio::test]
    async fn a_failed_fold_of_a_preflight_shrink_leaves_the_marker_alone() {
        let mut conv = conv_of_pairs(40);
        let requested_start = window_start(&conv.messages, MAX_CONTEXT_MESSAGES);
        conv.compacted_through = requested_start;
        let shrunk_start = window_start(&conv.messages, MIN_CONTEXT_MESSAGES);

        let folded =
            compact_preflight_shrink(&mut conv, shrunk_start, MAX_CONTEXT_MESSAGES, &FailingLlm)
                .await;

        assert_eq!(
            folded,
            PreflightFold::Declined,
            "a needed fold the summariser refused is not the same as no fold: it \
             uses up the turn's one attempt"
        );
        assert_eq!(conv.compacted_through, requested_start);
        assert_eq!(conv.context_summary, "");
    }

    /// A single enormous user message cannot dominate the fold either.
    #[test]
    fn the_summariser_transcript_cuts_an_oversized_user_message() {
        let messages = vec![Message::new(Role::User, "u".repeat(50_000))];
        let transcript = summary_transcript(&messages);
        assert!(
            transcript.len() < SUMMARY_PROSE_BYTES + 100,
            "the transcript must carry a head, not the whole message, got {} bytes",
            transcript.len()
        );
        assert!(transcript.contains("...[truncated]"));
    }

    #[tokio::test]
    async fn summariser_prompt_requires_active_task_header() {
        use std::sync::{Arc, Mutex};
        // The system prompt used by the rolling summariser must require the
        // model to lead with an "Active task:" line so the goal survives
        // even when the layer-3b injection conditions are misjudged.
        struct CapturingSummariserLlm {
            seen: Arc<Mutex<Option<Vec<Message>>>>,
        }
        #[async_trait::async_trait]
        impl LlmClient for CapturingSummariserLlm {
            async fn stream_completion(
                &self,
                messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                *self.seen.lock().unwrap() = Some(messages);
                Ok(LlmResponse::text("Active task: stub.\n- a"))
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let llm = CapturingSummariserLlm {
            seen: Arc::clone(&seen),
        };
        let messages = vec![
            Message::new(Role::User, "first user prompt"),
            Message::new(Role::Assistant, "first assistant reply"),
        ];
        let _ = generate_context_summary("", &messages, &llm).await;

        let captured = seen
            .lock()
            .unwrap()
            .clone()
            .expect("summariser LLM should have been invoked");
        let system = captured
            .iter()
            .find(|m| m.role == Role::System)
            .expect("summariser must send a system message");
        assert!(
            system.content.contains("Active task:"),
            "summariser system prompt must contain the Active task: directive, got: {:?}",
            system.content
        );
    }

    // --- System block budget tests (issue #66) ---

    /// Build a tool list whose enumerated names alone are large enough that
    /// the assembled system block exceeds 20% of the supplied budget. The
    /// chars/4 default estimator counts each name as `chars/4` tokens, so we
    /// pad each tool name with enough characters to reach the threshold.
    fn make_huge_tool_set(count: usize, name_pad: usize) -> Vec<ToolDefinition> {
        (0..count)
            .map(|i| {
                let padded = format!("tool_{i}_{}", "x".repeat(name_pad));
                ToolDefinition::new(padded, "desc", serde_json::json!({"type": "object"}))
            })
            .collect()
    }

    #[test]
    fn system_block_demoted_when_oversized() {
        use crate::ports::llm::BudgetSource;
        // Budget of 1000 tokens means a 20% threshold of 200 tokens. Build
        // a tool set whose enumeration alone overshoots that, then assert
        // the assembled system block carries the demoted "There are N
        // tools" wording rather than the full enumeration.
        let tools = make_huge_tool_set(60, 64);
        let msgs = vec![Message::new(Role::User, "hi")];
        let budget = ContextBudget {
            max_input_tokens: 1_000,
            source: BudgetSource::ConnectorTable,
        };

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext {
                tool_defs: &tools,
                ..Default::default()
            },
            &TurnAnchors::default(),
            Some(budget),
            &default_estimate,
        );

        let system = result
            .iter()
            .find(|m| m.role == Role::System)
            .expect("system message must be present");
        assert!(
            system
                .content
                .contains(&format!("There are {} tools across", tools.len())),
            "demoted system block must include 'There are <N> tools' wording, \
             got: {:?}",
            system.content
        );
        assert!(
            !system
                .content
                .contains(&format!("Available tools in this turn: {}", tools[0].name)),
            "demoted system block must not enumerate every tool name, got: {:?}",
            system.content
        );
    }

    #[test]
    fn system_block_full_when_under_threshold() {
        use crate::ports::llm::BudgetSource;
        // Generous budget + tiny tool list — the full enumeration must be
        // preserved verbatim.
        let tools = vec![ToolDefinition::new(
            "ping",
            "Ping a host",
            serde_json::json!({"type": "object"}),
        )];
        let msgs = vec![Message::new(Role::User, "hi")];
        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext {
                tool_defs: &tools,
                ..Default::default()
            },
            &TurnAnchors::default(),
            Some(budget),
            &default_estimate,
        );

        let system = result
            .iter()
            .find(|m| m.role == Role::System)
            .expect("system message must be present");
        assert!(
            system
                .content
                .contains("Available tools in this turn: ping."),
            "full enumeration must be preserved, got: {:?}",
            system.content
        );
        assert!(
            !system.content.contains("There are 1 tools across"),
            "demoted summary must not appear when under threshold, got: {:?}",
            system.content
        );
    }

    #[test]
    fn system_block_full_when_no_budget() {
        // No budget installed — the threshold check is skipped and the
        // full enumeration is returned regardless of how many tools there
        // are. Preserves backward compatibility for test contexts and
        // background jobs that don't route through `with_context_budget`.
        let tools = make_huge_tool_set(60, 64);
        let msgs = vec![Message::new(Role::User, "hi")];

        let result = assemble_for_test(
            &ConversationView {
                messages: &msgs,
                ..Default::default()
            },
            &ToolContext {
                tool_defs: &tools,
                ..Default::default()
            },
            &TurnAnchors::default(),
            None,
            &default_estimate,
        );

        let system = result
            .iter()
            .find(|m| m.role == Role::System)
            .expect("system message must be present");
        // Look for the first tool name in the enumeration — its presence
        // proves the full listing was emitted rather than the demoted
        // summary.
        assert!(
            system.content.contains(tools[0].name.as_str()),
            "full enumeration must be present when no budget installed"
        );
        assert!(
            !system
                .content
                .contains(&format!("There are {} tools across", tools.len())),
            "demoted summary must not appear when no budget installed"
        );
    }

    // --- Tool execution-locality (issue #243) ------------------------------

    /// Build a context that relies on the **transport** co-location heuristic
    /// (`co_located: None`), i.e. an older client that reported no system id.
    /// This preserves the Phase-1 (#243) behaviour the existing assertions
    /// cover.
    fn locality_ctx(
        transport: TransportKind,
        host: &str,
        server_names: &[&str],
        client_names: &[&str],
    ) -> ToolLocalityContext {
        ToolLocalityContext {
            co_located: None,
            transport,
            host: host.to_string(),
            daemon_on_workstation: true,
            client_label: "your device".to_string(),
            server_tool_names: server_names.iter().map(|s| s.to_string()).collect(),
            client_tool_names: client_names.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Build a context with an authoritative system-id co-location result
    /// (#248). `co_located` overrides the transport heuristic — used to assert
    /// id-match co-locates even over WebSocket, and id-mismatch keeps localities
    /// distinct even over a "local" transport.
    fn locality_ctx_with_id(
        co_located: bool,
        transport: TransportKind,
        host: &str,
        server_names: &[&str],
        client_names: &[&str],
    ) -> ToolLocalityContext {
        ToolLocalityContext {
            co_located: Some(co_located),
            transport,
            host: host.to_string(),
            daemon_on_workstation: true,
            client_label: "your device".to_string(),
            server_tool_names: server_names.iter().map(|s| s.to_string()).collect(),
            client_tool_names: client_names.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn resolve_localities_co_located_collapses_duplicates() {
        // `terminal` exists on both server and client; `voice_stop` is
        // client-only; `kb_search` is server-only. Co-located (UDS) ⇒ the
        // duplicate collapses to the single server-side tool.
        let ctx = locality_ctx(
            TransportKind::Uds,
            "daemon-host",
            &["terminal", "kb_search"],
            &["terminal", "voice_stop"],
        );
        let entries = resolve_tool_localities(&["terminal", "kb_search", "voice_stop"], &ctx);
        // terminal exists both sides → only the server entry survives.
        let terminal: Vec<_> = entries.iter().filter(|e| e.name == "terminal").collect();
        assert_eq!(
            terminal.len(),
            1,
            "co-located duplicate must collapse to one"
        );
        assert!(terminal[0].locality.is_server());
        // voice_stop is client-only → present once, client locality.
        let voice: Vec<_> = entries.iter().filter(|e| e.name == "voice_stop").collect();
        assert_eq!(voice.len(), 1);
        assert!(voice[0].locality.is_client());
        // kb_search is server-only.
        assert!(
            entries
                .iter()
                .any(|e| e.name == "kb_search" && e.locality.is_server())
        );
    }

    #[test]
    fn a_shadowed_client_tool_is_not_advertised_as_an_alternative() {
        // `terminal` on both machines, over a remote transport. The turn loop
        // offers the model only the server-side definition and routes the name
        // to the server executor, so naming a client alternative would point at
        // a tool that cannot be called.
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal", "kb_search"],
            &["terminal"],
        );
        let entries = resolve_tool_localities(&["terminal", "kb_search"], &ctx);
        let terminal: Vec<_> = entries.iter().filter(|e| e.name == "terminal").collect();
        assert_eq!(
            terminal.len(),
            1,
            "a shadowed client tool must not be advertised: {terminal:?}"
        );
        assert!(terminal[0].locality.is_server());
        assert!(terminal[0].primary);

        // And the rendered note carries no phantom alternative.
        let rendered = render_locality_list(&entries, false);
        assert!(
            !rendered.contains("(alternative)"),
            "no unreachable alternative may be named: {rendered}"
        );
    }

    #[test]
    fn resolve_localities_id_match_co_locates_over_websocket() {
        // #248: an authoritative system-id MATCH co-locates even on WebSocket —
        // overriding the transport heuristic (which would treat WS as remote).
        // The duplicate `terminal` collapses to the single server-side tool.
        let ctx = locality_ctx_with_id(
            true,
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal", "kb_search"],
            &["terminal"],
        );
        let entries = resolve_tool_localities(&["terminal", "kb_search"], &ctx);
        let terminal: Vec<_> = entries.iter().filter(|e| e.name == "terminal").collect();
        assert_eq!(
            terminal.len(),
            1,
            "id-match must co-locate (collapse the duplicate) even over WebSocket"
        );
        assert!(terminal[0].locality.is_server());
    }

    #[test]
    fn resolve_localities_id_mismatch_still_shadows_the_client_twin() {
        // #248: an authoritative system-id MISMATCH keeps the two machines
        // distinct even over a nominally-local transport. That changes the
        // topology, not what can be dispatched: the client `terminal` is still
        // shadowed by the server one, so only the server entry is advertised.
        let ctx = locality_ctx_with_id(
            false,
            TransportKind::Uds,
            "daemon-host",
            &["terminal"],
            &["terminal"],
        );
        let entries = resolve_tool_localities(&["terminal"], &ctx);
        let terminal: Vec<_> = entries.iter().filter(|e| e.name == "terminal").collect();
        assert_eq!(terminal.len(), 1);
        assert!(terminal[0].locality.is_server() && terminal[0].primary);
        // The distinct-machines finding still shows in the context itself, so
        // the topology section and the client-only tools remain correct.
        assert!(!ctx.is_co_located());
    }

    #[test]
    fn a_client_only_tool_keeps_its_client_locality_on_every_transport() {
        // Shadowing applies to a name the server also holds. A client-only
        // capability is unaffected and stays tagged to the user's machine.
        for (co_located, transport) in [
            (Some(false), TransportKind::WebSocket),
            (Some(true), TransportKind::Uds),
            (None, TransportKind::WebSocket),
        ] {
            let ctx = ToolLocalityContext {
                co_located,
                transport,
                host: "daemon-host".to_string(),
                daemon_on_workstation: true,
                client_label: "user-laptop".to_string(),
                server_tool_names: vec!["kb_search".to_string()],
                client_tool_names: vec!["device_terminal".to_string()],
            };
            let entries = resolve_tool_localities(&["device_terminal", "kb_search"], &ctx);
            assert_eq!(entries.len(), 2);
            assert!(
                entries[0].locality.is_client(),
                "a client-only tool stays on the client for {transport:?}"
            );
            assert!(entries[1].locality.is_server());
        }
    }

    #[test]
    fn resolve_localities_no_id_falls_back_to_transport() {
        // #248: with no system id reported (`co_located: None`), co-location is
        // the Phase-1 transport heuristic — WebSocket remote, UDS local. This is
        // the backward-compat path for older clients. The signal shows in how a
        // client-only tool is labelled: a remote connection names the user's
        // machine, a co-located one lists a plain name.
        let ws = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["kb_search"],
            &["device_terminal"],
        );
        assert!(!ws.is_co_located(), "WebSocket must fall back to remote");
        let rendered = render_locality_list(
            &resolve_tool_localities(&["device_terminal"], &ws),
            ws.is_co_located(),
        );
        assert!(
            rendered.contains("your device"),
            "a remote connection labels the user's machine: {rendered}"
        );

        let uds = locality_ctx(
            TransportKind::Uds,
            "daemon-host",
            &["kb_search"],
            &["device_terminal"],
        );
        assert!(uds.is_co_located(), "UDS must fall back to co-located");
        let rendered = render_locality_list(
            &resolve_tool_localities(&["device_terminal"], &uds),
            uds.is_co_located(),
        );
        assert_eq!(
            rendered, "device_terminal",
            "one machine draws no per-machine distinction"
        );
    }

    #[test]
    fn build_tool_note_co_located_omits_locality_labels() {
        // Co-located: plain name list, no "server '...'" / "your device" labels.
        let tools = vec![
            ToolDefinition::new("terminal", "run", serde_json::json!({})),
            ToolDefinition::new("kb_search", "search", serde_json::json!({})),
        ];
        let ctx = locality_ctx(
            TransportKind::Uds,
            "daemon-host",
            &["terminal", "kb_search"],
            &[],
        );
        let note = build_full_tool_note(&tools, &[], Some(&ctx));
        assert!(note.contains("Available tools in this turn: terminal, kb_search."));
        assert!(!note.contains("server 'daemon-host'"), "note: {note}");
        assert!(!note.contains("your device"), "note: {note}");
    }

    #[test]
    fn build_tool_note_remote_labels_each_tool_by_machine() {
        // Two machines: every tool carries the machine it runs on, so the model
        // reads the location beside the name. How to choose between them is
        // stated once, in the topology section, not repeated here.
        let tools = vec![
            ToolDefinition::new("terminal", "run", serde_json::json!({})),
            ToolDefinition::new("device_terminal", "run here", serde_json::json!({})),
        ];
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal"],
            &["device_terminal"],
        );
        let note = build_full_tool_note(&tools, &[], Some(&ctx));
        assert!(
            note.contains("terminal — server 'daemon-host'"),
            "note: {note}"
        );
        assert!(
            note.contains("device_terminal — your device 'your device'")
                || note.contains("device_terminal — your device"),
            "note: {note}"
        );
    }

    #[test]
    fn build_tool_note_remote_client_only_labels_without_routing_hint() {
        // A client-only capability over a remote transport still gets a
        // locality label, but there's no duplicated capability so no routing
        // hint is emitted.
        let tools = vec![ToolDefinition::new(
            "voice_stop",
            "stop",
            serde_json::json!({}),
        )];
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &[],
            &["voice_stop"],
        );
        let note = build_full_tool_note(&tools, &[], Some(&ctx));
        assert!(note.contains("voice_stop — your device"), "note: {note}");
        assert!(
            !note.contains("ask which machine"),
            "no routing hint without a duplicated capability: {note}"
        );
    }

    #[test]
    fn build_tool_note_none_locality_is_plain_list() {
        // No locality context (legacy callers) → byte-identical plain list.
        let tools = vec![ToolDefinition::new(
            "terminal",
            "run",
            serde_json::json!({}),
        )];
        let with_none = build_full_tool_note(&tools, &[], None);
        let co_located = locality_ctx(TransportKind::Uds, "daemon-host", &["terminal"], &[]);
        let with_local = build_full_tool_note(&tools, &[], Some(&co_located));
        assert_eq!(
            with_none, with_local,
            "co-located note must match the no-context plain list"
        );
        assert!(with_none.contains("Available tools in this turn: terminal."));
    }
}
