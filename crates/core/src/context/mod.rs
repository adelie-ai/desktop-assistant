//! Context-window management for the conversation handler.
//!
//! This module groups three related concerns that conspire to keep the
//! prompt under the model's input-token budget:
//!
//! - **Assembly** ([`llm_messages_for_turn`], [`assemble_messages_inner`]):
//!   Builds the per-turn `Vec<Message>` from conversation history,
//!   summaries, tool definitions, and the active-task anchor — applying
//!   pre-flight token-budget checks and shrinking the window when the
//!   estimated cost exceeds the threshold.
//! - **Recovery** ([`recover_from_overflow`]): When the provider rejects a
//!   turn with [`crate::CoreError::ContextOverflow`], runs a structured
//!   recovery ladder (truncate the largest tool result → trim oldest tool
//!   pairs → summarise-and-shrink) before the dispatch loop retries.
//! - **Summarisation** ([`generate_context_summary`]): Asks the LLM for a
//!   bullet-point summary of dropped messages and merges it with any
//!   existing rolling summary, so windowed-out history is not lost.
//!
//! Constants exposed here are tuning knobs read by the dispatch loop in
//! `service.rs` to mirror this module's defaults (e.g., the floor on
//! window size, the compaction-token-pressure threshold).

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
/// [`llm_messages_for_turn`] when the assembled prompt exceeds the budget.
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
    pub tool_rounds_since_anchor: u32,
}

/// Build the message list for a single turn, optionally enforcing a
/// pre-flight token budget by shrinking the window before any LLM call.
///
/// Why a separate wrapper around [`assemble_messages_inner`]: assembly is
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
pub(crate) fn llm_messages_for_turn_with_plan(
    conversation: &ConversationView,
    tools: &ToolContext,
    anchors: &TurnAnchors,
    max_messages: usize,
    system_refinement: &str,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> Vec<Message> {
    // One assembly pass at a given window size. The only thing that varies
    // across the shrink loop is `max_messages`, so everything else is captured
    // once here and the two call sites collapse to `assemble(current_max)`.
    let assemble = |max: usize| {
        assemble_messages_inner(
            conversation,
            tools,
            anchors,
            max,
            system_refinement,
            budget,
            estimate,
        )
    };

    let mut current_max = max_messages;
    let mut assembled = assemble(current_max);

    let Some(active_budget) = budget else {
        return assembled;
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
        let message_tokens: u64 = assembled.iter().map(|m| estimate(&m.content)).sum();
        let assembled_tokens = message_tokens + tool_schema_tokens;
        if assembled_tokens <= threshold {
            return assembled;
        }
        // Already at the floor — further halving has no effect, so stop.
        if current_max <= MIN_CONTEXT_MESSAGES {
            return assembled;
        }
        let new_max = (current_max / 2).max(MIN_CONTEXT_MESSAGES);
        if new_max == current_max {
            return assembled;
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

    assembled
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
    /// Label shown for a client tool's machine in the remote tool note (e.g.
    /// `your device`, or a hostname the client reported in the handshake, #248).
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
    fn is_co_located(&self) -> bool {
        self.co_located
            .unwrap_or_else(|| self.transport.is_co_located())
    }

    fn is_server(&self, name: &str) -> bool {
        self.server_tool_names.iter().any(|n| n == name)
    }

    fn is_client(&self, name: &str) -> bool {
        self.client_tool_names.iter().any(|n| n == name)
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
/// Behaviour:
/// - **Co-located** (UDS / D-Bus): the client and daemon are the same machine,
///   so a server tool and a client tool with the same name are physically the
///   same capability. We collapse them — keep the server-side tool, drop the
///   duplicate client one — so the LLM sees one tool per capability and the
///   note carries no confusing per-machine distinction.
/// - **Remote** (WebSocket): server and client are distinct hosts. Both tools
///   of a duplicated capability are exposed, each tagged with its locality, and
///   the **server-side** one is marked primary (daemon-side execution is the
///   safe default; the prompt rule tells the model to prefer the client tool
///   for work on the user's own device and to ask when genuinely ambiguous).
///
/// Tool order is preserved (server entries keep their position; in the remote
/// case the matching client entry is appended right after its server twin).
pub(crate) fn resolve_tool_localities(
    tool_names: &[&str],
    ctx: &ToolLocalityContext,
) -> Vec<ToolLocalityEntry> {
    let co_located = ctx.is_co_located();
    let mut entries: Vec<ToolLocalityEntry> = Vec::with_capacity(tool_names.len());

    for &name in tool_names {
        let is_server = ctx.is_server(name);
        let is_client = ctx.is_client(name);
        let duplicated = is_server && is_client;

        if duplicated {
            // Capability on both machines. Co-located ⇒ the two are physically
            // the same, so keep only the server-side tool. Remote ⇒ expose both
            // with the server tool as the primary and the client tool as the
            // labelled alternative.
            entries.push(ToolLocalityEntry {
                name: name.to_string(),
                locality: ToolLocality::server(&ctx.host),
                primary: true,
            });
            if !co_located {
                entries.push(ToolLocalityEntry {
                    name: name.to_string(),
                    locality: ToolLocality::client(name, &ctx.client_label),
                    primary: false,
                });
            }
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
                format!("{} — your device '{label}'{alt}", e.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Concise test entry point for assembly: takes the grouped inputs and fills
/// in the two values every test holds constant (`MAX_CONTEXT_MESSAGES` and no
/// system refinement). Adding a per-turn field to any input struct leaves this
/// helper and its callers untouched — the whole point of the grouping.
#[cfg(test)]
fn assemble_for_test(
    conversation: &ConversationView,
    tools: &ToolContext,
    anchors: &TurnAnchors,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> Vec<Message> {
    llm_messages_for_turn_with_plan(
        conversation,
        tools,
        anchors,
        MAX_CONTEXT_MESSAGES,
        "",
        budget,
        estimate,
    )
}

/// Build the full tool-availability note enumerating every tool name and
/// the deferred-namespace index. Returned by default; demoted to a
/// namespace-only summary by [`build_demoted_tool_note`] when the
/// assembled system block exceeds [`SYSTEM_BLOCK_BUDGET_RATIO`].
///
/// When `locality` is `Some`, tools are tagged with where they run (issue
/// #243): co-located connections (UDS / D-Bus) get a plain list because
/// everything is on this machine, while a remote (WebSocket) connection gets
/// per-tool locality labels and a short routing hint. When `None` (callers
/// that don't thread a transport context) the listing is the plain name list,
/// byte-identical to the pre-#243 behaviour.
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
        let (names, remote_routing_hint) = match locality {
            Some(ctx) => {
                let tool_names: Vec<&str> = tool_defs.iter().map(|t| t.name.as_str()).collect();
                let entries = resolve_tool_localities(&tool_names, ctx);
                let co_located = ctx.is_co_located();
                let rendered = render_locality_list(&entries, co_located);
                // Only emit a routing hint when a capability is genuinely
                // duplicated across distinct machines (remote case).
                let has_remote_dup =
                    !co_located && entries.iter().any(|e| !e.primary && e.locality.is_client());
                let hint = if has_remote_dup {
                    " Some capabilities exist on both the server and your device — \
                     prefer the tool on your device for work on your own machine, the \
                     server tool for daemon-side work, and ask which machine when it's \
                     genuinely ambiguous."
                } else {
                    ""
                };
                (rendered, hint)
            }
            None => (
                tool_defs
                    .iter()
                    .map(|t| t.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                "",
            ),
        };
        if has_tool_search {
            note = format!(
                "Available tools in this turn: {names}.{remote_routing_hint} \
                 Additional tools may be available — use builtin_tool_search to discover \
                 tools for tasks not covered by the tools listed above."
            );
        } else {
            note = format!("Available tools in this turn: {names}.{remote_routing_hint}");
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
fn build_demoted_tool_note(
    tool_defs: &[ToolDefinition],
    deferred_namespaces: &[ToolNamespace],
) -> String {
    let total_tools: usize = tool_defs.len()
        + deferred_namespaces
            .iter()
            .map(|ns| ns.tools.len())
            .sum::<usize>();
    let namespace_count = deferred_namespaces.len();
    format!(
        "There are {total_tools} tools across {namespace_count} namespaces. \
         Use builtin_tool_search to discover a tool for any task you need."
    )
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
fn assemble_system_instruction(tool_note: String, system_refinement: &str) -> String {
    use crate::prompts::{self, PromptSection, PromptSectionKind};
    let mut sections = prompts::static_sections();

    // Personality disposition (#226): read the active personality the same way
    // the per-turn refinement is read (a task-local installed by the daemon
    // dispatch wrapper). Injected *before* the tool note and the per-turn
    // refinement so the standing disposition is established up front while a
    // one-turn refinement can still adjust tone last. Always rendered — the
    // blurb at minimum carries the adaptation clause — so every turn carries a
    // personality, with the default disposition for callers that install no
    // scope.
    let personality_blurb = crate::prompts::render_blurb(&crate::ports::llm::current_personality());
    if !personality_blurb.trim().is_empty() {
        sections.push(PromptSection::new(
            PromptSectionKind::Personality,
            personality_blurb,
        ));
    }

    sections.push(PromptSection::new(
        PromptSectionKind::ToolAvailability,
        tool_note,
    ));
    let trimmed = system_refinement.trim();
    if !trimmed.is_empty() {
        sections.push(PromptSection::new(
            PromptSectionKind::SystemRefinement,
            trimmed.to_string(),
        ));
    }
    prompts::assemble(&sections)
}

// Why allow: this builder coordinates several independent prompt slices
// (windowed messages, summaries, tool sets, context summary, anchor) that
// don't naturally cluster into a single struct. Bundling them just to
// satisfy the lint would obscure the code at every call site.
fn assemble_messages_inner(
    conversation: &ConversationView,
    tools: &ToolContext,
    anchors: &TurnAnchors,
    max_messages: usize,
    system_refinement: &str,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> Vec<Message> {
    // Destructure into the local names the body below already uses, so the
    // assembly logic reads unchanged after the parameter grouping.
    let ConversationView {
        messages: conversation_messages,
        summaries,
        context_summary,
    } = *conversation;
    let ToolContext {
        tool_defs,
        deferred_namespaces,
        locality,
    } = *tools;
    let TurnAnchors {
        active_task,
        plan,
        scratchpad_index,
        tool_rounds_since_anchor,
    } = *anchors;

    let tool_note = build_full_tool_note(tool_defs, deferred_namespaces, locality);
    let system_instruction = assemble_system_instruction(tool_note, system_refinement);

    // Measure the system block. The system instruction is always
    // re-included in every turn, so any space it claims is permanently
    // displaced from conversation history. When a budget is installed,
    // record the size on every turn (observability) and demote the tool
    // listing to a namespace-only summary if the block exceeds
    // `SYSTEM_BLOCK_BUDGET_RATIO`.
    let system_instruction = if let Some(b) = budget {
        let system_tokens_before = estimate(&system_instruction);
        tracing::info!(
            system_tokens = system_tokens_before,
            budget = b.max_input_tokens,
            ratio = (system_tokens_before as f64 / b.max_input_tokens as f64),
            "system block size"
        );
        let threshold = (b.max_input_tokens as f64 * SYSTEM_BLOCK_BUDGET_RATIO) as u64;
        if system_tokens_before > threshold {
            let demoted_note = build_demoted_tool_note(tool_defs, deferred_namespaces);
            let demoted_system = assemble_system_instruction(demoted_note, system_refinement);
            let system_tokens_after = estimate(&demoted_system);
            tracing::warn!(
                original_tokens = system_tokens_before,
                demoted_tokens = system_tokens_after,
                budget = b.max_input_tokens,
                "system block exceeded budget threshold; demoted tool listing"
            );
            demoted_system
        } else {
            system_instruction
        }
    } else {
        system_instruction
    };

    // Apply context windowing: if the conversation exceeds the limit, keep
    // only the most recent messages, snapping the cut point forward to a
    // genuine User message so we never split tool-call/result pairs.
    let start = window_start(conversation_messages, max_messages);
    let windowed = &conversation_messages[start..];
    let is_windowed = start > 0;

    // Track which summary IDs are active so we know to skip their messages.
    let active_summary_ids: std::collections::HashSet<&str> =
        summaries.iter().map(|s| s.id.as_str()).collect();

    let mut messages = Vec::with_capacity(windowed.len() + 2);
    messages.push(Message::new(Role::System, system_instruction));

    // Ambient "now": a tiny, always-present line giving the assistant a
    // sense of the current date/time without spending a `builtin_sys_props`
    // tool round to find out. Installed per turn by the daemon dispatch wrapper
    // as a task-local (rendered from the same `NowSnapshot` that backs the
    // tool, so the two never disagree) and empty for callers that don't route
    // through it (tests, dreaming jobs). Pushed here as a per-turn system
    // message — deliberately NOT folded into the cached system instruction
    // above — so the volatile timestamp never busts the prompt-prefix cache.
    let now_context = crate::ports::llm::current_now_context();
    if !now_context.is_empty() {
        messages.push(Message::new(Role::System, format!("[Now] {now_context}")));
    }

    // Inject rolling context summary when windowing is active and summary exists.
    if is_windowed && !context_summary.is_empty() {
        messages.push(Message::new(
            Role::System,
            format!("[Summary of earlier conversation]\n{context_summary}"),
        ));
    }

    // Re-inject the active-task anchor when the original prompt has drifted
    // out of the model's view. Three triggers, any one of which is enough:
    //   1. Windowing is active and the anchor user message has been windowed
    //      out (heuristic: no User message with matching content in `windowed`).
    //   2. The anchor user message is still in `windowed` but has been
    //      collapsed behind an active summary (its `summary_id` is set), so
    //      the model only sees the summary text in this turn.
    //   3. The dispatch loop has gone through more than
    //      `ACTIVE_TASK_ROUND_THRESHOLD` tool rounds in the current turn — even
    //      if the anchor is still visible, surfacing it again keeps the model
    //      on-task during long agentic loops.
    //
    // Why: a long tool-calling session can bury the user's goal under many
    // tool results; an explicit `[Current task]` re-statement keeps the
    // assistant aligned with the original intent across compaction and
    // windowing events.
    // Shared "context is starting to drop" signal used both by the
    // `[Current task]` re-injection and the `[Scratchpad]` index (#340): once a
    // long agentic loop has run past the round threshold, surfacing durable
    // anchors again keeps the model on-task even if they're nominally visible.
    let many_tool_rounds = tool_rounds_since_anchor > ACTIVE_TASK_ROUND_THRESHOLD;

    if let Some(task) = active_task.filter(|t| !t.is_empty()) {
        // Find a non-collapsed User message in the window whose content
        // matches the anchor. Messages with an active `summary_id` are
        // about to be replaced by summary text below, so they don't count
        // as "visible" for the purpose of this check.
        let anchor_visible = windowed.iter().any(|m| {
            m.role == Role::User
                && m.content == task
                && !m
                    .summary_id
                    .as_deref()
                    .is_some_and(|sid| active_summary_ids.contains(sid))
        });

        if !anchor_visible || many_tool_rounds {
            messages.push(Message::new(Role::System, format!("[Current task] {task}")));
        }
    }

    // Surface the open plan (#240) right after the task anchor. The dispatch
    // loop renders the conversation's `todo` notes into a compact tree each
    // round, so the plan stays in view (cheap) while the verbose raw work that
    // produced it is evicted from the message log. Request-scoped — never
    // persisted to `conv.messages`.
    if let Some(plan) = plan.filter(|p| !p.is_empty()) {
        messages.push(Message::new(Role::System, format!("[Plan]\n{plan}")));
    }

    // Advertise the free-form scratchpad note keys (#340) right after the plan.
    // These notes are durable in storage but otherwise invisible once the
    // message that wrote them is windowed/compacted away — nothing re-surfaces a
    // general note, so the model never thinks to `builtin_scratchpad_search` for
    // it. The index lists the keys (recognition over recall), gated on the SAME
    // "context is dropping" condition as `[Current task]`: windowing has begun
    // (which also covers collapse-behind-summary, since summaries are only
    // injected when windowed) OR the turn has run past the round threshold.
    // Before that, the note content is usually still in the live conversation,
    // so the index would only burn tokens.
    if let Some(index) = scratchpad_index.filter(|s| !s.is_empty())
        && (is_windowed || many_tool_rounds)
    {
        messages.push(Message::new(Role::System, format!("[Scratchpad] {index}")));
    }

    // Track which summaries have already been injected.
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
                    // Recover the absolute ordinal range from the message
                    // positions tagged with this summary_id. The window may
                    // not contain the full range; fall back to a
                    // range-less label when no tagged message is visible.
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
                    messages.push(Message::new(Role::System, body));
                }
            }
            continue;
        }

        messages.push(msg.clone());
    }

    messages
}

/// Locate the largest `Role::Tool` message whose `content` length (bytes)
/// is at least `min_bytes`. Returns `None` if no tool message clears the
/// threshold — small tool results aren't worth truncating because the
/// truncation notice may be larger than the original.
///
/// Why estimated tokens (not bytes): non-ASCII payloads (CJK, emoji,
/// JSON-with-deep-escapes, base64) have wildly different byte-vs-token
/// ratios. Sorting by bytes mis-targets those cases. Step 1 of
/// [`recover_from_overflow`] aims to free the most prompt-token budget,
/// not the most filesystem bytes, so we measure with the same currency
/// the LLM pays in.
fn find_largest_tool_result_above(
    messages: &[Message],
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
            let tokens = estimate(&m.content);
            if tokens >= min_tokens {
                Some((i, tokens))
            } else {
                None
            }
        })
        .max_by_key(|(_, tokens)| *tokens)
        .map(|(i, _)| i)
}

/// Outcome of [`trim_tool_pairs`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct TrimResult {
    /// Total messages removed from the list.
    total_removed: usize,
    /// How many of the removed messages lay at indices `< compacted_through`,
    /// i.e. inside the already-summarized prefix. The caller decrements
    /// `compacted_through` by exactly this — never by `total_removed` — so the
    /// marker keeps pointing at the same logical boundary (DA-11 / #298).
    removed_before_marker: usize,
}

/// Remove the oldest assistant(tool_calls)+tool_result groups from a message
/// list to reduce context size. Keeps the first user message and the most
/// recent tool interaction intact.
///
/// `compacted_through` is the caller's summary boundary: messages at indices
/// `< compacted_through` have already been folded into the rolling context
/// summary. Removing them shifts the boundary, but removing messages *after* it
/// does not — so the returned [`TrimResult::removed_before_marker`] counts only
/// the removals inside the summarized prefix. The previous code decremented
/// `compacted_through` by the *total* removed, which re-summarized
/// already-summarized messages whenever a removed group lay past the marker
/// (DA-11 / #298).
fn trim_tool_pairs(messages: &mut Vec<Message>, compacted_through: usize) -> TrimResult {
    // Find ranges of (assistant-with-tool-calls, tool_result, ..., tool_result)
    // groups and remove roughly the oldest half.
    let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        if messages[i].role == Role::Assistant && !messages[i].tool_calls.is_empty() {
            let start = i;
            i += 1;
            while i < messages.len() && messages[i].role == Role::Tool {
                i += 1;
            }
            groups.push(start..i);
        } else {
            i += 1;
        }
    }

    if groups.len() <= 1 {
        // Nothing safe to remove — keep the most recent group
        return TrimResult::default();
    }

    // Remove the oldest half of groups
    let remove_count = groups.len() / 2;
    let groups_to_remove: Vec<_> = groups[..remove_count].to_vec();

    // Remove in reverse order to keep indices stable. Count, separately, how
    // many removed messages sat inside the already-summarized prefix
    // (`index < compacted_through`) — that, not the total, is the marker shift.
    let mut result = TrimResult::default();
    for range in groups_to_remove.into_iter().rev() {
        result.total_removed += range.len();
        result.removed_before_marker +=
            range.clone().filter(|idx| *idx < compacted_through).count();
        messages.drain(range);
    }

    result
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

/// Ask the LLM to produce a bullet-point summary of dropped messages, merged
/// with any existing summary. Falls back to the existing summary on failure.
pub(crate) async fn generate_context_summary<L: LlmClient>(
    existing_summary: &str,
    messages: &[Message],
    llm: &L,
) -> String {
    // Build a transcript of User/Assistant messages only; skip Tool/System.
    let mut transcript = String::new();
    for msg in messages {
        match msg.role {
            Role::User => {
                transcript.push_str("User: ");
                transcript.push_str(&msg.content);
                transcript.push('\n');
            }
            Role::Assistant if !msg.content.is_empty() => {
                transcript.push_str("Assistant: ");
                if msg.content.len() > 2000 {
                    // Char-boundary-safe cut: a naive byte slice panics when
                    // byte 2000 lands inside a multibyte character (DA-2).
                    transcript.push_str(&planning::truncate_on_char_boundary(&msg.content, 2000));
                    transcript.push_str("...[truncated]");
                } else {
                    transcript.push_str(&msg.content);
                }
                transcript.push('\n');
            }
            _ => {}
        }
    }

    if transcript.is_empty() {
        return existing_summary.to_string();
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
        Ok(response) if !response.text.trim().is_empty() => response.text.trim().to_string(),
        Ok(_) => {
            tracing::warn!("context summary generation returned empty");
            existing_summary.to_string()
        }
        Err(e) => {
            tracing::warn!("context summary generation failed: {e}");
            existing_summary.to_string()
        }
    }
}

/// Recover from a `ContextOverflow` error by reducing prompt size.
///
/// The ladder runs steps in order until one frees space:
///   1. Truncate the largest tool result with a chunking notice (preserves
///      the tool_call/result pair so the model sees what it tried).
///   2. If no tool result is large enough to be worth truncating, trim the
///      oldest tool-pair groups via [`trim_tool_pairs`].
///   3. If nothing to trim, summarise-and-shrink the active window
///      (delegates to the same logic that path A uses on the success branch).
///
/// Why this order: step 1 is the cleanest because it preserves history;
/// step 3 is the last resort. The retry counter in `send_prompt` bounds
/// total attempts across all steps so a persistently-oversized request
/// can't loop indefinitely.
pub(crate) async fn recover_from_overflow<L: LlmClient>(
    conv: &mut Conversation,
    prompt_tokens: Option<u64>,
    max_tokens: Option<u64>,
    target_window: &mut usize,
    task_llm: &L,
    estimate: &(dyn Fn(&str) -> u64 + Send + Sync),
) {
    // Step 1: largest tool result, if it's >= MIN_TRUNCATION_TOKENS.
    if let Some(idx) =
        find_largest_tool_result_above(&conv.messages, MIN_TRUNCATION_TOKENS, estimate)
    {
        let original_bytes = conv.messages[idx].content.len();
        conv.messages[idx].content =
            overflow_truncation_notice(original_bytes, prompt_tokens, max_tokens);
        tracing::warn!(
            tool_result_index = idx,
            original_bytes,
            prompt_tokens = ?prompt_tokens,
            max_tokens = ?max_tokens,
            "context overflow — truncating tool result (step 1)"
        );
        return;
    }

    // Step 2: trim oldest tool-pair groups. Decrement `compacted_through` only
    // by the removals inside the already-summarized prefix — not the total —
    // so trimming groups that lie *after* the marker doesn't drag it backwards
    // and re-summarize messages already folded into the summary (DA-11 / #298).
    let trimmed = trim_tool_pairs(&mut conv.messages, conv.compacted_through);
    if trimmed.total_removed > 0 {
        conv.compacted_through = conv
            .compacted_through
            .saturating_sub(trimmed.removed_before_marker);
        tracing::warn!(
            removed = trimmed.total_removed,
            removed_before_marker = trimmed.removed_before_marker,
            "context overflow — trimmed oldest tool pairs (step 2)"
        );
        return;
    }

    // Step 3: summarise and shrink the active window. Mirrors the
    // proactive token-pressure path on the success branch so the
    // conversation can keep progressing when there's nothing to trim.
    let new_window = (*target_window / 2).max(MIN_CONTEXT_MESSAGES);
    if new_window < *target_window {
        *target_window = new_window;
    }
    if let Some((from, to)) = compaction_range(conv, *target_window) {
        let summary =
            generate_context_summary(&conv.context_summary, &conv.messages[from..to], task_llm)
                .await;
        conv.context_summary = summary;
        conv.compacted_through = to;
        tracing::warn!(
            new_window = *target_window,
            from,
            to,
            "context overflow — summarised and shrank window (step 3)"
        );
    } else {
        tracing::warn!(
            new_window = *target_window,
            "context overflow — no recovery action available"
        );
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

    #[test]
    fn assemble_system_instruction_appends_refinement_last() {
        let base = assemble_system_instruction("TOOLNOTE".to_string(), "");
        let refined =
            assemble_system_instruction("TOOLNOTE".to_string(), "Respond briefly, by voice.");

        // Empty refinement is byte-identical to no refinement.
        assert_eq!(
            base,
            assemble_system_instruction("TOOLNOTE".to_string(), "   "),
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

    #[tokio::test]
    async fn assemble_system_instruction_injects_personality_before_tools_and_refinement() {
        use crate::ports::llm::with_personality;
        use crate::prompts::{Personality, PersonalityLevel};

        // A personality with a recognizable trait so we can locate the
        // injected section in the assembled output.
        let personality = Personality {
            sarcasm: PersonalityLevel::Always,
            ..Personality::default()
        };
        let assembled = with_personality(personality, async {
            assemble_system_instruction("TOOLNOTE".to_string(), "REFINEMENT")
        })
        .await;

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

    #[tokio::test]
    async fn assemble_system_instruction_default_personality_present_without_scope() {
        // No `with_personality` scope installed → the default disposition is
        // still injected (global personality applies to every turn).
        let assembled = assemble_system_instruction("TOOLNOTE".to_string(), "");
        let default_blurb = crate::prompts::render_blurb(&crate::prompts::Personality::default());
        assert!(
            assembled.contains(&default_blurb),
            "default personality must be injected even without a scope:\n{assembled}"
        );
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

    // --- trim_tool_pairs tests ---

    #[test]
    fn trim_tool_pairs_removes_oldest_half() {
        let mut messages = vec![
            Message::new(Role::User, "hello"),
            // Group 1
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "tool_a", "{}")]),
            Message::tool_result("c1", "result_1"),
            // Group 2
            Message::assistant_with_tool_calls(vec![ToolCall::new("c2", "tool_a", "{}")]),
            Message::tool_result("c2", "result_2"),
            // Group 3
            Message::assistant_with_tool_calls(vec![ToolCall::new("c3", "tool_a", "{}")]),
            Message::tool_result("c3", "result_3"),
            // Group 4
            Message::assistant_with_tool_calls(vec![ToolCall::new("c4", "tool_a", "{}")]),
            Message::tool_result("c4", "result_4"),
        ];

        // compacted_through = 0 → nothing was summarized, so the marker
        // adjustment is 0 even though 4 messages are removed.
        let trimmed = trim_tool_pairs(&mut messages, 0);
        // 4 groups, remove oldest half (2 groups = 4 messages)
        assert_eq!(trimmed.total_removed, 4);
        assert_eq!(trimmed.removed_before_marker, 0);
        // Should keep: user + group3 + group4
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].tool_calls[0].id, "c3");
    }

    #[test]
    fn trim_tool_pairs_keeps_single_group() {
        let mut messages = vec![
            Message::new(Role::User, "hello"),
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "tool_a", "{}")]),
            Message::tool_result("c1", "result"),
        ];

        let trimmed = trim_tool_pairs(&mut messages, 0);
        assert_eq!(trimmed.total_removed, 0);
        assert_eq!(trimmed.removed_before_marker, 0);
        assert_eq!(messages.len(), 3);
    }

    // --- DA-11: marker-aware trim so recovery doesn't corrupt compacted_through ---

    #[test]
    fn trim_counts_only_removals_before_the_marker() {
        // Marker at index 4: messages 0..4 (the user msg + group 1 + group 2's
        // first message) are summarized. The two removed groups (1 and 2) span
        // indices 1..5; of those, indices 1,2,3 lie at < 4, so the marker must
        // drop by exactly 3 — NOT by the full 4 removed.
        let mut messages = vec![
            Message::new(Role::User, "hello"), // 0
            // Group 1 (indices 1,2) — fully before marker(4)
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "t", "{}")]), // 1
            Message::tool_result("c1", "r1"),                                         // 2
            // Group 2 (indices 3,4) — straddles the marker(4): index 3 < 4, 4 not
            Message::assistant_with_tool_calls(vec![ToolCall::new("c2", "t", "{}")]), // 3
            Message::tool_result("c2", "r2"),                                         // 4
            // Group 3 (indices 5,6) — kept (most-recent half)
            Message::assistant_with_tool_calls(vec![ToolCall::new("c3", "t", "{}")]), // 5
            Message::tool_result("c3", "r3"),                                         // 6
            // Group 4 (indices 7,8) — kept
            Message::assistant_with_tool_calls(vec![ToolCall::new("c4", "t", "{}")]), // 7
            Message::tool_result("c4", "r4"),                                         // 8
        ];
        let compacted_through = 4;
        let trimmed = trim_tool_pairs(&mut messages, compacted_through);
        assert_eq!(trimmed.total_removed, 4, "two oldest groups removed");
        assert_eq!(
            trimmed.removed_before_marker, 3,
            "indices 1,2 (group1) + index 3 (group2's first msg) lie at < compacted_through(4)"
        );
    }

    #[test]
    fn trim_marker_adjustment_zero_when_all_removals_after_marker() {
        // Marker at 1: only the leading user message is summarized. Every
        // removed tool group lies at indices >= 1, so the marker must NOT move,
        // or already-summarized messages would be re-summarized (the DA-11 bug).
        let mut messages = vec![
            Message::new(Role::User, "hello"), // 0
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "t", "{}")]), // 1
            Message::tool_result("c1", "r1"),  // 2
            Message::assistant_with_tool_calls(vec![ToolCall::new("c2", "t", "{}")]), // 3
            Message::tool_result("c2", "r2"),  // 4
            Message::assistant_with_tool_calls(vec![ToolCall::new("c3", "t", "{}")]), // 5
            Message::tool_result("c3", "r3"),  // 6
            Message::assistant_with_tool_calls(vec![ToolCall::new("c4", "t", "{}")]), // 7
            Message::tool_result("c4", "r4"),  // 8
        ];
        let trimmed = trim_tool_pairs(&mut messages, 1);
        assert_eq!(trimmed.total_removed, 4);
        assert_eq!(
            trimmed.removed_before_marker, 0,
            "removals at indices >= compacted_through must not move the marker"
        );
    }

    #[tokio::test]
    async fn recover_step2_does_not_drag_marker_back_for_post_marker_trims() {
        // End-to-end (DA-11 / #298): a conversation whose tool results are all
        // small (so step 1 doesn't fire) and whose summary marker sits before
        // the trimmed groups. Step 2 must trim but leave `compacted_through`
        // pointing at the same logical boundary, not re-summarize already-
        // summarized messages.
        let mut conv = Conversation::new("c1", "t");
        conv.messages = vec![
            Message::new(Role::User, "hello"), // 0 — summarized
            // Group 1 (1,2)
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "t", "{}")]),
            Message::tool_result("c1", "small"),
            // Group 2 (3,4)
            Message::assistant_with_tool_calls(vec![ToolCall::new("c2", "t", "{}")]),
            Message::tool_result("c2", "small"),
            // Group 3 (5,6) — kept
            Message::assistant_with_tool_calls(vec![ToolCall::new("c3", "t", "{}")]),
            Message::tool_result("c3", "small"),
            // Group 4 (7,8) — kept
            Message::assistant_with_tool_calls(vec![ToolCall::new("c4", "t", "{}")]),
            Message::tool_result("c4", "small"),
        ];
        // Only the leading user message is summarized; every removed group lies
        // after the marker. Pre-fix this dropped to 0 (4 - 4, saturating);
        // post-fix it must stay at 1.
        conv.compacted_through = 1;

        let mut target_window = MAX_CONTEXT_MESSAGES;
        recover_from_overflow(
            &mut conv,
            Some(100_000),
            Some(8_000),
            &mut target_window,
            &FailingLlm,
            &default_estimate,
        )
        .await;

        assert_eq!(conv.messages.len(), 5, "two oldest groups trimmed");
        assert_eq!(
            conv.compacted_through, 1,
            "marker must not move when trimmed groups lie after it"
        );
    }

    #[test]
    fn trim_tool_pairs_no_groups() {
        let mut messages = vec![
            Message::new(Role::User, "hello"),
            Message::new(Role::Assistant, "hi there"),
        ];

        let trimmed = trim_tool_pairs(&mut messages, 0);
        assert_eq!(trimmed.total_removed, 0);
        assert_eq!(trimmed.removed_before_marker, 0);
        assert_eq!(messages.len(), 2);
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

    // --- Pure assembly tests (issue #65 + earlier) ---

    #[test]
    fn llm_messages_for_turn_returns_all_when_under_limit() {
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
    fn llm_messages_for_turn_windows_when_over_limit() {
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
    fn llm_messages_for_turn_snaps_to_user_boundary() {
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

        // The base system prompt dominates (~13.7k chars); size the budget so
        // its threshold (0.85 * budget) clears the full tiny-schema turn but
        // not the fat-schema one (+20k schema chars).
        let budget = ContextBudget {
            max_input_tokens: 22_000,
            source: BudgetSource::ConnectorTable,
        };
        let one_per_char = |s: &str| s.chars().count() as u64;

        let assemble = |tool: &ToolDefinition| {
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

        let with_tiny = assemble(&tiny);
        let with_fat = assemble(&fat);

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
    fn llm_messages_for_turn_injects_summary_when_windowing() {
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
    fn llm_messages_for_turn_omits_summary_when_under_limit() {
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
    fn llm_messages_for_turn_omits_empty_summary_when_windowing() {
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

    // --- Message summary (collapsing) tests ---

    #[test]
    fn llm_messages_for_turn_collapses_summarized_range() {
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
    fn llm_messages_for_turn_no_summaries_passes_through() {
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
    fn llm_messages_for_turn_multiple_summaries() {
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
    fn llm_messages_for_turn_renders_absolute_ordinals_when_windowed() {
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
    fn llm_messages_for_turn_skips_summary_when_all_tagged_messages_outside_window() {
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
        assert_eq!(result, "- Discussed Rust and lifetimes");
    }

    #[tokio::test]
    async fn generate_context_summary_falls_back_on_failure() {
        let messages = vec![
            Message::new(Role::User, "Hello"),
            Message::new(Role::Assistant, "Hi"),
        ];
        let llm = FailingLlm;
        let result = generate_context_summary("existing summary", &messages, &llm).await;
        assert_eq!(result, "existing summary");
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
        assert_eq!(result, "summary of long message");
    }

    #[tokio::test]
    async fn generate_context_summary_returns_existing_for_tool_only_messages() {
        let messages = vec![
            Message::assistant_with_tool_calls(vec![ToolCall::new("c1", "tool_a", "{}")]),
            Message::tool_result("c1", "result"),
        ];
        let llm = MockLlm::new(vec!["should not be called"]);
        let result = generate_context_summary("old summary", &messages, &llm).await;
        assert_eq!(result, "old summary");
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
    fn resolve_localities_remote_exposes_both_with_primary_server() {
        // `terminal` on both server and client + a remote (WebSocket)
        // transport: both are exposed, server is primary, client is the
        // labelled alternative.
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal", "kb_search"],
            &["terminal"],
        );
        let entries = resolve_tool_localities(&["terminal", "kb_search"], &ctx);
        let terminal: Vec<_> = entries.iter().filter(|e| e.name == "terminal").collect();
        assert_eq!(terminal.len(), 2, "remote duplicate must expose both tools");
        // Exactly one is server (primary), one is client (alternative).
        let server = terminal.iter().find(|e| e.locality.is_server()).unwrap();
        let client = terminal.iter().find(|e| e.locality.is_client()).unwrap();
        assert!(
            server.primary,
            "server side is the primary in the remote case"
        );
        assert!(
            !client.primary,
            "client side is the non-primary alternative"
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
    fn resolve_localities_id_mismatch_keeps_distinct_over_local_transport() {
        // #248: an authoritative system-id MISMATCH keeps the localities
        // distinct even on a nominally-local transport — overriding the
        // transport heuristic (which would co-locate). Both `terminal` entries
        // survive, server primary + client alternative.
        let ctx = locality_ctx_with_id(
            false,
            TransportKind::Uds,
            "daemon-host",
            &["terminal"],
            &["terminal"],
        );
        let entries = resolve_tool_localities(&["terminal"], &ctx);
        let terminal: Vec<_> = entries.iter().filter(|e| e.name == "terminal").collect();
        assert_eq!(
            terminal.len(),
            2,
            "id-mismatch must keep both tools distinct even over a local transport"
        );
        assert!(terminal.iter().any(|e| e.locality.is_server() && e.primary));
        assert!(
            terminal
                .iter()
                .any(|e| e.locality.is_client() && !e.primary)
        );
    }

    #[test]
    fn resolve_localities_no_id_falls_back_to_transport() {
        // #248: with no system id reported (`co_located: None`), co-location is
        // the Phase-1 transport heuristic — WS remote (distinct), UDS local
        // (collapsed). This is the backward-compat path for older clients.
        let ws = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal"],
            &["terminal"],
        );
        assert_eq!(
            resolve_tool_localities(&["terminal"], &ws)
                .iter()
                .filter(|e| e.name == "terminal")
                .count(),
            2,
            "no-id + WebSocket must stay remote (transport fallback, distinct tools)"
        );
        let uds = locality_ctx(
            TransportKind::Uds,
            "daemon-host",
            &["terminal"],
            &["terminal"],
        );
        assert_eq!(
            resolve_tool_localities(&["terminal"], &uds)
                .iter()
                .filter(|e| e.name == "terminal")
                .count(),
            1,
            "no-id + UDS must co-locate (transport fallback, collapsed)"
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
    fn build_tool_note_remote_labels_localities_and_routes() {
        // Remote duplicate: per-tool locality labels plus a routing hint. The
        // flat list carries the deduped (server) `terminal`; the context marks
        // it as also client-registered, so the note twins it.
        let tools = vec![ToolDefinition::new(
            "terminal",
            "run",
            serde_json::json!({}),
        )];
        let ctx = locality_ctx(
            TransportKind::WebSocket,
            "daemon-host",
            &["terminal"],
            &["terminal"],
        );
        let note = build_full_tool_note(&tools, &[], Some(&ctx));
        assert!(
            note.contains("terminal — server 'daemon-host'"),
            "note: {note}"
        );
        assert!(note.contains("your device"), "note: {note}");
        // Routing hint for the duplicated capability.
        assert!(
            note.contains("prefer the tool on your device") && note.contains("ask which machine"),
            "remote routing hint must be present: {note}"
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
