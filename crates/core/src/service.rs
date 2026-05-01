use crate::CoreError;
use crate::domain::{
    Conversation, ConversationId, ConversationSummary, Message, MessageSummary, Role,
    ToolDefinition, ToolNamespace,
};
use crate::ports::inbound::ConversationService;
use crate::ports::llm::{
    ChunkCallback, ContextBudget, LlmClient, ReasoningConfig, StatusCallback,
    current_context_budget,
};
use crate::ports::store::ConversationStore;
use crate::ports::tools::ToolExecutor;
use chrono::{Duration, Local};

/// Maximum number of tool-calling rounds before giving up.
const MAX_TOOL_ROUNDS: usize = 200;

/// Default maximum number of conversation messages sent to the LLM per turn.
/// When the conversation exceeds this limit, only the most recent messages
/// are included, with the cut point snapped forward to a genuine `Role::User`
/// message to avoid splitting tool-call/result pairs.
const MAX_CONTEXT_MESSAGES: usize = 40;

/// Lower bound applied when the window is shrunk in response to token pressure.
/// Keeps enough room for at least the current user prompt plus a tool round.
const MIN_CONTEXT_MESSAGES: usize = 8;

/// Minimum number of newly-dropped messages before re-compacting the summary.
const COMPACTION_INTERVAL: usize = 20;

/// Fraction of the model's prompt-token budget at which proactive compaction
/// triggers. Checked against `LlmResponse.usage.input_tokens` after each
/// successful LLM call.
const COMPACTION_TOKEN_RATIO: f64 = 0.85;

/// Maximum number of `CoreError::ContextOverflow` recoveries allowed within
/// a single `send_prompt` call. Each recovery applies one step of the
/// context-recovery ladder; if successive calls still overflow we surface
/// the error rather than loop.
const MAX_OVERFLOW_RETRIES: u32 = 3;

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
const MIN_TRUNCATION_TOKENS: u64 = 1024;

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

/// Fraction of the prompt-token budget below which the full tool listing
/// is considered cheap enough to enumerate without LLM-driven
/// categorization. When the raw `(name + description)` cost of every tool
/// fits within this slice, [`categorize_tool_namespaces`] returns the
/// input unchanged and skips the categorization round-trip.
///
/// Why 0.10: leaves headroom under [`SYSTEM_BLOCK_BUDGET_RATIO`] (the
/// demotion threshold for the assembled system block) so a listing that
/// passes this check is also unlikely to trigger demotion at assembly
/// time. Above the threshold the categorization LLM call is worth the
/// expense — it compresses the listing — but below it the round-trip
/// just adds latency and tokens.
const FULL_LISTING_FIT_RATIO: f64 = 0.10;

/// Build the replacement content used when a tool result is truncated in
/// response to a `ContextOverflow` error. The text is addressed to the
/// model so it learns to chunk subsequent requests more narrowly.
fn overflow_truncation_notice(
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

fn now_timestamp() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn cutoff_timestamp(max_age_days: u32) -> String {
    (Local::now() - Duration::days(i64::from(max_age_days)))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

/// Number of consecutive tool rounds within a single `send_prompt` call after
/// which the active-task anchor must be re-injected even if it is still in
/// the windowed message list. Why: long agentic loops drift away from the
/// goal; surfacing it again every few rounds keeps the model on-task.
const ACTIVE_TASK_ROUND_THRESHOLD: u32 = 5;

/// Maximum number of pre-flight shrink iterations attempted by
/// `llm_messages_for_turn` when the assembled prompt exceeds the budget.
/// Why bounded: each iteration halves the message window, so 5 iterations
/// already drop the count by 32x — enough to reach `MIN_CONTEXT_MESSAGES`
/// from any plausible starting point. The bound also guarantees termination
/// regardless of estimator behaviour.
const MAX_PREFLIGHT_SHRINK_ITERATIONS: u32 = 5;

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
// Why allow: the inner builder coordinates several independent prompt slices
// (windowed messages, summaries, tool sets, context summary, anchor); the
// outer wrapper threads the same set plus the budget pair. Bundling them
// just to satisfy the lint would obscure the code at every call site.
#[allow(clippy::too_many_arguments)]
fn llm_messages_for_turn(
    conversation_messages: &[Message],
    summaries: &[MessageSummary],
    tool_defs: &[ToolDefinition],
    deferred_namespaces: &[ToolNamespace],
    context_summary: &str,
    max_messages: usize,
    active_task: Option<&str>,
    tool_rounds_since_anchor: u32,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> Vec<Message> {
    let mut current_max = max_messages;
    let mut assembled = assemble_messages_inner(
        conversation_messages,
        summaries,
        tool_defs,
        deferred_namespaces,
        context_summary,
        current_max,
        active_task,
        tool_rounds_since_anchor,
        budget,
        estimate,
    );

    let Some(budget) = budget else {
        return assembled;
    };

    // Pre-flight token estimate: sum the cost of every assembled message's
    // body. The threshold mirrors `COMPACTION_TOKEN_RATIO` used by the
    // post-call token-pressure path so the two checks agree on what
    // counts as "near the limit".
    let max_input_tokens = budget.max_input_tokens;
    let threshold = (max_input_tokens as f64 * COMPACTION_TOKEN_RATIO) as u64;

    for _ in 0..MAX_PREFLIGHT_SHRINK_ITERATIONS {
        let assembled_tokens: u64 = assembled
            .iter()
            .map(|m| estimate(&m.content))
            .sum();
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
        assembled = assemble_messages_inner(
            conversation_messages,
            summaries,
            tool_defs,
            deferred_namespaces,
            context_summary,
            current_max,
            active_task,
            tool_rounds_since_anchor,
            Some(budget),
            estimate,
        );
    }

    assembled
}

/// Build the full tool-availability note enumerating every tool name and
/// the deferred-namespace index. Returned by default; demoted to a
/// namespace-only summary by [`build_demoted_tool_note`] when the
/// assembled system block exceeds [`SYSTEM_BLOCK_BUDGET_RATIO`].
fn build_full_tool_note(
    tool_defs: &[ToolDefinition],
    deferred_namespaces: &[ToolNamespace],
) -> String {
    if tool_defs.is_empty() && deferred_namespaces.is_empty() {
        return "No tools are available in this turn.".to_string();
    }

    let has_tool_search = tool_defs.iter().any(|t| t.name == "builtin_tool_search");
    let mut note = String::new();

    if !tool_defs.is_empty() {
        let names = tool_defs
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
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
/// final tool-availability section. Centralised so the demotion path
/// rebuilds the same shape as the default path.
fn assemble_system_instruction(tool_note: String) -> String {
    use crate::prompts::{self, PromptSection, PromptSectionKind};
    let mut sections = prompts::static_sections();
    sections.push(PromptSection::new(
        PromptSectionKind::ToolAvailability,
        tool_note,
    ));
    prompts::assemble(&sections)
}

// Why allow: this builder coordinates several independent prompt slices
// (windowed messages, summaries, tool sets, context summary, anchor) that
// don't naturally cluster into a single struct. Bundling them just to
// satisfy the lint would obscure the code at every call site.
#[allow(clippy::too_many_arguments)]
fn assemble_messages_inner(
    conversation_messages: &[Message],
    summaries: &[MessageSummary],
    tool_defs: &[ToolDefinition],
    deferred_namespaces: &[ToolNamespace],
    context_summary: &str,
    max_messages: usize,
    active_task: Option<&str>,
    tool_rounds_since_anchor: u32,
    budget: Option<ContextBudget>,
    estimate: &dyn Fn(&str) -> u64,
) -> Vec<Message> {
    let tool_note = build_full_tool_note(tool_defs, deferred_namespaces);
    let system_instruction = assemble_system_instruction(tool_note);

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
        let threshold =
            (b.max_input_tokens as f64 * SYSTEM_BLOCK_BUDGET_RATIO) as u64;
        if system_tokens_before > threshold {
            let demoted_note = build_demoted_tool_note(tool_defs, deferred_namespaces);
            let demoted_system = assemble_system_instruction(demoted_note);
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
        let many_tool_rounds = tool_rounds_since_anchor > ACTIVE_TASK_ROUND_THRESHOLD;

        if !anchor_visible || many_tool_rounds {
            messages.push(Message::new(
                Role::System,
                format!("[Current task] {task}"),
            ));
        }
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
                        (Some(f), Some(l)) => format!(
                            "[Summary of messages {}\u{2013}{}] {}",
                            f, l, s.summary
                        ),
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

fn sanitize_assistant_text(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::with_capacity(text.len());

    loop {
        let Some(start) = remaining.find("<think>") else {
            output.push_str(remaining);
            break;
        };

        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];

        match after_start.find("</think>") {
            Some(end) => {
                remaining = &after_start[end + "</think>".len()..];
            }
            None => {
                break;
            }
        }
    }

    let mut sanitized = output.trim().to_string();
    while sanitized.contains("\n\n\n") {
        sanitized = sanitized.replace("\n\n\n", "\n\n");
    }
    sanitized
}

/// Generate a short, human-readable status message for a tool call.
fn tool_status_message(tool_name: &str, arguments: &serde_json::Value) -> String {
    match tool_name {
        "builtin_knowledge_base_search" => {
            if let Some(q) = arguments.get("query").and_then(|v| v.as_str()) {
                let truncated: String = q.chars().take(60).collect();
                format!("Searching knowledge base: {truncated}")
            } else {
                "Searching knowledge base".into()
            }
        }
        "builtin_knowledge_base_write" => "Saving to knowledge base".into(),
        "builtin_knowledge_base_delete" => "Removing knowledge base entry".into(),
        "builtin_sys_props" => "Checking system properties".into(),
        "builtin_db_query" => "Querying database".into(),
        "builtin_tool_search" => {
            if let Some(q) = arguments.get("query").and_then(|v| v.as_str()) {
                let truncated: String = q.chars().take(60).collect();
                format!("Searching for tools: {truncated}")
            } else {
                "Searching for tools".into()
            }
        }
        "builtin_mcp_control" => "Managing tool servers".into(),
        name => {
            // For MCP/dynamic tools, humanize the snake_case name.
            let friendly = name.replace('_', " ");
            format!("Running {friendly}")
        }
    }
}

/// Translate a [`CoreError`] into a user-visible explanation suitable
/// for surfacing in chat. Each LLM-domain variant maps to a tailored
/// message; non-LLM variants and the bare `Llm(detail)` fallback share a
/// generic "I hit an LLM backend error..." line that includes the raw
/// detail for debugging.
fn user_visible_llm_error_message(error: &CoreError) -> String {
    match error {
        CoreError::ContextOverflow { detail, .. } => format!(
            "The conversation exceeded the model's context window. We'll truncate older content and retry. Details: {detail}"
        ),
        CoreError::RateLimited { detail, .. } => format!(
            "The API rate limit was exceeded. Please wait a moment and try again. Details: {detail}"
        ),
        CoreError::QuotaExceeded { detail } => format!(
            "Your API quota is exhausted. Top up the account or switch to a different API key. Details: {detail}"
        ),
        CoreError::ModelLoading { detail } => format!(
            "The model is still downloading or loading. Please wait a moment and try again. Details: {detail}"
        ),
        CoreError::ToolsUnsupported { detail } => format!(
            "This model does not support tool use. Please switch to a tool-capable model or disable tools for this chat. Details: {detail}"
        ),
        // Bare LLM error and any non-LLM variant share the generic
        // fallback. This intentionally does NOT enumerate every
        // CoreError variant — `Display` already produces a readable
        // string and the surrounding service layer is the right place
        // to add tailored messages for non-LLM domains.
        _ => format!(
            "I hit an LLM backend error and could not complete this request. Details: {error}"
        ),
    }
}

fn sanitize_assistant_text_for_stream(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::with_capacity(text.len());

    loop {
        let Some(start) = remaining.find("<think>") else {
            output.push_str(remaining);
            break;
        };

        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];

        match after_start.find("</think>") {
            Some(end) => {
                remaining = &after_start[end + "</think>".len()..];
            }
            None => {
                break;
            }
        }
    }

    let partial_len = trailing_tag_prefix_len(&output, "<think>");
    if partial_len > 0 {
        output.truncate(output.len() - partial_len);
    }

    output
}

fn trailing_tag_prefix_len(text: &str, tag: &str) -> usize {
    for len in (1..tag.len()).rev() {
        if text.ends_with(&tag[..len]) {
            return len;
        }
    }
    0
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

/// Remove the oldest assistant(tool_calls)+tool_result groups from a message
/// list to reduce context size. Keeps the first user message and the most
/// recent tool interaction intact. Returns the number of messages removed.
fn trim_tool_pairs(messages: &mut Vec<Message>) -> usize {
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
        return 0;
    }

    // Remove the oldest half of groups
    let remove_count = groups.len() / 2;
    let groups_to_remove: Vec<_> = groups[..remove_count].to_vec();

    // Remove in reverse order to keep indices stable
    let mut removed = 0;
    for range in groups_to_remove.into_iter().rev() {
        let len = range.len();
        messages.drain(range);
        removed += len;
    }

    removed
}

/// Strip surrounding quotes/backticks and trailing punctuation from a raw LLM title,
/// then limit to at most 8 words as a guard-rail.
fn sanitize_generated_title(raw: &str) -> String {
    let first_line = raw.lines().next().unwrap_or("").trim();
    let stripped = first_line
        .trim_matches(|c| matches!(c, '"' | '\'' | '`'))
        .trim_end_matches(['.', ',', ';', '!', '?']);
    stripped
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Ask the LLM for a concise 3-5-word channel name based on the initial prompt.
/// Returns an empty string on failure so the caller can keep the existing title.
async fn generate_conversation_title<L: LlmClient>(initial_prompt: &str, llm: &L) -> String {
    let messages = vec![
        Message::new(
            Role::System,
            "Generate a concise channel name for a new conversation. \
             Use 3-5 words. Front-load the most specific and meaningful words first — \
             the name may be truncated at the end. Use title case. No punctuation at \
             the edges, no quotes, no explanation. Respond with ONLY the channel name.",
        ),
        Message::new(
            Role::User,
            format!("First message in the conversation: {initial_prompt}"),
        ),
    ];
    match llm
        .stream_completion(messages, &[], ReasoningConfig::default(), Box::new(|_| true))
        .await
    {
        Ok(response) => sanitize_generated_title(&response.text),
        Err(e) => {
            tracing::warn!("conversation title generation failed: {e}");
            String::new()
        }
    }
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
fn window_start(messages: &[Message], max_messages: usize) -> usize {
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
    search
        .iter()
        .position(|m| m.role != Role::Tool)
        .map_or(tentative, |offset| tentative + offset)
}

/// Determine which message range (if any) should be compacted into the
/// rolling context summary. Returns `Some((from, to))` when there are
/// enough newly-dropped messages, or `None` otherwise.
fn compaction_range(conv: &Conversation, max_messages: usize) -> Option<(usize, usize)> {
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
async fn generate_context_summary<L: LlmClient>(
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
                    transcript.push_str(&msg.content[..2000]);
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
async fn recover_from_overflow<L: LlmClient>(
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

    // Step 2: trim oldest tool-pair groups.
    let removed = trim_tool_pairs(&mut conv.messages);
    if removed > 0 {
        conv.compacted_through = conv.compacted_through.saturating_sub(removed);
        tracing::warn!(
            removed,
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

/// Use the LLM to semantically categorize tools into descriptive namespaces.
///
/// Takes the raw tool namespaces (typically grouped by MCP server) and asks
/// the LLM to reorganize them into ≤10-tool categories with descriptive names.
/// Falls back to the original namespaces on failure.
///
/// Skips the LLM round-trip when `budget` is `Some(b)` and the raw
/// `(name + description)` cost of the full listing fits within
/// [`FULL_LISTING_FIT_RATIO`] of `b.max_input_tokens`. Why: categorization
/// is itself an LLM call carrying the full manifest in its prompt — only
/// worth the cost when the raw listing pressure justifies the expense.
async fn categorize_tool_namespaces<L: LlmClient>(
    namespaces: Vec<ToolNamespace>,
    llm: &L,
    budget: Option<ContextBudget>,
) -> Vec<ToolNamespace> {
    // Collect all tools across namespaces. If there are very few, skip categorization.
    let all_tools: Vec<&ToolDefinition> = namespaces.iter().flat_map(|ns| &ns.tools).collect();
    if all_tools.len() <= 10 {
        return namespaces;
    }

    // Skip categorization if the full listing already fits comfortably
    // in the budget. Categorization costs an LLM call AND injects category
    // headers into the prompt; the cost is only justified when the raw
    // listing would meaningfully crowd the budget.
    if let Some(b) = budget {
        let full_listing_tokens: u64 = all_tools
            .iter()
            .map(|t| llm.estimate_tokens(&t.name) + llm.estimate_tokens(&t.description))
            .sum();
        let threshold = (b.max_input_tokens as f64 * FULL_LISTING_FIT_RATIO) as u64;
        if full_listing_tokens < threshold {
            tracing::debug!(
                full_listing_tokens,
                threshold,
                budget = b.max_input_tokens,
                "full tool listing fits budget; skipping categorization"
            );
            return namespaces;
        }
    }

    // Build a tool manifest for the LLM
    let tool_list: String = all_tools
        .iter()
        .map(|t| format!("- {} : {}", t.name, t.description))
        .collect::<Vec<_>>()
        .join("\n");

    let messages = vec![
        Message::new(
            Role::System,
            "You organize tools into semantic categories for an AI assistant's hosted tool search. \
             The search system matches a natural-language query against category descriptions to \
             decide which tools to surface, so descriptions are critical.\n\n\
             Rules:\n\
             - Each category: around 10 tools. A few more is fine if splitting \
               would break a natural workflow grouping.\n\
             - \"name\": short snake_case identifier.\n\
             - \"description\": one or two sentences listing the KEY ACTIONS and VERBS a user \
               would search for. Include synonyms and related terms. \
               Example: \"Start, stop, and manage time-tracking sessions — clock in, clock out, \
               log hours, backdate entries, query timesheets, and correct recorded time.\"\n\
             - Every tool must appear in exactly one category. Do not add or remove tools.\n\
             - Keep related read AND write operations together in the same category.\n\
             - Group tools by WORKFLOW, not by abstract type. Tools that are typically \
               used together in the same task belong in the same category — a user \
               performing a workflow should find everything they need in one namespace.\n\n\
             Respond with ONLY valid JSON: an array of objects with \"name\" (snake_case), \
             \"description\" (string), and \"tools\" (array of tool name strings).",
        ),
        Message::new(
            Role::User,
            format!("Organize these tools into categories (max 10 tools each):\n\n{tool_list}"),
        ),
    ];

    let response = match llm
        .stream_completion(messages, &[], ReasoningConfig::default(), Box::new(|_| true))
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("tool namespace categorization failed: {e}");
            return namespaces;
        }
    };

    // Parse the LLM's JSON response
    let text = response.text.trim();
    // Strip markdown code fences if present
    let json_str = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(text)
        .trim();

    let categories: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("failed to parse tool categorization response: {e}");
            return namespaces;
        }
    };

    // Build a lookup from tool name to tool definition
    let tool_map: std::collections::HashMap<&str, &ToolDefinition> =
        all_tools.iter().map(|t| (t.name.as_str(), *t)).collect();

    let mut result = Vec::new();
    for cat in &categories {
        let Some(name) = cat.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(description) = cat.get("description").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(tool_names) = cat.get("tools").and_then(|v| v.as_array()) else {
            continue;
        };

        let tools: Vec<ToolDefinition> = tool_names
            .iter()
            .filter_map(|v| v.as_str())
            .filter_map(|name| tool_map.get(name).map(|t| (*t).clone()))
            .collect();

        if !tools.is_empty() {
            result.push(ToolNamespace::new(name, description, tools));
        }
    }

    // Sanity check: if the LLM dropped tools, fall back to original
    let result_tool_count: usize = result.iter().map(|ns| ns.tools.len()).sum();
    if result_tool_count < all_tools.len() {
        tracing::warn!(
            "LLM categorization lost tools ({result_tool_count} vs {}), using original namespaces",
            all_tools.len()
        );
        return namespaces;
    }

    tracing::info!(
        "LLM categorized {} tools into {} namespaces",
        result_tool_count,
        result.len()
    );
    for ns in &result {
        tracing::debug!(
            "namespace {:?}: {:?} (tools: {})",
            ns.name,
            ns.description,
            ns.tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    result
}

/// A no-op tool executor for use when no MCP servers are configured.
pub struct NoopToolExecutor;

impl ToolExecutor for NoopToolExecutor {
    async fn core_tools(&self) -> Vec<crate::domain::ToolDefinition> {
        Vec::new()
    }

    async fn search_tools(
        &self,
        _query: &str,
    ) -> Result<Vec<crate::domain::ToolDefinition>, CoreError> {
        Ok(vec![])
    }

    async fn tool_definition(
        &self,
        _name: &str,
    ) -> Result<Option<crate::domain::ToolDefinition>, CoreError> {
        Ok(None)
    }

    async fn execute_tool(
        &self,
        name: &str,
        _arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        Err(CoreError::ToolExecution(format!(
            "no tool executor configured, cannot execute '{name}'"
        )))
    }
}

/// Compute a stable hash over the tool set (names AND descriptions),
/// sorted by name so input ordering does not affect the hash.
///
/// Why: The hash is the cache key for `categorize_tool_namespaces`. That LLM
/// call sees both names and descriptions in its prompt, so a description
/// change can produce a different categorization. Hashing names alone would
/// hide such a change and serve a stale categorization. Re-categorizing on
/// any name OR description edit keeps the cache honest.
fn tool_set_hash(namespaces: &[ToolNamespace]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut entries: Vec<(&str, &str)> = namespaces
        .iter()
        .flat_map(|ns| &ns.tools)
        .map(|t| (t.name.as_str(), t.description.as_str()))
        .collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (name, description) in &entries {
        name.hash(&mut hasher);
        description.hash(&mut hasher);
    }
    hasher.finish()
}

/// Core service implementing conversation management.
/// Generic over store, LLM, and tool executor backends for testability.
pub struct ConversationHandler<S, L, T = NoopToolExecutor> {
    store: S,
    llm: L,
    backend_llm: Option<L>,
    tools: T,
    id_generator: Box<dyn Fn() -> String + Send + Sync>,
    /// Memoized result of `categorize_tool_namespaces`, keyed by `tool_set_hash`.
    ///
    /// Why: Categorization is an LLM call carrying the full tool manifest
    /// (often ≥1K input tokens). Re-running it every turn is wasteful when
    /// the underlying tools have not changed. Lifetime is per-handler
    /// (process-lifetime — there is no eviction); invalidation happens
    /// implicitly when the hash of the current tool set differs from the
    /// stored one. The hash covers tool names AND descriptions, so any
    /// edit to either triggers a fresh categorization.
    namespace_cache: std::sync::Mutex<Option<(u64, Vec<ToolNamespace>)>>,
}

impl<S, L> ConversationHandler<S, L, NoopToolExecutor> {
    pub fn new(store: S, llm: L, id_generator: Box<dyn Fn() -> String + Send + Sync>) -> Self {
        Self {
            store,
            llm,
            backend_llm: None,
            tools: NoopToolExecutor,
            id_generator,
            namespace_cache: std::sync::Mutex::new(None),
        }
    }
}

impl<S, L, T> ConversationHandler<S, L, T> {
    pub fn with_tools(
        store: S,
        llm: L,
        tools: T,
        id_generator: Box<dyn Fn() -> String + Send + Sync>,
    ) -> Self {
        Self {
            store,
            llm,
            backend_llm: None,
            tools,
            id_generator,
            namespace_cache: std::sync::Mutex::new(None),
        }
    }

    /// Set a separate LLM for backend tasks (title generation, context summary).
    /// Falls back to the primary LLM when not set.
    pub fn with_backend_llm(mut self, llm: L) -> Self {
        self.backend_llm = Some(llm);
        self
    }
}

impl<S, L: LlmClient, T> ConversationHandler<S, L, T> {
    /// Returns the backend-tasks LLM if configured, otherwise the primary LLM.
    fn task_llm(&self) -> &L {
        self.backend_llm.as_ref().unwrap_or(&self.llm)
    }
}

impl<S: ConversationStore, L: LlmClient, T: ToolExecutor> ConversationService
    for ConversationHandler<S, L, T>
{
    async fn create_conversation(&self, title: String) -> Result<Conversation, CoreError> {
        let id = (self.id_generator)();
        let mut conv = Conversation::new(id, title);
        let timestamp = now_timestamp();
        conv.created_at = timestamp.clone();
        conv.updated_at = timestamp;
        self.store.create(conv.clone()).await?;
        Ok(conv)
    }

    async fn list_conversations(
        &self,
        max_age_days: Option<u32>,
        include_archived: bool,
    ) -> Result<Vec<ConversationSummary>, CoreError> {
        let mut convs = self.store.list().await?;

        if !include_archived {
            convs.retain(|conv| conv.archived_at.is_none());
        }

        if let Some(days) = max_age_days.filter(|days| *days > 0) {
            let cutoff = cutoff_timestamp(days);
            convs.retain(|conv| !conv.updated_at.is_empty() && conv.updated_at >= cutoff);
        }

        convs.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| left.title.cmp(&right.title))
                .then_with(|| left.id.0.cmp(&right.id.0))
        });

        Ok(convs.iter().map(ConversationSummary::from).collect())
    }

    async fn get_conversation(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
        self.store.get(id).await
    }

    async fn delete_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.store.delete(id).await
    }

    async fn rename_conversation(
        &self,
        id: &ConversationId,
        title: String,
    ) -> Result<(), CoreError> {
        let mut conv = self.store.get(id).await?;
        conv.title = title;
        conv.updated_at = now_timestamp();
        self.store.update(conv).await
    }

    async fn archive_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.store.archive(id).await
    }

    async fn unarchive_conversation(&self, id: &ConversationId) -> Result<(), CoreError> {
        self.store.unarchive(id).await
    }

    async fn clear_all_history(&self) -> Result<u32, CoreError> {
        let conversations = self.store.list().await?;
        let mut deleted = 0u32;

        for conversation in conversations {
            self.store.delete(&conversation.id).await?;
            deleted += 1;
        }

        Ok(deleted)
    }

    async fn send_prompt(
        &self,
        conversation_id: &ConversationId,
        prompt: String,
        mut on_chunk: ChunkCallback,
        mut on_status: StatusCallback,
    ) -> Result<String, CoreError> {
        let mut conv = self.store.get(conversation_id).await?;
        let is_first_message = conv.messages.is_empty();
        conv.messages.push(Message::new(Role::User, &prompt));
        // Capture the prompt as the active-task anchor for this turn. It is
        // re-injected in `llm_messages_for_turn` when conditions indicate
        // the original message has drifted out of the model's view.
        conv.active_task = Some(prompt.clone());

        // Effective window size for this turn. May shrink further if the
        // provider reports input-token usage above COMPACTION_TOKEN_RATIO.
        let mut target_window = MAX_CONTEXT_MESSAGES;

        // Count of in-turn ContextOverflow recoveries. Bounded so a
        // persistently-oversized request doesn't loop indefinitely.
        let mut overflow_retries: u32 = 0;

        // Run compaction if enough messages have been dropped by windowing.
        if let Some((from, to)) = compaction_range(&conv, target_window) {
            let summary = generate_context_summary(
                &conv.context_summary,
                &conv.messages[from..to],
                self.task_llm(),
            )
            .await;
            conv.context_summary = summary;
            conv.compacted_through = to;
        }

        // Dynamic tool discovery: start with core tools, activate more via tool_search.
        let use_hosted_search = self.llm.supports_hosted_tool_search();
        let namespaces: Vec<ToolNamespace> = if use_hosted_search {
            let raw_namespaces = self.tools.tool_namespaces().await;
            if raw_namespaces.is_empty() {
                vec![]
            } else {
                let hash = tool_set_hash(&raw_namespaces);
                let cached_hit = {
                    let cached = self.namespace_cache.lock().unwrap();
                    cached
                        .as_ref()
                        .filter(|(h, _)| *h == hash)
                        .map(|(_, ns)| ns.clone())
                };
                if let Some(ns) = cached_hit {
                    tracing::debug!(
                        hash,
                        namespace_count = ns.len(),
                        "tool categorization cache hit"
                    );
                    ns
                } else {
                    tracing::debug!(hash, "tool categorization cache miss; invoking LLM");
                    let result = categorize_tool_namespaces(
                        raw_namespaces,
                        self.task_llm(),
                        current_context_budget(),
                    )
                    .await;
                    *self.namespace_cache.lock().unwrap() = Some((hash, result.clone()));
                    result
                }
            }
        } else {
            vec![]
        };

        let core_tools = self.tools.core_tools().await;
        // When hosted search is active and we have namespaces, remove
        // builtin_tool_search from core tools — the provider handles discovery.
        let core_tools_for_llm: Vec<ToolDefinition> = if use_hosted_search && !namespaces.is_empty()
        {
            core_tools
                .iter()
                .filter(|t| t.name != "builtin_tool_search")
                .cloned()
                .collect()
        } else {
            core_tools.clone()
        };

        let mut activated_tools: std::collections::HashMap<String, ToolDefinition> =
            std::collections::HashMap::new();
        // Track whether hosted search has been demoted to local fallback.
        let mut hosted_search_demoted = false;

        for round in 0..MAX_TOOL_ROUNDS {
            // Build the tool set: core + dynamically activated.
            // When hosted search has been demoted, use the full core set
            // (which includes builtin_tool_search) instead of the filtered one.
            let mut tool_defs: Vec<ToolDefinition> = if hosted_search_demoted {
                core_tools.clone()
            } else {
                core_tools_for_llm.clone()
            };
            tool_defs.extend(activated_tools.values().cloned());

            let deferred_ns: &[ToolNamespace] = if !hosted_search_demoted {
                &namespaces
            } else {
                &[]
            };
            // `tool_rounds_since_anchor` doubles as "how many tool rounds
            // have we executed in this turn". Each completed round increments
            // the count, and the anchor was just (re)set at the start of
            // `send_prompt` — so this is exactly the round counter we want
            // to thread into the active-task injection check.
            let tool_rounds_since_anchor = u32::try_from(round).unwrap_or(u32::MAX);
            // The estimator borrows `&self.llm` so the closure is built
            // each iteration; constructing it is cheap (no allocation).
            let estimate = |text: &str| self.llm.estimate_tokens(text);
            let llm_messages = llm_messages_for_turn(
                &conv.messages,
                &conv.summaries,
                &tool_defs,
                deferred_ns,
                &conv.context_summary,
                target_window,
                conv.active_task.as_deref(),
                tool_rounds_since_anchor,
                current_context_budget(),
                &estimate,
            );
            let mut raw_stream = String::new();
            let mut emitted_visible_len = 0usize;
            let mut visible_chunk_callback = on_chunk;
            let filtered_chunk_callback: ChunkCallback = Box::new(move |chunk| {
                raw_stream.push_str(&chunk);
                let sanitized = sanitize_assistant_text_for_stream(&raw_stream);

                if sanitized.len() < emitted_visible_len {
                    emitted_visible_len = sanitized.len();
                    return true;
                }

                if sanitized.len() <= emitted_visible_len {
                    return true;
                }

                let visible = sanitized[emitted_visible_len..].to_string();
                emitted_visible_len = sanitized.len();

                if visible.is_empty() {
                    true
                } else {
                    visible_chunk_callback(visible)
                }
            });

            // Reasoning config is threaded through a task-local set by
            // the daemon-side routing wrapper (`RoutingConversationHandler`)
            // before it calls `send_prompt`. In tests / standalone uses
            // with no wrapper, the slot is unset and we pass the default
            // empty config, matching the pre-issue-18 behaviour.
            let reasoning = crate::ports::llm::current_reasoning_config();

            let response =
                match if use_hosted_search && !namespaces.is_empty() && !hosted_search_demoted {
                    self.llm
                        .stream_completion_with_namespaces(
                            llm_messages,
                            &tool_defs,
                            &namespaces,
                            reasoning,
                            filtered_chunk_callback,
                        )
                        .await
                } else {
                    self.llm
                        .stream_completion(
                            llm_messages,
                            &tool_defs,
                            reasoning,
                            filtered_chunk_callback,
                        )
                        .await
                } {
                    Ok(r) => r,
                    Err(CoreError::ContextOverflow {
                        prompt_tokens,
                        max_tokens,
                        detail: _,
                    }) if overflow_retries < MAX_OVERFLOW_RETRIES => {
                        // The provider rejected this turn's prompt for
                        // exceeding its context window. Run the recovery
                        // ladder (truncate large tool result → trim old
                        // pairs → summarise-and-shrink) and retry. The
                        // counter bounds total attempts across all steps
                        // so persistently-oversized requests can't loop.
                        overflow_retries += 1;
                        tracing::warn!(
                            attempt = overflow_retries,
                            max_attempts = MAX_OVERFLOW_RETRIES,
                            prompt_tokens = ?prompt_tokens,
                            max_tokens = ?max_tokens,
                            "context overflow — running recovery ladder"
                        );
                        recover_from_overflow(
                            &mut conv,
                            prompt_tokens,
                            max_tokens,
                            &mut target_window,
                            self.task_llm(),
                            &estimate,
                        )
                        .await;
                        on_chunk = Box::new(|_| true);
                        continue;
                    }
                    Err(e) => {
                        // Anything else — including exhausted overflow
                        // retries — surfaces as a user-visible message.
                        // Non-context errors are no longer trimmed-and-prayed
                        // through old path C; that swallowed transient
                        // failures (rate limits, server errors, malformed
                        // tool calls) by mutating conversation state.
                        let friendly = user_visible_llm_error_message(&e);
                        conv.messages.push(Message::new(Role::Assistant, &friendly));
                        conv.updated_at = now_timestamp();
                        self.store.update(conv).await?;
                        return Ok(friendly);
                    }
                };

            // Token-pressure check: if the provider reports input tokens
            // above COMPACTION_TOKEN_RATIO of its context window, shrink the
            // effective message window and compact the newly-dropped range
            // before building the next turn's prompt.
            //
            // The budget is resolved once at dispatch entry by the daemon's
            // routing wrapper (issue #63) and read here via the
            // `CONTEXT_BUDGET` task-local. When the slot is unset (test
            // contexts, background jobs that don't route through the
            // wrapper), token-based compaction skips — same behaviour as
            // when the connector previously returned `None` from
            // `max_context_tokens()`.
            if let (Some(budget), Some(usage)) =
                (current_context_budget(), response.usage.as_ref())
                && let Some(input_tokens) = usage.input_tokens
            {
                let max_tokens = budget.max_input_tokens;
                let threshold = (max_tokens as f64 * COMPACTION_TOKEN_RATIO) as u64;
                if input_tokens > threshold {
                    let new_window = (target_window / 2).max(MIN_CONTEXT_MESSAGES);
                    if new_window < target_window {
                        tracing::info!(
                            input_tokens,
                            max_tokens,
                            prev_window = target_window,
                            new_window,
                            "context pressure — shrinking window and compacting"
                        );
                        target_window = new_window;
                        if let Some((from, to)) = compaction_range(&conv, target_window) {
                            let summary = generate_context_summary(
                                &conv.context_summary,
                                &conv.messages[from..to],
                                self.task_llm(),
                            )
                            .await;
                            conv.context_summary = summary;
                            conv.compacted_through = to;
                        }
                    } else {
                        tracing::debug!(
                            input_tokens,
                            max_tokens,
                            window = target_window,
                            "context pressure with window already at minimum"
                        );
                    }
                }
            }

            if !response.has_tool_calls() {
                // Hosted-search fallback: if the model returned text-only
                // while hosted search was active, it likely couldn't invoke
                // deferred tools.  Demote to local builtin_tool_search and
                // let the model try again with the classic tool-discovery path.
                if use_hosted_search
                    && !namespaces.is_empty()
                    && !hosted_search_demoted
                    && round < 2
                {
                    tracing::warn!(
                        round,
                        "hosted tool search produced no tool calls — \
                         falling back to builtin_tool_search"
                    );
                    hosted_search_demoted = true;
                    // Keep the assistant text so the model has context,
                    // then inject a system nudge to use builtin_tool_search.
                    if !response.text.is_empty() {
                        conv.messages
                            .push(Message::new(Role::Assistant, &response.text));
                    }
                    conv.messages.push(Message::new(
                        Role::System,
                        "The server-side tool search was unable to surface the \
                         tools you need. You now have access to `builtin_tool_search` \
                         — call it with a query describing what you need.",
                    ));
                    on_chunk = Box::new(|_| true);
                    continue;
                }

                // Text-only response — we're done
                let mut visible_text = sanitize_assistant_text(&response.text);
                if visible_text.is_empty() {
                    tracing::warn!(
                        raw_len = response.text.len(),
                        raw_first_100 = %response.text.chars().take(100).collect::<String>(),
                        round,
                        "LLM returned empty visible text after sanitization"
                    );
                    if round > 0 {
                        visible_text =
                            "I wasn't able to complete this request — the tools I tried \
                             returned errors. Please check the conversation log or try again."
                                .to_string();
                    }
                }
                conv.messages
                    .push(Message::new(Role::Assistant, &visible_text));
                // On the first message, generate a descriptive title via the LLM
                // so the conversation list shows meaningful names rather than
                // timestamp-based placeholders.
                if is_first_message {
                    let generated = generate_conversation_title(&prompt, self.task_llm()).await;
                    if !generated.is_empty() {
                        conv.title = generated;
                    }
                }
                conv.updated_at = now_timestamp();
                self.store.update(conv).await?;
                return Ok(visible_text);
            }

            // LLM wants to call tools — record the assistant message with tool calls
            tracing::info!(
                "LLM requested {} tool call(s) (round {}/{})",
                response.tool_calls.len(),
                round + 1,
                MAX_TOOL_ROUNDS
            );
            conv.messages.push(Message::assistant_with_tool_calls(
                response.tool_calls.clone(),
            ));

            // Execute each tool call and append results
            for tool_call in &response.tool_calls {
                let arguments: serde_json::Value =
                    serde_json::from_str(&tool_call.arguments).unwrap_or_default();
                on_status(tool_status_message(&tool_call.name, &arguments));
                tracing::info!(tool = %tool_call.name, %arguments, "executing tool");
                let result = match self.tools.execute_tool(&tool_call.name, arguments).await {
                    Ok(output) => {
                        tracing::debug!(tool = %tool_call.name, output = %output, "tool result");
                        output
                    }
                    Err(e) => {
                        tracing::warn!(tool = %tool_call.name, error = %e, "tool execution failed");
                        format!("Error: {e}")
                    }
                };

                // Dynamic activation: if tool_search returned results,
                // activate the discovered tools for subsequent rounds.
                // Skip when hosted search is active (unless demoted to local fallback).
                if (!use_hosted_search || hosted_search_demoted)
                    && tool_call.name == "builtin_tool_search"
                    && let Ok(found) = serde_json::from_str::<serde_json::Value>(&result)
                    && let Some(tools_arr) = found.get("tools").and_then(|v| v.as_array())
                {
                    for tool_entry in tools_arr {
                        if let Some(name) = tool_entry.get("name").and_then(|v| v.as_str())
                            && !activated_tools.contains_key(name)
                            && !core_tools.iter().any(|t| t.name == name)
                            && let Ok(Some(def)) = self.tools.tool_definition(name).await
                        {
                            tracing::info!("dynamically activated tool: {}", def.name);
                            activated_tools.insert(def.name.clone(), def);
                        }
                    }
                }

                // When hosted search is active and the model calls a
                // deferred namespace tool, activate the entire namespace
                // so full schemas are available in subsequent rounds.
                if use_hosted_search
                    && !hosted_search_demoted
                    && !activated_tools.contains_key(&tool_call.name)
                    && !core_tools.iter().any(|t| t.name == tool_call.name)
                {
                    for ns in &namespaces {
                        if ns.tools.iter().any(|t| t.name == tool_call.name) {
                            for t in &ns.tools {
                                if !activated_tools.contains_key(&t.name)
                                    && !core_tools.iter().any(|ct| ct.name == t.name)
                                {
                                    tracing::info!(
                                        "activated deferred tool from namespace {:?}: {}",
                                        ns.name,
                                        t.name
                                    );
                                    activated_tools.insert(t.name.clone(), t.clone());
                                }
                            }
                            break;
                        }
                    }
                }

                conv.messages
                    .push(Message::tool_result(&tool_call.id, &result));
            }

            // Create a new noop callback for subsequent rounds
            // (the original callback was consumed by stream_completion)
            on_chunk = Box::new(|_| true);
        }

        // If we exhausted all rounds, return what we have
        Err(CoreError::Llm(format!(
            "tool calling loop exceeded maximum of {MAX_TOOL_ROUNDS} rounds"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{ToolCall, ToolDefinition};
    use crate::ports::llm::{BudgetSource, LlmResponse, TokenUsage};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    // --- Mock Store ---
    struct MockStore {
        data: Mutex<HashMap<String, Conversation>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                data: Mutex::new(HashMap::new()),
            }
        }
    }

    impl ConversationStore for MockStore {
        async fn create(&self, conv: Conversation) -> Result<(), CoreError> {
            self.data.lock().unwrap().insert(conv.id.0.clone(), conv);
            Ok(())
        }

        async fn get(&self, id: &ConversationId) -> Result<Conversation, CoreError> {
            self.data
                .lock()
                .unwrap()
                .get(&id.0)
                .cloned()
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
        }

        async fn list(&self) -> Result<Vec<Conversation>, CoreError> {
            Ok(self.data.lock().unwrap().values().cloned().collect())
        }

        async fn update(&self, conv: Conversation) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            if data.contains_key(&conv.id.0) {
                data.insert(conv.id.0.clone(), conv);
                Ok(())
            } else {
                Err(CoreError::ConversationNotFound(conv.id.0.clone()))
            }
        }

        async fn delete(&self, id: &ConversationId) -> Result<(), CoreError> {
            self.data
                .lock()
                .unwrap()
                .remove(&id.0)
                .map(|_| ())
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))
        }

        async fn archive(&self, id: &ConversationId) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            let conv = data
                .get_mut(&id.0)
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;
            conv.archived_at = Some("2026-01-01 00:00:00".to_string());
            Ok(())
        }

        async fn unarchive(&self, id: &ConversationId) -> Result<(), CoreError> {
            let mut data = self.data.lock().unwrap();
            let conv = data
                .get_mut(&id.0)
                .ok_or_else(|| CoreError::ConversationNotFound(id.0.clone()))?;
            conv.archived_at = None;
            Ok(())
        }

        async fn create_summary(
            &self,
            _conversation_id: &ConversationId,
            _summary: String,
            _start_ordinal: usize,
            _end_ordinal: usize,
        ) -> Result<String, CoreError> {
            Ok("mock-summary".to_string())
        }

        async fn expand_summary(&self, _summary_id: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    // --- Mock LLM ---
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

    fn make_handler(chunks: Vec<&str>) -> ConversationHandler<MockStore, MockLlm> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        ConversationHandler::new(
            MockStore::new(),
            MockLlm::new(chunks),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
    }

    fn noop_callback() -> ChunkCallback {
        Box::new(|_| true)
    }

    fn noop_status() -> StatusCallback {
        Box::new(|_| {})
    }

    /// Token estimator used by the existing assembly tests. Mirrors the
    /// `LlmClient::estimate_tokens` default so tests don't depend on any
    /// connector and behave identically to the real default-impl path.
    fn default_estimate(s: &str) -> u64 {
        (s.chars().count() as u64).div_ceil(4)
    }

    struct ListOnlyStore {
        conversations: Vec<Conversation>,
    }

    impl ConversationStore for ListOnlyStore {
        async fn create(&self, _conv: Conversation) -> Result<(), CoreError> {
            Ok(())
        }

        async fn get(&self, _id: &ConversationId) -> Result<Conversation, CoreError> {
            Err(CoreError::ConversationNotFound("unused".to_string()))
        }

        async fn list(&self) -> Result<Vec<Conversation>, CoreError> {
            Ok(self.conversations.clone())
        }

        async fn update(&self, _conv: Conversation) -> Result<(), CoreError> {
            Ok(())
        }

        async fn delete(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn archive(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn unarchive(&self, _id: &ConversationId) -> Result<(), CoreError> {
            Ok(())
        }

        async fn create_summary(
            &self,
            _conversation_id: &ConversationId,
            _summary: String,
            _start_ordinal: usize,
            _end_ordinal: usize,
        ) -> Result<String, CoreError> {
            Ok("mock-summary".to_string())
        }

        async fn expand_summary(&self, _summary_id: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn create_assigns_unique_ids() {
        let handler = make_handler(vec![]);
        let c1 = handler.create_conversation("A".into()).await.unwrap();
        let c2 = handler.create_conversation("B".into()).await.unwrap();
        assert_ne!(c1.id, c2.id);
        assert_eq!(c1.id.as_str(), "conv-1");
        assert_eq!(c2.id.as_str(), "conv-2");
    }

    #[tokio::test]
    async fn create_sets_human_readable_timestamps() {
        let handler = make_handler(vec![]);
        let conv = handler.create_conversation("A".into()).await.unwrap();
        assert!(!conv.created_at.is_empty());
        assert!(!conv.updated_at.is_empty());
        assert_eq!(conv.created_at.len(), 19);
        assert_eq!(conv.updated_at.len(), 19);
        assert_eq!(conv.created_at, conv.updated_at);
    }

    #[tokio::test]
    async fn create_stores_conversation() {
        let handler = make_handler(vec![]);
        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let retrieved = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(retrieved.title, "Test");
    }

    #[tokio::test]
    async fn list_returns_summaries() {
        let handler = make_handler(vec![]);
        handler.create_conversation("A".into()).await.unwrap();
        handler.create_conversation("B".into()).await.unwrap();

        let summaries = handler.list_conversations(None, false).await.unwrap();
        assert_eq!(summaries.len(), 2);
        for s in &summaries {
            assert_eq!(s.message_count, 0);
        }
    }

    #[tokio::test]
    async fn list_filters_by_age_and_sorts_descending() {
        let now = Local::now();

        let mut old_conv = Conversation::new("old", "Old");
        old_conv.created_at = (now - Duration::days(30))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        old_conv.updated_at = old_conv.created_at.clone();

        let mut newer_conv = Conversation::new("newer", "Newer");
        newer_conv.created_at = (now - Duration::days(2))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        newer_conv.updated_at = newer_conv.created_at.clone();

        let mut newest_conv = Conversation::new("newest", "Newest");
        newest_conv.created_at = (now - Duration::hours(1))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        newest_conv.updated_at = newest_conv.created_at.clone();

        let handler = ConversationHandler::new(
            ListOnlyStore {
                conversations: vec![old_conv, newer_conv, newest_conv],
            },
            MockLlm::new(vec![]),
            Box::new(|| "unused".to_string()),
        );

        let filtered = handler.list_conversations(Some(7), false).await.unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].id.as_str(), "newest");
        assert_eq!(filtered[1].id.as_str(), "newer");
    }

    #[tokio::test]
    async fn delete_removes_conversation() {
        let handler = make_handler(vec![]);
        let conv = handler.create_conversation("Gone".into()).await.unwrap();
        handler.delete_conversation(&conv.id).await.unwrap();

        let result = handler.get_conversation(&conv.id).await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }

    #[tokio::test]
    async fn clear_all_history_removes_all_conversations() {
        let handler = make_handler(vec![]);
        handler.create_conversation("A".into()).await.unwrap();
        handler.create_conversation("B".into()).await.unwrap();

        let deleted = handler.clear_all_history().await.unwrap();
        assert_eq!(deleted, 2);

        let summaries = handler.list_conversations(None, false).await.unwrap();
        assert!(summaries.is_empty());
    }

    #[tokio::test]
    async fn send_prompt_adds_messages_to_history() {
        let handler = make_handler(vec!["Hello", " there"]);
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

        let response = handler
            .send_prompt(&conv.id, "Hi".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(response, "Hello there");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages.len(), 2);
        assert_eq!(updated.messages[0].role, Role::User);
        assert_eq!(updated.messages[0].content, "Hi");
        assert_eq!(updated.messages[1].role, Role::Assistant);
        assert_eq!(updated.messages[1].content, "Hello there");
    }

    #[tokio::test]
    async fn send_prompt_streams_chunks() {
        let handler = make_handler(vec!["a", "b", "c"]);
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

        let chunks = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = Arc::clone(&chunks);
        let response = handler
            .send_prompt(
                &conv.id,
                "test".into(),
                Box::new(move |chunk| {
                    chunks_clone.lock().unwrap().push(chunk);
                    true
                }),
                noop_status(),
            )
            .await
            .unwrap();
        assert_eq!(response, "abc");
        assert_eq!(*chunks.lock().unwrap(), vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn send_prompt_hides_thinking_blocks_in_final_response() {
        let handler = make_handler(vec!["<think>internal reasoning</think>\n\nVisible answer"]);
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

        let response = handler
            .send_prompt(&conv.id, "Hi".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(response, "Visible answer");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages[1].role, Role::Assistant);
        assert_eq!(updated.messages[1].content, "Visible answer");
    }

    #[tokio::test]
    async fn send_prompt_hides_thinking_blocks_in_streamed_chunks() {
        let handler = make_handler(vec!["Visible ", "<th", "ink>internal</think>", "answer"]);
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

        let chunks = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = Arc::clone(&chunks);
        let response = handler
            .send_prompt(
                &conv.id,
                "Hi".into(),
                Box::new(move |chunk| {
                    chunks_clone.lock().unwrap().push(chunk);
                    true
                }),
                noop_status(),
            )
            .await
            .unwrap();

        assert_eq!(response, "Visible answer");
        assert_eq!(*chunks.lock().unwrap(), vec!["Visible ", "answer"]);
    }

    #[test]
    fn sanitize_assistant_text_handles_unclosed_think_block() {
        let input = "Visible before <think>internal";
        let output = sanitize_assistant_text(input);
        assert_eq!(output, "Visible before");
    }

    #[tokio::test]
    async fn send_prompt_nonexistent_conversation_fails() {
        let handler = make_handler(vec![]);
        let result = handler
            .send_prompt(
                &ConversationId::from("nope"),
                "hi".into(),
                noop_callback(),
                noop_status(),
            )
            .await;
        assert!(matches!(result, Err(CoreError::ConversationNotFound(_))));
    }

    // --- Tool calling tests ---

    /// Mock LLM that returns tool calls on first invocation, then text.
    struct ToolCallingLlm {
        /// Responses to return in sequence. Each call to stream_completion
        /// pops the first response.
        responses: Mutex<Vec<LlmResponse>>,
    }

    impl ToolCallingLlm {
        fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl LlmClient for ToolCallingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let response = {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Ok(LlmResponse::text("fallback"));
                }
                responses.remove(0)
            };
            // Stream any text content
            if !response.text.is_empty() {
                on_chunk(response.text.clone());
            }
            Ok(response)
        }
    }

    /// Mock tool executor that returns predictable results.
    struct MockToolExecutor {
        tools: Vec<ToolDefinition>,
        results: Mutex<HashMap<String, String>>,
    }

    impl MockToolExecutor {
        fn new(tools: Vec<ToolDefinition>, results: HashMap<String, String>) -> Self {
            Self {
                tools,
                results: Mutex::new(results),
            }
        }
    }

    impl ToolExecutor for MockToolExecutor {
        async fn core_tools(&self) -> Vec<ToolDefinition> {
            self.tools.clone()
        }

        async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(vec![])
        }

        async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
            Ok(self.tools.iter().find(|t| t.name == name).cloned())
        }

        async fn execute_tool(
            &self,
            name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            self.results
                .lock()
                .unwrap()
                .get(name)
                .cloned()
                .ok_or_else(|| CoreError::ToolExecution(format!("unknown tool: {name}")))
        }
    }

    fn make_tool_handler(
        responses: Vec<LlmResponse>,
        tools: Vec<ToolDefinition>,
        tool_results: HashMap<String, String>,
    ) -> ConversationHandler<MockStore, ToolCallingLlm, MockToolExecutor> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        ConversationHandler::with_tools(
            MockStore::new(),
            ToolCallingLlm::new(responses),
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
    }

    #[tokio::test]
    async fn tool_loop_executes_tool_and_returns_final_text() {
        let tool_def = ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        );
        let tool_call = ToolCall::new("call-1", "read_file", r#"{"path": "/tmp/test"}"#);

        let responses = vec![
            // First: LLM requests a tool call
            LlmResponse::with_tool_calls("", vec![tool_call]),
            // Second: LLM returns final text after seeing tool result
            LlmResponse::text("The file contains: hello world"),
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("read_file".to_string(), "hello world".to_string());

        let handler = make_tool_handler(responses, vec![tool_def], tool_results);
        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Read /tmp/test".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();
        assert_eq!(result, "The file contains: hello world");

        // Verify conversation history has all messages
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages.len(), 4);
        assert_eq!(updated.messages[0].role, Role::User);
        assert_eq!(updated.messages[1].role, Role::Assistant); // tool call request
        assert!(!updated.messages[1].tool_calls.is_empty());
        assert_eq!(updated.messages[2].role, Role::Tool); // tool result
        assert_eq!(updated.messages[2].content, "hello world");
        assert_eq!(updated.messages[3].role, Role::Assistant); // final response
        assert_eq!(
            updated.messages[3].content,
            "The file contains: hello world"
        );
    }

    #[tokio::test]
    async fn tool_loop_handles_multiple_tool_calls() {
        let tools = vec![
            ToolDefinition::new("tool_a", "Tool A", serde_json::json!({})),
            ToolDefinition::new("tool_b", "Tool B", serde_json::json!({})),
        ];

        let responses = vec![
            LlmResponse::with_tool_calls(
                "",
                vec![
                    ToolCall::new("c1", "tool_a", "{}"),
                    ToolCall::new("c2", "tool_b", "{}"),
                ],
            ),
            LlmResponse::text("Done with both tools"),
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("tool_a".to_string(), "result_a".to_string());
        tool_results.insert("tool_b".to_string(), "result_b".to_string());

        let handler = make_tool_handler(responses, tools, tool_results);
        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(&conv.id, "Do both".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "Done with both tools");

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        // User + assistant(tool_calls) + tool_result_a + tool_result_b + assistant(final)
        assert_eq!(updated.messages.len(), 5);
    }

    #[tokio::test]
    async fn tool_loop_handles_tool_error_gracefully() {
        let tools = vec![ToolDefinition::new(
            "bad_tool",
            "Fails",
            serde_json::json!({}),
        )];

        let responses = vec![
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "bad_tool", "{}")]),
            LlmResponse::text("Tool failed, but I can continue"),
        ];

        // No results configured — tool will return error
        let handler = make_tool_handler(responses, tools, HashMap::new());
        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Try bad tool".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();
        assert_eq!(result, "Tool failed, but I can continue");

        // The tool error should be in the conversation as a tool result message
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages[2].role, Role::Tool);
        assert!(updated.messages[2].content.starts_with("Error:"));
    }

    #[tokio::test]
    async fn tool_loop_respects_max_rounds() {
        let tools = vec![ToolDefinition::new(
            "loop_tool",
            "Loops",
            serde_json::json!({}),
        )];

        // LLM always returns tool calls — never text
        let responses: Vec<LlmResponse> = (0..MAX_TOOL_ROUNDS + 1)
            .map(|i| {
                LlmResponse::with_tool_calls(
                    "",
                    vec![ToolCall::new(format!("c{i}"), "loop_tool", "{}")],
                )
            })
            .collect();

        let mut tool_results = HashMap::new();
        tool_results.insert("loop_tool".to_string(), "ok".to_string());

        let handler = make_tool_handler(responses, tools, tool_results);
        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Loop forever".into(),
                noop_callback(),
                noop_status(),
            )
            .await;
        assert!(matches!(result, Err(CoreError::Llm(_))));
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

        let removed = trim_tool_pairs(&mut messages);
        // 4 groups, remove oldest half (2 groups = 4 messages)
        assert_eq!(removed, 4);
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

        let removed = trim_tool_pairs(&mut messages);
        assert_eq!(removed, 0);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn trim_tool_pairs_no_groups() {
        let mut messages = vec![
            Message::new(Role::User, "hello"),
            Message::new(Role::Assistant, "hi there"),
        ];

        let removed = trim_tool_pairs(&mut messages);
        assert_eq!(removed, 0);
        assert_eq!(messages.len(), 2);
    }

    // --- Context recovery test ---

    /// Mock LLM that fails on a specific call index.
    struct FailingLlm {
        responses: Mutex<Vec<LlmResponse>>,
        fail_on_call: usize,
        call_count: Mutex<usize>,
        error_factory: Box<dyn Fn() -> CoreError + Send + Sync>,
    }

    impl FailingLlm {
        fn new(responses: Vec<LlmResponse>, fail_on_call: usize) -> Self {
            Self {
                responses: Mutex::new(responses),
                fail_on_call,
                call_count: Mutex::new(0),
                // Default to a generic LLM error; tests that need a
                // specific structured variant call `with_error_variant`.
                error_factory: Box::new(|| CoreError::Llm("context_length_exceeded".into())),
            }
        }

        /// Substitute the variant produced on the failing call. Used by
        /// tests that exercise control-flow paths keyed on the specific
        /// `CoreError` variant (e.g. `RateLimited` skipping the trim
        /// branch).
        fn with_error_variant<F>(mut self, factory: F) -> Self
        where
            F: Fn() -> CoreError + Send + Sync + 'static,
        {
            self.error_factory = Box::new(factory);
            self
        }
    }

    impl LlmClient for FailingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let call_idx = {
                let mut count = self.call_count.lock().unwrap();
                let idx = *count;
                *count += 1;
                idx
            };

            if call_idx == self.fail_on_call {
                return Err((self.error_factory)());
            }

            let response = {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Ok(LlmResponse::text("fallback"));
                }
                responses.remove(0)
            };
            if !response.text.is_empty() {
                on_chunk(response.text.clone());
            }
            Ok(response)
        }
    }

    #[tokio::test]
    async fn non_context_error_after_round_zero_surfaces_directly() {
        // Old path C trimmed-and-retried any non-retryable, non-rate-limit
        // error after round 0 — including transient or malformed-call
        // failures that had nothing to do with context size. Now that the
        // recovery ladder is gated on `CoreError::ContextOverflow`, those
        // errors must surface to the user immediately instead of mutating
        // the conversation state.
        let tools = vec![ToolDefinition::new(
            "my_tool",
            "A tool",
            serde_json::json!({}),
        )];

        let responses = vec![
            // Round 0: LLM requests tool call.
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "my_tool", "{}")]),
            // Round 1: fails with a generic LLM error (call index 1 below).
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("my_tool".to_string(), "result".to_string());

        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            FailingLlm::new(responses, 1)
                .with_error_variant(|| CoreError::Llm("context_length_exceeded".into())),
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Use my tool".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        // The user-visible error mentions the underlying detail; what
        // matters is that we don't pretend to have recovered.
        assert!(result.contains("LLM backend error"));
        assert!(result.contains("context_length_exceeded"));

        // No system trim notice was injected — path C is gone.
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let has_trim_msg = updated
            .messages
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("context became too long"));
        assert!(
            !has_trim_msg,
            "non-context errors must not trigger context trimming"
        );
    }

    #[tokio::test]
    async fn first_round_llm_error_is_saved_as_assistant_message() {
        // If the first LLM call fails, return a user-visible assistant message
        let tools = vec![];

        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            FailingLlm::new(vec![], 0), // fail on 1st call
            MockToolExecutor::new(tools, HashMap::new()),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(&conv.id, "hello".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert!(result.contains("LLM backend error"));

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.messages.len(), 2);
        assert_eq!(updated.messages[1].role, Role::Assistant);
        assert!(updated.messages[1].content.contains("LLM backend error"));
    }

    #[test]
    fn user_visible_error_for_unsupported_tools() {
        let err = CoreError::ToolsUnsupported {
            detail: "phi4:14b does not support tools".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("does not support tool use"));
    }

    #[test]
    fn user_visible_error_for_loading_model() {
        let err = CoreError::ModelLoading {
            detail: "model is currently loading".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("still downloading or loading"));
    }

    #[test]
    fn user_visible_error_for_rate_limit_429() {
        let err = CoreError::RateLimited {
            retry_after: None,
            detail: "Rate limited".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("rate limit was exceeded"));
    }

    #[test]
    fn user_visible_error_for_overloaded_529() {
        let err = CoreError::RateLimited {
            retry_after: None,
            detail: "overloaded".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("rate limit was exceeded"));
    }

    #[test]
    fn user_visible_error_for_quota_exceeded() {
        let err = CoreError::QuotaExceeded {
            detail: "insufficient_quota".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("quota is exhausted"));
    }

    #[test]
    fn user_visible_error_for_context_overflow() {
        let err = CoreError::ContextOverflow {
            prompt_tokens: Some(203_524),
            max_tokens: Some(200_000),
            detail: "prompt is too long".into(),
        };
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("context window"));
    }

    #[test]
    fn user_visible_error_for_generic_llm() {
        let err = CoreError::Llm("invalid API key".into());
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("LLM backend error"));
        assert!(msg.contains("invalid API key"));
    }

    #[tokio::test]
    async fn rate_limit_error_mid_loop_does_not_trim_context() {
        let tools = vec![ToolDefinition::new(
            "my_tool",
            "A tool",
            serde_json::json!({}),
        )];

        let responses = vec![
            // Round 0: LLM requests tool call
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "my_tool", "{}")]),
            // Round 1: fails with 429 (simulated by FailingLlm, call index 1)
            // — should NOT trim, should surface as user-visible error
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("my_tool".to_string(), "result".to_string());

        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            FailingLlm::new(responses, 1).with_error_variant(|| CoreError::RateLimited {
                retry_after: None,
                detail: "Anthropic API error (HTTP 429 Too Many Requests): rate_limit_error"
                    .into(),
            }),
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "Use my tool".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        // Should get a rate-limit user-visible message, not "adjusted my approach"
        assert!(result.contains("rate limit was exceeded"));

        // Verify NO system message about trimming was added
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let has_trim_msg = updated
            .messages
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("context became too long"));
        assert!(
            !has_trim_msg,
            "rate limit error should not trigger context trimming"
        );
    }

    #[tokio::test]
    async fn noop_executor_returns_empty_tools() {
        let executor = NoopToolExecutor;
        assert!(executor.core_tools().await.is_empty());
    }

    #[tokio::test]
    async fn noop_executor_returns_error() {
        let executor = NoopToolExecutor;
        let result = executor
            .execute_tool("anything", serde_json::json!({}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    struct CapturingLlm {
        seen_messages: Arc<Mutex<Vec<Message>>>,
    }

    impl LlmClient for CapturingLlm {
        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            _on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            // Only capture the first call (the main LLM turn). The second call
            // triggered by title generation must not overwrite the captured state
            // that the test assertions rely on.
            let mut seen = self.seen_messages.lock().unwrap();
            if seen.is_empty() {
                *seen = messages;
            }
            Ok(LlmResponse::text("ok"))
        }
    }

    #[tokio::test]
    async fn llm_input_includes_runtime_instruction_message() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let seen = Arc::new(Mutex::new(Vec::<Message>::new()));
        let counter = Arc::new(AtomicU64::new(0));

        let handler = ConversationHandler::new(
            MockStore::new(),
            CapturingLlm {
                seen_messages: Arc::clone(&seen),
            },
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let _ = handler
            .send_prompt(&conv.id, "hello".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let messages = seen.lock().unwrap();
        assert!(!messages.is_empty());
        assert_eq!(messages[0].role, Role::System);
        assert!(messages[0].content.contains(
            "You are Adele, a desktop assistant named in reference to the Adélie penguin"
        ));
        assert!(messages[0].content.contains("Your name is Adele"));
        assert!(
            messages[0]
                .content
                .contains("Follow these rules in priority order")
        );
        assert!(
            messages[0]
                .content
                .contains("Current-turn user instructions override all stored data")
        );
        assert!(messages[0].content.contains(
            "search the knowledge base first (project scope, then global) before using other tools"
        ));
        assert!(
            messages[0]
                .content
                .contains("If still unclear, ask one brief clarifying question and do not assume")
        );
        assert!(
            messages[0]
                .content
                .contains("do a short internal preflight")
        );
        assert!(
            messages[0]
                .content
                .contains("Validate facts relevant to the request before relying on them")
        );
        assert!(
            messages[0]
                .content
                .contains("No tools are available in this turn.")
        );
        assert!(messages[0].content.contains("non-blocking launch pattern"));
        assert!(messages[0].content.contains("check PATH"));
        assert!(messages[0].content.contains("check Flatpak and Snap"));
        assert!(
            messages[0]
                .content
                .contains("builtin_knowledge_base_write/search/delete")
        );
        assert!(messages[0].content.contains("builtin_sys_props"));
        assert!(messages[0].content.contains("builtin_tool_search"));
        assert!(messages[0].content.contains("Never fabricate tool outputs"));
    }

    #[test]
    fn runtime_instruction_enforces_kb_first_for_user_specific_requests() {
        use crate::prompts;

        let instruction = prompts::assemble(&prompts::static_sections());

        let priority_rule = "Current-turn user instructions override all stored data.";
        let kb_first = "If a request is user-specific, project-specific, or a reference is unclear, search the knowledge base first (project scope, then global) before using other tools.";
        let ambiguous_reference =
            "If still unclear, ask one brief clarifying question and do not assume.";
        let tool_fallback = "For tool-relevant requests (terminal, filesystem, D-Bus, network/web), use the best-fit available tool.";
        let no_guessing = "Do not guess user-specific details (project path, run command, package manager, editor, service name, account, or host).";
        let verify_relevant_facts = "Validate facts relevant to the request before relying on them, especially temporally variable details (machine settings, current date/time).";
        let no_fabrication =
            "Never fabricate tool outputs or claim a tool succeeded when it did not.";
        let tool_search_discovery = "Use builtin_tool_search to discover additional tools when the user's request might need capabilities beyond your current set.";

        assert!(instruction.contains(priority_rule));
        assert!(instruction.contains(kb_first));
        assert!(instruction.contains(ambiguous_reference));
        assert!(instruction.contains(no_guessing));
        assert!(instruction.contains(verify_relevant_facts));
        assert!(instruction.contains(tool_fallback));
        assert!(instruction.contains(no_fabrication));
        assert!(instruction.contains(tool_search_discovery));

        let priority_rule_pos = instruction.find(priority_rule).unwrap();
        let kb_first_pos = instruction.find(kb_first).unwrap();
        let ambiguous_reference_pos = instruction.find(ambiguous_reference).unwrap();
        let no_guessing_pos = instruction.find(no_guessing).unwrap();
        let verify_relevant_facts_pos = instruction.find(verify_relevant_facts).unwrap();
        let tool_fallback_pos = instruction.find(tool_fallback).unwrap();

        assert!(
            priority_rule_pos < kb_first_pos,
            "priority rule must remain before knowledge base decision rules"
        );
        assert!(
            kb_first_pos < tool_fallback_pos,
            "kb-first rule must remain before tool fallback rule"
        );
        assert!(
            tool_fallback_pos < no_guessing_pos,
            "tool usage rules must come before discovery/verification guardrails"
        );
        assert!(
            no_guessing_pos < ambiguous_reference_pos,
            "no-guessing guardrail must come before ambiguity guardrail"
        );
        assert!(
            ambiguous_reference_pos < verify_relevant_facts_pos,
            "ambiguity guardrail must come before fact-validation guardrail"
        );
    }

    // --- Title generation tests ---

    #[test]
    fn sanitize_generated_title_basic() {
        assert_eq!(
            sanitize_generated_title("Weather Forecast Today"),
            "Weather Forecast Today"
        );
    }

    #[test]
    fn sanitize_generated_title_strips_quotes_and_punctuation() {
        assert_eq!(
            sanitize_generated_title("\"Fix Broken Build Pipeline\""),
            "Fix Broken Build Pipeline"
        );
        assert_eq!(
            sanitize_generated_title("'Deploy to Production.'"),
            "Deploy to Production"
        );
    }

    #[test]
    fn sanitize_generated_title_takes_first_line_only() {
        assert_eq!(
            sanitize_generated_title("Rust Memory Debug\nSome explanation here"),
            "Rust Memory Debug"
        );
    }

    #[test]
    fn sanitize_generated_title_limits_to_eight_words() {
        let long = "One Two Three Four Five Six Seven Eight Nine Ten";
        assert_eq!(
            sanitize_generated_title(long),
            "One Two Three Four Five Six Seven Eight"
        );
    }

    #[tokio::test]
    async fn send_prompt_generates_title_on_first_message() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            ToolCallingLlm::new(vec![
                LlmResponse::text("That sounds great!"),   // main response
                LlmResponse::text("Plan Weekend Getaway"), // title generation
            ]),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );
        let conv = handler
            .create_conversation("New Chat".into())
            .await
            .unwrap();
        assert_eq!(conv.title, "New Chat");

        handler
            .send_prompt(
                &conv.id,
                "Let's plan a trip!".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.title, "Plan Weekend Getaway");
    }

    #[tokio::test]
    async fn send_prompt_does_not_overwrite_title_on_second_message() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            ToolCallingLlm::new(vec![
                LlmResponse::text("First response"), // main response round 1
                LlmResponse::text("Generated Title Here"), // title generation round 1
                LlmResponse::text("Second response"), // main response round 2
            ]),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );
        let conv = handler
            .create_conversation("New Chat".into())
            .await
            .unwrap();

        // First prompt — sets the title
        handler
            .send_prompt(&conv.id, "Hello".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        let after_first = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(after_first.title, "Generated Title Here");

        // Second prompt — title must remain unchanged
        handler
            .send_prompt(
                &conv.id,
                "Follow-up question".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();
        let after_second = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(after_second.title, "Generated Title Here");
    }

    #[tokio::test]
    async fn send_prompt_keeps_original_title_when_generation_returns_empty() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            ToolCallingLlm::new(vec![
                LlmResponse::text("Sure, I can help."), // main response
                LlmResponse::text(""),                  // title generation returns empty
            ]),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );
        let conv = handler.create_conversation("My Chat".into()).await.unwrap();

        handler
            .send_prompt(&conv.id, "Hi".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let updated = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.title, "My Chat");
    }

    #[tokio::test]
    async fn llm_input_runtime_instruction_lists_available_tools() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let seen = Arc::new(Mutex::new(Vec::<Message>::new()));
        let counter = Arc::new(AtomicU64::new(0));

        let tools = vec![ToolDefinition::new(
            "terminal",
            "Run terminal command",
            serde_json::json!({"type": "object"}),
        )];
        let tool_results = HashMap::new();

        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            CapturingLlm {
                seen_messages: Arc::clone(&seen),
            },
            MockToolExecutor::new(tools, tool_results),
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let _ = handler
            .send_prompt(&conv.id, "hello".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let messages = seen.lock().unwrap();
        assert!(!messages.is_empty());
        assert_eq!(messages[0].role, Role::System);
        assert!(
            messages[0]
                .content
                .contains("Available tools in this turn: terminal.")
        );
    }

    // --- Context windowing tests ---

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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES, None, 0, None, &default_estimate);
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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES, None, 0, None, &default_estimate);
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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES, None, 0, None, &default_estimate);

        // The first conversation message (after System) must be a User message.
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].role, Role::User);

        // The tail must be preserved intact.
        let last = result.last().unwrap();
        assert_eq!(last.content, "final-reply");
    }

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

    // --- Pre-flight token-budget assembly tests (issue #65) ---

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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
            None,
            &default_estimate,
        );

        // 1 system message + MAX_CONTEXT_MESSAGES conversation messages.
        assert_eq!(result.len(), MAX_CONTEXT_MESSAGES + 1);
    }

    #[test]
    fn assembly_shrinks_when_over_token_budget() {
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
        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
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
    fn assembly_does_not_shrink_below_min_context_messages() {
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
        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
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

    #[tokio::test]
    async fn recovery_picks_largest_by_token_estimate_not_bytes() {
        // Two tool results with the same byte length but different
        // token-estimate weights (using the chars/4 default):
        //
        //  - `ascii`  = 256 ASCII bytes = 256 chars → 64 estimated tokens
        //  - `emoji`  = 64 emoji × 4 bytes = 256 bytes / 64 chars → 16 tokens
        //
        // With the byte-length picker (the pre-#65 logic) both ties: the
        // first one to enumerate would win. With token-estimate ranking
        // the ASCII result wins unambiguously.
        use std::sync::atomic::AtomicU64;

        let call_count = Arc::new(AtomicU32::new(0));
        let llm = OverflowThenSucceedLlm {
            remaining_overflows: Mutex::new(1),
            call_count: Arc::clone(&call_count),
            ok_text: "ok".into(),
        };
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        // Build two payloads where byte length and estimated tokens
        // give different rankings. Both clear MIN_TRUNCATION_TOKENS so
        // either could be picked by step 1; the picker must choose by
        // token estimate, not bytes.
        //
        //   ASCII:  8192 chars × 1 byte  =  8192 bytes / 2048 est. tokens
        //   Emoji:  4096 chars × 4 bytes = 16384 bytes / 1024 est. tokens
        //
        // Bytes alone would pick emoji (more bytes); tokens pick ASCII
        // (more estimated cost). That's the regression this guards.
        let ascii_payload: String = "A".repeat(8192);
        let emoji_one = "\u{1F600}"; // 4 bytes, 1 char
        let emoji_payload: String = emoji_one.repeat(4096);
        assert!(
            emoji_payload.len() > ascii_payload.len(),
            "emoji payload must have more bytes so byte-picker would mis-target"
        );
        assert!(
            ascii_payload.chars().count() > emoji_payload.chars().count(),
            "ASCII payload must have more chars so token-picker prefers it"
        );

        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "ascii", "t", "{}",
            )]));
        stored
            .messages
            .push(Message::tool_result("ascii", &ascii_payload));
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "emoji", "t", "{}",
            )]));
        stored
            .messages
            .push(Message::tool_result("emoji", &emoji_payload));
        handler.store.update(stored).await.unwrap();

        handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let after = handler.get_conversation(&conv.id).await.unwrap();
        let ascii_after = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("ascii"))
            .expect("ascii result preserved");
        let emoji_after = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("emoji"))
            .expect("emoji result preserved");
        assert!(
            ascii_after.content.starts_with("<tool output omitted"),
            "token-estimate picker should target the ASCII result, got: {:?}",
            &ascii_after.content
        );
        assert_eq!(
            emoji_after.content, emoji_payload,
            "emoji result must be preserved verbatim — fewer estimated tokens"
        );
    }

    // --- Compaction tests ---

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

    // --- Token-pressure compaction tests ---

    /// Mock LLM that reports configurable token usage and a declared
    /// `max_context_tokens`, used to drive the token-pressure path in
    /// `send_prompt`.
    struct TokenReportingLlm {
        text: String,
        input_tokens: u64,
        max_context: Option<u64>,
    }

    impl LlmClient for TokenReportingLlm {
        fn max_context_tokens(&self) -> Option<u64> {
            self.max_context
        }

        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            on_chunk(self.text.clone());
            let usage = TokenUsage {
                input_tokens: Some(self.input_tokens),
                output_tokens: Some(10),
                ..Default::default()
            };
            Ok(LlmResponse::text(self.text.clone()).with_usage(usage))
        }
    }

    #[tokio::test]
    async fn send_prompt_shrinks_window_on_token_pressure() {
        use crate::ports::llm::{BudgetSource, ContextBudget, with_context_budget};
        use std::sync::atomic::{AtomicU64, Ordering};

        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            TokenReportingLlm {
                text: "ok".into(),
                input_tokens: 180_000, // 90% of 200K — above 85% threshold
                max_context: Some(200_000),
            },
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        // Prime the conversation with enough messages to exceed the default
        // window, so shrinking it triggers a new compaction range.
        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        for i in 0..(MAX_CONTEXT_MESSAGES + 20) {
            if i % 2 == 0 {
                stored
                    .messages
                    .push(Message::new(Role::User, format!("u-{i}")));
            } else {
                stored
                    .messages
                    .push(Message::new(Role::Assistant, format!("a-{i}")));
            }
        }
        handler.store.update(stored).await.unwrap();

        let before = handler.get_conversation(&conv.id).await.unwrap();
        let baseline_compacted = before.compacted_through;

        // Install the resolved budget the daemon's wrapper would set
        // (issue #63) so the token-pressure check fires. Without the
        // wrapper, `current_context_budget()` returns `None` and the
        // token-pressure branch skips.
        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            // Drive a turn that will receive high token usage and trigger
            // the token-pressure shrink + compaction path.
            handler
                .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
                .await
                .unwrap();
        })
        .await;

        let after = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            after.compacted_through > baseline_compacted,
            "token pressure should have advanced compacted_through"
        );
    }

    #[tokio::test]
    async fn send_prompt_no_shrink_when_tokens_under_threshold() {
        use crate::ports::llm::{BudgetSource, ContextBudget, with_context_budget};
        use std::sync::atomic::{AtomicU64, Ordering};

        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            TokenReportingLlm {
                text: "ok".into(),
                input_tokens: 100_000, // 50% — below threshold
                max_context: Some(200_000),
            },
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();
        // Small conversation: no windowing, no compaction expected.
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        for _ in 0..5 {
            stored.messages.push(Message::new(Role::User, "hi"));
        }
        handler.store.update(stored).await.unwrap();

        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            handler
                .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
                .await
                .unwrap();
        })
        .await;

        let after = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(
            after.compacted_through, 0,
            "no compaction expected when token usage is below threshold"
        );
    }

    // --- Overflow-recovery tests ---

    #[test]
    fn overflow_truncation_notice_includes_byte_count_and_hint() {
        let notice = overflow_truncation_notice(12_345, Some(203_524), Some(200_000));
        assert!(notice.contains("12345 bytes"));
        assert!(notice.contains("203524"));
        assert!(notice.contains("200000"));
        assert!(notice.contains("narrower") || notice.contains("chunk") || notice.contains("smaller"));
    }

    #[test]
    fn overflow_truncation_notice_omits_counts_when_unknown() {
        let notice = overflow_truncation_notice(500, None, None);
        assert!(notice.contains("500 bytes"));
        assert!(!notice.contains("prompt was"));
    }

    /// LLM that returns `ContextOverflow` for a configurable number of
    /// calls before succeeding. Tracks call count so tests can assert on it.
    struct OverflowThenSucceedLlm {
        remaining_overflows: Mutex<u32>,
        call_count: Arc<AtomicU32>,
        ok_text: String,
    }

    impl LlmClient for OverflowThenSucceedLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            let mut remaining = self.remaining_overflows.lock().unwrap();
            if *remaining > 0 {
                *remaining -= 1;
                return Err(CoreError::ContextOverflow {
                    prompt_tokens: Some(203_524),
                    max_tokens: Some(200_000),
                    detail: "Bedrock validation error: prompt is too long".into(),
                });
            }
            drop(remaining);
            on_chunk(self.ok_text.clone());
            Ok(LlmResponse::text(self.ok_text.clone()))
        }
    }

    #[tokio::test]
    async fn recovery_step1_truncates_largest_tool_result() {
        // Step 1 of the ladder: when there is at least one tool result
        // bigger than MIN_TRUNCATION_TOKENS (in estimated tokens), truncate
        // the largest and retry.
        // Smaller tool results stay untouched.
        use std::sync::atomic::AtomicU64;

        let call_count = Arc::new(AtomicU32::new(0));
        let llm = OverflowThenSucceedLlm {
            remaining_overflows: Mutex::new(1),
            call_count: Arc::clone(&call_count),
            ok_text: "all done".into(),
        };
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        // Prime the conversation with three tool results: one tiny, one
        // medium-but-still-below-threshold, one well above the threshold.
        // Only the third should be truncated.
        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1", "tiny", "{}",
            )]));
        stored.messages.push(Message::tool_result("c1", "ok"));
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c2", "medium", "{}",
            )]));
        // 2048 chars ≈ 512 tokens (chars/4 default) — below the
        // 1024-token threshold, so step 1 should leave it alone.
        let medium_content = "m".repeat((MIN_TRUNCATION_TOKENS * 2) as usize);
        stored
            .messages
            .push(Message::tool_result("c2", &medium_content));
        // 16384 chars ≈ 4096 tokens — well above the 1024-token threshold.
        let big_content = "X".repeat((MIN_TRUNCATION_TOKENS * 16) as usize);
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c3", "big", "{}",
            )]));
        stored.messages.push(Message::tool_result("c3", &big_content));
        handler.store.update(stored).await.unwrap();

        let result = handler
            .send_prompt(
                &conv.id,
                "what happened?".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();

        assert_eq!(result, "all done");
        assert_eq!(
            call_count.load(Ordering::Relaxed),
            2,
            "expected one overflow + one retry"
        );

        let after = handler.get_conversation(&conv.id).await.unwrap();
        let small = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("c1"))
            .expect("small tool result present");
        assert_eq!(small.content, "ok", "small tool result must be untouched");
        let medium = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("c2"))
            .expect("medium tool result present");
        assert_eq!(
            medium.content, medium_content,
            "below-threshold tool result must be untouched"
        );
        let big = after
            .messages
            .iter()
            .find(|m| m.tool_call_id.as_deref() == Some("c3"))
            .expect("big tool result present");
        assert!(
            big.content.starts_with("<tool output omitted"),
            "expected truncation notice, got: {:?}",
            &big.content
        );
        assert!(big.content.contains(&format!("{} bytes", big_content.len())));
    }

    #[tokio::test]
    async fn recovery_step2_trims_oldest_pairs_when_no_large_results() {
        // Step 2 of the ladder: when no tool result is large enough to be
        // worth truncating but multiple tool-pair groups exist, drop the
        // oldest groups via `trim_tool_pairs`. The most recent group must
        // survive.
        use std::sync::atomic::AtomicU64;

        let call_count = Arc::new(AtomicU32::new(0));
        let llm = OverflowThenSucceedLlm {
            remaining_overflows: Mutex::new(1),
            call_count: Arc::clone(&call_count),
            ok_text: "ok".into(),
        };
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        // Four tool-pair groups, all with tiny results (below
        // MIN_TRUNCATION_TOKENS in estimated tokens) so step 1 declines.
        for i in 1..=4 {
            stored
                .messages
                .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                    format!("c{i}"),
                    "tiny",
                    "{}",
                )]));
            stored
                .messages
                .push(Message::tool_result(format!("c{i}"), format!("res-{i}")));
        }
        handler.store.update(stored).await.unwrap();

        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert_eq!(result, "ok");

        let after = handler.get_conversation(&conv.id).await.unwrap();
        // The most recent group must remain intact, regardless of how many
        // older groups got trimmed.
        let kept_recent = after
            .messages
            .iter()
            .any(|m| m.tool_call_id.as_deref() == Some("c4"));
        assert!(kept_recent, "the most recent tool group must survive");
        let dropped_oldest = !after
            .messages
            .iter()
            .any(|m| m.tool_call_id.as_deref() == Some("c1"));
        assert!(dropped_oldest, "the oldest tool group must be trimmed");
    }

    #[tokio::test]
    async fn recovery_step3_summarises_when_nothing_to_trim() {
        // Step 3 of the ladder: with no tool results to truncate and no
        // tool-pair groups to trim, recovery falls through to summarising
        // and shrinking the active window. The rolling summary on the
        // conversation should advance after recovery runs.
        use std::sync::atomic::AtomicU64;

        struct OverflowThenSucceedWithSummary {
            remaining_overflows: Mutex<u32>,
            ok_text: String,
            summary_text: String,
            call_count: Arc<AtomicU32>,
        }

        impl LlmClient for OverflowThenSucceedWithSummary {
            fn max_context_tokens(&self) -> Option<u64> {
                Some(200_000)
            }

            async fn stream_completion(
                &self,
                messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                mut on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                // The summary-generation call passes a system prompt that
                // contains "conversation summarizer". Detect it and reply
                // with the canned summary text instead of the OK text.
                let is_summary_call = messages.iter().any(|m| {
                    m.role == Role::System && m.content.contains("conversation summarizer")
                });
                if is_summary_call {
                    on_chunk(self.summary_text.clone());
                    return Ok(LlmResponse::text(self.summary_text.clone()));
                }
                self.call_count.fetch_add(1, Ordering::Relaxed);
                let mut remaining = self.remaining_overflows.lock().unwrap();
                if *remaining > 0 {
                    *remaining -= 1;
                    return Err(CoreError::ContextOverflow {
                        prompt_tokens: Some(300_000),
                        max_tokens: Some(200_000),
                        detail: "prompt too long".into(),
                    });
                }
                drop(remaining);
                on_chunk(self.ok_text.clone());
                Ok(LlmResponse::text(self.ok_text.clone()))
            }
        }

        let call_count = Arc::new(AtomicU32::new(0));
        let counter = Arc::new(AtomicU64::new(0));
        let llm = OverflowThenSucceedWithSummary {
            remaining_overflows: Mutex::new(1),
            ok_text: "done".into(),
            summary_text: "- recovery summary".into(),
            call_count: Arc::clone(&call_count),
        };
        let handler = ConversationHandler::new(
            MockStore::new(),
            llm,
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();
        // Prime with enough plain User/Assistant turns to push the window
        // past `MAX_CONTEXT_MESSAGES`, so step 3 has a non-empty range to
        // summarise. No tool calls are present, so steps 1 and 2 decline.
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        for i in 0..(MAX_CONTEXT_MESSAGES + 4) {
            if i % 2 == 0 {
                stored
                    .messages
                    .push(Message::new(Role::User, format!("u-{i}")));
            } else {
                stored
                    .messages
                    .push(Message::new(Role::Assistant, format!("a-{i}")));
            }
        }
        handler.store.update(stored).await.unwrap();
        let baseline = handler
            .get_conversation(&conv.id)
            .await
            .unwrap()
            .context_summary
            .clone();

        let result = handler
            .send_prompt(
                &conv.id,
                "follow-up".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();
        assert_eq!(result, "done");
        // One overflow + one retry from the main path; the inner summary
        // call doesn't bump call_count.
        assert_eq!(
            call_count.load(Ordering::Relaxed),
            2,
            "expected one overflow + one retry"
        );

        let after = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            after.context_summary != baseline && !after.context_summary.is_empty(),
            "step 3 must update the rolling summary; got: {:?}",
            after.context_summary
        );
    }

    #[tokio::test]
    async fn recovery_exhausts_retries_then_surfaces() {
        // After MAX_OVERFLOW_RETRIES recoveries the loop must surface a
        // user-visible error rather than spin forever.
        struct AlwaysOverflowLlm {
            call_count: Arc<AtomicU32>,
        }
        impl LlmClient for AlwaysOverflowLlm {
            async fn stream_completion(
                &self,
                _messages: Vec<Message>,
                _tools: &[ToolDefinition],
                _reasoning: ReasoningConfig,
                _on_chunk: ChunkCallback,
            ) -> Result<LlmResponse, CoreError> {
                self.call_count.fetch_add(1, Ordering::Relaxed);
                Err(CoreError::ContextOverflow {
                    prompt_tokens: Some(300_000),
                    max_tokens: Some(200_000),
                    detail: "prompt is too long".into(),
                })
            }
        }

        use std::sync::atomic::AtomicU64;
        let call_count = Arc::new(AtomicU32::new(0));
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::new(
            MockStore::new(),
            AlwaysOverflowLlm {
                call_count: Arc::clone(&call_count),
            },
            Box::new(move || {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        );

        let conv = handler.create_conversation("Test".into()).await.unwrap();
        let mut stored = handler.get_conversation(&conv.id).await.unwrap();
        stored
            .messages
            .push(Message::assistant_with_tool_calls(vec![ToolCall::new(
                "c1", "t", "{}",
            )]));
        stored
            .messages
            .push(Message::tool_result(
                "c1",
                "x".repeat((MIN_TRUNCATION_TOKENS * 8) as usize),
            ));
        handler.store.update(stored).await.unwrap();

        let result = handler
            .send_prompt(&conv.id, "go".into(), noop_callback(), noop_status())
            .await
            .unwrap();
        assert!(result.to_ascii_lowercase().contains("context"));

        // MAX_OVERFLOW_RETRIES + 1 calls total: the recovered attempts plus
        // the final one whose error gets surfaced.
        assert_eq!(
            call_count.load(Ordering::Relaxed),
            MAX_OVERFLOW_RETRIES + 1,
            "should stop after bounded retries"
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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "- User prefers dark mode",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "- Some summary",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES, None, 0, None, &default_estimate);

        // System prompt directly followed by windowed messages — no summary
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].role, Role::User);
    }

    // --- Active-task anchor tests ---

    #[tokio::test]
    async fn active_task_anchor_set_on_user_prompt() {
        let handler = make_handler(vec!["ok"]);
        let conv = handler.create_conversation("Chat".into()).await.unwrap();

        handler
            .send_prompt(
                &conv.id,
                "refactor the auth module".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .expect("send_prompt succeeds");

        let stored = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(
            stored.active_task.as_deref(),
            Some("refactor the auth module"),
            "the user's prompt should be captured as the active-task anchor"
        );
    }

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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            Some(task),
            0,
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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            Some(task),
            0,
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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            Some(task),
            6,
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
        let result =
            llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES, None, 0, None, &default_estimate);

        let any_anchor = result
            .iter()
            .any(|m| m.role == Role::System && m.content.starts_with("[Current task]"));
        assert!(!any_anchor, "no anchor should be injected when active_task is None");
    }

    #[test]
    fn active_task_not_injected_when_empty_string() {
        let msgs = vec![Message::new(Role::User, "hello")];
        let result =
            llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES, Some(""), 99, None, &default_estimate);

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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "- earlier conversation summary",
            MAX_CONTEXT_MESSAGES,
            Some(task),
            0,
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

    #[tokio::test]
    async fn summariser_prompt_requires_active_task_header() {
        // The system prompt used by the rolling summariser must require the
        // model to lead with an "Active task:" line so the goal survives
        // even when the layer-3b injection conditions are misjudged.
        struct CapturingSummariserLlm {
            seen: Arc<Mutex<Option<Vec<Message>>>>,
        }
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

        let result =
            llm_messages_for_turn(&msgs, &summaries, &[], &[], "", MAX_CONTEXT_MESSAGES, None, 0, None, &default_estimate);

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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES, None, 0, None, &default_estimate);
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

        let result =
            llm_messages_for_turn(&msgs, &summaries, &[], &[], "", MAX_CONTEXT_MESSAGES, None, 0, None, &default_estimate);
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

        let result = llm_messages_for_turn(
            &msgs,
            &summaries,
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
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

        let result = llm_messages_for_turn(
            &msgs,
            &summaries,
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
            None,
            &default_estimate,
        );

        assert!(
            result.iter().all(|m| !m.content.contains("Old context.")),
            "summary whose tagged messages fall outside the window must not be injected"
        );
    }

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
        let llm = FailingLlm::new(vec![], 0);
        let result = generate_context_summary("existing summary", &messages, &llm).await;
        assert_eq!(result, "existing summary");
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

    // --- Tool namespace categorization cache tests ---
    //
    // These exercise the `namespace_cache` on `ConversationHandler` by driving
    // `send_prompt` end-to-end with a mock LLM that supports hosted tool
    // search. The mock recognises the categorization call by its system prompt
    // and counts how many times the categorizer is invoked across calls.

    /// System-prompt fragment unique to `categorize_tool_namespaces`.
    /// Used by the mock LLM to distinguish categorization calls from
    /// regular completion calls.
    const CATEGORIZATION_SYSTEM_FRAGMENT: &str = "You organize tools into semantic categories";

    /// Mock LLM that:
    /// - Reports hosted tool search support so the cache path runs.
    /// - Counts categorization calls (system prompt fragment match) and
    ///   returns a deterministic JSON categorization for them.
    /// - For all other calls returns plain text so `send_prompt` exits.
    struct CategorizingLlm {
        categorization_calls: Arc<AtomicU32>,
        category_payload: Mutex<String>,
    }

    impl CategorizingLlm {
        fn new(category_payload: String) -> Self {
            Self {
                categorization_calls: Arc::new(AtomicU32::new(0)),
                category_payload: Mutex::new(category_payload),
            }
        }

        fn calls(&self) -> Arc<AtomicU32> {
            Arc::clone(&self.categorization_calls)
        }
    }

    impl LlmClient for CategorizingLlm {
        fn supports_hosted_tool_search(&self) -> bool {
            true
        }

        async fn stream_completion(
            &self,
            messages: Vec<Message>,
            _tools: &[ToolDefinition],
            _reasoning: ReasoningConfig,
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let is_categorization = messages.iter().any(|m| {
                matches!(m.role, Role::System) && m.content.contains(CATEGORIZATION_SYSTEM_FRAGMENT)
            });
            if is_categorization {
                self.categorization_calls.fetch_add(1, Ordering::SeqCst);
                let payload = self.category_payload.lock().unwrap().clone();
                return Ok(LlmResponse::text(payload));
            }
            let text = "ok".to_string();
            on_chunk(text.clone());
            Ok(LlmResponse::text(text))
        }
    }

    /// Mock tool executor with a mutable namespace set so individual tests
    /// can edit names/descriptions between `send_prompt` calls.
    struct NamespacedToolExecutor {
        namespaces: Mutex<Vec<ToolNamespace>>,
    }

    impl NamespacedToolExecutor {
        fn new(namespaces: Vec<ToolNamespace>) -> Self {
            Self {
                namespaces: Mutex::new(namespaces),
            }
        }

        fn mutate<F: FnOnce(&mut Vec<ToolNamespace>)>(&self, f: F) {
            let mut guard = self.namespaces.lock().unwrap();
            f(&mut guard);
        }
    }

    impl ToolExecutor for NamespacedToolExecutor {
        async fn core_tools(&self) -> Vec<ToolDefinition> {
            Vec::new()
        }

        async fn search_tools(&self, _query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(Vec::new())
        }

        async fn tool_definition(
            &self,
            _name: &str,
        ) -> Result<Option<ToolDefinition>, CoreError> {
            Ok(None)
        }

        async fn tool_namespaces(&self) -> Vec<ToolNamespace> {
            self.namespaces.lock().unwrap().clone()
        }

        async fn execute_tool(
            &self,
            name: &str,
            _arguments: serde_json::Value,
        ) -> Result<String, CoreError> {
            Err(CoreError::ToolExecution(format!("unexpected exec: {name}")))
        }
    }

    /// Build a single namespace containing `count` distinct tools so the
    /// total tool count exceeds `categorize_tool_namespaces`'s skip threshold.
    fn make_oversized_namespace(count: usize) -> ToolNamespace {
        let tools: Vec<ToolDefinition> = (0..count)
            .map(|i| {
                ToolDefinition::new(
                    format!("tool_{i}"),
                    format!("Description for tool {i}"),
                    serde_json::json!({"type": "object"}),
                )
            })
            .collect();
        ToolNamespace::new("seed_namespace", "Seed namespace for tests", tools)
    }

    /// Categorization payload that puts every `tool_*` into one bucket.
    /// Construction matches `make_oversized_namespace` so the LLM-shaped
    /// JSON is internally consistent and `categorize_tool_namespaces`
    /// accepts it (every tool appears in exactly one category).
    fn make_categorization_payload(count: usize) -> String {
        let names: Vec<String> = (0..count).map(|i| format!("\"tool_{i}\"")).collect();
        format!(
            r#"[{{"name":"all","description":"All tools","tools":[{}]}}]"#,
            names.join(",")
        )
    }

    fn build_categorization_handler(
        executor: NamespacedToolExecutor,
        llm: CategorizingLlm,
    ) -> ConversationHandler<MockStore, CategorizingLlm, NamespacedToolExecutor> {
        use std::sync::atomic::{AtomicU64, Ordering as IdOrdering};
        let counter = Arc::new(AtomicU64::new(0));
        ConversationHandler::with_tools(
            MockStore::new(),
            llm,
            executor,
            Box::new(move || {
                let n = counter.fetch_add(1, IdOrdering::Relaxed) + 1;
                format!("conv-{n}")
            }),
        )
    }

    #[tokio::test]
    async fn categorization_cache_hits_on_unchanged_tools() {
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        handler
            .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: first send_prompt with valid conv must succeed");
        handler
            .send_prompt(&conv.id, "second".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: second send_prompt with valid conv must succeed");

        // Cache hit on second call: categorizer runs at most once.
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "categorizer should run once and be served from cache thereafter"
        );
    }

    #[tokio::test]
    async fn categorization_cache_invalidates_on_description_change() {
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        handler
            .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: first send_prompt must succeed");

        // Mutate a description without changing any name. Without
        // descriptions in the hash, the cache would falsely hit.
        handler.tools.mutate(|namespaces| {
            namespaces[0].tools[0].description = "Description for tool 0 (edited)".to_string();
        });

        handler
            .send_prompt(&conv.id, "second".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: second send_prompt must succeed");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "description change must invalidate the categorization cache"
        );
    }

    #[tokio::test]
    async fn categorization_cache_invalidates_on_tool_addition() {
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        // Pre-build a payload that covers the post-addition tool set so the
        // second categorization call returns valid JSON for `count + 1` tools.
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        handler
            .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: first send_prompt must succeed");

        // Add a tool, then update the LLM's stored payload so the second
        // categorization succeeds (and thus actually runs end-to-end).
        handler.tools.mutate(|namespaces| {
            namespaces[0].tools.push(ToolDefinition::new(
                format!("tool_{count}"),
                "Description for added tool",
                serde_json::json!({"type": "object"}),
            ));
        });
        *handler.llm.category_payload.lock().unwrap() = make_categorization_payload(count + 1);

        handler
            .send_prompt(&conv.id, "second".into(), noop_callback(), noop_status())
            .await
            .expect("invariant: second send_prompt must succeed");

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "tool addition must invalidate the categorization cache"
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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &tools,
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
            Some(budget),
            &default_estimate,
        );

        let system = result
            .iter()
            .find(|m| m.role == Role::System)
            .expect("system message must be present");
        assert!(
            system.content.contains(&format!("There are {} tools across", tools.len())),
            "demoted system block must include 'There are <N> tools' wording, \
             got: {:?}",
            system.content
        );
        assert!(
            !system.content.contains(&format!("Available tools in this turn: {}", tools[0].name)),
            "demoted system block must not enumerate every tool name, got: {:?}",
            system.content
        );
    }

    #[test]
    fn system_block_full_when_under_threshold() {
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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &tools,
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
            Some(budget),
            &default_estimate,
        );

        let system = result
            .iter()
            .find(|m| m.role == Role::System)
            .expect("system message must be present");
        assert!(
            system.content.contains("Available tools in this turn: ping."),
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

        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &tools,
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            None,
            0,
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
            !system.content.contains(&format!("There are {} tools across", tools.len())),
            "demoted summary must not appear when no budget installed"
        );
    }

    #[tokio::test]
    async fn categorization_skipped_when_listing_fits_budget() {
        use crate::ports::llm::with_context_budget;

        // Generous budget + short tool descriptions — the raw listing
        // sums well below 10% of the budget, so categorization should
        // skip the LLM round-trip and return the input namespaces.
        let count = 12;
        let executor = NamespacedToolExecutor::new(vec![make_oversized_namespace(count)]);
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            handler
                .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
                .await
                .expect("invariant: first send_prompt must succeed");
        })
        .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "categorizer must not be called when the full listing fits the budget"
        );
    }

    #[tokio::test]
    async fn categorization_runs_when_listing_too_large() {
        use crate::ports::llm::with_context_budget;

        // Same setup but with very long tool descriptions, so the raw
        // listing pushes past the 10% threshold and categorization runs.
        let count = 12;
        let big_desc = "DESCRIPTION ".repeat(1000); // ~12 KB per tool
        let tools: Vec<ToolDefinition> = (0..count)
            .map(|i| {
                ToolDefinition::new(
                    format!("tool_{i}"),
                    big_desc.clone(),
                    serde_json::json!({"type": "object"}),
                )
            })
            .collect();
        let namespace = ToolNamespace::new("seed", "seed namespace", tools);
        let executor = NamespacedToolExecutor::new(vec![namespace]);
        let llm = CategorizingLlm::new(make_categorization_payload(count));
        let calls = llm.calls();
        let handler = build_categorization_handler(executor, llm);

        let conv = handler.create_conversation("Test".into()).await.unwrap();

        let budget = ContextBudget {
            max_input_tokens: 200_000,
            source: BudgetSource::ConnectorTable,
        };
        with_context_budget(budget, async {
            handler
                .send_prompt(&conv.id, "first".into(), noop_callback(), noop_status())
                .await
                .expect("invariant: first send_prompt must succeed");
        })
        .await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "categorizer must run once when the raw listing exceeds the budget threshold"
        );
    }
}
