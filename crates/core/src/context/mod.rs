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

use crate::domain::{Conversation, Message, MessageSummary, Role, ToolDefinition, ToolNamespace};
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
pub(crate) fn llm_messages_for_turn(
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
        let assembled_tokens: u64 = assembled.iter().map(|m| estimate(&m.content)).sum();
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
        let threshold = (b.max_input_tokens as f64 * SYSTEM_BLOCK_BUDGET_RATIO) as u64;
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
            messages.push(Message::new(Role::System, format!("[Current task] {task}")));
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
    search
        .iter()
        .position(|m| m.role != Role::Tool)
        .map_or(tentative, |offset| tentative + offset)
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
        let result = llm_messages_for_turn(
            &msgs,
            &[],
            &[],
            &[],
            "",
            MAX_CONTEXT_MESSAGES,
            Some(""),
            99,
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
            !system
                .content
                .contains(&format!("There are {} tools across", tools.len())),
            "demoted summary must not appear when no budget installed"
        );
    }
}
