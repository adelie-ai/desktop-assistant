use crate::CoreError;
use crate::domain::{
    Conversation, ConversationId, ConversationSummary, Message, MessageSummary, Role,
    ToolDefinition, ToolNamespace,
};
use crate::ports::inbound::ConversationService;
use crate::ports::llm::{ChunkCallback, LlmClient, StatusCallback};
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

fn now_timestamp() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn cutoff_timestamp(max_age_days: u32) -> String {
    (Local::now() - Duration::days(i64::from(max_age_days)))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

fn llm_messages_for_turn(
    conversation_messages: &[Message],
    summaries: &[MessageSummary],
    tool_defs: &[ToolDefinition],
    deferred_namespaces: &[ToolNamespace],
    context_summary: &str,
    max_messages: usize,
) -> Vec<Message> {
    use crate::prompts::{self, PromptSection, PromptSectionKind};

    let has_tool_search = tool_defs.iter().any(|t| t.name == "builtin_tool_search");
    let tool_note = if tool_defs.is_empty() && deferred_namespaces.is_empty() {
        "No tools are available in this turn.".to_string()
    } else {
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
    };

    // Assemble system prompt from static sections + dynamic tool availability.
    let mut sections = prompts::static_sections();
    sections.push(PromptSection::new(
        PromptSectionKind::ToolAvailability,
        tool_note,
    ));
    let system_instruction = prompts::assemble(&sections);

    // Apply context windowing: if the conversation exceeds the limit, keep
    // only the most recent messages, snapping the cut point forward to a
    // genuine User message so we never split tool-call/result pairs.
    let start = window_start(conversation_messages, max_messages);
    let windowed = &conversation_messages[start..];
    let is_windowed = start > 0;

    // Build a map from start_ordinal -> summary for active summaries in the window.
    let summary_map: std::collections::HashMap<usize, &MessageSummary> =
        summaries.iter().map(|s| (s.start_ordinal, s)).collect();

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

    // Track which summaries have already been injected.
    let mut injected_summaries: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for (i, msg) in windowed.iter().enumerate() {
        let ordinal = start + i;

        if let Some(sid) = &msg.summary_id
            && active_summary_ids.contains(sid.as_str())
        {
            // This message is collapsed. Inject the summary at the first
            // collapsed message we encounter for this summary.
            if !injected_summaries.contains(sid.as_str()) {
                injected_summaries.insert(sid);
                // Look up by ordinal first, then fall back to finding by ID
                // (the window may not start at the summary's start_ordinal).
                let found = summary_map
                    .get(&ordinal)
                    .copied()
                    .or_else(|| summaries.iter().find(|s| s.id == *sid));
                if let Some(s) = found {
                    messages.push(Message::new(
                        Role::System,
                        format!(
                            "[Summary of messages {}\u{2013}{}] {}",
                            s.start_ordinal, s.end_ordinal, s.summary
                        ),
                    ));
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

use crate::ports::llm::is_retryable_error;

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

fn user_visible_llm_error_message(error: &CoreError) -> String {
    let raw = error.to_string();
    let normalized = raw.to_ascii_lowercase();

    if normalized.contains("429")
        || normalized.contains("rate_limit")
        || normalized.contains("529")
        || normalized.contains("overloaded")
    {
        return format!(
            "The API rate limit was exceeded. Please wait a moment and try again. Details: {raw}"
        );
    }

    if normalized.contains("does not support tools") {
        return format!(
            "This Ollama model does not support tool use. Please switch to a tool-capable model or disable tools for this chat. Details: {raw}"
        );
    }

    if normalized.contains("unable to load model")
        || normalized.contains("model not found")
        || normalized.contains("pull model manifest")
        || normalized.contains("no such file")
    {
        return format!(
            "The selected model could not be loaded or found. Please verify the model name and that it is installed in Ollama. Details: {raw}"
        );
    }

    if normalized.contains("downloading")
        || normalized.contains("currently loading")
        || normalized.contains("is loading")
        || normalized.contains("loading model")
    {
        return format!(
            "The model is still downloading or loading. Please wait a moment and try again. Details: {raw}"
        );
    }

    format!("I hit an LLM backend error and could not complete this request. Details: {raw}")
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
        .stream_completion(messages, &[], Box::new(|_| true))
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
            "You are a conversation summarizer. Produce a concise bullet-point summary of the \
             key points, decisions, user preferences, and established facts from the conversation. \
             Merge with any existing summary provided. Keep the summary under 500 words. \
             Output ONLY the bullet-point summary, no preamble.",
        ),
        Message::new(Role::User, prompt),
    ];

    match llm
        .stream_completion(llm_messages, &[], Box::new(|_| true))
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

/// Use the LLM to semantically categorize tools into descriptive namespaces.
///
/// Takes the raw tool namespaces (typically grouped by MCP server) and asks
/// the LLM to reorganize them into ≤10-tool categories with descriptive names.
/// Falls back to the original namespaces on failure.
async fn categorize_tool_namespaces<L: LlmClient>(
    namespaces: Vec<ToolNamespace>,
    llm: &L,
) -> Vec<ToolNamespace> {
    // Collect all tools across namespaces. If there are very few, skip categorization.
    let all_tools: Vec<&ToolDefinition> = namespaces.iter().flat_map(|ns| &ns.tools).collect();
    if all_tools.len() <= 10 {
        return namespaces;
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
        .stream_completion(messages, &[], Box::new(|_| true))
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

/// Compute a stable hash over the sorted tool names in a set of namespaces.
fn tool_set_hash(namespaces: &[ToolNamespace]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut names: Vec<&str> = namespaces
        .iter()
        .flat_map(|ns| &ns.tools)
        .map(|t| t.name.as_str())
        .collect();
    names.sort_unstable();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    names.hash(&mut hasher);
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

        // Effective window size for this turn. May shrink further if the
        // provider reports input-token usage above COMPACTION_TOKEN_RATIO.
        let mut target_window = MAX_CONTEXT_MESSAGES;

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
                    tracing::debug!("reusing cached namespace categorization (hash={hash:#x})");
                    ns
                } else {
                    let result = categorize_tool_namespaces(raw_namespaces, self.task_llm()).await;
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
            let llm_messages = llm_messages_for_turn(
                &conv.messages,
                &conv.summaries,
                &tool_defs,
                deferred_ns,
                &conv.context_summary,
                target_window,
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

            let response =
                match if use_hosted_search && !namespaces.is_empty() && !hosted_search_demoted {
                    self.llm
                        .stream_completion_with_namespaces(
                            llm_messages,
                            &tool_defs,
                            &namespaces,
                            filtered_chunk_callback,
                        )
                        .await
                } else {
                    self.llm
                        .stream_completion(llm_messages, &tool_defs, filtered_chunk_callback)
                        .await
                } {
                    Ok(r) => r,
                    Err(e)
                        if round > 0
                            && !is_retryable_error(&e)
                            && !user_visible_llm_error_message(&e)
                                .contains("rate limit was exceeded") =>
                    {
                        // Mid-loop LLM error (e.g. context too long) — trim old
                        // tool call/result pairs and tell the LLM what happened
                        // so it can adjust its approach.
                        tracing::warn!(
                            "LLM call failed on round {}/{}, trimming context: {e}",
                            round + 1,
                            MAX_TOOL_ROUNDS
                        );
                        let removed = trim_tool_pairs(&mut conv.messages);
                        conv.compacted_through = conv.compacted_through.saturating_sub(removed);
                        tracing::info!("removed {removed} messages to reduce context");
                        if removed == 0 {
                            // Nothing left to trim — retrying won't help.
                            let friendly = user_visible_llm_error_message(&e);
                            conv.messages.push(Message::new(Role::Assistant, &friendly));
                            conv.updated_at = now_timestamp();
                            self.store.update(conv).await?;
                            return Ok(friendly);
                        }
                        conv.messages.push(Message::new(
                            Role::System,
                            format!(
                                "Your previous tool call could not be processed because \
                             the context became too long. {removed} older messages were \
                             trimmed. The original error was: {e}\n\
                             Please adjust your approach — for example, request less \
                             output or take a different path."
                            ),
                        ));
                        on_chunk = Box::new(|_| true);
                        continue;
                    }
                    Err(e) => {
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
            if let (Some(max_tokens), Some(usage)) =
                (self.llm.max_context_tokens(), response.usage.as_ref())
                && let Some(input_tokens) = usage.input_tokens
            {
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
    use crate::ports::llm::{LlmResponse, TokenUsage};
    use std::collections::HashMap;
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
        error_message: String,
    }

    impl FailingLlm {
        fn new(responses: Vec<LlmResponse>, fail_on_call: usize) -> Self {
            Self {
                responses: Mutex::new(responses),
                fail_on_call,
                call_count: Mutex::new(0),
                error_message: "context_length_exceeded".into(),
            }
        }

        fn with_error(mut self, msg: &str) -> Self {
            self.error_message = msg.into();
            self
        }
    }

    impl LlmClient for FailingLlm {
        async fn stream_completion(
            &self,
            _messages: Vec<Message>,
            _tools: &[ToolDefinition],
            mut on_chunk: ChunkCallback,
        ) -> Result<LlmResponse, CoreError> {
            let call_idx = {
                let mut count = self.call_count.lock().unwrap();
                let idx = *count;
                *count += 1;
                idx
            };

            if call_idx == self.fail_on_call {
                return Err(CoreError::Llm(self.error_message.clone()));
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
    async fn tool_loop_recovers_from_context_length_error() {
        let tools = vec![ToolDefinition::new(
            "big_tool",
            "Returns lots of data",
            serde_json::json!({}),
        )];

        let responses = vec![
            // Round 0: LLM requests first tool call
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c1", "big_tool", "{}")]),
            // Round 1: LLM requests second tool call (creates 2 groups so trim can remove one)
            LlmResponse::with_tool_calls("", vec![ToolCall::new("c2", "big_tool", "{}")]),
            // Round 2: fails (simulated by FailingLlm, call index 2)
            // Round 3 (retry after trim): LLM succeeds with final text
            LlmResponse::text("I adjusted my approach"),
        ];

        let mut tool_results = HashMap::new();
        tool_results.insert("big_tool".to_string(), "x".repeat(1000));

        use std::sync::atomic::{AtomicU64, Ordering};
        let counter = Arc::new(AtomicU64::new(0));
        let handler = ConversationHandler::with_tools(
            MockStore::new(),
            FailingLlm::new(responses, 2), // fail on 3rd LLM call (after 2 tool groups exist)
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
                "Use big tool".into(),
                noop_callback(),
                noop_status(),
            )
            .await
            .unwrap();
        assert_eq!(result, "I adjusted my approach");

        // Verify the conversation has a system message about trimming
        let updated = handler.get_conversation(&conv.id).await.unwrap();
        let has_system_msg = updated
            .messages
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("context became too long"));
        assert!(has_system_msg);
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
        let err = CoreError::Llm(
            r#"Ollama API error (HTTP 400 Bad Request): {\"error\":\"registry.ollama.ai/library/phi4:14b does not support tools\"}"#
                .to_string(),
        );
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("does not support tool use"));
    }

    #[test]
    fn user_visible_error_for_missing_model() {
        let err = CoreError::Llm(
            r#"Ollama API error (HTTP 500 Internal Server Error): {\"error\":\"unable to load model\"}"#
                .to_string(),
        );
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("could not be loaded or found"));
    }

    #[test]
    fn user_visible_error_for_loading_model() {
        let err = CoreError::Llm(
            r#"Ollama API error (HTTP 503 Service Unavailable): {\"error\":\"model is currently loading\"}"#
                .to_string(),
        );
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("still downloading or loading"));
    }

    #[test]
    fn user_visible_error_for_rate_limit_429() {
        let err = CoreError::Llm(
            r#"Anthropic API error (HTTP 429 Too Many Requests): {"error":{"type":"rate_limit_error","message":"Rate limited"}}"#
                .to_string(),
        );
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("rate limit was exceeded"));
    }

    #[test]
    fn user_visible_error_for_overloaded_529() {
        let err = CoreError::Llm("Anthropic API error (HTTP 529): overloaded".to_string());
        let msg = user_visible_llm_error_message(&err);
        assert!(msg.contains("rate limit was exceeded"));
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
            FailingLlm::new(responses, 1)
                .with_error("Anthropic API error (HTTP 429 Too Many Requests): rate_limit_error"),
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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES);
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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES);
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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES);

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

        // Drive a turn that will receive high token usage and trigger
        // the token-pressure shrink + compaction path.
        handler
            .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let after = handler.get_conversation(&conv.id).await.unwrap();
        assert!(
            after.compacted_through > baseline_compacted,
            "token pressure should have advanced compacted_through"
        );
    }

    #[tokio::test]
    async fn send_prompt_no_shrink_when_tokens_under_threshold() {
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

        handler
            .send_prompt(&conv.id, "next".into(), noop_callback(), noop_status())
            .await
            .unwrap();

        let after = handler.get_conversation(&conv.id).await.unwrap();
        assert_eq!(
            after.compacted_through, 0,
            "no compaction expected when token usage is below threshold"
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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES);

        // System prompt directly followed by windowed messages — no summary
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].role, Role::User);
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
            start_ordinal: 1,
            end_ordinal: 3,
        }];

        let result = llm_messages_for_turn(&msgs, &summaries, &[], &[], "", MAX_CONTEXT_MESSAGES);

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

        let result = llm_messages_for_turn(&msgs, &[], &[], &[], "", MAX_CONTEXT_MESSAGES);
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
                start_ordinal: 1,
                end_ordinal: 2,
            },
            MessageSummary {
                id: "s2".to_string(),
                summary: "Second batch.".to_string(),
                start_ordinal: 4,
                end_ordinal: 5,
            },
        ];

        let result = llm_messages_for_turn(&msgs, &summaries, &[], &[], "", MAX_CONTEXT_MESSAGES);
        // System + "start" + summary1 + "middle" + summary2 + "end" = 6
        assert_eq!(result.len(), 6);
        assert!(result[2].content.contains("First batch."));
        assert_eq!(result[3].content, "middle");
        assert!(result[4].content.contains("Second batch."));
        assert_eq!(result[5].content, "end");
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
}
