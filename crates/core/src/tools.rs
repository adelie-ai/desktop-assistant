//! Tool helpers shared by the conversation handler.
//!
//! Covers three concerns:
//! - LLM-driven categorization of raw tool namespaces into ≤10-tool buckets,
//!   with a budget-aware skip path for small listings.
//! - A stable hash over (name, description) pairs used as the cache key
//!   for categorization.
//! - Human-readable status hints rendered by the dispatch loop while a
//!   tool is executing.
//! - [`NoopToolExecutor`], the default executor for handlers built without
//!   an MCP backend.

use crate::CoreError;
use crate::domain::{Message, Role, ToolDefinition, ToolNamespace};
use crate::ports::llm::{ContextBudget, LlmClient, ReasoningConfig};
use crate::ports::tools::ToolExecutor;

/// Fraction of the prompt-token budget below which the full tool listing
/// is considered cheap enough to enumerate without LLM-driven
/// categorization. When the raw `(name + description)` cost of every tool
/// fits within this slice, [`categorize_tool_namespaces`] returns the
/// input unchanged and skips the categorization round-trip.
///
/// Why 0.10: leaves headroom under the system-block demotion threshold so
/// a listing that passes this check is also unlikely to trigger demotion
/// at assembly time. Above the threshold the categorization LLM call is
/// worth the expense — it compresses the listing — but below it the
/// round-trip just adds latency and tokens.
const FULL_LISTING_FIT_RATIO: f64 = 0.10;

/// Generate a short, human-readable status message for a tool call.
pub(crate) fn tool_status_message(tool_name: &str, arguments: &serde_json::Value) -> String {
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
pub(crate) async fn categorize_tool_namespaces<L: LlmClient>(
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

/// Compute a stable hash over the tool set (names AND descriptions),
/// sorted by name so input ordering does not affect the hash.
///
/// Why: The hash is the cache key for `categorize_tool_namespaces`. That LLM
/// call sees both names and descriptions in its prompt, so a description
/// change can produce a different categorization. Hashing names alone would
/// hide such a change and serve a stale categorization. Re-categorizing on
/// any name OR description edit keeps the cache honest.
pub(crate) fn tool_set_hash(namespaces: &[ToolNamespace]) -> u64 {
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
