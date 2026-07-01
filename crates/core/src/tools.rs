//! Tool helpers shared by the conversation handler.
//!
//! Covers three concerns:
//! - LLM-driven categorization of raw tool namespaces into ≤10-tool buckets,
//!   with a budget-aware skip path for small listings.
//! - A stable hash over (name, description) pairs used as the cache key
//!   for categorization.
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

/// Max characters of a tool's arguments/result surfaced to a tool-activity
/// observer. Long enough to be informative in a log line, short enough to keep
/// the per-task log ring cheap and each entry single-line.
const TOOL_EVENT_SUMMARY_MAX: usize = 200;

/// Placeholder substituted for a sensitive argument value before it reaches
/// the activity feed (which is broadcast over WebSocket and D-Bus, issue #253).
const REDACTED: &str = "‹redacted›";

/// Whether an object key names a sensitive value that must be redacted before
/// it leaves the process in an activity-feed summary (issue #253). Matches
/// case-insensitively: keys *containing* a secret-bearing substring
/// (`key`, `token`, `secret`, `password`, `passwd`, `credential`) or keys that
/// *equal* a short auth marker (`auth`, `authorization`) where a substring
/// match would catch too much. Named so it can be unit-tested directly.
pub(crate) fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    const CONTAINS: [&str; 6] = ["key", "token", "secret", "password", "passwd", "credential"];
    if CONTAINS.iter().any(|needle| lower.contains(needle)) {
        return true;
    }
    matches!(lower.as_str(), "auth" | "authorization")
}

/// Compact, single-line rendering of a tool call's JSON arguments for an
/// activity feed. Objects render as `key=value` pairs (the common, readable
/// case); other shapes fall back to compact JSON. Empty for no-argument calls.
/// Values under a [`is_sensitive_key`] key are replaced with [`REDACTED`]
/// (issue #253), including inside nested objects. Truncated to
/// [`TOOL_EVENT_SUMMARY_MAX`] characters.
pub(crate) fn summarize_tool_value(args: &serde_json::Value) -> String {
    let rendered = match args {
        serde_json::Value::Null => String::new(),
        serde_json::Value::Object(map) if map.is_empty() => String::new(),
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, v)| {
                if is_sensitive_key(k) {
                    format!("{k}={REDACTED}")
                } else {
                    format!("{k}={}", compact_scalar(v))
                }
            })
            .collect::<Vec<_>>()
            .join(", "),
        other => other.to_string(),
    };
    truncate_single_line(&rendered, TOOL_EVENT_SUMMARY_MAX)
}

/// Compact, single-line rendering of a tool result (or error) string for an
/// activity feed. Truncated to [`TOOL_EVENT_SUMMARY_MAX`] characters.
pub(crate) fn summarize_tool_text(text: &str) -> String {
    truncate_single_line(text, TOOL_EVENT_SUMMARY_MAX)
}

/// Render a single JSON value compactly: strings unquoted (so `path=/tmp/x`
/// rather than `path="/tmp/x"`), everything else as compact JSON so a nested
/// argument can't expand the summary's structure. Nested objects have their
/// own keys checked for sensitivity (issue #253) so a secret tucked one level
/// down (e.g. `headers.authorization`) is still redacted.
fn compact_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => {
            let inner = map
                .iter()
                .map(|(k, val)| {
                    if is_sensitive_key(k) {
                        format!("{k:?}:{REDACTED:?}")
                    } else {
                        format!("{k:?}:{}", compact_value_json(val))
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
        // #433: an array value must be rendered through the redacting
        // renderer, not `to_string()`. A top-level arg like
        // `{"headers":[{"authorization":"Bearer …"}]}` would otherwise emit
        // the token verbatim to the activity feed (the Object arm redacts,
        // but arrays fell through to the raw `other` arm below).
        serde_json::Value::Array(_) => compact_value_json(v),
        other => other.to_string(),
    }
}

/// Recursive compact JSON rendering that redacts sensitive keys at every
/// object level (issue #253). Used for values nested inside an argument object
/// so a secret in a deeply nested structure is never emitted verbatim.
fn compact_value_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let inner = map
                .iter()
                .map(|(k, val)| {
                    if is_sensitive_key(k) {
                        format!("{k:?}:{REDACTED:?}")
                    } else {
                        format!("{k:?}:{}", compact_value_json(val))
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
        serde_json::Value::Array(items) => {
            let inner = items
                .iter()
                .map(compact_value_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        other => other.to_string(),
    }
}

/// Collapse internal whitespace runs to single spaces and cap the length,
/// appending an ellipsis when truncated. Keeps activity-feed lines tidy and
/// bounded regardless of how a tool formats its output.
fn truncate_single_line(s: &str, max: usize) -> String {
    let collapsed = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max {
        collapsed
    } else {
        let kept: String = collapsed.chars().take(max).collect();
        format!("{kept}…")
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
        .stream_completion(
            messages,
            &[],
            ReasoningConfig::default(),
            Box::new(|_| true),
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sensitive_key_predicate_matches_case_insensitively() {
        for k in [
            "key",
            "api_key",
            "API_KEY",
            "token",
            "AccessToken",
            "secret",
            "client_secret",
            "password",
            "Passwd",
            "credential",
            "credentials",
            "auth",
            "Authorization",
        ] {
            assert!(is_sensitive_key(k), "{k} should be sensitive");
        }
        for k in ["query", "path", "url", "author", "limit", "name"] {
            assert!(!is_sensitive_key(k), "{k} should not be sensitive");
        }
        // `author` contains "auth" only as a prefix, not the whole key, so the
        // equality arm (not a substring arm) keeps it clear.
        assert!(!is_sensitive_key("author"));
    }

    #[test]
    fn summarize_redacts_sensitive_values() {
        let out = summarize_tool_value(&json!({
            "query": "weather",
            "api_key": "sk-supersecret-12345",
        }));
        // Keys render in sorted order (serde_json::Map preserves insertion
        // order by default, but assert on substrings to stay order-agnostic).
        assert!(out.contains("query=weather"), "got: {out}");
        assert!(out.contains("api_key=‹redacted›"), "got: {out}");
        assert!(!out.contains("supersecret"), "secret value leaked: {out}");
    }

    #[test]
    fn summarize_redaction_is_case_insensitive() {
        let out = summarize_tool_value(&json!({ "Authorization": "Bearer abc.def" }));
        assert_eq!(out, "Authorization=‹redacted›");
        assert!(!out.contains("Bearer"), "got: {out}");
    }

    #[test]
    fn summarize_redacts_nested_sensitive_values() {
        let out = summarize_tool_value(&json!({
            "config": {
                "host": "example.com",
                "password": "hunter2",
            }
        }));
        assert!(out.contains("example.com"), "got: {out}");
        assert!(!out.contains("hunter2"), "nested secret leaked: {out}");
        assert!(out.contains("‹redacted›"), "got: {out}");
    }

    #[test]
    fn summarize_passes_through_non_sensitive_args() {
        // serde_json::Map renders keys in sorted order by default, so the
        // pairs come out alphabetically — assert on that ordering.
        let out = summarize_tool_value(&json!({ "path": "/tmp/x", "limit": 5 }));
        assert_eq!(out, "limit=5, path=/tmp/x");
    }

    #[test]
    fn summarize_redacts_secret_in_array_of_objects() {
        // #433: a secret nested inside an array-valued arg must be redacted.
        // Before the fix, the array fell through to `to_string()` and the
        // token leaked verbatim into the activity feed.
        let out = summarize_tool_value(&json!({
            "headers": [
                { "name": "Accept", "value": "application/json" },
                { "authorization": "Bearer sk-abc123secret" },
            ]
        }));
        assert!(
            !out.contains("sk-abc123secret"),
            "secret in array-of-objects leaked: {out}"
        );
        assert!(out.contains("‹redacted›"), "expected redaction marker: {out}");
    }

    #[test]
    fn summarize_redacts_secret_in_nested_array_under_object() {
        // Array nested one level deeper (object -> object -> array) must
        // also redact, exercising the recursive array walk.
        let out = summarize_tool_value(&json!({
            "request": { "cookies": [ { "session_token": "s3cr3t-value" } ] }
        }));
        assert!(
            !out.contains("s3cr3t-value"),
            "secret in nested array leaked: {out}"
        );
        assert!(out.contains("‹redacted›"), "expected redaction marker: {out}");
    }

    #[test]
    fn summarize_renders_non_sensitive_array_readably() {
        // Regression: non-sensitive array args must still render (not be
        // dropped or over-redacted).
        let out = summarize_tool_value(&json!({ "ids": [1, 2, 3] }));
        assert!(out.contains("ids="), "got: {out}");
        assert!(out.contains('1') && out.contains('3'), "array content lost: {out}");
    }
}
