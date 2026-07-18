use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::ToolDefinition;

/// Outbound port for the tool definition registry (Postgres-backed).
/// Stores tool definitions with embeddings for hybrid search.
pub trait ToolRegistryStore: Send + Sync {
    /// Register (upsert) tool definitions from a source (e.g. an MCP server name or "builtin").
    /// Embeddings are chunk arrays (one Vec<f32> per chunk) to match the vector[] column.
    ///
    /// `provider` is the batch-constant provider identity (an MCP server's
    /// namespace/name, or a builtin group) written to every row in the batch —
    /// we register once per provider, so a single value per call is correct
    /// (mirrors `source`). `None` leaves the column unclassified. A batch may
    /// carry the provider's own synthetic `provider:<provider>` row; any *other*
    /// row literally named `provider:*` is rejected (it must never be dispatchable).
    fn register_tools(
        &self,
        tools: Vec<ToolDefinition>,
        source: &str,
        is_core: bool,
        provider: Option<&str>,
        embeddings: Vec<Option<Vec<Vec<f32>>>>,
        embedding_model: Option<String>,
    ) -> impl Future<Output = Result<(), CoreError>> + Send;

    /// Remove all tool definitions registered by a given source.
    fn unregister_source(&self, source: &str)
    -> impl Future<Output = Result<(), CoreError>> + Send;

    /// Return tool definitions marked as core (always sent to LLM).
    fn core_tools(&self) -> impl Future<Output = Result<Vec<ToolDefinition>, CoreError>> + Send;

    /// Hybrid search for tool definitions using vector similarity + full-text search via RRF.
    fn search_tools(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ToolDefinition>, CoreError>> + Send;

    /// Look up a single tool definition by name.
    fn tool_definition(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<Option<ToolDefinition>, CoreError>> + Send;
}

/// Boxed async closure for searching tool definitions through non-generic boundaries.
pub type ToolSearchFn = Arc<
    dyn Fn(
            String,
            Vec<f32>,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolDefinition>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for looking up a single tool definition.
pub type ToolDefinitionFn = Arc<
    dyn Fn(
            String,
        )
            -> Pin<Box<dyn Future<Output = Result<Option<ToolDefinition>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// A provider (an MCP server or a builtin group) and its member tools, handed
/// across the reindex boundary so the daemon can register each provider's tools
/// *and* a synthetic, searchable `provider:<name>` row (see [`Self::synthetic_row`]).
///
/// Widening the reindex payload from a flat `Vec<ToolDefinition>` to per-provider
/// groups is what lets tool-search surface a whole server's/group's tools when
/// its provider row matches — the unifying concept across external MCP servers
/// and Adele's own builtins.
pub struct ReindexProvider {
    /// Stable provider identity — an MCP server's namespace/name, or a builtin
    /// group. Becomes the `provider` column value and the `provider:<name>` row.
    pub name: String,
    /// Row source for the persistence sweep: `"mcp"` or `"builtin"`.
    pub source: &'static str,
    /// Resolved provider description (server instructions, config description, or
    /// an authored builtin blurb) seeding the synthetic row's searchable text.
    pub description: String,
    /// The provider's member tools.
    pub tools: Vec<ToolDefinition>,
}

impl ReindexProvider {
    /// The synthetic, searchable `provider:<name>` row for this provider: the
    /// description followed by the member tool names, so a tool-search query that
    /// hits the description or any member name matches the provider and boosts
    /// its members. Non-routable (registered `is_core = FALSE`, excluded from
    /// search results, and never dispatched — see the guards in `crates/storage`).
    pub fn synthetic_row(&self) -> ToolDefinition {
        let members = self
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        ToolDefinition::new(
            format!("provider:{}", self.name),
            format!("{} Tools: {members}.", self.description),
            serde_json::json!({}),
        )
    }
}

/// Boxed async closure for re-writing the persistent tool-search index with a
/// fresh set of provider groups.
///
/// Why: runtime MCP enable/disable changes the connected-tool set, but
/// `crates/mcp-client` must not depend on `crates/storage` (its only workspace
/// dep is `desktop-assistant-core`) and `ToolRegistryStore` is not
/// dyn-compatible (RPIT in trait position), so a boxed closure - not
/// `Arc<dyn ToolRegistryStore>` - is the boundary. The daemon injects a closure
/// that owns the storage-touching policy (delete-then-reinsert the `"mcp"`
/// source, registering each provider's tools plus its synthetic row, with NULL
/// embeddings for the background backfill to fill); the executor only hands over
/// the current [`ReindexProvider`] groups. Mirrors [`ToolSearchFn`] /
/// [`ToolDefinitionFn`].
pub type ToolReindexFn = Arc<
    dyn Fn(Vec<ReindexProvider>) -> Pin<Box<dyn Future<Output = Result<(), CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    struct MockToolRegistry;

    impl ToolRegistryStore for MockToolRegistry {
        async fn register_tools(
            &self,
            _tools: Vec<ToolDefinition>,
            _source: &str,
            _is_core: bool,
            _provider: Option<&str>,
            _embeddings: Vec<Option<Vec<Vec<f32>>>>,
            _embedding_model: Option<String>,
        ) -> Result<(), CoreError> {
            Ok(())
        }

        async fn unregister_source(&self, _source: &str) -> Result<(), CoreError> {
            Ok(())
        }

        async fn core_tools(&self) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(vec![])
        }

        async fn search_tools(
            &self,
            _query: &str,
            _query_embedding: Vec<f32>,
            _limit: usize,
        ) -> Result<Vec<ToolDefinition>, CoreError> {
            Ok(vec![])
        }

        async fn tool_definition(&self, _name: &str) -> Result<Option<ToolDefinition>, CoreError> {
            Ok(None)
        }
    }

    #[tokio::test]
    async fn mock_registry_core_tools_empty() {
        let registry = MockToolRegistry;
        let tools = registry.core_tools().await.unwrap();
        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn mock_registry_search_returns_empty() {
        let registry = MockToolRegistry;
        let tools = registry.search_tools("test", vec![0.0], 10).await.unwrap();
        assert!(tools.is_empty());
    }

    #[test]
    fn provider_row_description_includes_member_tool_names() {
        // The synthetic row's searchable text is the provider description plus
        // every member tool name, so a query for a member name matches the
        // provider (and boosts all its members).
        let provider = ReindexProvider {
            name: "weather".to_string(),
            source: "mcp",
            description: "Live weather and forecasts.".to_string(),
            tools: vec![
                ToolDefinition::new("weather__forecast", "d", serde_json::json!({})),
                ToolDefinition::new("weather__alerts", "d", serde_json::json!({})),
            ],
        };
        let row = provider.synthetic_row();
        assert_eq!(
            row.name, "provider:weather",
            "synthetic name is provider:<name>"
        );
        assert_eq!(
            row.description,
            "Live weather and forecasts. Tools: weather__forecast, weather__alerts.",
            "the row text carries the description AND the member tool names"
        );
        assert_eq!(
            row.parameters,
            serde_json::json!({}),
            "no callable parameters"
        );
    }

    fn _assert_tool_registry<T: ToolRegistryStore>() {}
}
