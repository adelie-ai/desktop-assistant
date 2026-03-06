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
    fn register_tools(
        &self,
        tools: Vec<ToolDefinition>,
        source: &str,
        is_core: bool,
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

    fn _assert_tool_registry<T: ToolRegistryStore>() {}
}
