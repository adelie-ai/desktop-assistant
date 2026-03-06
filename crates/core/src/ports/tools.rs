use crate::CoreError;
use crate::domain::{ToolDefinition, ToolNamespace};

/// Outbound port for executing tools (e.g., via MCP servers).
///
/// Supports dynamic tool discovery: `core_tools()` returns the small set
/// always sent to the LLM, while `search_tools()` and `tool_definition()`
/// allow on-demand lookup of additional tools.
pub trait ToolExecutor: Send + Sync {
    /// Returns the core tools that should always be included in LLM requests.
    fn core_tools(&self) -> impl std::future::Future<Output = Vec<ToolDefinition>> + Send;

    /// Search for tools matching a query. Used by the `builtin_tool_search` tool.
    fn search_tools(
        &self,
        query: &str,
    ) -> impl std::future::Future<Output = Result<Vec<ToolDefinition>, CoreError>> + Send;

    /// Look up a single tool definition by name.
    fn tool_definition(
        &self,
        name: &str,
    ) -> impl std::future::Future<Output = Result<Option<ToolDefinition>, CoreError>> + Send;

    /// Returns tools grouped into namespaces for hosted tool search.
    ///
    /// Default returns empty — connectors that don't support hosted tool search
    /// ignore this entirely. When non-empty, the service layer can pass these
    /// to `LlmClient::stream_completion_with_namespaces()`.
    fn tool_namespaces(&self) -> impl std::future::Future<Output = Vec<ToolNamespace>> + Send {
        async { vec![] }
    }

    /// Execute a tool by name with the given arguments.
    /// Returns the tool's text output.
    fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> impl std::future::Future<Output = Result<String, CoreError>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockToolExecutor {
        tools: Vec<ToolDefinition>,
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
            if self.tools.iter().any(|t| t.name == name) {
                Ok(format!("result from {name}"))
            } else {
                Err(CoreError::ToolExecution(format!("unknown tool: {name}")))
            }
        }
    }

    #[tokio::test]
    async fn mock_executor_returns_tools() {
        let executor = MockToolExecutor {
            tools: vec![ToolDefinition::new(
                "test_tool",
                "A test",
                serde_json::json!({"type": "object"}),
            )],
        };
        let tools = executor.core_tools().await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "test_tool");
    }

    #[tokio::test]
    async fn mock_executor_executes_known_tool() {
        let executor = MockToolExecutor {
            tools: vec![ToolDefinition::new(
                "read_file",
                "Read a file",
                serde_json::json!({}),
            )],
        };
        let result = executor
            .execute_tool("read_file", serde_json::json!({"path": "/tmp/test"}))
            .await
            .unwrap();
        assert_eq!(result, "result from read_file");
    }

    #[tokio::test]
    async fn mock_executor_rejects_unknown_tool() {
        let executor = MockToolExecutor { tools: vec![] };
        let result = executor
            .execute_tool("nonexistent", serde_json::json!({}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn mock_executor_searches_tools() {
        let executor = MockToolExecutor { tools: vec![] };
        let results = executor.search_tools("test").await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn mock_executor_looks_up_tool_definition() {
        let executor = MockToolExecutor {
            tools: vec![ToolDefinition::new(
                "my_tool",
                "desc",
                serde_json::json!({}),
            )],
        };
        let def = executor.tool_definition("my_tool").await.unwrap();
        assert!(def.is_some());
        assert_eq!(def.unwrap().name, "my_tool");

        let missing = executor.tool_definition("missing").await.unwrap();
        assert!(missing.is_none());
    }
}
