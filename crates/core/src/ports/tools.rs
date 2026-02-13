use crate::CoreError;
use crate::domain::ToolDefinition;

/// Outbound port for executing tools (e.g., via MCP servers).
pub trait ToolExecutor: Send + Sync {
    /// Returns all available tools from all connected tool providers.
    fn available_tools(&self) -> impl std::future::Future<Output = Vec<ToolDefinition>> + Send;

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
        async fn available_tools(&self) -> Vec<ToolDefinition> {
            self.tools.clone()
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
        let tools = executor.available_tools().await;
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
}
