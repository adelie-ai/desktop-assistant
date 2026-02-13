use std::collections::HashMap;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::tools::ToolExecutor;
use tokio::sync::Mutex;

use crate::{McpClient, McpError};

/// Configuration for an MCP server.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

/// Adapter implementing `ToolExecutor` by managing multiple MCP server connections.
/// Routes tool calls to the correct MCP server based on tool name.
pub struct McpToolExecutor {
    configs: Vec<McpServerConfig>,
    /// Map from tool name to the index of the server that provides it.
    tool_routing: Mutex<HashMap<String, usize>>,
    /// Connected MCP client instances, indexed by config position.
    clients: Mutex<Vec<Option<McpClient>>>,
    /// Cached list of all available tools.
    cached_tools: Mutex<Vec<ToolDefinition>>,
}

impl McpToolExecutor {
    pub fn new(configs: Vec<McpServerConfig>) -> Self {
        let clients: Vec<Option<McpClient>> = (0..configs.len()).map(|_| None).collect();
        Self {
            configs,
            tool_routing: Mutex::new(HashMap::new()),
            clients: Mutex::new(clients),
            cached_tools: Mutex::new(Vec::new()),
        }
    }

    /// Connect to all configured MCP servers, discover their tools,
    /// and build the routing table.
    pub async fn start(&self) -> Result<(), McpError> {
        let mut clients = self.clients.lock().await;
        let mut routing = self.tool_routing.lock().await;
        let mut all_tools = Vec::new();

        for (idx, config) in self.configs.iter().enumerate() {
            tracing::info!(
                "connecting to MCP server '{}': {}",
                config.name,
                config.command
            );

            match McpClient::connect(&config.command, &config.args).await {
                Ok(mut client) => match client.list_tools().await {
                    Ok(tools) => {
                        tracing::info!(
                            "MCP server '{}' provides {} tools",
                            config.name,
                            tools.len()
                        );
                        for tool in &tools {
                            tracing::debug!("  tool: {}", tool.name);
                            routing.insert(tool.name.clone(), idx);
                        }
                        all_tools.extend(tools);
                        clients[idx] = Some(client);
                    }
                    Err(e) => {
                        tracing::error!(
                            "failed to list tools from MCP server '{}': {e}",
                            config.name
                        );
                        client.shutdown().await;
                    }
                },
                Err(e) => {
                    tracing::error!("failed to connect to MCP server '{}': {e}", config.name);
                }
            }
        }

        *self.cached_tools.lock().await = all_tools;
        Ok(())
    }

    /// Shut down all connected MCP servers.
    pub async fn shutdown(self) {
        let mut clients = self.clients.lock().await;
        for client in clients.iter_mut() {
            if let Some(c) = client.take() {
                c.shutdown().await;
            }
        }
    }
}

impl ToolExecutor for McpToolExecutor {
    async fn available_tools(&self) -> Vec<ToolDefinition> {
        self.cached_tools.lock().await.clone()
    }

    async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        let routing = self.tool_routing.lock().await;
        let server_idx = routing
            .get(name)
            .ok_or_else(|| CoreError::ToolExecution(format!("unknown tool: {name}")))?;
        let idx = *server_idx;
        drop(routing);

        let mut clients = self.clients.lock().await;
        let client = clients[idx].as_mut().ok_or_else(|| {
            CoreError::ToolExecution(format!("MCP server for tool '{name}' is not connected"))
        })?;

        client
            .call_tool(name, arguments)
            .await
            .map_err(|e| CoreError::ToolExecution(format!("tool '{name}' failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_creation_with_empty_configs() {
        let executor = McpToolExecutor::new(vec![]);
        // Should not panic
        assert!(executor.configs.is_empty());
    }

    #[test]
    fn server_config_construction() {
        let config = McpServerConfig {
            name: "fileio".into(),
            command: "fileio-mcp".into(),
            args: vec![],
        };
        assert_eq!(config.name, "fileio");
        assert_eq!(config.command, "fileio-mcp");
        assert!(config.args.is_empty());
    }

    #[test]
    fn server_config_with_args() {
        let config = McpServerConfig {
            name: "genmcp".into(),
            command: "genmcp".into(),
            args: vec!["--config".into(), "/path/to/config.toml".into()],
        };
        assert_eq!(config.args.len(), 2);
    }

    #[tokio::test]
    async fn executor_no_configs_returns_empty_tools() {
        let executor = McpToolExecutor::new(vec![]);
        let tools = executor.available_tools().await;
        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn executor_unknown_tool_returns_error() {
        let executor = McpToolExecutor::new(vec![]);
        let result = executor
            .execute_tool("nonexistent", serde_json::json!({}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }
}
