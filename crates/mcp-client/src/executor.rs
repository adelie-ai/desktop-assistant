use std::collections::HashMap;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::embedding::EmbedFn;
use desktop_assistant_core::ports::tools::ToolExecutor;
use tokio::sync::Mutex;

use crate::builtin::BuiltinToolService;
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
    /// Cached metadata for MCP resources across all connected servers.
    cached_resources: Mutex<Vec<serde_json::Value>>,
    /// Cached metadata for MCP prompts across all connected servers.
    cached_prompts: Mutex<Vec<serde_json::Value>>,
    /// Built-in in-process tools (preferences + factual memory).
    builtin_tools: BuiltinToolService,
}

impl McpToolExecutor {
    pub fn new(configs: Vec<McpServerConfig>) -> Self {
        Self::with_builtin_tools(configs, BuiltinToolService::from_default_paths())
    }

    pub fn new_with_embedding(
        configs: Vec<McpServerConfig>,
        embed_fn: EmbedFn,
        embedding_model: String,
    ) -> Self {
        let builtin_tools =
            BuiltinToolService::from_default_paths().with_embedding(embed_fn, embedding_model);
        Self::with_builtin_tools(configs, builtin_tools)
    }

    fn with_builtin_tools(
        configs: Vec<McpServerConfig>,
        builtin_tools: BuiltinToolService,
    ) -> Self {
        let clients: Vec<Option<McpClient>> = (0..configs.len()).map(|_| None).collect();
        Self {
            configs,
            tool_routing: Mutex::new(HashMap::new()),
            clients: Mutex::new(clients),
            cached_tools: Mutex::new(Vec::new()),
            cached_resources: Mutex::new(Vec::new()),
            cached_prompts: Mutex::new(Vec::new()),
            builtin_tools,
        }
    }

    /// Connect to all configured MCP servers, discover their tools,
    /// and build the routing table.
    pub async fn start(&self) -> Result<(), McpError> {
        {
            let mut clients = self.clients.lock().await;

            for (idx, config) in self.configs.iter().enumerate() {
                tracing::info!(
                    "connecting to MCP server '{}': {}",
                    config.name,
                    config.command
                );

                match McpClient::connect(&config.command, &config.args).await {
                    Ok(client) => {
                        clients[idx] = Some(client);
                    }
                    Err(e) => {
                        tracing::error!("failed to connect to MCP server '{}': {e}", config.name);
                    }
                }
            }
        }

        self.refresh_all_metadata().await?;
        Ok(())
    }

    async fn maybe_refresh_metadata(&self) -> Result<(), McpError> {
        let (tools_changed, resources_changed, prompts_changed) = {
            let clients = self.clients.lock().await;
            (
                clients
                    .iter()
                    .flatten()
                    .any(|client| client.tools_list_changed()),
                clients
                    .iter()
                    .flatten()
                    .any(|client| client.resources_list_changed()),
                clients
                    .iter()
                    .flatten()
                    .any(|client| client.prompts_list_changed()),
            )
        };

        if tools_changed {
            tracing::info!("MCP reported tools/list_changed, refreshing tool cache");
            self.refresh_tool_cache().await?;
        }

        if resources_changed {
            tracing::info!("MCP reported resources/list_changed, refreshing resources cache");
            self.refresh_resources_cache().await?;
        }

        if prompts_changed {
            tracing::info!("MCP reported prompts/list_changed, refreshing prompts cache");
            self.refresh_prompts_cache().await?;
        }

        Ok(())
    }

    async fn refresh_all_metadata(&self) -> Result<(), McpError> {
        self.refresh_tool_cache().await?;
        self.refresh_resources_cache().await?;
        self.refresh_prompts_cache().await?;
        Ok(())
    }

    async fn refresh_tool_cache(&self) -> Result<(), McpError> {
        let mut all_tools = Vec::new();
        let mut new_routing = HashMap::new();

        {
            let mut clients = self.clients.lock().await;
            for (idx, client_slot) in clients.iter_mut().enumerate() {
                let Some(client) = client_slot.as_mut() else {
                    continue;
                };

                match client.list_tools().await {
                    Ok(tools) => {
                        tracing::info!(
                            "MCP server '{}' provides {} tools",
                            self.configs[idx].name,
                            tools.len()
                        );
                        for tool in &tools {
                            tracing::debug!("  tool: {}", tool.name);
                            new_routing.insert(tool.name.clone(), idx);
                        }
                        all_tools.extend(tools);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "failed to refresh tools from MCP server '{}': {e}",
                            self.configs[idx].name
                        );
                    }
                }
            }
        }

        *self.tool_routing.lock().await = new_routing;
        *self.cached_tools.lock().await = all_tools;

        Ok(())
    }

    async fn refresh_resources_cache(&self) -> Result<(), McpError> {
        let mut all_resources = Vec::new();

        let mut clients = self.clients.lock().await;
        for (idx, client_slot) in clients.iter_mut().enumerate() {
            let Some(client) = client_slot.as_mut() else {
                continue;
            };

            match client.list_resources().await {
                Ok(resources) => {
                    tracing::info!(
                        "MCP server '{}' provides {} resources",
                        self.configs[idx].name,
                        resources.len()
                    );
                    all_resources.extend(resources);
                }
                Err(e) if is_method_not_found(&e) => {
                    tracing::debug!(
                        "MCP server '{}' does not implement resources/list",
                        self.configs[idx].name
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "failed to refresh resources from MCP server '{}': {e}",
                        self.configs[idx].name
                    );
                }
            }
        }

        *self.cached_resources.lock().await = all_resources;
        Ok(())
    }

    async fn refresh_prompts_cache(&self) -> Result<(), McpError> {
        let mut all_prompts = Vec::new();

        let mut clients = self.clients.lock().await;
        for (idx, client_slot) in clients.iter_mut().enumerate() {
            let Some(client) = client_slot.as_mut() else {
                continue;
            };

            match client.list_prompts().await {
                Ok(prompts) => {
                    tracing::info!(
                        "MCP server '{}' provides {} prompts",
                        self.configs[idx].name,
                        prompts.len()
                    );
                    all_prompts.extend(prompts);
                }
                Err(e) if is_method_not_found(&e) => {
                    tracing::debug!(
                        "MCP server '{}' does not implement prompts/list",
                        self.configs[idx].name
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "failed to refresh prompts from MCP server '{}': {e}",
                        self.configs[idx].name
                    );
                }
            }
        }

        *self.cached_prompts.lock().await = all_prompts;
        Ok(())
    }

    pub async fn available_resources(&self) -> Vec<serde_json::Value> {
        if let Err(e) = self.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP resources cache: {e}");
        }
        self.cached_resources.lock().await.clone()
    }

    pub async fn available_prompts(&self) -> Vec<serde_json::Value> {
        if let Err(e) = self.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP prompts cache: {e}");
        }
        self.cached_prompts.lock().await.clone()
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
        if let Err(e) = self.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP tools cache: {e}");
        }
        let mut tools = self.cached_tools.lock().await.clone();
        tools.extend(self.builtin_tools.tool_definitions());
        tools
    }

    async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        if BuiltinToolService::supports_tool(name) {
            return self.builtin_tools.execute_tool(name, arguments).await;
        }

        self.maybe_refresh_metadata()
            .await
            .map_err(|e| CoreError::ToolExecution(format!("failed to refresh tools: {e}")))?;

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

fn is_method_not_found(error: &McpError) -> bool {
    matches!(error, McpError::ServerError { code: -32601, .. })
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
        assert!(!tools.is_empty());
        assert!(
            tools
                .iter()
                .any(|tool| tool.name == "builtin_preferences_remember")
        );
    }

    #[tokio::test]
    async fn executor_no_configs_returns_empty_resources_and_prompts() {
        let executor = McpToolExecutor::new(vec![]);
        let resources = executor.available_resources().await;
        let prompts = executor.available_prompts().await;
        assert!(resources.is_empty());
        assert!(prompts.is_empty());
    }

    #[tokio::test]
    async fn executor_unknown_tool_returns_error() {
        let executor = McpToolExecutor::new(vec![]);
        let result = executor
            .execute_tool("nonexistent", serde_json::json!({}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn executor_includes_builtin_tools() {
        let executor = McpToolExecutor::new(vec![]);
        let tools = executor.available_tools().await;
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"builtin_preferences_remember"));
        assert!(names.contains(&"builtin_memory_update"));
    }

    #[tokio::test]
    async fn executor_executes_builtin_tool() {
        let executor = McpToolExecutor::new(vec![]);
        let result = executor
            .execute_tool(
                "builtin_preferences_remember",
                serde_json::json!({
                    "key": "editor",
                    "value": "vscode"
                }),
            )
            .await
            .unwrap();
        assert!(result.contains("\"ok\":true"));
    }
}
