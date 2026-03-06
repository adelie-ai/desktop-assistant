use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{ToolDefinition, ToolNamespace};
use desktop_assistant_core::ports::tools::ToolExecutor;
use tokio::sync::{Mutex, RwLock};

pub use crate::builtin::BuiltinToolService;
use crate::config::save_mcp_configs;
use crate::{McpClient, McpError};

fn default_enabled() -> bool {
    true
}

/// Configuration for an MCP server.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Optional namespace prefix. When set, all tools from this server are
    /// exposed as `{namespace}__{tool_name}`. When absent, tool names are
    /// passed through unchanged.
    pub namespace: Option<String>,
    /// Whether this server is enabled. Disabled servers are not started.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

/// Status information for an MCP server.
#[derive(Debug, Clone, serde::Serialize)]
pub struct McpServerStatusInfo {
    pub name: String,
    pub command: String,
    pub enabled: bool,
    pub status: String,
    pub tool_count: u32,
}

/// Shared mutable state for MCP servers, accessible via `McpControlHandle`.
pub struct McpExecutorState {
    configs: RwLock<Vec<McpServerConfig>>,
    /// Connected MCP client instances, indexed by config position.
    clients: Mutex<Vec<Option<McpClient>>>,
    /// Map from namespaced tool name (`server__original`) to (server index, original tool name).
    tool_routing: Mutex<HashMap<String, (usize, String)>>,
    /// Cached list of all available tools.
    cached_tools: Mutex<Vec<ToolDefinition>>,
    /// Cached metadata for MCP resources across all connected servers.
    cached_resources: Mutex<Vec<serde_json::Value>>,
    /// Cached metadata for MCP prompts across all connected servers.
    cached_prompts: Mutex<Vec<serde_json::Value>>,
    /// Path to the MCP config file (for persisting changes).
    config_path: PathBuf,
}

impl McpExecutorState {
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
            let configs = self.configs.read().await;
            let mut clients = self.clients.lock().await;
            for (idx, client_slot) in clients.iter_mut().enumerate() {
                let Some(client) = client_slot.as_mut() else {
                    continue;
                };

                match client.list_tools().await {
                    Ok(tools) => {
                        tracing::info!(
                            "MCP server '{}' provides {} tools",
                            configs[idx].name,
                            tools.len()
                        );
                        let ns = configs[idx].namespace.as_deref();
                        for tool in tools {
                            let exposed_name = match ns {
                                Some(prefix) => format!("{}__{}", prefix, tool.name),
                                None => tool.name.clone(),
                            };
                            if ns.is_some() {
                                tracing::debug!(
                                    "  tool: {} (exposed as {})",
                                    tool.name,
                                    exposed_name
                                );
                            } else {
                                tracing::debug!("  tool: {}", tool.name);
                            }
                            new_routing.insert(exposed_name.clone(), (idx, tool.name.clone()));
                            all_tools.push(ToolDefinition::new(
                                exposed_name,
                                tool.description,
                                tool.parameters,
                            ));
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "failed to refresh tools from MCP server '{}': {e}",
                            configs[idx].name
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

        let configs = self.configs.read().await;
        let mut clients = self.clients.lock().await;
        for (idx, client_slot) in clients.iter_mut().enumerate() {
            let Some(client) = client_slot.as_mut() else {
                continue;
            };

            match client.list_resources().await {
                Ok(resources) => {
                    tracing::info!(
                        "MCP server '{}' provides {} resources",
                        configs[idx].name,
                        resources.len()
                    );
                    all_resources.extend(resources);
                }
                Err(e) if is_method_not_found(&e) => {
                    tracing::debug!(
                        "MCP server '{}' does not implement resources/list",
                        configs[idx].name
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "failed to refresh resources from MCP server '{}': {e}",
                        configs[idx].name
                    );
                }
            }
        }

        *self.cached_resources.lock().await = all_resources;
        Ok(())
    }

    async fn refresh_prompts_cache(&self) -> Result<(), McpError> {
        let mut all_prompts = Vec::new();

        let configs = self.configs.read().await;
        let mut clients = self.clients.lock().await;
        for (idx, client_slot) in clients.iter_mut().enumerate() {
            let Some(client) = client_slot.as_mut() else {
                continue;
            };

            match client.list_prompts().await {
                Ok(prompts) => {
                    tracing::info!(
                        "MCP server '{}' provides {} prompts",
                        configs[idx].name,
                        prompts.len()
                    );
                    all_prompts.extend(prompts);
                }
                Err(e) if is_method_not_found(&e) => {
                    tracing::debug!(
                        "MCP server '{}' does not implement prompts/list",
                        configs[idx].name
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "failed to refresh prompts from MCP server '{}': {e}",
                        configs[idx].name
                    );
                }
            }
        }

        *self.cached_prompts.lock().await = all_prompts;
        Ok(())
    }

    /// Connect a single server by index.
    async fn connect_server(&self, idx: usize) -> Result<(), McpError> {
        let configs = self.configs.read().await;
        let config = configs.get(idx).ok_or_else(|| {
            McpError::UnexpectedResponse(format!("server index {idx} out of range"))
        })?;

        tracing::info!(
            "connecting to MCP server '{}': {}",
            config.name,
            config.command
        );

        match McpClient::connect(&config.command, &config.args).await {
            Ok(client) => {
                let mut clients = self.clients.lock().await;
                clients[idx] = Some(client);
                Ok(())
            }
            Err(e) => {
                tracing::error!("failed to connect to MCP server '{}': {e}", config.name);
                Err(e)
            }
        }
    }

    /// Disconnect a single server by index.
    async fn disconnect_server(&self, idx: usize) {
        let mut clients = self.clients.lock().await;
        if let Some(Some(_)) = clients.get_mut(idx) {
            let client = clients[idx].take().unwrap();
            drop(clients);
            client.shutdown().await;
        }
    }

    /// Find the index of a server by name.
    async fn find_server_index(&self, name: &str) -> Option<usize> {
        let configs = self.configs.read().await;
        configs.iter().position(|c| c.name == name)
    }
}

/// Clonable handle for runtime control of MCP servers.
///
/// Created via `McpToolExecutor::control_handle()` before the executor is
/// moved into `ConversationHandler`.
#[derive(Clone)]
pub struct McpControlHandle {
    state: Arc<McpExecutorState>,
}

impl McpControlHandle {
    /// Get status for one or all servers.
    pub async fn status(&self, server: Option<&str>) -> Vec<McpServerStatusInfo> {
        let configs = self.state.configs.read().await;
        let clients = self.state.clients.lock().await;
        let routing = self.state.tool_routing.lock().await;

        let indices: Vec<usize> = if let Some(name) = server {
            configs
                .iter()
                .position(|c| c.name == name)
                .into_iter()
                .collect()
        } else {
            (0..configs.len()).collect()
        };

        indices
            .into_iter()
            .filter_map(|idx| {
                let config = configs.get(idx)?;
                let connected = clients.get(idx).is_some_and(|c| c.is_some());
                let tool_count = routing
                    .values()
                    .filter(|(server_idx, _)| *server_idx == idx)
                    .count() as u32;

                let status = if !config.enabled {
                    "disabled"
                } else if connected {
                    "running"
                } else {
                    "stopped"
                };

                Some(McpServerStatusInfo {
                    name: config.name.clone(),
                    command: config.command.clone(),
                    enabled: config.enabled,
                    status: status.to_string(),
                    tool_count,
                })
            })
            .collect()
    }

    /// Start one or all servers.
    pub async fn start_server(&self, server: Option<&str>) -> Result<String, McpError> {
        let indices = self.resolve_indices(server).await?;
        let mut started = Vec::new();

        for idx in indices {
            let configs = self.state.configs.read().await;
            let config = &configs[idx];
            if !config.enabled {
                continue;
            }
            let name = config.name.clone();
            drop(configs);

            // Skip if already connected
            {
                let clients = self.state.clients.lock().await;
                if clients[idx].is_some() {
                    continue;
                }
            }

            if self.state.connect_server(idx).await.is_ok() {
                started.push(name);
            }
        }

        self.state.refresh_all_metadata().await?;

        if started.is_empty() {
            Ok("no servers started".to_string())
        } else {
            Ok(format!("started: {}", started.join(", ")))
        }
    }

    /// Stop one or all servers.
    pub async fn stop_server(&self, server: Option<&str>) -> Result<String, McpError> {
        let indices = self.resolve_indices(server).await?;
        let mut stopped = Vec::new();

        for idx in indices {
            let was_connected = {
                let clients = self.state.clients.lock().await;
                clients[idx].is_some()
            };

            if was_connected {
                let name = {
                    let configs = self.state.configs.read().await;
                    configs[idx].name.clone()
                };
                self.state.disconnect_server(idx).await;
                stopped.push(name);
            }
        }

        self.state.refresh_all_metadata().await?;

        if stopped.is_empty() {
            Ok("no servers stopped".to_string())
        } else {
            Ok(format!("stopped: {}", stopped.join(", ")))
        }
    }

    /// Restart one or all servers.
    pub async fn restart_server(&self, server: Option<&str>) -> Result<String, McpError> {
        self.stop_server(server).await?;
        self.start_server(server).await
    }

    /// Add a server config, persist to TOML, and auto-start if enabled.
    pub async fn add_server(&self, config: McpServerConfig) -> Result<(), McpError> {
        let auto_start = config.enabled;
        let idx = {
            let mut configs = self.state.configs.write().await;

            // Check for duplicate name
            if configs.iter().any(|c| c.name == config.name) {
                return Err(McpError::UnexpectedResponse(format!(
                    "server '{}' already exists",
                    config.name
                )));
            }

            configs.push(config);
            let idx = configs.len() - 1;

            // Extend clients vec to match
            let mut clients = self.state.clients.lock().await;
            clients.push(None);

            idx
        };

        self.persist_configs().await?;

        if auto_start {
            let _ = self.state.connect_server(idx).await;
            let _ = self.state.refresh_all_metadata().await;
        }

        Ok(())
    }

    /// Remove a server by name: auto-stop, remove config, persist.
    pub async fn remove_server(&self, name: &str) -> Result<(), McpError> {
        let idx =
            self.state.find_server_index(name).await.ok_or_else(|| {
                McpError::UnexpectedResponse(format!("server '{name}' not found"))
            })?;

        // Stop if connected
        self.state.disconnect_server(idx).await;

        // Remove from configs and clients
        {
            let mut configs = self.state.configs.write().await;
            configs.remove(idx);

            let mut clients = self.state.clients.lock().await;
            clients.remove(idx);
        }

        // Rebuild routing since indices shifted
        let _ = self.state.refresh_all_metadata().await;
        self.persist_configs().await?;

        Ok(())
    }

    /// Enable a server: set enabled=true, auto-start, persist.
    pub async fn enable_server(&self, name: &str) -> Result<(), McpError> {
        let idx =
            self.state.find_server_index(name).await.ok_or_else(|| {
                McpError::UnexpectedResponse(format!("server '{name}' not found"))
            })?;

        {
            let mut configs = self.state.configs.write().await;
            configs[idx].enabled = true;
        }

        self.persist_configs().await?;
        let _ = self.state.connect_server(idx).await;
        let _ = self.state.refresh_all_metadata().await;

        Ok(())
    }

    /// Disable a server: auto-stop, set enabled=false, persist.
    pub async fn disable_server(&self, name: &str) -> Result<(), McpError> {
        let idx =
            self.state.find_server_index(name).await.ok_or_else(|| {
                McpError::UnexpectedResponse(format!("server '{name}' not found"))
            })?;

        self.state.disconnect_server(idx).await;

        {
            let mut configs = self.state.configs.write().await;
            configs[idx].enabled = false;
        }

        let _ = self.state.refresh_all_metadata().await;
        self.persist_configs().await?;

        Ok(())
    }

    /// Persist current configs to the TOML file.
    pub async fn persist_configs(&self) -> Result<(), McpError> {
        let configs = self.state.configs.read().await;
        save_mcp_configs(&self.state.config_path, &configs)
    }

    async fn resolve_indices(&self, server: Option<&str>) -> Result<Vec<usize>, McpError> {
        let configs = self.state.configs.read().await;
        if let Some(name) = server {
            let idx = configs.iter().position(|c| c.name == name).ok_or_else(|| {
                McpError::UnexpectedResponse(format!("server '{name}' not found"))
            })?;
            Ok(vec![idx])
        } else {
            Ok((0..configs.len()).collect())
        }
    }
}

/// Adapter implementing `ToolExecutor` by managing multiple MCP server connections.
/// Routes tool calls to the correct MCP server based on tool name.
pub struct McpToolExecutor {
    state: Arc<McpExecutorState>,
    /// Built-in in-process tools (knowledge base + tool search + sys props).
    builtin_tools: BuiltinToolService,
}

impl McpToolExecutor {
    pub fn new(configs: Vec<McpServerConfig>) -> Self {
        Self::with_builtin_tools(configs, BuiltinToolService::new())
    }

    pub fn with_builtin_tools(
        configs: Vec<McpServerConfig>,
        builtin_tools: BuiltinToolService,
    ) -> Self {
        let clients: Vec<Option<McpClient>> = (0..configs.len()).map(|_| None).collect();
        Self {
            state: Arc::new(McpExecutorState {
                configs: RwLock::new(configs),
                clients: Mutex::new(clients),
                tool_routing: Mutex::new(HashMap::new()),
                cached_tools: Mutex::new(Vec::new()),
                cached_resources: Mutex::new(Vec::new()),
                cached_prompts: Mutex::new(Vec::new()),
                config_path: PathBuf::new(),
            }),
            builtin_tools,
        }
    }

    pub fn with_builtin_tools_and_config_path(
        configs: Vec<McpServerConfig>,
        builtin_tools: BuiltinToolService,
        config_path: PathBuf,
    ) -> Self {
        let clients: Vec<Option<McpClient>> = (0..configs.len()).map(|_| None).collect();
        Self {
            state: Arc::new(McpExecutorState {
                configs: RwLock::new(configs),
                clients: Mutex::new(clients),
                tool_routing: Mutex::new(HashMap::new()),
                cached_tools: Mutex::new(Vec::new()),
                cached_resources: Mutex::new(Vec::new()),
                cached_prompts: Mutex::new(Vec::new()),
                config_path,
            }),
            builtin_tools,
        }
    }

    /// Get a clonable handle for runtime control of MCP servers.
    ///
    /// Call this before moving the executor into `ConversationHandler`.
    pub fn control_handle(&self) -> McpControlHandle {
        McpControlHandle {
            state: Arc::clone(&self.state),
        }
    }

    /// Get a mutable reference to the builtin tool service.
    pub fn builtin_tools_mut(&mut self) -> &mut BuiltinToolService {
        &mut self.builtin_tools
    }

    /// Connect to all configured MCP servers, discover their tools,
    /// and build the routing table.
    pub async fn start(&self) -> Result<(), McpError> {
        {
            let configs = self.state.configs.read().await;
            let mut clients = self.state.clients.lock().await;

            for (idx, config) in configs.iter().enumerate() {
                if !config.enabled {
                    tracing::info!("skipping disabled MCP server '{}'", config.name);
                    continue;
                }

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

        self.state.refresh_all_metadata().await?;
        Ok(())
    }

    pub async fn available_resources(&self) -> Vec<serde_json::Value> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP resources cache: {e}");
        }
        self.state.cached_resources.lock().await.clone()
    }

    pub async fn available_prompts(&self) -> Vec<serde_json::Value> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP prompts cache: {e}");
        }
        self.state.cached_prompts.lock().await.clone()
    }

    /// Returns every registered tool as a `(service_name, tool_name)` pair.
    ///
    /// MCP server tools are labelled with their configured service name.
    /// Built-in tools are labelled `"builtin"`.
    /// Intended for startup diagnostics.
    pub async fn tools_by_service(&self) -> Vec<(String, String)> {
        let configs = self.state.configs.read().await;
        let routing = self.state.tool_routing.lock().await;
        let cached = self.state.cached_tools.lock().await;

        let mut entries: Vec<(String, String)> = cached
            .iter()
            .map(|tool| {
                let service = routing
                    .get(&tool.name)
                    .and_then(|(idx, _)| configs.get(*idx))
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                (service, tool.name.clone())
            })
            .collect();

        for tool in self.builtin_tools.tool_definitions() {
            entries.push(("builtin".to_string(), tool.name));
        }

        entries
    }

    /// Return all MCP (non-builtin) tool definitions.
    pub async fn all_mcp_tools(&self) -> Vec<ToolDefinition> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP tools cache: {e}");
        }
        self.state.cached_tools.lock().await.clone()
    }

    /// Shut down all connected MCP servers.
    pub async fn shutdown(&self) {
        let mut clients = self.state.clients.lock().await;
        for client in clients.iter_mut() {
            if let Some(c) = client.take() {
                c.shutdown().await;
            }
        }
    }
}

impl ToolExecutor for McpToolExecutor {
    async fn core_tools(&self) -> Vec<ToolDefinition> {
        // Only return builtin tools as core. MCP tools are discovered
        // dynamically via builtin_tool_search to avoid bloating every
        // request with dozens of tool definitions.
        self.builtin_tools.tool_definitions()
    }

    async fn tool_namespaces(&self) -> Vec<ToolNamespace> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP tools cache: {e}");
        }

        let mut namespaces = Vec::new();

        // Builtins are always sent as core tools, so skip them here.
        // Only MCP server tools go into deferred namespaces.

        // MCP server tool namespaces — grouped by server
        let configs = self.state.configs.read().await;
        let cached = self.state.cached_tools.lock().await;
        let routing = self.state.tool_routing.lock().await;

        for (idx, config) in configs.iter().enumerate() {
            let server_tools: Vec<ToolDefinition> = cached
                .iter()
                .filter(|tool| {
                    routing
                        .get(&tool.name)
                        .is_some_and(|(server_idx, _)| *server_idx == idx)
                })
                .cloned()
                .collect();

            if !server_tools.is_empty() {
                let ns_name = config
                    .namespace
                    .as_deref()
                    .unwrap_or(&config.name)
                    .to_string();
                namespaces.push(ToolNamespace::new(
                    &ns_name,
                    format!("Tools from the {} MCP server", config.name),
                    server_tools,
                ));
            }
        }

        namespaces
    }

    async fn search_tools(&self, query: &str) -> Result<Vec<ToolDefinition>, CoreError> {
        if let Err(e) = self.state.maybe_refresh_metadata().await {
            tracing::warn!("failed to refresh MCP tools cache: {e}");
        }
        let cached = self.state.cached_tools.lock().await;
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let results: Vec<ToolDefinition> = cached
            .iter()
            .filter(|tool| {
                let name = tool.name.to_lowercase();
                let desc = tool.description.to_lowercase();
                keywords
                    .iter()
                    .any(|kw| name.contains(kw) || desc.contains(kw))
            })
            .cloned()
            .collect();

        Ok(results)
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        // Check builtins first
        if BuiltinToolService::supports_tool(name) {
            return Ok(self
                .builtin_tools
                .tool_definitions()
                .into_iter()
                .find(|t| t.name == name));
        }

        // Check cached MCP tools
        let cached = self.state.cached_tools.lock().await;
        Ok(cached.iter().find(|t| t.name == name).cloned())
    }

    async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        if BuiltinToolService::supports_tool(name) {
            return self.builtin_tools.execute_tool(name, arguments).await;
        }

        self.state
            .maybe_refresh_metadata()
            .await
            .map_err(|e| CoreError::ToolExecution(format!("failed to refresh tools: {e}")))?;

        let routing = self.state.tool_routing.lock().await;
        let (idx, original_name) = routing
            .get(name)
            .ok_or_else(|| {
                // Find tools with a similar prefix to help the model self-correct.
                let prefix = name.find('_').map(|i| &name[..i]).unwrap_or(name);
                let similar: Vec<&str> = routing
                    .keys()
                    .filter(|k| k.starts_with(prefix))
                    .map(|k| k.as_str())
                    .collect();
                if similar.is_empty() {
                    CoreError::ToolExecution(format!("unknown tool: {name}"))
                } else {
                    CoreError::ToolExecution(format!(
                        "unknown tool: {name}. Similar tools available: {}",
                        similar.join(", ")
                    ))
                }
            })?
            .clone();
        drop(routing);

        let mut clients = self.state.clients.lock().await;
        let client = clients[idx].as_mut().ok_or_else(|| {
            CoreError::ToolExecution(format!("MCP server for tool '{name}' is not connected"))
        })?;

        client
            .call_tool(&original_name, arguments)
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
        let rt = tokio::runtime::Runtime::new().unwrap();
        let configs = rt.block_on(executor.state.configs.read());
        assert!(configs.is_empty());
    }

    #[test]
    fn server_config_construction() {
        let config = McpServerConfig {
            name: "fileio".into(),
            command: "fileio-mcp".into(),
            args: vec![],
            namespace: None,
            enabled: true,
        };
        assert_eq!(config.name, "fileio");
        assert_eq!(config.command, "fileio-mcp");
        assert!(config.args.is_empty());
        assert!(config.namespace.is_none());
        assert!(config.enabled);
    }

    #[test]
    fn server_config_with_namespace() {
        let config = McpServerConfig {
            name: "tickets-jira".into(),
            command: "jira-mcp".into(),
            args: vec![],
            namespace: Some("jira".into()),
            enabled: true,
        };
        assert_eq!(config.namespace.as_deref(), Some("jira"));
    }

    #[test]
    fn server_config_with_args() {
        let config = McpServerConfig {
            name: "genmcp".into(),
            command: "genmcp".into(),
            args: vec!["--config".into(), "/path/to/config.toml".into()],
            namespace: None,
            enabled: true,
        };
        assert_eq!(config.args.len(), 2);
    }

    #[tokio::test]
    async fn executor_no_configs_returns_builtin_tools() {
        let executor = McpToolExecutor::new(vec![]);
        let tools = executor.core_tools().await;
        assert!(!tools.is_empty());
        assert!(
            tools
                .iter()
                .any(|tool| tool.name == "builtin_knowledge_base_write")
        );
        assert!(tools.iter().any(|tool| tool.name == "builtin_tool_search"));
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
        let tools = executor.core_tools().await;
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"builtin_knowledge_base_write"));
        assert!(names.contains(&"builtin_knowledge_base_search"));
        assert!(names.contains(&"builtin_knowledge_base_delete"));
        assert!(names.contains(&"builtin_tool_search"));
        assert!(names.contains(&"builtin_sys_props"));
    }

    #[tokio::test]
    async fn executor_executes_builtin_sys_props() {
        let executor = McpToolExecutor::new(vec![]);
        let result = executor
            .execute_tool("builtin_sys_props", serde_json::json!({}))
            .await
            .unwrap();
        assert!(result.contains("\"ok\":true"));
    }

    #[tokio::test]
    async fn control_handle_status_empty() {
        let executor = McpToolExecutor::new(vec![]);
        let handle = executor.control_handle();
        let status = handle.status(None).await;
        assert!(status.is_empty());
    }

    #[tokio::test]
    async fn control_handle_status_shows_configs() {
        let configs = vec![
            McpServerConfig {
                name: "fileio".into(),
                command: "fileio-mcp".into(),
                args: vec![],
                namespace: None,
                enabled: true,
            },
            McpServerConfig {
                name: "jira".into(),
                command: "jira-mcp".into(),
                args: vec![],
                namespace: Some("jira".into()),
                enabled: false,
            },
        ];
        let executor = McpToolExecutor::new(configs);
        let handle = executor.control_handle();
        let status = handle.status(None).await;
        assert_eq!(status.len(), 2);
        assert_eq!(status[0].name, "fileio");
        assert_eq!(status[0].status, "stopped");
        assert!(status[0].enabled);
        assert_eq!(status[1].name, "jira");
        assert_eq!(status[1].status, "disabled");
        assert!(!status[1].enabled);
    }

    #[tokio::test]
    async fn control_handle_status_by_name() {
        let configs = vec![McpServerConfig {
            name: "fileio".into(),
            command: "fileio-mcp".into(),
            args: vec![],
            namespace: None,
            enabled: true,
        }];
        let executor = McpToolExecutor::new(configs);
        let handle = executor.control_handle();
        let status = handle.status(Some("fileio")).await;
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].name, "fileio");

        let empty = handle.status(Some("nonexistent")).await;
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn tool_namespaces_excludes_builtins() {
        let executor = McpToolExecutor::new(vec![]);
        let namespaces = executor.tool_namespaces().await;

        // With no MCP servers, namespaces should be empty —
        // builtins are always core tools, not deferred.
        assert!(namespaces.is_empty());
    }

    #[tokio::test]
    async fn shutdown_non_consuming() {
        let executor = McpToolExecutor::new(vec![]);
        executor.shutdown().await;
        // Can still access after shutdown
        let tools = executor.core_tools().await;
        assert!(!tools.is_empty());
    }
}
