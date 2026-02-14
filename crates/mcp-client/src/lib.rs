pub mod config;
pub mod executor;
mod jsonrpc;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use desktop_assistant_core::domain::ToolDefinition;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use jsonrpc::{JsonRpcRequest, JsonRpcResponse};

/// Error type for MCP client operations.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("failed to spawn MCP server process: {0}")]
    SpawnFailed(std::io::Error),

    #[error("MCP server stdin not available")]
    NoStdin,

    #[error("MCP server stdout not available")]
    NoStdout,

    #[error("I/O error communicating with MCP server: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("MCP server returned error: code={code}, message={message}")]
    ServerError { code: i64, message: String },

    #[error("unexpected response from MCP server: {0}")]
    UnexpectedResponse(String),

    #[error("MCP client is not connected")]
    NotConnected,
}

/// Client for a single MCP server process, communicating via JSON-RPC over stdio.
pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: AtomicU64,
    tools_list_changed: AtomicBool,
    resources_list_changed: AtomicBool,
    prompts_list_changed: AtomicBool,
}

impl McpClient {
    /// Spawn an MCP server process and perform the initialize handshake.
    pub async fn connect(command: &str, args: &[String]) -> Result<Self, McpError> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(McpError::SpawnFailed)?;

        let stdin = child.stdin.take().ok_or(McpError::NoStdin)?;
        let stdout = child.stdout.take().ok_or(McpError::NoStdout)?;
        let reader = BufReader::new(stdout);

        let mut client = Self {
            child,
            stdin,
            reader,
            next_id: AtomicU64::new(1),
            tools_list_changed: AtomicBool::new(false),
            resources_list_changed: AtomicBool::new(false),
            prompts_list_changed: AtomicBool::new(false),
        };

        // Perform initialize handshake
        client.initialize().await?;

        Ok(client)
    }

    async fn initialize(&mut self) -> Result<(), McpError> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "desktop-assistant",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let _response = self.send_request("initialize", Some(params)).await?;

        // Send initialized notification (no id, no response expected)
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        let mut line = serde_json::to_string(&notification)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;

        Ok(())
    }

    /// List all tools available from this MCP server.
    pub async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, McpError> {
        let response = self.send_request("tools/list", None).await?;

        let tools_value = response
            .get("tools")
            .ok_or_else(|| McpError::UnexpectedResponse("missing 'tools' field".into()))?;

        let raw_tools: Vec<RawToolDef> = serde_json::from_value(tools_value.clone())?;

        self.tools_list_changed.store(false, Ordering::Relaxed);

        Ok(raw_tools
            .into_iter()
            .map(|t| {
                ToolDefinition::new(
                    t.name,
                    t.description.unwrap_or_default(),
                    t.input_schema
                        .unwrap_or(serde_json::json!({"type": "object"})),
                )
            })
            .collect())
    }

    /// List all resources available from this MCP server.
    pub async fn list_resources(&mut self) -> Result<Vec<serde_json::Value>, McpError> {
        let response = self.send_request("resources/list", None).await?;
        let resources = extract_list_field(&response, "resources")?;
        self.resources_list_changed.store(false, Ordering::Relaxed);
        Ok(resources)
    }

    /// List all prompts available from this MCP server.
    pub async fn list_prompts(&mut self) -> Result<Vec<serde_json::Value>, McpError> {
        let response = self.send_request("prompts/list", None).await?;
        let prompts = extract_list_field(&response, "prompts")?;
        self.prompts_list_changed.store(false, Ordering::Relaxed);
        Ok(prompts)
    }

    /// Returns true if this client has observed a tools list change notification
    /// since the last successful `list_tools` refresh.
    pub fn tools_list_changed(&self) -> bool {
        self.tools_list_changed.load(Ordering::Relaxed)
    }

    /// Returns true if this client has observed a resources list change notification.
    pub fn resources_list_changed(&self) -> bool {
        self.resources_list_changed.load(Ordering::Relaxed)
    }

    /// Returns true if this client has observed a prompts list change notification.
    pub fn prompts_list_changed(&self) -> bool {
        self.prompts_list_changed.load(Ordering::Relaxed)
    }

    /// Call a tool on this MCP server.
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, McpError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });

        let response = self.send_request("tools/call", Some(params)).await?;

        // Extract content from the response
        if let Some(content) = response.get("content")
            && let Some(arr) = content.as_array()
        {
            let text_parts: Vec<String> = arr
                .iter()
                .filter_map(|item| {
                    if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                        item.get("text").and_then(|t| t.as_str()).map(String::from)
                    } else {
                        Some(
                            serde_json::to_string_pretty(item).unwrap_or_else(|_| item.to_string()),
                        )
                    }
                })
                .collect();
            if !text_parts.is_empty() {
                return Ok(text_parts.join("\n"));
            }
        }

        // Fallback: return raw JSON
        Ok(serde_json::to_string(&response)?)
    }

    /// Shut down the MCP server process gracefully.
    pub async fn shutdown(mut self) {
        // Try to kill the child process
        let _ = self.child.kill().await;
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    async fn send_request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let id = self.next_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let mut line = serde_json::to_string(&request)?;
        line.push('\n');

        tracing::debug!("MCP request: {}", line.trim());
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;

        // Read response lines until we get one with matching id
        loop {
            let mut buf = String::new();
            let bytes_read = self.reader.read_line(&mut buf).await?;
            if bytes_read == 0 {
                return Err(McpError::UnexpectedResponse(
                    "MCP server closed stdout".into(),
                ));
            }

            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }

            tracing::debug!("MCP response: {trimmed}");

            let message: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(value) => value,
                Err(_) => {
                    tracing::debug!("skipping non-JSON line from MCP server");
                    continue;
                }
            };

            if let Some(list_kind) = list_kind_from_notification(&message) {
                tracing::debug!("received {list_kind} list changed notification");
                self.mark_list_changed_for_kind(list_kind);
                continue;
            }

            // Try to parse as JSON-RPC response
            let response: JsonRpcResponse = match serde_json::from_value(message.clone()) {
                Ok(r) => r,
                Err(_) => {
                    // Could be a notification, skip it
                    tracing::debug!("skipping non-response line from MCP server");
                    continue;
                }
            };

            // Check if this response matches our request id
            if response.id != Some(serde_json::Value::Number(id.into())) {
                // Not our response, could be a notification or out-of-order
                tracing::debug!("skipping response with non-matching id");
                continue;
            }

            if let Some(error) = response.error {
                return Err(McpError::ServerError {
                    code: error.code,
                    message: error.message,
                });
            }

            let result = response.result.unwrap_or(serde_json::Value::Null);
            if result_has_list_changed(&result) {
                self.mark_list_changed_for_method(method);
            }

            return Ok(result);
        }
    }

    fn mark_list_changed_for_method(&self, method: &str) {
        match method {
            "tools/list" => self.tools_list_changed.store(true, Ordering::Relaxed),
            "resources/list" => self.resources_list_changed.store(true, Ordering::Relaxed),
            "prompts/list" => self.prompts_list_changed.store(true, Ordering::Relaxed),
            _ => {}
        }
    }

    fn mark_list_changed_for_kind(&self, list_kind: ListKind) {
        match list_kind {
            ListKind::Tools => self.tools_list_changed.store(true, Ordering::Relaxed),
            ListKind::Resources => self.resources_list_changed.store(true, Ordering::Relaxed),
            ListKind::Prompts => self.prompts_list_changed.store(true, Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ListKind {
    Tools,
    Resources,
    Prompts,
}

impl std::fmt::Display for ListKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tools => write!(f, "tools"),
            Self::Resources => write!(f, "resources"),
            Self::Prompts => write!(f, "prompts"),
        }
    }
}

fn list_kind_from_notification(message: &serde_json::Value) -> Option<ListKind> {
    let method = message.get("method").and_then(serde_json::Value::as_str)?;
    match method {
        "notifications/tools/list_changed" | "tools/list_changed" => Some(ListKind::Tools),
        "notifications/resources/list_changed" | "resources/list_changed" => {
            Some(ListKind::Resources)
        }
        "notifications/prompts/list_changed" | "prompts/list_changed" => Some(ListKind::Prompts),
        _ => None,
    }
}

fn result_has_list_changed(result: &serde_json::Value) -> bool {
    result
        .get("listChanged")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn extract_list_field(
    response: &serde_json::Value,
    field_name: &str,
) -> Result<Vec<serde_json::Value>, McpError> {
    let field_value = response
        .get(field_name)
        .ok_or_else(|| McpError::UnexpectedResponse(format!("missing '{field_name}' field")))?;

    let items = field_value
        .as_array()
        .ok_or_else(|| {
            McpError::UnexpectedResponse(format!("'{field_name}' field is not an array"))
        })?
        .clone();

    Ok(items)
}

/// Raw tool definition as returned by MCP servers.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawToolDef {
    name: String,
    description: Option<String>,
    input_schema: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_error_display() {
        let err = McpError::ServerError {
            code: -32600,
            message: "Invalid request".into(),
        };
        assert!(err.to_string().contains("-32600"));
        assert!(err.to_string().contains("Invalid request"));
    }

    #[test]
    fn raw_tool_def_deserialize() {
        let json = r#"{
            "name": "read_file",
            "description": "Read a file",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }
        }"#;
        let tool: RawToolDef = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "read_file");
        assert_eq!(tool.description.as_deref(), Some("Read a file"));
        assert!(tool.input_schema.is_some());
    }

    #[test]
    fn raw_tool_def_without_optional_fields() {
        let json = r#"{"name": "simple_tool"}"#;
        let tool: RawToolDef = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "simple_tool");
        assert!(tool.description.is_none());
        assert!(tool.input_schema.is_none());
    }

    #[test]
    fn raw_tool_to_tool_definition() {
        let raw = RawToolDef {
            name: "test".into(),
            description: Some("A test tool".into()),
            input_schema: Some(serde_json::json!({"type": "object"})),
        };
        let def = ToolDefinition::new(
            raw.name,
            raw.description.unwrap_or_default(),
            raw.input_schema
                .unwrap_or(serde_json::json!({"type": "object"})),
        );
        assert_eq!(def.name, "test");
        assert_eq!(def.description, "A test tool");
    }

    #[test]
    fn raw_tool_without_description_defaults_to_empty() {
        let raw = RawToolDef {
            name: "bare".into(),
            description: None,
            input_schema: None,
        };
        let def = ToolDefinition::new(
            raw.name,
            raw.description.unwrap_or_default(),
            raw.input_schema
                .unwrap_or(serde_json::json!({"type": "object"})),
        );
        assert_eq!(def.description, "");
    }

    #[test]
    fn detects_tools_list_changed_notifications() {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed"
        });
        assert_eq!(
            list_kind_from_notification(&notification),
            Some(ListKind::Tools)
        );

        let short_form = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "tools/list_changed"
        });
        assert_eq!(
            list_kind_from_notification(&short_form),
            Some(ListKind::Tools)
        );
    }

    #[test]
    fn detects_resources_and_prompts_list_changed_notifications() {
        let resources_notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/resources/list_changed"
        });
        assert_eq!(
            list_kind_from_notification(&resources_notification),
            Some(ListKind::Resources)
        );

        let prompts_notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "prompts/list_changed"
        });
        assert_eq!(
            list_kind_from_notification(&prompts_notification),
            Some(ListKind::Prompts)
        );
    }

    #[test]
    fn ignores_non_tools_list_changed_notifications() {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        assert_eq!(list_kind_from_notification(&notification), None);
    }

    #[test]
    fn detects_result_list_changed_flag() {
        assert!(result_has_list_changed(
            &serde_json::json!({"listChanged": true})
        ));
        assert!(!result_has_list_changed(
            &serde_json::json!({"listChanged": false})
        ));
        assert!(!result_has_list_changed(
            &serde_json::json!({"other": true})
        ));
    }

    #[test]
    fn extract_list_field_reads_arrays() {
        let response = serde_json::json!({
            "resources": [
                {"uri": "file:///tmp/a.txt"},
                {"uri": "file:///tmp/b.txt"}
            ]
        });

        let resources = extract_list_field(&response, "resources").unwrap();
        assert_eq!(resources.len(), 2);
    }

    #[test]
    fn extract_list_field_requires_existing_array_field() {
        let missing = serde_json::json!({"other": []});
        let err = extract_list_field(&missing, "prompts").unwrap_err();
        assert!(err.to_string().contains("missing 'prompts' field"));

        let wrong_type = serde_json::json!({"prompts": {"name": "x"}});
        let err = extract_list_field(&wrong_type, "prompts").unwrap_err();
        assert!(err.to_string().contains("'prompts' field is not an array"));
    }
}
