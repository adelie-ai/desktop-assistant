pub mod executor;
mod jsonrpc;

use std::sync::atomic::{AtomicU64, Ordering};

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

        // Extract text content from the response
        if let Some(content) = response.get("content") {
            if let Some(arr) = content.as_array() {
                let text_parts: Vec<&str> = arr
                    .iter()
                    .filter_map(|item| {
                        if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                            item.get("text").and_then(|t| t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
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

            // Try to parse as JSON-RPC response
            let response: JsonRpcResponse = match serde_json::from_str(trimmed) {
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

            return Ok(response.result.unwrap_or(serde_json::Value::Null));
        }
    }
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
}
