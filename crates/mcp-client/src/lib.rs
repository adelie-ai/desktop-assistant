//! Model Context Protocol (MCP) client for discovering and invoking external tool servers.

mod builtin;
pub mod config;
pub mod executor;
mod jsonrpc;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use desktop_assistant_core::domain::ToolDefinition;
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use jsonrpc::{JsonRpcRequest, JsonRpcResponse};

/// Default maximum silent gap while waiting for a response line from an MCP
/// server. The window resets whenever the server sends *any* line
/// (notifications count as liveness), so long-running tools that emit
/// progress notifications are not cut off. Generous because tool calls can
/// legitimately take minutes (e.g. terminal commands); the point is that a
/// silently wedged server fails the turn instead of hanging it forever
/// (DS-3, same standard the LLM providers got in #220).
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Cap on the `initialize` handshake: a server that can't even complete the
/// handshake within this window is treated as broken. Mirrors the LLM
/// connectors' 30s connect timeout (#220).
const INIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum accepted length of a single response line from an MCP server.
/// Anything larger is a protocol violation (or a runaway tool result) and is
/// surfaced as an error instead of buffering unbounded memory (DS-4).
const MAX_LINE_BYTES: u64 = 8 * 1024 * 1024;

/// Error type for MCP client operations.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("invalid MCP server command: {0}")]
    InvalidCommand(String),

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

    #[error("MCP request '{method}' timed out after {after:?} of silence")]
    Timeout { method: String, after: Duration },

    #[error("HTTP transport error: {0}")]
    Http(String),
}

/// Characters that could cause unintended behaviour if they appear in the
/// command name.  `Command::new` does not invoke a shell, but rejecting these
/// in the command string catches obvious misuse (e.g. `"cmd; rm -rf /"`) and
/// enforces that the command field is a simple program name or path.
///
/// Arguments are **not** checked because they are passed directly to `execve`
/// and are never shell-interpreted.
const SHELL_META: &str = ";&|<>$(){}!#`\n\r";

/// Validate an MCP command before spawning.
fn validate_command(command: &str, _args: &[String]) -> Result<(), McpError> {
    if command.is_empty() {
        return Err(McpError::InvalidCommand("command is empty".into()));
    }
    if command.contains(|c: char| SHELL_META.contains(c)) {
        return Err(McpError::InvalidCommand(format!(
            "command contains shell metacharacters: {command}"
        )));
    }
    Ok(())
}

/// "List changed" notification flags, shared between an `McpClient` and the
/// executor that owns it. Kept behind an `Arc` so the executor can poll the
/// flags without locking the client itself — a client busy with a slow tool
/// call must not block status/refresh checks (DS-1).
#[derive(Default)]
pub struct ListChangeFlags {
    tools: AtomicBool,
    resources: AtomicBool,
    prompts: AtomicBool,
}

impl ListChangeFlags {
    /// True if a tools list change notification was observed since the last
    /// successful `list_tools` refresh.
    pub fn tools_changed(&self) -> bool {
        self.tools.load(Ordering::Relaxed)
    }

    /// True if a resources list change notification was observed.
    pub fn resources_changed(&self) -> bool {
        self.resources.load(Ordering::Relaxed)
    }

    /// True if a prompts list change notification was observed.
    pub fn prompts_changed(&self) -> bool {
        self.prompts.load(Ordering::Relaxed)
    }
}

/// Client for a single MCP server, speaking JSON-RPC over a pluggable
/// [`Transport`] — either a spawned stdio child process or a remote
/// streamable-HTTP endpoint.
pub struct McpClient {
    transport: Transport,
    next_id: AtomicU64,
    flags: Arc<ListChangeFlags>,
    /// Maximum silent gap while waiting for a response; see
    /// [`DEFAULT_REQUEST_TIMEOUT`].
    request_timeout: Duration,
}

impl McpClient {
    /// Spawn an MCP server process and perform the initialize handshake.
    ///
    /// The command is validated before spawning: it must be a single
    /// program name or absolute path and must not contain shell
    /// metacharacters. Arguments are checked individually as well.
    pub async fn connect(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self, McpError> {
        Self::connect_with_request_timeout(command, args, env, DEFAULT_REQUEST_TIMEOUT).await
    }

    /// [`Self::connect`] with an explicit per-request silence timeout.
    pub async fn connect_with_request_timeout(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
        request_timeout: Duration,
    ) -> Result<Self, McpError> {
        let transport = Transport::Stdio(StdioTransport::spawn(command, args, env)?);
        Self::from_transport(transport, request_timeout).await
    }

    /// Connect to a remote MCP server over streamable-HTTP and perform the
    /// initialize handshake.
    ///
    /// `bearer`, when set, is sent verbatim as an `Authorization: Bearer`
    /// header on every request. Acquiring/refreshing that token (e.g. via
    /// Google OAuth) is the caller's concern and out of scope here.
    pub async fn connect_http(url: &str, bearer: Option<String>) -> Result<Self, McpError> {
        Self::connect_http_with_request_timeout(url, bearer, DEFAULT_REQUEST_TIMEOUT).await
    }

    /// [`Self::connect_http`] with an explicit per-request silence timeout.
    pub async fn connect_http_with_request_timeout(
        url: &str,
        bearer: Option<String>,
        request_timeout: Duration,
    ) -> Result<Self, McpError> {
        let transport = Transport::Http(HttpTransport::new(url, bearer)?);
        Self::from_transport(transport, request_timeout).await
    }

    /// Wrap a ready transport and run the initialize handshake, bounded so a
    /// wedged server fails startup instead of stalling it (DS-3). On error the
    /// transport is dropped here, tearing down any child process (DS-2).
    async fn from_transport(
        transport: Transport,
        request_timeout: Duration,
    ) -> Result<Self, McpError> {
        let mut client = Self {
            transport,
            next_id: AtomicU64::new(1),
            flags: Arc::new(ListChangeFlags::default()),
            request_timeout,
        };

        let init_timeout = INIT_TIMEOUT.min(request_timeout);
        tokio::time::timeout(init_timeout, client.initialize())
            .await
            .map_err(|_| McpError::Timeout {
                method: "initialize".into(),
                after: init_timeout,
            })??;

        Ok(client)
    }

    /// Shared handle to this client's list-change notification flags.
    pub fn list_change_flags(&self) -> Arc<ListChangeFlags> {
        Arc::clone(&self.flags)
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

        // Send initialized notification (no id, no response expected).
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        self.transport.send_notification(&notification).await?;

        Ok(())
    }

    /// List all tools available from this MCP server.
    pub async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, McpError> {
        let response = self.send_request("tools/list", None).await?;

        let tools_value = response
            .get("tools")
            .ok_or_else(|| McpError::UnexpectedResponse("missing 'tools' field".into()))?;

        let raw_tools: Vec<RawToolDef> = serde_json::from_value(tools_value.clone())?;

        self.flags.tools.store(false, Ordering::Relaxed);

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
        self.flags.resources.store(false, Ordering::Relaxed);
        Ok(resources)
    }

    /// List all prompts available from this MCP server.
    pub async fn list_prompts(&mut self) -> Result<Vec<serde_json::Value>, McpError> {
        let response = self.send_request("prompts/list", None).await?;
        let prompts = extract_list_field(&response, "prompts")?;
        self.flags.prompts.store(false, Ordering::Relaxed);
        Ok(prompts)
    }

    /// Returns true if this client has observed a tools list change notification
    /// since the last successful `list_tools` refresh.
    pub fn tools_list_changed(&self) -> bool {
        self.flags.tools_changed()
    }

    /// Returns true if this client has observed a resources list change notification.
    pub fn resources_list_changed(&self) -> bool {
        self.flags.resources_changed()
    }

    /// Returns true if this client has observed a prompts list change notification.
    pub fn prompts_list_changed(&self) -> bool {
        self.flags.prompts_changed()
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

    /// Shut down the MCP server gracefully (kills a stdio child; a no-op for
    /// HTTP, which has no process to reap).
    pub async fn shutdown(&mut self) {
        self.transport.shutdown().await;
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

        let result = self
            .transport
            .round_trip(&request, self.request_timeout, &self.flags)
            .await?;

        if result_has_list_changed(&result) {
            mark_list_changed_for_method(&self.flags, method);
        }

        Ok(result)
    }
}

/// Transport backing an [`McpClient`]: a spawned stdio child process, or a
/// remote streamable-HTTP endpoint. The MCP request/response layer above is
/// transport-agnostic — both variants expose the same round-trip surface.
enum Transport {
    Stdio(StdioTransport),
    Http(HttpTransport),
}

impl Transport {
    /// Send `request` and return its JSON-RPC `result`, marking `flags` for any
    /// interleaved list-changed notifications observed along the way.
    async fn round_trip(
        &mut self,
        request: &JsonRpcRequest,
        timeout: Duration,
        flags: &ListChangeFlags,
    ) -> Result<serde_json::Value, McpError> {
        match self {
            Transport::Stdio(t) => t.round_trip(request, timeout, flags).await,
            Transport::Http(t) => t.round_trip(request, timeout, flags).await,
        }
    }

    async fn send_notification(
        &mut self,
        notification: &serde_json::Value,
    ) -> Result<(), McpError> {
        match self {
            Transport::Stdio(t) => t.send_notification(notification).await,
            Transport::Http(t) => t.send_notification(notification).await,
        }
    }

    async fn shutdown(&mut self) {
        match self {
            Transport::Stdio(t) => t.shutdown().await,
            // HTTP has no process to reap.
            Transport::Http(_) => {}
        }
    }
}

/// JSON-RPC over the stdio of a spawned MCP server child process.
struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

impl StdioTransport {
    /// Spawn the server process with piped stdio. The command is validated
    /// first: it must be a single program name or path with no shell
    /// metacharacters (arguments go straight to `execve`, so they are not
    /// checked).
    fn spawn(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self, McpError> {
        validate_command(command, args)?;

        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            // DS-2: make the kernel reap the server if this transport is
            // dropped without an explicit `shutdown` (panic, cancelled task,
            // error mid-connect).
            .kill_on_drop(true);
        for (key, value) in env {
            cmd.env(key, value);
        }
        let mut child = cmd.spawn().map_err(McpError::SpawnFailed)?;

        let stdin = child.stdin.take().ok_or(McpError::NoStdin)?;
        let stdout = child.stdout.take().ok_or(McpError::NoStdout)?;
        let reader = BufReader::new(stdout);

        Ok(Self {
            child,
            stdin,
            reader,
        })
    }

    async fn round_trip(
        &mut self,
        request: &JsonRpcRequest,
        timeout: Duration,
        flags: &ListChangeFlags,
    ) -> Result<serde_json::Value, McpError> {
        let mut line = serde_json::to_string(request)?;
        line.push('\n');
        tracing::debug!("MCP request: {}", line.trim());

        let id = request.id;
        let method = request.method.as_str();
        let stdin = &mut self.stdin;
        let reader = &mut self.reader;

        // Write and read concurrently (DS-4): a request larger than the pipe
        // buffer sent to a server itself blocked writing a large message would
        // otherwise deadlock. `try_join!` also short-circuits when the read
        // side times out.
        let write_fut = async move {
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await?;
            Ok::<(), McpError>(())
        };

        let read_fut = async move {
            // Read response lines until we get one with a matching id. Each
            // read is bounded by the request timeout (DS-3); any line from the
            // server (including notifications) resets the window.
            loop {
                let next = tokio::time::timeout(timeout, read_line_bounded(reader, MAX_LINE_BYTES))
                    .await
                    .map_err(|_| McpError::Timeout {
                        method: method.to_string(),
                        after: timeout,
                    })??;
                let Some(buf) = next else {
                    return Err(McpError::UnexpectedResponse(
                        "MCP server closed stdout".into(),
                    ));
                };

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
                    mark_list_changed_for_kind(flags, list_kind);
                    continue;
                }

                let response: JsonRpcResponse = match serde_json::from_value(message) {
                    Ok(r) => r,
                    Err(_) => {
                        tracing::debug!("skipping non-response line from MCP server");
                        continue;
                    }
                };
                if response.id != Some(serde_json::Value::Number(id.into())) {
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
        };

        let ((), result) = tokio::try_join!(write_fut, read_fut)?;
        Ok(result)
    }

    async fn send_notification(
        &mut self,
        notification: &serde_json::Value,
    ) -> Result<(), McpError> {
        let mut line = serde_json::to_string(notification)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn shutdown(&mut self) {
        let _ = self.child.kill().await;
    }
}

impl Drop for StdioTransport {
    /// Belt-and-suspenders teardown (DS-2): `kill_on_drop` already arranges for
    /// the runtime to reap the child, but issuing `start_kill()` here sends the
    /// signal immediately even if the runtime is shutting down. Harmless if the
    /// process already exited or `shutdown()` was called.
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// JSON-RPC over a remote streamable-HTTP MCP endpoint. Each request is a POST
/// whose reply is either a single JSON body or a `text/event-stream` (SSE)
/// sequence of JSON-RPC messages.
struct HttpTransport {
    client: reqwest::Client,
    url: String,
    /// Verbatim bearer token for `Authorization`, if the endpoint requires one.
    bearer: Option<String>,
    /// `Mcp-Session-Id` assigned by the server on initialize; echoed on
    /// subsequent requests when present.
    session_id: Option<String>,
}

impl HttpTransport {
    fn new(url: &str, bearer: Option<String>) -> Result<Self, McpError> {
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err(McpError::Http(format!(
                "remote MCP url must be http(s): {url}"
            )));
        }
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| McpError::Http(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            client,
            url: url.to_string(),
            bearer,
            session_id: None,
        })
    }

    /// Attach the shared headers (accept + auth + session) to a request.
    fn prepare(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut builder = builder.header(ACCEPT, "application/json, text/event-stream");
        if let Some(bearer) = &self.bearer {
            builder = builder.bearer_auth(bearer);
        }
        if let Some(session) = &self.session_id {
            builder = builder.header("Mcp-Session-Id", session);
        }
        builder
    }

    /// Capture the `Mcp-Session-Id` header the first time the server assigns one.
    fn capture_session(&mut self, response: &reqwest::Response) {
        if self.session_id.is_none()
            && let Some(session) = response
                .headers()
                .get("mcp-session-id")
                .and_then(|v| v.to_str().ok())
        {
            self.session_id = Some(session.to_string());
        }
    }

    async fn round_trip(
        &mut self,
        request: &JsonRpcRequest,
        timeout: Duration,
        flags: &ListChangeFlags,
    ) -> Result<serde_json::Value, McpError> {
        let builder = self.prepare(self.client.post(&self.url).json(request));
        let response = tokio::time::timeout(timeout, builder.send())
            .await
            .map_err(|_| McpError::Timeout {
                method: request.method.clone(),
                after: timeout,
            })?
            .map_err(|e| McpError::Http(format!("request to {} failed: {e}", self.url)))?;

        self.capture_session(&response);

        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body = tokio::time::timeout(timeout, response.text())
            .await
            .map_err(|_| McpError::Timeout {
                method: request.method.clone(),
                after: timeout,
            })?
            .map_err(|e| McpError::Http(format!("reading response body failed: {e}")))?;

        if !status.is_success() {
            return Err(McpError::Http(format!(
                "{} returned HTTP {status}: {}",
                self.url,
                body.chars().take(500).collect::<String>()
            )));
        }

        let messages = if content_type.contains("text/event-stream") {
            parse_sse_messages(&body)
        } else if body.trim().is_empty() {
            Vec::new()
        } else {
            match serde_json::from_str::<serde_json::Value>(body.trim())? {
                serde_json::Value::Array(items) => items,
                other => vec![other],
            }
        };

        for message in messages {
            if let Some(list_kind) = list_kind_from_notification(&message) {
                mark_list_changed_for_kind(flags, list_kind);
                continue;
            }
            let response: JsonRpcResponse = match serde_json::from_value(message) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if response.id != Some(serde_json::Value::Number(request.id.into())) {
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

        Err(McpError::UnexpectedResponse(format!(
            "no JSON-RPC response with id {} in HTTP reply from {}",
            request.id, self.url
        )))
    }

    async fn send_notification(
        &mut self,
        notification: &serde_json::Value,
    ) -> Result<(), McpError> {
        let builder = self.prepare(self.client.post(&self.url).json(notification));
        let response = builder
            .send()
            .await
            .map_err(|e| McpError::Http(format!("notification to {} failed: {e}", self.url)))?;
        self.capture_session(&response);
        // A notification carries no response payload; 200/202 are both fine and
        // any body is intentionally ignored.
        Ok(())
    }
}

/// Parse an SSE (`text/event-stream`) body into the JSON values carried by its
/// `data:` fields. Events are separated by blank lines; multiple `data:` lines
/// within one event are joined with newlines (per the SSE spec).
fn parse_sse_messages(body: &str) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    for block in body.split("\n\n") {
        let mut data = String::new();
        for raw in block.lines() {
            let raw = raw.strip_suffix('\r').unwrap_or(raw);
            if let Some(rest) = raw.strip_prefix("data:") {
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest);
            }
        }
        let trimmed = data.trim();
        if !trimmed.is_empty()
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
        {
            out.push(value);
        }
    }
    out
}

/// Read one newline-terminated line from `reader`, capped at `max` bytes
/// (DS-4). Returns `Ok(None)` on EOF; a line exceeding the cap is an error —
/// the stream is no longer parseable at that point, so the connection is
/// effectively dead.
async fn read_line_bounded(
    reader: &mut BufReader<ChildStdout>,
    max: u64,
) -> Result<Option<String>, McpError> {
    let mut buf = Vec::new();
    let n = (&mut *reader)
        .take(max + 1)
        .read_until(b'\n', &mut buf)
        .await?;
    if n == 0 {
        return Ok(None);
    }
    if buf.last() != Some(&b'\n') && n as u64 > max {
        return Err(McpError::UnexpectedResponse(format!(
            "MCP server sent a line exceeding the {max}-byte cap"
        )));
    }
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

fn mark_list_changed_for_method(flags: &ListChangeFlags, method: &str) {
    match method {
        "tools/list" => flags.tools.store(true, Ordering::Relaxed),
        "resources/list" => flags.resources.store(true, Ordering::Relaxed),
        "prompts/list" => flags.prompts.store(true, Ordering::Relaxed),
        _ => {}
    }
}

fn mark_list_changed_for_kind(flags: &ListChangeFlags, list_kind: ListKind) {
    match list_kind {
        ListKind::Tools => flags.tools.store(true, Ordering::Relaxed),
        ListKind::Resources => flags.resources.store(true, Ordering::Relaxed),
        ListKind::Prompts => flags.prompts.store(true, Ordering::Relaxed),
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
    fn validate_command_rejects_empty() {
        assert!(validate_command("", &[]).is_err());
    }

    #[test]
    fn validate_command_rejects_shell_metacharacters() {
        assert!(validate_command("cmd; rm -rf /", &[]).is_err());
        assert!(validate_command("$(whoami)", &[]).is_err());
        assert!(validate_command("cmd | cat", &[]).is_err());
        assert!(validate_command("cmd > /tmp/out", &[]).is_err());
        assert!(validate_command("cmd &", &[]).is_err());
        assert!(validate_command("`whoami`", &[]).is_err());
    }

    #[test]
    fn validate_command_allows_metacharacters_in_args() {
        // Arguments are passed directly to execve, not shell-interpreted.
        assert!(validate_command("safe-cmd", &["-c".into(), "echo $HOME".into()]).is_ok());
    }

    #[test]
    fn validate_command_accepts_safe_commands() {
        assert!(validate_command("fileio-mcp", &[]).is_ok());
        assert!(validate_command("/usr/bin/fileio-mcp", &[]).is_ok());
        assert!(
            validate_command(
                "genmcp",
                &["--config".into(), "/path/to/config.toml".into()]
            )
            .is_ok()
        );
    }

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
