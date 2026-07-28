//! Model Context Protocol (MCP) client for discovering and invoking external tool servers.

mod builtin;
pub mod config;
pub mod executor;
mod jsonrpc;
#[cfg(feature = "http")]
pub mod oauth;
#[cfg(feature = "http")]
pub mod url_policy;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use desktop_assistant_core::domain::ToolDefinition;
#[cfg(feature = "http")]
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

/// Cap on a remote (HTTP) response body — the streamable-HTTP analogue of
/// [`MAX_LINE_BYTES`]. A remote server (or a hostile endpoint impersonating
/// one) cannot make the daemon buffer unbounded memory; anything larger fails
/// the request. Generous, because an SSE reply can carry a whole tool result.
#[cfg(feature = "http")]
const MAX_HTTP_BODY_BYTES: usize = 16 * 1024 * 1024;

/// The MCP spec revision this client requests at `initialize`. The spec says a
/// client SHOULD request the latest version it supports; bumping this is the
/// one-line edit that adopts a new revision.
const REQUESTED_PROTOCOL_VERSION: &str = "2025-11-25";

/// Revisions this client can reason about, oldest first.
///
/// Deliberately wider than what our own servers advertise: `mcp-core` retired
/// `2024-11-05` and `2025-03-26`, but this client also talks to third-party
/// servers that have not. Every revision listed carries `initialize`,
/// `tools/list`, `tools/call` and `resources/list` identically, which is why a
/// downgrade to any of them is safe to proceed on.
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];

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

    #[error("invalid MCP configuration: {0}")]
    InvalidConfig(String),

    #[error("MCP client is not connected")]
    NotConnected,

    #[error(
        "MCP server negotiated protocol version '{got}', which this client does not \
         support (requested '{requested}')"
    )]
    UnsupportedProtocolVersion { got: String, requested: String },

    #[error("MCP request '{method}' timed out after {after:?} of silence")]
    Timeout { method: String, after: Duration },

    #[cfg(feature = "http")]
    #[error("HTTP transport error: {0}")]
    Http(String),

    #[cfg(feature = "http")]
    #[error("OAuth error: {0}")]
    OAuth(#[from] oauth::OAuthError),
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

/// Names read from *this process's own* environment and passed through to
/// **every** spawned stdio MCP child — including third-party ones this
/// client is designed to talk to (this file talks to servers well beyond
/// the shipped fleet, e.g. [`SUPPORTED_PROTOCOL_VERSIONS`]'s note on
/// third-party servers, and [`ServerMetadata`] treats every server-declared
/// field as untrusted) — on top of [`Command::env_clear`] and the server's
/// own configured `env` (applied afterward, so a server's config always
/// wins over the ambient value it happens to share a name with).
///
/// An explicit ALLOWLIST, not a denylist (#910): a `*_SECRET` / `*_TOKEN` /
/// `*_PASSWORD` pattern match fails open on the next variable nobody thought
/// of, which is exactly the failure mode that let a spawned child inherit
/// `DESKTOP_ASSISTANT_DATABASE_URL` — the application role's Postgres DSN —
/// and reach straight past the #721/#722 `scratch`-schema sandbox.
///
/// This same code path spawns servers for **two** deployment shapes: the
/// daemon's own fleet (headless, containers) via `crates/mcp-client/src/executor.rs`,
/// and the **client-side** MCP host (`crates/client-common/src/mcp_host/host.rs`),
/// which runs on a real desktop session where D-Bus and audio genuinely
/// exist. Weigh both when adding or refusing an entry, not just the fleet
/// container.
///
/// **This list is deliberately narrower than "every variable some shipped
/// server wants".** A variable that would hand *every* spawned server —
/// including a third-party one an operator adds — a route to something
/// sensitive belongs on [`McpServerConfig::inherit_env`] instead, scoped to
/// the one server that needs it. `DBUS_SESSION_BUS_ADDRESS` and
/// `XDG_RUNTIME_DIR` are the concrete case: both are exactly what a stock
/// D-Bus client library uses to auto-discover the session bus, which fronts
/// the freedesktop Secret Service holding connector API keys and MCP OAuth
/// tokens — see [`McpServerConfig::inherit_env`]'s doc for the full
/// reasoning. They are granted to `tasks-mcp` and `internet-radio-mcp`
/// specifically, not here.
///
/// [`McpServerConfig::inherit_env`]: crate::executor::McpServerConfig::inherit_env
///
/// Every entry is named for the server(s) — in the shipped fleet
/// (`deploy/mcp/mcp_servers.default.toml`, `Dockerfile.fleet`) or documented
/// by the server itself — that need it. Add a new entry only with that same
/// kind of evidence, not "it might be useful" — see
/// `crates/mcp-client/tests/env_isolation.rs` for the test that must
/// accompany it (one per variable, named for the server it protects, so a
/// later tightening fails the test that names what it broke).
const ENV_PASSTHROUGH_ALLOWLIST: &[&str] = &[
    // Every spawned server needs these to run at all.
    "PATH", // resolve its own subprocess dependencies (chromium, shell tools, mpv, ...)
    "HOME", // XDG fallback dir + any library that resolves `~` for its own config/cache
    // terminal-mcp's own defence-in-depth env scrub reads exactly this set -
    // PATH, HOME, USER, TMPDIR, TERM, LANG - from its process environment
    // before running a command. Supplying only some of those six would
    // silently drop the rest regardless of terminal-mcp's own settings.
    "USER",
    "TMPDIR",
    "TERM",
    // Locale/time, named directly in #910's fix-shape.
    "LANG",
    "TZ",
    // Outbound HTTP through a proxy: weather-forecast, geocode, openstreetmap,
    // cve, and web all call an external service. Both casings, because
    // different HTTP client libraries check different spellings.
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    // XDG base dirs. tasks-mcp and timeclock-mcp default their persistent
    // storage under $XDG_DATA_HOME (docs/mcp-services.md); the k8s
    // deployment repoints XDG_DATA_HOME/XDG_CONFIG_HOME at the
    // persistent-volume state dir specifically so daemon-side state
    // survives a pod restart (deploy/k8s/base/daemon.yaml). Without
    // pass-through, a spawned server would silently fall back to $HOME on
    // the ephemeral container filesystem and lose its data on the next
    // restart. Pass the whole XDG family together - a compliant program
    // isn't designed to reason about a partial view of it.
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
    // NOTE: XDG_RUNTIME_DIR and DBUS_SESSION_BUS_ADDRESS are deliberately
    // NOT here — see this constant's doc comment above and
    // `McpServerConfig::inherit_env`. Granting them globally would give
    // every spawned server, including a third-party one, the standard
    // auto-discovery route to the session D-Bus bus and, through it, the
    // freedesktop Secret Service credential store.
    //
    // Named single-server dependencies from the shipped fleet image
    // (Dockerfile.fleet, deploy/mcp/mcp_servers.default.toml) or documented
    // by the server itself.
    "WEB_CHROME_PATH",  // web-mcp: bundled headless-Chrome binary
    "SKILLS_MCP_ROOTS", // skills-mcp: skill root search path
    // skills-mcp (enabled by default): documented override for where NEW
    // skills are written when the default root isn't writable (e.g. a
    // read-only container filesystem) - the sibling of SKILLS_MCP_ROOTS
    // above, for writes rather than reads.
    "SKILLS_MCP_WRITE_ROOT",
];

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
    /// The `instructions` string the server returned from `initialize`, if any
    /// (trimmed, non-empty). Captured once at connect and used as the primary
    /// seed for the server's provider description in tool-search surfacing.
    server_instructions: Option<String>,
    /// The protocol revision this session actually negotiated, captured from
    /// the `initialize` result. `None` only before the handshake runs — a
    /// completed `initialize` always sets it or fails.
    protocol_version: Option<String>,
    /// What the server declared about itself in `serverInfo` (SEP-973). Empty
    /// for every server that has not opted in.
    server_metadata: ServerMetadata,
}

/// What a server declared about *itself* in `initialize`'s `serverInfo`
/// (SEP-973, spec revision 2025-11-25), beyond the required `name`/`version`.
///
/// Grouped rather than carried as three loose `Option<String>`s because the
/// three are parsed together and travel together through `ClientHandle`,
/// `McpServerStatusInfo` and on to the settings surface — three call sites, so
/// the type is earned rather than speculative. It stays a *domain* shape: the
/// wire type flattens it, matching its neighbouring fields.
///
/// Every field is untrusted input from whatever process the server config
/// points at. They are trimmed and empty-filtered here; a consumer that renders
/// them owes them the same scrutiny it gives any other remote string.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerMetadata {
    /// Human-facing display name, where `name` is the programmatic identity.
    pub title: Option<String>,
    /// What the server offers. Distinct from `instructions`, which is usage
    /// guidance aimed at the model.
    pub description: Option<String>,
    /// The server's home page.
    pub website_url: Option<String>,
}

impl ServerMetadata {
    /// True when the server declared none of the optional fields — the case for
    /// every server that has not opted in.
    pub fn is_empty(&self) -> bool {
        self.title.is_none() && self.description.is_none() && self.website_url.is_none()
    }
}

/// Extract the optional [`ServerMetadata`] from an MCP `initialize` result.
///
/// Applies the same rule as [`parse_server_instructions`]: trimmed, and blank
/// treated as absent, so a whitespace-only value falls through to the next
/// description source rather than seeding an empty one. A non-string value is
/// dropped rather than coerced.
pub fn parse_server_metadata(result: &serde_json::Value) -> ServerMetadata {
    let field = |key: &str| {
        result
            .get("serverInfo")
            .and_then(|info| info.get(key))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
    };
    ServerMetadata {
        title: field("title"),
        description: field("description"),
        website_url: field("websiteUrl"),
    }
}

/// Extract the trimmed, non-empty `instructions` string from an MCP
/// `initialize` result, or `None` when absent or blank. Servers may include
/// human-facing usage instructions here (MCP spec); Adele seeds a server's
/// provider description from it when the server declares no
/// `serverInfo.description`.
pub fn parse_server_instructions(result: &serde_json::Value) -> Option<String> {
    result
        .get("instructions")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

/// Validate the `protocolVersion` an MCP server answered `initialize` with, and
/// return the revision this session is on.
///
/// Why warn-and-proceed rather than disconnect: the spec says a client that
/// does not support the returned version SHOULD disconnect. We deliberately do
/// not, for a version we recognise. Every revision in
/// [`SUPPORTED_PROTOCOL_VERSIONS`] carries the entire surface this client uses
/// identically, so hard-failing would break working third-party servers — and
/// every fleet server whose `mcp-core` pin has not moved yet — in exchange for
/// nothing. We fail only where we genuinely cannot reason about the session:
/// a version we do not know, a non-string value, or no value at all.
fn negotiated_protocol_version(result: &serde_json::Value) -> Result<String, McpError> {
    let Some(version) = result.get("protocolVersion") else {
        return Err(McpError::UnexpectedResponse(
            "initialize result is missing 'protocolVersion'".into(),
        ));
    };
    let Some(version) = version.as_str() else {
        return Err(McpError::UnexpectedResponse(format!(
            "initialize result has a non-string 'protocolVersion': {version}"
        )));
    };

    if !SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
        return Err(McpError::UnsupportedProtocolVersion {
            got: version.to_string(),
            requested: REQUESTED_PROTOCOL_VERSION.to_string(),
        });
    }

    if version != REQUESTED_PROTOCOL_VERSION {
        let server = result
            .get("serverInfo")
            .and_then(|i| i.get("name"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unnamed>");
        tracing::warn!(
            server,
            negotiated = version,
            requested = REQUESTED_PROTOCOL_VERSION,
            "MCP server negotiated an older protocol revision; proceeding"
        );
    }

    Ok(version.to_string())
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
    #[cfg(feature = "http")]
    pub async fn connect_http(url: &str, bearer: Option<String>) -> Result<Self, McpError> {
        Self::connect_http_with_request_timeout(url, bearer, DEFAULT_REQUEST_TIMEOUT).await
    }

    /// [`Self::connect_http`] with an explicit per-request silence timeout.
    #[cfg(feature = "http")]
    pub async fn connect_http_with_request_timeout(
        url: &str,
        bearer: Option<String>,
        request_timeout: Duration,
    ) -> Result<Self, McpError> {
        Self::connect_http_credential(url, Credential::from_bearer(bearer), request_timeout).await
    }

    /// Connect to a remote MCP server over streamable-HTTP, authenticating with
    /// an OAuth 2.0 [`TokenProvider`](oauth::TokenProvider). The provider mints
    /// and refreshes access tokens on demand, and the transport retries once
    /// with a fresh token if the server answers `401`.
    #[cfg(feature = "http")]
    pub async fn connect_http_oauth(
        url: &str,
        provider: Arc<oauth::TokenProvider>,
    ) -> Result<Self, McpError> {
        Self::connect_http_oauth_with_request_timeout(url, provider, DEFAULT_REQUEST_TIMEOUT).await
    }

    /// [`Self::connect_http_oauth`] with an explicit per-request silence timeout.
    #[cfg(feature = "http")]
    pub async fn connect_http_oauth_with_request_timeout(
        url: &str,
        provider: Arc<oauth::TokenProvider>,
        request_timeout: Duration,
    ) -> Result<Self, McpError> {
        Self::connect_http_credential(url, Credential::OAuth(provider), request_timeout).await
    }

    #[cfg(feature = "http")]
    async fn connect_http_credential(
        url: &str,
        credential: Credential,
        request_timeout: Duration,
    ) -> Result<Self, McpError> {
        let transport = Transport::Http(HttpTransport::new(url, credential)?);
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
            server_instructions: None,
            protocol_version: None,
            server_metadata: ServerMetadata::default(),
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

    /// The server's `initialize` instructions (trimmed, non-empty), if it sent
    /// any. Seeds this server's provider description for tool-search surfacing.
    pub fn server_instructions(&self) -> Option<&str> {
        self.server_instructions.as_deref()
    }

    /// The MCP protocol revision this session negotiated, as reported by the
    /// server's `initialize` result — which may be older than
    /// [`REQUESTED_PROTOCOL_VERSION`]. `None` only before the handshake has
    /// run; a connected client always has one.
    pub fn protocol_version(&self) -> Option<&str> {
        self.protocol_version.as_deref()
    }

    /// What the server declared about itself in `initialize`'s `serverInfo`
    /// (SEP-973). Empty for any server that has not opted in, which is all of
    /// them until they do.
    pub fn server_metadata(&self) -> &ServerMetadata {
        &self.server_metadata
    }

    async fn initialize(&mut self) -> Result<(), McpError> {
        let params = serde_json::json!({
            "protocolVersion": REQUESTED_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "desktop-assistant",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let response = self.send_request("initialize", Some(params)).await?;
        self.server_metadata = parse_server_metadata(&response);
        let negotiated = negotiated_protocol_version(&response)?;
        // Before the `initialized` notification, which is itself the first
        // post-initialize request and so must already carry the header.
        self.transport.set_protocol_version(&negotiated);
        self.protocol_version = Some(negotiated);
        self.server_instructions = parse_server_instructions(&response);

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
    #[cfg(feature = "http")]
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
            #[cfg(feature = "http")]
            Transport::Http(t) => t.round_trip(request, timeout, flags).await,
        }
    }

    /// Record the revision `initialize` negotiated, so transports that carry it
    /// on the wire can start doing so.
    ///
    /// Only Streamable HTTP has somewhere to put it (`MCP-Protocol-Version`,
    /// required on every post-initialize request since 2025-06-18). stdio
    /// negotiates once per process and has no per-message envelope, so this is
    /// a no-op there.
    fn set_protocol_version(&mut self, version: &str) {
        match self {
            Transport::Stdio(_) => {
                // Explicitly discarded rather than ignored: with the `http`
                // feature off this is the only arm, and an unused parameter is
                // a hard error under the workspace lint table.
                let _ = version;
            }
            #[cfg(feature = "http")]
            Transport::Http(t) => t.protocol_version = Some(version.to_string()),
        }
    }

    async fn send_notification(
        &mut self,
        notification: &serde_json::Value,
    ) -> Result<(), McpError> {
        match self {
            Transport::Stdio(t) => t.send_notification(notification).await,
            #[cfg(feature = "http")]
            Transport::Http(t) => t.send_notification(notification).await,
        }
    }

    async fn shutdown(&mut self) {
        match self {
            Transport::Stdio(t) => t.shutdown().await,
            // HTTP has no process to reap.
            #[cfg(feature = "http")]
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
    ///
    /// The child gets an explicit environment, not the daemon's whole one
    /// (#910): [`Command::env_clear`], then [`ENV_PASSTHROUGH_ALLOWLIST`]
    /// read from this process's own environment, then `env` (the server's
    /// configured `env`/`env_secrets`, already resolved by the caller) on
    /// top — so a server's own config always wins over an ambient value it
    /// shares a name with.
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
            .kill_on_drop(true)
            .env_clear();
        for key in ENV_PASSTHROUGH_ALLOWLIST {
            // `var_os`, not `var`: `var` returns `Err` both when a variable
            // is absent and when its value is not valid UTF-8, which would
            // silently drop a well-formed-but-non-UTF-8 value (e.g. a path
            // with non-UTF-8 bytes) instead of passing it through.
            if let Some(value) = std::env::var_os(key) {
                cmd.env(key, value);
            }
        }
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

        match tokio::try_join!(write_fut, read_fut) {
            Ok(((), result)) => Ok(result),
            Err(err) => Err(Self::enrich_with_exit_status(&mut self.child, err).await),
        }
    }

    /// Bound on waiting for the child's exit status in
    /// [`Self::enrich_with_exit_status`]. Generous because it only runs on an
    /// already-failed path (a few extra seconds before reporting a failure
    /// that already happened is a fair trade for naming its cause), and it
    /// can never hang past the *caller's* own handshake timeout ([`INIT_TIMEOUT`]
    /// / the configured request timeout in `McpClient::from_transport`),
    /// since this whole wait runs inside that outer bound.
    const EXIT_STATUS_WAIT: Duration = Duration::from_secs(10);

    /// If `err` is the generic "server closed stdout" error and the child has
    /// exited (or exits promptly), replace it with one naming the exit status.
    ///
    /// Why: a spawned server that exits immediately (missing dependency, bad
    /// config, or — since #910 — an environment variable it needed that
    /// isn't on [`ENV_PASSTHROUGH_ALLOWLIST`] or its own configured `env`)
    /// used to surface only "MCP server closed stdout", which does not say
    /// the process is even gone. This message flows verbatim into
    /// `McpServerStatusInfo::detail` (the settings/KCM panel's honest-state
    /// field) and the daemon's `ERROR failed to connect to MCP server` log
    /// line, so naming the exit status here is what makes a too-tight
    /// allowlist diagnosable instead of a silent "server just isn't there".
    ///
    /// Uses `wait()` rather than the non-blocking `try_wait()`: the pipe's
    /// read end closes (which is what produced `err`) as soon as the kernel
    /// tears down the process's file descriptors, but tokio's own SIGCHLD
    /// -driven reaping runs on its own background task and can lag that under
    /// scheduler contention — `try_wait()` observed here raced `Ok(None)`
    /// ("still running") under a fully parallel `cargo test --workspace` run,
    /// though the process had in fact already exited. `wait()` blocks until
    /// the reap actually completes, bounded by [`Self::EXIT_STATUS_WAIT`] so
    /// this can never hang the caller if stdout closed for some other reason
    /// and the process is still alive.
    async fn enrich_with_exit_status(child: &mut Child, err: McpError) -> McpError {
        let McpError::UnexpectedResponse(ref msg) = err else {
            return err;
        };
        if msg != "MCP server closed stdout" {
            return err;
        }
        let Ok(Ok(status)) = tokio::time::timeout(Self::EXIT_STATUS_WAIT, child.wait()).await
        else {
            return err;
        };
        let detail = match status.code() {
            Some(code) => format!("exited with status {code}"),
            None => format!("was terminated by a signal ({status})"),
        };
        McpError::UnexpectedResponse(format!(
            "MCP server {detail} before completing the handshake; if it needs an \
             environment variable, set it in this server's own `env` config (see \
             docs/mcp-services.md#environment-variables) rather than relying on it \
             being inherited"
        ))
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

/// How an [`HttpTransport`] authenticates each request.
#[cfg(feature = "http")]
enum Credential {
    /// No `Authorization` header (e.g. a single-user local endpoint).
    None,
    /// A verbatim bearer token from `secrets.toml` — used as-is, never refreshed.
    Static(String),
    /// An OAuth 2.0 provider that mints and refreshes access tokens on demand.
    OAuth(Arc<oauth::TokenProvider>),
}

#[cfg(feature = "http")]
impl Credential {
    fn from_bearer(bearer: Option<String>) -> Self {
        match bearer {
            Some(token) => Credential::Static(token),
            None => Credential::None,
        }
    }

    fn is_oauth(&self) -> bool {
        matches!(self, Credential::OAuth(_))
    }

    /// The bearer token to attach right now — refreshing an OAuth token first
    /// if it is missing or near expiry. `None` means send no `Authorization`.
    async fn token(&self) -> Result<Option<String>, McpError> {
        Ok(match self {
            Credential::None => None,
            Credential::Static(token) => Some(token.clone()),
            Credential::OAuth(provider) => Some(provider.current_token().await?),
        })
    }

    /// Force a fresh OAuth token (after a `401`); other credential kinds have
    /// nothing to refresh and just return their current token.
    async fn refreshed_token(&self) -> Result<Option<String>, McpError> {
        match self {
            Credential::OAuth(provider) => Ok(Some(provider.force_refresh().await?)),
            other => other.token().await,
        }
    }
}

/// JSON-RPC over a remote streamable-HTTP MCP endpoint. Each request is a POST
/// whose reply is either a single JSON body or a `text/event-stream` (SSE)
/// sequence of JSON-RPC messages.
#[cfg(feature = "http")]
struct HttpTransport {
    client: reqwest::Client,
    url: String,
    /// How to authenticate each request: none, a static bearer, or an OAuth
    /// provider that mints/refreshes access tokens on demand.
    credential: Credential,
    /// `Mcp-Session-Id` assigned by the server on initialize; echoed on
    /// subsequent requests when present.
    session_id: Option<String>,
    /// The negotiated revision, sent as `MCP-Protocol-Version` on every request
    /// after initialize (required since spec revision 2025-06-18).
    ///
    /// `None` until the handshake resolves, which is exactly the window in
    /// which the header must *not* be sent: nothing has been negotiated yet.
    protocol_version: Option<String>,
}

#[cfg(feature = "http")]
impl HttpTransport {
    fn new(url: &str, credential: Credential) -> Result<Self, McpError> {
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
            credential,
            session_id: None,
            protocol_version: None,
        })
    }

    /// Send one POST and return `(status, content_type, body)`. Captures the
    /// session id and caps the body size. `token`, when set, is attached as
    /// `Authorization: Bearer`.
    async fn send_once(
        &mut self,
        payload: &serde_json::Value,
        method: &str,
        token: Option<&str>,
        timeout: Duration,
    ) -> Result<(reqwest::StatusCode, String, String), McpError> {
        let mut builder = self
            .client
            .post(&self.url)
            .header(ACCEPT, "application/json, text/event-stream")
            .json(payload);
        if let Some(token) = token {
            builder = builder.bearer_auth(token);
        }
        if let Some(session) = &self.session_id {
            builder = builder.header("Mcp-Session-Id", session);
        }
        if let Some(version) = &self.protocol_version {
            builder = builder.header("MCP-Protocol-Version", version);
        }

        let response = tokio::time::timeout(timeout, builder.send())
            .await
            .map_err(|_| McpError::Timeout {
                method: method.to_string(),
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
        let body = read_body_capped(response, timeout, method, MAX_HTTP_BODY_BYTES).await?;
        Ok((status, content_type, body))
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
        let payload = serde_json::to_value(request)?;
        let method = request.method.as_str();

        let token = self.credential.token().await?;
        let (status, content_type, body) = self
            .send_once(&payload, method, token.as_deref(), timeout)
            .await?;

        // If the resource server rejects the (possibly stale) access token,
        // mint a fresh one and retry once. Only OAuth credentials can refresh;
        // a static bearer or no-auth request returns the 401 as-is.
        let (status, content_type, body) =
            if status == reqwest::StatusCode::UNAUTHORIZED && self.credential.is_oauth() {
                tracing::info!(
                    "MCP HTTP endpoint {} returned 401; refreshing OAuth token and retrying",
                    self.url
                );
                let token = self.credential.refreshed_token().await?;
                self.send_once(&payload, method, token.as_deref(), timeout)
                    .await?
            } else {
                (status, content_type, body)
            };

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
        // Resolve (and, for OAuth, proactively refresh) the token first.
        let token = self.credential.token().await?;
        let mut builder = self
            .client
            .post(&self.url)
            .header(ACCEPT, "application/json, text/event-stream")
            .json(notification);
        if let Some(token) = &token {
            builder = builder.bearer_auth(token);
        }
        if let Some(session) = &self.session_id {
            builder = builder.header("Mcp-Session-Id", session);
        }
        if let Some(version) = &self.protocol_version {
            builder = builder.header("MCP-Protocol-Version", version);
        }
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

/// Read an HTTP response body, bounding both total size ([`MAX_HTTP_BODY_BYTES`],
/// the streamable-HTTP analogue of the stdio `MAX_LINE_BYTES` cap) and the
/// overall read time, so a slow or oversized remote reply fails the request
/// instead of hanging or exhausting memory.
#[cfg(feature = "http")]
async fn read_body_capped(
    mut response: reqwest::Response,
    timeout: Duration,
    method: &str,
    max: usize,
) -> Result<String, McpError> {
    let read = async {
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| McpError::Http(format!("reading response body failed: {e}")))?
        {
            if buf.len() + chunk.len() > max {
                return Err(McpError::Http(format!(
                    "response body from remote MCP server exceeded {max} bytes"
                )));
            }
            buf.extend_from_slice(&chunk);
        }
        String::from_utf8(buf)
            .map_err(|e| McpError::Http(format!("response body was not valid UTF-8: {e}")))
    };
    tokio::time::timeout(timeout, read)
        .await
        .map_err(|_| McpError::Timeout {
            method: method.to_string(),
            after: timeout,
        })?
}

/// Parse an SSE (`text/event-stream`) body into the JSON values carried by its
/// `data:` fields. Events are separated by blank lines; multiple `data:` lines
/// within one event are joined with newlines (per the SSE spec).
#[cfg(feature = "http")]
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

    // ----- HttpTransport::new applies the shared remote-URL policy (#804) -----
    //
    // `HttpTransport::new` used to accept any URL that merely started with
    // "http://" or "https://" — including a plain-http URL to any host,
    // which sends the bearer/OAuth token attached to every request in the
    // clear. These exercise the fix directly, without a mock server: the
    // refusal happens before any request is sent.

    #[cfg(feature = "http")]
    #[test]
    fn http_transport_new_rejects_plain_http_to_a_public_host() {
        // `HttpTransport` is not `Debug`, so match instead of `expect_err`.
        match HttpTransport::new("http://evil.example.com/mcp", Credential::None) {
            Ok(_) => panic!("plain http to a non-loopback host must be refused"),
            Err(McpError::Http(_)) => {}
            Err(other) => panic!("expected a transport error, got {other:?}"),
        }
    }

    #[cfg(feature = "http")]
    #[test]
    fn http_transport_new_accepts_plain_http_to_loopback() {
        // Loopback stays permitted: this is also what keeps the httpmock
        // integration tests in tests/http_transport.rs working, since
        // httpmock binds to 127.0.0.1.
        HttpTransport::new("http://127.0.0.1:8080/mcp", Credential::None)
            .expect("plain http to loopback must still connect");
    }

    #[cfg(feature = "http")]
    #[test]
    fn http_transport_new_accepts_https() {
        HttpTransport::new("https://mcp.example.com/mcp", Credential::None)
            .expect("https to a public host must be accepted");
    }

    #[cfg(feature = "http")]
    #[test]
    fn http_transport_new_rejects_a_disallowed_scheme() {
        match HttpTransport::new("ftp://mcp.example.com/mcp", Credential::None) {
            Ok(_) => panic!("a non-http(s) scheme must be refused"),
            Err(McpError::Http(_)) => {}
            Err(other) => panic!("expected a transport error, got {other:?}"),
        }
    }

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
    fn parse_server_instructions_captures() {
        // The `instructions` field of an initialize result becomes the server's
        // description seed, trimmed of surrounding whitespace.
        let result = serde_json::json!({
            "serverInfo": {"name": "weather"},
            "instructions": "  Query live weather and forecasts.  "
        });
        assert_eq!(
            parse_server_instructions(&result).as_deref(),
            Some("Query live weather and forecasts.")
        );
    }

    #[test]
    fn parse_server_instructions_absent_is_none() {
        let result = serde_json::json!({"serverInfo": {"name": "weather"}});
        assert_eq!(parse_server_instructions(&result), None);
    }

    #[test]
    fn parse_server_instructions_blank_is_none() {
        // A whitespace-only instructions string carries no signal — treat it as
        // absent so description resolution falls through to the config/boilerplate.
        let result = serde_json::json!({"instructions": "   \n\t "});
        assert_eq!(parse_server_instructions(&result), None);
    }

    #[test]
    fn parse_server_metadata_captures_all_three() {
        let result = serde_json::json!({
            "serverInfo": {
                "name": "weather",
                "version": "1.0",
                "title": "  Weather Service  ",
                "description": "  Live weather and forecasts.  ",
                "websiteUrl": "  https://example.com/weather  "
            }
        });
        let meta = parse_server_metadata(&result);
        assert_eq!(meta.title.as_deref(), Some("Weather Service"));
        assert_eq!(
            meta.description.as_deref(),
            Some("Live weather and forecasts.")
        );
        assert_eq!(
            meta.website_url.as_deref(),
            Some("https://example.com/weather")
        );
    }

    #[test]
    fn parse_server_metadata_absent_is_none() {
        let result = serde_json::json!({"serverInfo": {"name": "weather", "version": "1.0"}});
        let meta = parse_server_metadata(&result);
        assert_eq!(meta.title, None);
        assert_eq!(meta.description, None);
        assert_eq!(meta.website_url, None);
    }

    #[test]
    fn parse_server_metadata_blank_is_none() {
        // Same rule as instructions: whitespace-only carries no signal, so
        // description resolution must fall through rather than seed a blank.
        let result = serde_json::json!({
            "serverInfo": {"title": "  ", "description": "\n\t", "websiteUrl": ""}
        });
        let meta = parse_server_metadata(&result);
        assert_eq!(meta.title, None);
        assert_eq!(meta.description, None);
        assert_eq!(meta.website_url, None);
    }

    #[test]
    fn parse_server_metadata_ignores_non_string_values() {
        // Malformed input from an untrusted peer must be dropped, not coerced.
        let result = serde_json::json!({
            "serverInfo": {"title": 42, "description": ["a"], "websiteUrl": {"a": 1}}
        });
        let meta = parse_server_metadata(&result);
        assert_eq!(meta.title, None);
        assert_eq!(meta.description, None);
        assert_eq!(meta.website_url, None);
    }

    #[test]
    fn parse_server_metadata_without_server_info_is_none() {
        let meta = parse_server_metadata(&serde_json::json!({"protocolVersion": "2025-11-25"}));
        assert!(meta.is_empty());
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
