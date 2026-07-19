//! The client-side MCP host: spawn the configured local MCP servers, discover
//! their tools, and expose them as client-side tool registrations the daemon
//! can invoke — routing each call back to the owning server.
//!
//! Reuses [`McpClient`] (stdio spawn + `list_tools` + `call_tool` + `shutdown`)
//! verbatim; the host adds only orchestration: per-server spawn/supervise, tool
//! namespacing, a routing table, and the size cap the daemon enforces on
//! client-tool results.

use desktop_assistant_api_model::ClientToolRegistration;
use desktop_assistant_mcp_client::McpClient;
use mcp_core::McpService;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::config::McpServerConfig;

/// Cap on a client-tool result field. The daemon rejects anything larger
/// (`MAX_CLIENT_TOOL_RESULT_BYTES`), which would fail the turn — so we truncate
/// to keep the result usable and the turn alive.
const MAX_RESULT_BYTES: usize = 1024 * 1024;

/// Separator between a server's namespace and a tool name (`{ns}__{tool}`),
/// matching the daemon's own MCP tool-namespacing convention.
const NAMESPACE_SEP: &str = "__";

/// A built-in MCP server hosted **in-process**: an [`McpService`] the host calls
/// directly, with no subprocess. Its tools are namespaced, registered, routed,
/// counted, and shut down exactly like a spawned server's — downstream code
/// cannot tell the two apart.
pub struct BuiltinServer {
    /// Diagnostic name (mirrors a subprocess server's `cfg.name`).
    pub name: String,
    /// Tool-namespace prefix this server's tools are advertised under.
    pub namespace: String,
    /// The in-process service implementation.
    pub service: Arc<dyn McpService>,
}

impl BuiltinServer {
    /// Build an in-process built-in server from a service implementation.
    pub fn new(
        name: impl Into<String>,
        namespace: impl Into<String>,
        service: Arc<dyn McpService>,
    ) -> Self {
        Self {
            name: name.into(),
            namespace: namespace.into(),
            service,
        }
    }
}

/// How a hosted server executes tool calls: a spawned subprocess reached over
/// stdio, or an in-process [`McpService`] called directly. Every other host
/// operation (namespacing, routing, counting) is backend-agnostic.
enum ServerBackend {
    /// A spawned stdio server. `list_tools`/`call_tool` need `&mut McpClient`,
    /// so it sits behind its own mutex (a slow call on one server never blocks
    /// another). Boxed: an `McpClient` is far larger than the `InProcess`
    /// variant, so boxing keeps every `HostedServer` uniformly small.
    Subprocess(Box<Mutex<McpClient>>),
    /// A built-in service run in-process (no subprocess). Shared immutably; its
    /// `call_tool` takes `&self`, so no mutex is needed.
    InProcess(Arc<dyn McpService>),
}

/// A running local MCP server, whether spawned as a subprocess or hosted
/// in-process.
struct HostedServer {
    /// Config name, for diagnostics.
    name: String,
    /// Tool-namespace prefix this server's tools are advertised under
    /// (`cfg.namespace`, or the name when unset) — the key [`McpHost::tool_counts`]
    /// reports per-server totals against.
    namespace: String,
    /// The execution backend for this server's tool calls.
    backend: ServerBackend,
}

/// A set of running local MCP servers whose tools are exposed to the daemon as
/// client-side tools. Each tool is advertised as `{namespace}__{tool}` and
/// routed back to its owning server on invocation.
pub struct McpHost {
    servers: Vec<HostedServer>,
    /// Namespaced tool name -> (server index, original tool name).
    routes: HashMap<String, (usize, String)>,
    registrations: Vec<ClientToolRegistration>,
}

impl McpHost {
    /// Start the given subprocess servers: spawn each, list its tools, and build
    /// the registration set + routing table.
    ///
    /// Degrades: a server that fails to start (or list its tools) is logged and
    /// skipped — the host still serves every healthy server. Never panics on a
    /// bad server.
    pub async fn start(servers: &[McpServerConfig]) -> Self {
        Self::start_with(servers, Vec::new()).await
    }

    /// Like [`start`](Self::start), but additionally hosts each `builtins`
    /// entry **in-process** (no subprocess). Built-in tools are namespaced,
    /// registered, routed, counted, and shut down identically to subprocess
    /// tools; a built-in never "fails to start". Subprocess servers are set up
    /// first, so on a namespaced-name collision the built-in tool is the one
    /// skipped.
    pub async fn start_with(servers: &[McpServerConfig], builtins: Vec<BuiltinServer>) -> Self {
        let mut hosted: Vec<HostedServer> = Vec::new();
        let mut routes: HashMap<String, (usize, String)> = HashMap::new();
        let mut registrations: Vec<ClientToolRegistration> = Vec::new();

        for cfg in servers {
            if !cfg.env_secrets.is_empty() {
                tracing::warn!(
                    "client MCP server '{}' declares env_secrets; client-side secret \
                     resolution is not implemented yet, so those vars will be unset",
                    cfg.name
                );
            }

            let mut client = match McpClient::connect(&cfg.command, &cfg.args, &cfg.env).await {
                Ok(client) => client,
                Err(err) => {
                    tracing::warn!(
                        "client MCP server '{}' failed to start: {err}; skipping",
                        cfg.name
                    );
                    continue;
                }
            };

            let tools = match client.list_tools().await {
                Ok(tools) => tools,
                Err(err) => {
                    tracing::warn!(
                        "client MCP server '{}' failed to list tools: {err}; skipping",
                        cfg.name
                    );
                    client.shutdown().await;
                    continue;
                }
            };

            let namespace = cfg.namespace.clone().unwrap_or_else(|| cfg.name.clone());
            let index = hosted.len();
            let mut hosted_any = false;
            for tool in tools {
                let namespaced = format!("{namespace}{NAMESPACE_SEP}{}", tool.name);
                if routes.contains_key(&namespaced) {
                    tracing::warn!(
                        "duplicate client tool name '{namespaced}' from server '{}'; skipping it",
                        cfg.name
                    );
                    continue;
                }
                routes.insert(namespaced.clone(), (index, tool.name.clone()));
                registrations.push(ClientToolRegistration {
                    name: namespaced,
                    description: tool.description,
                    input_schema: tool.parameters,
                });
                hosted_any = true;
            }

            hosted.push(HostedServer {
                name: cfg.name.clone(),
                namespace: namespace.clone(),
                backend: ServerBackend::Subprocess(Box::new(Mutex::new(client))),
            });
            tracing::info!(
                "hosting client MCP server '{}' as namespace '{}'{}",
                cfg.name,
                namespace,
                if hosted_any { "" } else { " (no tools)" }
            );
        }

        // In-process built-ins use the same namespacing / dedup / routing /
        // registration path as subprocess servers; only the backend differs. A
        // built-in never fails to start.
        for builtin in builtins {
            let namespace = builtin.namespace.clone();
            let index = hosted.len();
            let mut hosted_any = false;
            for tool in builtin.service.tools() {
                let namespaced = format!("{namespace}{NAMESPACE_SEP}{}", tool.name);
                if routes.contains_key(&namespaced) {
                    tracing::warn!(
                        "duplicate client tool name '{namespaced}' from built-in server '{}'; \
                         skipping it",
                        builtin.name
                    );
                    continue;
                }
                routes.insert(namespaced.clone(), (index, tool.name.clone()));
                registrations.push(ClientToolRegistration {
                    name: namespaced,
                    description: tool.description,
                    input_schema: tool.input_schema,
                });
                hosted_any = true;
            }

            hosted.push(HostedServer {
                name: builtin.name.clone(),
                namespace: namespace.clone(),
                backend: ServerBackend::InProcess(builtin.service),
            });
            tracing::info!(
                "hosting built-in client MCP server '{}' as namespace '{}'{}",
                builtin.name,
                namespace,
                if hosted_any { "" } else { " (no tools)" }
            );
        }

        Self {
            servers: hosted,
            routes,
            registrations,
        }
    }

    /// The client-side tool registrations for every hosted server.
    pub fn registrations(&self) -> Vec<ClientToolRegistration> {
        self.registrations.clone()
    }

    /// The number of tools each hosted server currently exposes, keyed by the
    /// server's **namespace** (`cfg.namespace`, or the server name when it has
    /// none) — the same key the host uses to namespace that server's tools. A
    /// server that started but advertised no tools maps to `0`; a server that
    /// failed to start is absent (it hosts nothing).
    ///
    /// Client UIs surface this as each client-hosted server's live tool count
    /// (adele-gtk#125), matching the per-server totals daemon rows already show.
    pub fn tool_counts(&self) -> HashMap<String, usize> {
        // Each route points at its owning server by index; tally per index (so a
        // hosted-but-toolless server stays at 0), then key by that server's
        // namespace. Servers that never started aren't in `self.servers`, so they
        // are naturally absent.
        let mut per_index = vec![0usize; self.servers.len()];
        for &(index, _) in self.routes.values() {
            per_index[index] += 1;
        }
        self.servers
            .iter()
            .zip(per_index)
            .map(|(server, count)| (server.namespace.clone(), count))
            .collect()
    }

    /// Whether `tool_name` is one this host serves (used to decide whether a
    /// `ClientToolCall` should be routed here).
    pub fn handles(&self, tool_name: &str) -> bool {
        self.routes.contains_key(tool_name)
    }

    /// Invoke a hosted tool by its namespaced name, routing to the owning
    /// server.
    ///
    /// `Err` on an unknown tool or a transport/JSON-RPC failure. An MCP
    /// tool-level error (`isError`) comes back as its text in `Ok` — the LLM
    /// sees the error content as the result, which is the intended behavior.
    /// Results are capped to the daemon's client-tool limit.
    ///
    /// In-process built-ins are mapped to be byte-for-byte indistinguishable
    /// from a subprocess: a [`CallError::Tool`](mcp_core::CallError::Tool)
    /// becomes `Ok(text)` (the `isError` content path), while
    /// [`InvalidParams`](mcp_core::CallError::InvalidParams) /
    /// [`Internal`](mcp_core::CallError::Internal) are protocol-level faults and
    /// become `Err` — the same split `McpClient` produces from the wire.
    pub async fn call(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, String> {
        let Some((index, original)) = self.routes.get(tool_name) else {
            return Err(format!("unknown client MCP tool: {tool_name}"));
        };
        match &self.servers[*index].backend {
            ServerBackend::Subprocess(client) => {
                let mut client = client.lock().await;
                match client.call_tool(original, arguments).await {
                    Ok(result) => Ok(cap_result(result)),
                    Err(err) => Err(format!("client MCP tool '{tool_name}' failed: {err}")),
                }
            }
            ServerBackend::InProcess(svc) => match svc.call_tool(original, &arguments).await {
                Ok(reply) => Ok(cap_result(render_reply(&reply))),
                Err(mcp_core::CallError::Tool(m)) => Ok(cap_result(m)),
                Err(err) => Err(format!("client MCP tool '{tool_name}' failed: {err}")),
            },
        }
    }

    /// Shut down every hosted server: kill each subprocess, and invoke each
    /// in-process service's optional `shutdown` hook.
    pub async fn shutdown(self) {
        for server in self.servers {
            match server.backend {
                ServerBackend::Subprocess(client) => client.into_inner().shutdown().await,
                ServerBackend::InProcess(svc) => svc.shutdown().await,
            }
            tracing::debug!("shut down client MCP server '{}'", server.name);
        }
    }
}

/// Render an in-process [`ToolReply`](mcp_core::ToolReply) to the same joined
/// text `McpClient` extracts from the wire: text blocks verbatim, non-text
/// (raw) blocks pretty-printed, joined with newlines. With no content blocks,
/// fall back to the pretty `structuredContent`, else the empty string.
fn render_reply(reply: &mcp_core::ToolReply) -> String {
    let parts: Vec<String> = reply
        .content
        .iter()
        .map(|c| match c {
            mcp_core::Content::Text(t) => t.clone(),
            mcp_core::Content::Raw(v) => {
                serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
            }
        })
        .collect();
    if parts.is_empty() {
        reply
            .structured_content
            .as_ref()
            .map(|v| serde_json::to_string_pretty(v).unwrap_or_default())
            .unwrap_or_default()
    } else {
        parts.join("\n")
    }
}

/// Truncate a result to the daemon's client-tool byte cap at a UTF-8 char
/// boundary, appending a marker when truncated.
fn cap_result(mut result: String) -> String {
    if result.len() <= MAX_RESULT_BYTES {
        return result;
    }
    const MARKER: &str = "\n…[truncated]";
    let mut end = MAX_RESULT_BYTES - MARKER.len();
    while end > 0 && !result.is_char_boundary(end) {
        end -= 1;
    }
    result.truncate(end);
    result.push_str(MARKER);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    enum CallMode {
        Ok,
        Error,
        Oversize,
    }

    /// Render a minimal `/bin/sh` fake MCP server (mirrors the pattern in
    /// `mcp-client/tests/robustness.rs`): answers `initialize`, lists a single
    /// `tool`, and replies to `tools/call` per `call`.
    fn fake_script(tool: &str, call: CallMode, pid_file: Option<&Path>) -> String {
        let pid_line = match pid_file {
            Some(p) => format!("echo $$ > '{}'\n", p.display()),
            None => String::new(),
        };
        let call_action = match call {
            CallMode::Ok => {
                r#"printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"done"}]}}\n' "$id""#.to_string()
            }
            CallMode::Error => {
                r#"printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32000,"message":"boom"}}\n' "$id""#.to_string()
            }
            CallMode::Oversize => {
                r#"printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"' "$id"
      head -c 2000000 /dev/zero | tr '\0' 'x'
      printf '"}]}}\n'"#.to_string()
            }
        };
        const TEMPLATE: &str = r#"#!/bin/sh
@PID@while IFS= read -r line; do
  id=$(printf %s "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"0.0"}}}\n' "$id" ;;
    *'"method":"notifications/initialized"'*) : ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"@TOOL@","description":"d","inputSchema":{"type":"object"}}]}}\n' "$id" ;;
    *'"method":"resources/list"'*|*'"method":"prompts/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"method not found"}}\n' "$id" ;;
    *'"method":"tools/call"'*)
      @CALL@ ;;
  esac
done
"#;
        TEMPLATE
            .replace("@PID@", &pid_line)
            .replace("@TOOL@", tool)
            .replace("@CALL@", &call_action)
    }

    fn sh_config(name: &str, script: &Path, namespace: Option<&str>) -> McpServerConfig {
        McpServerConfig {
            name: name.into(),
            command: "/bin/sh".into(),
            args: vec![script.display().to_string()],
            namespace: namespace.map(Into::into),
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
            description: None,
        }
    }

    /// A config whose command does not exist, so `connect` fails to spawn.
    fn broken_config(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.into(),
            command: "/nonexistent-mcp-binary-xyzzy".into(),
            args: vec![],
            namespace: Some("x".into()),
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
            http: None,
            description: None,
        }
    }

    fn pid_running(pid: u32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => {
                let after_comm = stat.rsplit(')').next().unwrap_or("");
                !after_comm.trim_start().starts_with('Z')
            }
            Err(_) => false,
        }
    }

    #[test]
    fn cap_result_passes_small() {
        assert_eq!(cap_result("hi".into()), "hi");
    }

    #[test]
    fn cap_result_truncates_large() {
        let capped = cap_result("x".repeat(MAX_RESULT_BYTES + 100));
        assert!(capped.len() <= MAX_RESULT_BYTES);
        assert!(capped.ends_with("[truncated]"));
    }

    #[tokio::test]
    async fn registers_namespaced_union_no_collision() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.sh");
        let b = dir.path().join("b.sh");
        std::fs::write(&a, fake_script("echo", CallMode::Ok, None)).unwrap();
        std::fs::write(&b, fake_script("echo", CallMode::Ok, None)).unwrap();

        let host = McpHost::start(&[
            sh_config("srv-a", &a, Some("a")),
            sh_config("srv-b", &b, Some("b")),
        ])
        .await;

        // Same tool name in both servers, but namespacing keeps them distinct.
        let mut names: Vec<String> = host.registrations().into_iter().map(|r| r.name).collect();
        names.sort();
        assert_eq!(names, vec!["a__echo".to_string(), "b__echo".to_string()]);
        host.shutdown().await;
    }

    #[tokio::test]
    async fn tool_counts_keyed_by_namespace_with_name_fallback() {
        // Two healthy servers, each advertising one tool. One declares a
        // namespace ("a"); the other has none, so it keys by its config name
        // ("plain") — the SAME key the host uses when namespacing that server's
        // tools. adele-gtk#125.
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.sh");
        let b = dir.path().join("b.sh");
        std::fs::write(&a, fake_script("echo", CallMode::Ok, None)).unwrap();
        std::fs::write(&b, fake_script("echo", CallMode::Ok, None)).unwrap();

        let host = McpHost::start(&[
            sh_config("srv-a", &a, Some("a")),
            sh_config("plain", &b, None),
        ])
        .await;

        let counts = host.tool_counts();
        assert_eq!(counts.get("a").copied(), Some(1), "namespace-keyed count");
        assert_eq!(
            counts.get("plain").copied(),
            Some(1),
            "name-keyed count when no namespace"
        );
        assert_eq!(counts.len(), 2, "exactly the two hosted namespaces");
        host.shutdown().await;
    }

    #[tokio::test]
    async fn tool_counts_excludes_server_that_failed_to_start() {
        // A broken server hosts nothing, so it never appears in the counts; the
        // healthy one is still reported with its tool total.
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.sh");
        std::fs::write(&good, fake_script("echo", CallMode::Ok, None)).unwrap();

        let host =
            McpHost::start(&[broken_config("broken"), sh_config("good", &good, Some("g"))]).await;

        let counts = host.tool_counts();
        assert_eq!(counts.get("g").copied(), Some(1));
        assert!(
            !counts.contains_key("x"),
            "the broken server's namespace must not appear"
        );
        assert_eq!(counts.len(), 1);
        host.shutdown().await;
    }

    #[tokio::test]
    async fn routes_call_and_unknown_errors() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("s.sh");
        std::fs::write(&script, fake_script("echo", CallMode::Ok, None)).unwrap();

        let host = McpHost::start(&[sh_config("srv", &script, Some("ns"))]).await;
        assert!(host.handles("ns__echo"));
        assert_eq!(
            host.call("ns__echo", serde_json::json!({})).await.unwrap(),
            "done"
        );
        assert!(
            host.call("ns__missing", serde_json::json!({}))
                .await
                .is_err()
        );
        host.shutdown().await;
    }

    #[tokio::test]
    async fn bad_server_degrades() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.sh");
        std::fs::write(&good, fake_script("echo", CallMode::Ok, None)).unwrap();

        let host =
            McpHost::start(&[broken_config("broken"), sh_config("good", &good, Some("g"))]).await;

        // The broken server is skipped; the healthy one still works.
        let names: Vec<String> = host.registrations().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["g__echo".to_string()]);
        assert_eq!(
            host.call("g__echo", serde_json::json!({})).await.unwrap(),
            "done"
        );
        host.shutdown().await;
    }

    #[tokio::test]
    async fn transport_error_maps_to_err() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("err.sh");
        std::fs::write(&script, fake_script("echo", CallMode::Error, None)).unwrap();

        let host = McpHost::start(&[sh_config("srv", &script, Some("ns"))]).await;
        // Tool lists fine, but tools/call returns a JSON-RPC error -> Err.
        assert!(host.call("ns__echo", serde_json::json!({})).await.is_err());
        host.shutdown().await;
    }

    #[tokio::test]
    async fn oversize_result_capped() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("big.sh");
        std::fs::write(&script, fake_script("echo", CallMode::Oversize, None)).unwrap();

        let host = McpHost::start(&[sh_config("srv", &script, Some("ns"))]).await;
        let result = host.call("ns__echo", serde_json::json!({})).await.unwrap();
        assert!(
            result.len() <= MAX_RESULT_BYTES,
            "result was {} bytes",
            result.len()
        );
        host.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_kills_children() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("pid.sh");
        let pid_file = dir.path().join("pid.txt");
        std::fs::write(&script, fake_script("echo", CallMode::Ok, Some(&pid_file))).unwrap();

        let host = McpHost::start(&[sh_config("srv", &script, Some("ns"))]).await;
        let pid: u32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert!(pid_running(pid), "server should be alive before shutdown");

        host.shutdown().await;

        // Give the kernel a beat to reap the killed process.
        for _ in 0..40 {
            if !pid_running(pid) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            !pid_running(pid),
            "server process should be dead after shutdown"
        );
    }

    // ----- In-process built-in servers (da#538 Phase B) -----

    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    /// What a [`FakeService`]'s single tool does when called.
    enum BuiltinBehavior {
        /// Succeed with a plain-text reply carrying this body.
        Text(String),
        /// A *tool-level* failure (`CallError::Tool`) — the subprocess-parity
        /// `isError` path, which must surface as `Ok(text)`.
        ToolError(String),
        /// A *protocol-level* fault (`CallError::Internal`) — must surface as `Err`.
        ProtocolError(String),
        /// A reply whose only block is Raw JSON (plus `structuredContent`).
        Raw(serde_json::Value),
        /// A text reply larger than [`MAX_RESULT_BYTES`].
        Oversize,
    }

    /// A configurable in-process [`mcp_core::McpService`] for host tests:
    /// advertises one tool and answers per `behavior`, and records whether its
    /// `shutdown` hook ran.
    struct FakeService {
        tool: String,
        behavior: BuiltinBehavior,
        shutdown_called: Arc<AtomicBool>,
    }

    impl FakeService {
        fn new(tool: &str, behavior: BuiltinBehavior) -> Self {
            Self {
                tool: tool.into(),
                behavior,
                shutdown_called: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    #[mcp_core::async_trait]
    impl mcp_core::McpService for FakeService {
        fn tools(&self) -> Vec<mcp_core::ToolDef> {
            vec![mcp_core::ToolDef::new(
                self.tool.clone(),
                "a fake in-process tool",
                serde_json::json!({ "type": "object" }),
            )]
        }

        async fn call_tool(
            &self,
            name: &str,
            _arguments: &serde_json::Value,
        ) -> Result<mcp_core::ToolReply, mcp_core::CallError> {
            assert_eq!(
                name, self.tool,
                "host must route the ORIGINAL (un-namespaced) tool name to the service"
            );
            match &self.behavior {
                BuiltinBehavior::Text(s) => Ok(mcp_core::ToolReply::text(s.clone())),
                BuiltinBehavior::ToolError(m) => Err(mcp_core::CallError::tool(m.clone())),
                BuiltinBehavior::ProtocolError(m) => Err(mcp_core::CallError::internal(m.clone())),
                BuiltinBehavior::Raw(v) => Ok(mcp_core::ToolReply {
                    content: vec![mcp_core::Content::Raw(v.clone())],
                    is_error: false,
                    structured_content: Some(v.clone()),
                    tools_list_changed: false,
                }),
                BuiltinBehavior::Oversize => Ok(mcp_core::ToolReply::text(
                    "x".repeat(MAX_RESULT_BYTES + 1024),
                )),
            }
        }

        async fn shutdown(&self) {
            self.shutdown_called.store(true, AtomicOrdering::SeqCst);
        }
    }

    /// Build a [`BuiltinServer`] wrapping a fresh [`FakeService`], returning the
    /// shared shutdown flag so a test can assert the hook ran.
    fn builtin(
        name: &str,
        namespace: &str,
        tool: &str,
        behavior: BuiltinBehavior,
    ) -> (BuiltinServer, Arc<AtomicBool>) {
        let svc = Arc::new(FakeService::new(tool, behavior));
        let flag = svc.shutdown_called.clone();
        (BuiltinServer::new(name, namespace, svc), flag)
    }

    #[tokio::test]
    async fn builtin_registers_namespaced_tools() {
        let (b, _flag) = builtin(
            "dev-server",
            "dev",
            "echo",
            BuiltinBehavior::Text("hi".into()),
        );
        let host = McpHost::start_with(&[], vec![b]).await;

        let names: Vec<String> = host.registrations().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["dev__echo".to_string()]);
        assert!(host.handles("dev__echo"));
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_call_returns_text() {
        let (b, _f) = builtin(
            "dev-server",
            "dev",
            "echo",
            BuiltinBehavior::Text("hello world".into()),
        );
        let host = McpHost::start_with(&[], vec![b]).await;
        assert_eq!(
            host.call("dev__echo", serde_json::json!({})).await.unwrap(),
            "hello world"
        );
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_tool_error_surfaced_as_ok_text() {
        // Parity with a subprocess `isError`: a tool-level failure comes back as
        // its text in `Ok`, so the LLM sees the error content as the result.
        let (b, _f) = builtin(
            "dev-server",
            "dev",
            "echo",
            BuiltinBehavior::ToolError("nope".into()),
        );
        let host = McpHost::start_with(&[], vec![b]).await;
        assert_eq!(
            host.call("dev__echo", serde_json::json!({})).await.unwrap(),
            "nope"
        );
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_protocol_error_maps_to_err() {
        // A protocol-level fault (`Internal`/`InvalidParams`) is a transport
        // failure, mapped to `Err` just like a subprocess JSON-RPC error.
        let (b, _f) = builtin(
            "dev-server",
            "dev",
            "echo",
            BuiltinBehavior::ProtocolError("kaboom".into()),
        );
        let host = McpHost::start_with(&[], vec![b]).await;
        let err = host
            .call("dev__echo", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err.contains("dev__echo"), "got: {err}");
        assert!(
            err.contains("kaboom"),
            "protocol error text should surface, got: {err}"
        );
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_raw_and_structured_rendering() {
        // A Raw JSON content block renders as its pretty-printed value, mirroring
        // `McpClient`'s extraction of non-text content blocks.
        let v = serde_json::json!({ "answer": 42, "nested": { "a": [1, 2, 3] } });
        let (b, _f) = builtin("dev-server", "dev", "calc", BuiltinBehavior::Raw(v.clone()));
        let host = McpHost::start_with(&[], vec![b]).await;
        let result = host.call("dev__calc", serde_json::json!({})).await.unwrap();
        assert_eq!(result, serde_json::to_string_pretty(&v).unwrap());
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_and_subprocess_coexist() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("s.sh");
        std::fs::write(&script, fake_script("echo", CallMode::Ok, None)).unwrap();
        let (b, _f) = builtin(
            "dev-server",
            "dev",
            "run",
            BuiltinBehavior::Text("built-in-ok".into()),
        );

        let host = McpHost::start_with(&[sh_config("srv", &script, Some("ns"))], vec![b]).await;

        // Both servers' tools are registered under their own namespaces.
        let mut names: Vec<String> = host.registrations().into_iter().map(|r| r.name).collect();
        names.sort();
        assert_eq!(names, vec!["dev__run".to_string(), "ns__echo".to_string()]);

        // Each call routes to its own backend.
        assert_eq!(
            host.call("ns__echo", serde_json::json!({})).await.unwrap(),
            "done"
        );
        assert_eq!(
            host.call("dev__run", serde_json::json!({})).await.unwrap(),
            "built-in-ok"
        );

        // tool_counts keys both namespaces, backend-agnostically.
        let counts = host.tool_counts();
        assert_eq!(counts.get("ns").copied(), Some(1), "subprocess namespace");
        assert_eq!(counts.get("dev").copied(), Some(1), "builtin namespace");
        assert_eq!(counts.len(), 2);
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_tool_counts_keyed_by_namespace() {
        let (b, _f) = builtin(
            "dev-server",
            "dev",
            "echo",
            BuiltinBehavior::Text("hi".into()),
        );
        let host = McpHost::start_with(&[], vec![b]).await;
        let counts = host.tool_counts();
        assert_eq!(counts.get("dev").copied(), Some(1));
        assert_eq!(counts.len(), 1);
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_oversize_result_capped() {
        let (b, _f) = builtin("dev-server", "dev", "big", BuiltinBehavior::Oversize);
        let host = McpHost::start_with(&[], vec![b]).await;
        let result = host.call("dev__big", serde_json::json!({})).await.unwrap();
        assert!(
            result.len() <= MAX_RESULT_BYTES,
            "result was {} bytes",
            result.len()
        );
        assert!(result.ends_with("[truncated]"));
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_shutdown_calls_hook() {
        let (b, flag) = builtin(
            "dev-server",
            "dev",
            "echo",
            BuiltinBehavior::Text("hi".into()),
        );
        let host = McpHost::start_with(&[], vec![b]).await;
        assert!(
            !flag.load(AtomicOrdering::SeqCst),
            "shutdown hook must not run before shutdown()"
        );
        host.shutdown().await;
        assert!(
            flag.load(AtomicOrdering::SeqCst),
            "shutdown() must invoke the in-process service's shutdown hook"
        );
    }

    #[tokio::test]
    async fn builtin_name_collision_with_subprocess_skipped() {
        // The subprocess registers `ns__echo` first; a builtin using the same
        // namespace + tool collides and is skipped by the existing dedup.
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("s.sh");
        std::fs::write(&script, fake_script("echo", CallMode::Ok, None)).unwrap();
        let (b, _f) = builtin(
            "dupe",
            "ns",
            "echo",
            BuiltinBehavior::Text("should-not-win".into()),
        );

        let host = McpHost::start_with(&[sh_config("srv", &script, Some("ns"))], vec![b]).await;

        // Exactly one `ns__echo`, routing to the subprocess (registered first).
        let names: Vec<String> = host.registrations().into_iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            vec!["ns__echo".to_string()],
            "collision must be skipped"
        );
        assert_eq!(
            host.call("ns__echo", serde_json::json!({})).await.unwrap(),
            "done"
        );
        host.shutdown().await;
    }

    #[tokio::test]
    async fn start_still_works_without_builtins() {
        // `start` delegates to `start_with(.., vec![])` and must behave as before.
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("s.sh");
        std::fs::write(&script, fake_script("echo", CallMode::Ok, None)).unwrap();

        let host = McpHost::start(&[sh_config("srv", &script, Some("ns"))]).await;
        let names: Vec<String> = host.registrations().into_iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["ns__echo".to_string()]);
        assert_eq!(
            host.call("ns__echo", serde_json::json!({})).await.unwrap(),
            "done"
        );
        host.shutdown().await;
    }

    // ----- Centralized built-in override + status (da#538 Phase D slice 2) -----

    #[tokio::test]
    async fn builtin_overridden_by_same_name_config_is_not_hosted() {
        // A configured client-mcp server named "fileio" shadows a built-in of the
        // same NAME: `start_with` hosts the configured one and leaves the built-in
        // dormant. Distinct namespaces ("cfg" vs "bi") ensure this exercises the
        // name-based override, not the namespaced-tool dedup — without the override
        // both tools would register.
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("fileio.sh");
        std::fs::write(&script, fake_script("read", CallMode::Ok, None)).unwrap();
        let (b, _f) = builtin(
            "fileio",
            "bi",
            "write",
            BuiltinBehavior::Text("built-in-should-not-run".into()),
        );

        let host = McpHost::start_with(&[sh_config("fileio", &script, Some("cfg"))], vec![b]).await;

        // Only the configured server's tool is hosted; the built-in's is not.
        let names: Vec<String> = host.registrations().into_iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            vec!["cfg__read".to_string()],
            "the configured server wins; the built-in's tool must be absent"
        );
        assert!(
            !host.handles("bi__write"),
            "the overridden built-in's tool must not be routed"
        );

        // The built-in is still reported, flagged as overridden by name.
        let status = host.builtin_status();
        let entry = status
            .iter()
            .find(|s| s.name == "fileio")
            .expect("built-in 'fileio' must appear in builtin_status");
        assert_eq!(entry.overridden_by, Some("fileio".to_string()));
        assert_eq!(entry.namespace, "bi");
        assert_eq!(
            entry.tool_count, 1,
            "reports the built-in's own advertised tool count"
        );
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_not_overridden_is_hosted_and_status_clean() {
        // No configured server named "tasks", so the built-in is hosted normally
        // and its status carries no override.
        let (b, _f) = builtin("tasks", "tasks", "list", BuiltinBehavior::Text("ok".into()));
        let host = McpHost::start_with(&[], vec![b]).await;

        assert!(host.handles("tasks__list"), "built-in must be hosted");

        let status = host.builtin_status();
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].name, "tasks");
        assert_eq!(status[0].namespace, "tasks");
        assert_eq!(status[0].tool_count, 1, "one registered tool");
        assert_eq!(status[0].overridden_by, None);
        host.shutdown().await;
    }

    #[tokio::test]
    async fn builtin_status_reports_all_passed_builtins() {
        // One overridden built-in + one active built-in: both appear in the
        // status, each with the correct override flag.
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("fileio.sh");
        std::fs::write(&script, fake_script("read", CallMode::Ok, None)).unwrap();
        let (overridden, _f1) = builtin("fileio", "bi", "write", BuiltinBehavior::Text("nope".into()));
        let (active, _f2) = builtin("tasks", "tasks", "list", BuiltinBehavior::Text("ok".into()));

        let host = McpHost::start_with(
            &[sh_config("fileio", &script, Some("cfg"))],
            vec![overridden, active],
        )
        .await;

        let status = host.builtin_status();
        assert_eq!(status.len(), 2, "both built-ins reported");
        let fileio = status
            .iter()
            .find(|s| s.name == "fileio")
            .expect("overridden built-in present");
        let tasks = status
            .iter()
            .find(|s| s.name == "tasks")
            .expect("active built-in present");
        assert_eq!(fileio.overridden_by, Some("fileio".to_string()));
        assert_eq!(tasks.overridden_by, None);
        assert_eq!(tasks.tool_count, 1);
        host.shutdown().await;
    }
}
