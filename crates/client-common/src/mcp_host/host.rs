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
use std::collections::HashMap;
use tokio::sync::Mutex;

use super::config::McpServerConfig;

/// Cap on a client-tool result field. The daemon rejects anything larger
/// (`MAX_CLIENT_TOOL_RESULT_BYTES`), which would fail the turn — so we truncate
/// to keep the result usable and the turn alive.
const MAX_RESULT_BYTES: usize = 1024 * 1024;

/// Separator between a server's namespace and a tool name (`{ns}__{tool}`),
/// matching the daemon's own MCP tool-namespacing convention.
const NAMESPACE_SEP: &str = "__";

/// A running local MCP server. `list_tools`/`call_tool` need `&mut McpClient`,
/// so each server sits behind its own mutex (a slow call on one server never
/// blocks another).
struct HostedServer {
    /// Config name, for diagnostics.
    name: String,
    client: Mutex<McpClient>,
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
    /// Start the given servers: spawn each, list its tools, and build the
    /// registration set + routing table.
    ///
    /// Degrades: a server that fails to start (or list its tools) is logged and
    /// skipped — the host still serves every healthy server. Never panics on a
    /// bad server.
    pub async fn start(servers: &[McpServerConfig]) -> Self {
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
                client: Mutex::new(client),
            });
            tracing::info!(
                "hosting client MCP server '{}' as namespace '{}'{}",
                cfg.name,
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

    /// Whether `tool_name` is one this host serves (used to decide whether a
    /// `ClientToolCall` should be routed here).
    pub fn handles(&self, tool_name: &str) -> bool {
        self.routes.contains_key(tool_name)
    }

    /// Invoke a hosted tool by its namespaced name, routing to the owning
    /// server.
    ///
    /// `Err` on an unknown tool or a transport/JSON-RPC failure. An MCP
    /// tool-level error (`isError`) comes back from `McpClient` as its text in
    /// `Ok` — the LLM sees the error content as the result, which is the
    /// intended behavior. Results are capped to the daemon's client-tool limit.
    pub async fn call(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, String> {
        let Some((index, original)) = self.routes.get(tool_name) else {
            return Err(format!("unknown client MCP tool: {tool_name}"));
        };
        let mut client = self.servers[*index].client.lock().await;
        match client.call_tool(original, arguments).await {
            Ok(result) => Ok(cap_result(result)),
            Err(err) => Err(format!("client MCP tool '{tool_name}' failed: {err}")),
        }
    }

    /// Shut down every hosted server process.
    pub async fn shutdown(self) {
        for server in self.servers {
            let mut client = server.client.into_inner();
            client.shutdown().await;
            tracing::debug!("shut down client MCP server '{}'", server.name);
        }
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
}
