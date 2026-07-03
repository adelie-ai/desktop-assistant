//! Glue between the [`McpHost`] and the daemon's client-tool wire path, so
//! wiring a client (phases 4-6) is a few lines: merge the host's tools with the
//! client's own built-in client tools into the single set the daemon expects,
//! and route an incoming `ClientToolCall` to the host.

use std::collections::HashSet;

use async_trait::async_trait;
use desktop_assistant_api_model::ClientToolRegistration;

use super::host::McpHost;
use crate::Connector;

/// Merge a client's built-in client-tool registrations with the MCP host's
/// tools into the single set to hand [`Connector::register_client_tools`] — the
/// daemon *replaces* the whole set per call, so hosted tools must be registered
/// together with the built-ins. Built-ins win on a name clash.
pub fn merge_registrations(
    builtins: Vec<ClientToolRegistration>,
    host_tools: Vec<ClientToolRegistration>,
) -> Vec<ClientToolRegistration> {
    let mut seen: HashSet<String> = builtins.iter().map(|t| t.name.clone()).collect();
    let mut merged = builtins;
    for tool in host_tools {
        if seen.insert(tool.name.clone()) {
            merged.push(tool);
        } else {
            tracing::warn!(
                "client MCP tool '{}' shadows a built-in client tool of the same name; \
                 keeping the built-in",
                tool.name
            );
        }
    }
    merged
}

/// Where a client-tool result is delivered. Implemented for [`Connector`]; the
/// trait exists so [`dispatch_client_tool_call`] is testable without a live
/// daemon connection.
#[async_trait]
pub trait ClientToolResultSink {
    async fn submit_result(&self, task_id: &str, tool_call_id: &str, result: Result<String, String>);
}

#[async_trait]
impl ClientToolResultSink for Connector {
    async fn submit_result(&self, task_id: &str, tool_call_id: &str, result: Result<String, String>) {
        if let Err(err) = self.submit_client_tool_result(task_id, tool_call_id, result).await {
            tracing::warn!("failed to submit client tool result (task {task_id}): {err}");
        }
    }
}

/// If `tool_name` is one the MCP host serves, invoke it and submit the outcome
/// via `sink`, returning `true`. Otherwise return `false` so the caller's own
/// client-tool dispatch handles it.
///
/// On a host tool it **always** submits — even on error — so the daemon's parked
/// turn resumes instead of waiting out the 120 s client-tool timeout.
pub async fn dispatch_client_tool_call(
    host: &McpHost,
    sink: &impl ClientToolResultSink,
    task_id: &str,
    tool_call_id: &str,
    tool_name: &str,
    arguments: serde_json::Value,
) -> bool {
    if !host.handles(tool_name) {
        return false;
    }
    let result = host.call(tool_name, arguments).await;
    sink.submit_result(task_id, tool_call_id, result).await;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::McpServerConfig;
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Mutex;

    fn reg(name: &str) -> ClientToolRegistration {
        ClientToolRegistration {
            name: name.into(),
            description: String::new(),
            input_schema: serde_json::json!({ "type": "object" }),
        }
    }

    /// Minimal `/bin/sh` fake MCP server (one `echo` tool). The fuller variant
    /// with pid/oversize modes lives in the `mcp_host::host` tests.
    fn fake_server(dir: &Path, error: bool) -> McpServerConfig {
        let script = dir.join(if error { "err.sh" } else { "ok.sh" });
        let call = if error {
            r#"printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32000,"message":"boom"}}\n' "$id""#
        } else {
            r#"printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"ok"}]}}\n' "$id""#
        };
        const TEMPLATE: &str = r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf %s "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  case "$line" in
    *'"method":"initialize"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"f","version":"0"}}}\n' "$id" ;;
    *'"method":"notifications/initialized"'*) : ;;
    *'"method":"tools/list"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"d","inputSchema":{"type":"object"}}]}}\n' "$id" ;;
    *'"method":"tools/call"'*) @CALL@ ;;
  esac
done
"#;
        std::fs::write(&script, TEMPLATE.replace("@CALL@", call)).unwrap();
        McpServerConfig {
            name: "s".into(),
            command: "/bin/sh".into(),
            args: vec![script.display().to_string()],
            namespace: Some("ns".into()),
            enabled: true,
            env: HashMap::new(),
            env_secrets: HashMap::new(),
        }
    }

    #[derive(Default)]
    struct FakeSink {
        calls: Mutex<Vec<(String, Result<String, String>)>>,
    }

    #[async_trait]
    impl ClientToolResultSink for FakeSink {
        async fn submit_result(&self, _task_id: &str, tool_call_id: &str, result: Result<String, String>) {
            self.calls.lock().unwrap().push((tool_call_id.into(), result));
        }
    }

    #[test]
    fn merge_includes_builtins_and_host() {
        let merged = merge_registrations(vec![reg("say_this")], vec![reg("ns__echo")]);
        let names: HashSet<&str> = merged.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains("say_this"));
        assert!(names.contains("ns__echo"));
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn merge_builtin_wins_on_clash() {
        let merged = merge_registrations(vec![reg("dup")], vec![reg("dup"), reg("ns__x")]);
        // The clashing host tool is dropped; the built-in survives once.
        assert_eq!(merged.iter().filter(|r| r.name == "dup").count(), 1);
        assert_eq!(merged.len(), 2);
    }

    #[tokio::test]
    async fn dispatch_ignores_foreign_tool() {
        let host = McpHost::start(&[]).await; // empty host serves nothing
        let sink = FakeSink::default();
        let handled = dispatch_client_tool_call(&host, &sink, "t", "c", "say_this", serde_json::json!({})).await;
        assert!(!handled);
        assert!(sink.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn dispatch_handles_host_tool() {
        let dir = tempfile::tempdir().unwrap();
        let host = McpHost::start(&[fake_server(dir.path(), false)]).await;
        let sink = FakeSink::default();
        let handled = dispatch_client_tool_call(&host, &sink, "t", "call-1", "ns__echo", serde_json::json!({})).await;
        host.shutdown().await;
        assert!(handled);
        // Lock only after all awaits so no guard is held across one.
        let calls = sink.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "call-1");
        assert_eq!(calls[0].1.as_deref(), Ok("ok"));
    }

    #[tokio::test]
    async fn dispatch_submits_on_host_error() {
        let dir = tempfile::tempdir().unwrap();
        let host = McpHost::start(&[fake_server(dir.path(), true)]).await;
        let sink = FakeSink::default();
        let handled = dispatch_client_tool_call(&host, &sink, "t", "call-2", "ns__echo", serde_json::json!({})).await;
        host.shutdown().await;
        assert!(handled, "a host tool is always handled, even on error");
        let calls = sink.calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "must submit so the parked turn resumes");
        assert!(calls[0].1.is_err());
    }
}
