//! Robustness tests for the MCP client / executor (review findings DS-1..DS-4).
//!
//! These tests drive the real `McpClient` against small `/bin/sh` fake MCP
//! servers, so they exercise the actual stdio transport:
//!
//! - DS-1: a slow/hung tool call on server A must not block tool calls on
//!   server B (the executor previously serialized ALL servers behind one
//!   global mutex).
//! - DS-2: the spawned server process must die when the client is dropped
//!   without an explicit `shutdown` (panic / cancelled task / failed connect).
//! - DS-3: a server that never replies must produce a timeout error, not a
//!   forever-hung turn.
//! - DS-4: an absurdly large response line must produce an error instead of
//!   buffering unbounded memory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use desktop_assistant_mcp_client::executor::{McpServerConfig, McpToolExecutor};
use desktop_assistant_mcp_client::{McpClient, McpError};
use desktop_assistant_core::ports::tools::ToolExecutor;

/// Unique temp file path for this test process.
fn temp_path(label: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "mcp-robustness-{}-{}-{}",
        std::process::id(),
        n,
        label
    ))
}

/// Behaviour knobs for the fake `/bin/sh` MCP server.
struct FakeServer {
    /// Write the shell PID to this file at startup.
    pid_file: Option<PathBuf>,
    /// Reply to `initialize` (a server that stays silent here simulates a
    /// wedged handshake).
    reply_initialize: bool,
    /// Shell commands to run before answering a `tools/call` (e.g. `sleep 3`).
    call_prelude: String,
    /// Reply mode for `tools/call`: `Some(tag)` replies with `done-<tag>`,
    /// `None` stays silent forever.
    call_reply_tag: Option<String>,
    /// When true, the `tools/call` reply is a single ~9 MB line.
    oversize_call_reply: bool,
}

impl Default for FakeServer {
    fn default() -> Self {
        Self {
            pid_file: None,
            reply_initialize: true,
            call_prelude: String::new(),
            call_reply_tag: Some("ok".into()),
            oversize_call_reply: false,
        }
    }
}

impl FakeServer {
    /// Render the server as a shell script and write it to a temp file.
    /// Returns the script path (pass as the single argument to `/bin/sh`).
    fn write(&self, label: &str) -> PathBuf {
        let pid_line = match &self.pid_file {
            Some(p) => format!("echo $$ > '{}'\n", p.display()),
            None => String::new(),
        };
        let init_action = if self.reply_initialize {
            r#"printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"0.0"}}}\n' "$id""#
        } else {
            ":"
        };
        let call_action = if self.oversize_call_reply {
            // One ~9 MB JSON line (no newline until the very end).
            r#"printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"' "$id"
      head -c 9000000 /dev/zero | tr '\0' 'x'
      printf '"}]}}\n'"#
                .to_string()
        } else {
            match &self.call_reply_tag {
                Some(tag) => format!(
                    r#"{prelude}
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"content":[{{"type":"text","text":"done-{tag}"}}]}}}}\n' "$id""#,
                    prelude = self.call_prelude,
                    tag = tag
                ),
                None => ":".to_string(),
            }
        };

        let script = format!(
            r#"#!/bin/sh
{pid_line}while IFS= read -r line; do
  id=$(printf %s "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  case "$line" in
    *'"method":"initialize"'*)
      {init_action}
      ;;
    *'"method":"notifications/initialized"'*)
      :
      ;;
    *'"method":"tools/list"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"tools":[{{"name":"echo","description":"echo tool","inputSchema":{{"type":"object"}}}}]}}}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      {call_action}
      ;;
  esac
done
"#
        );

        let path = temp_path(&format!("{label}.sh"));
        std::fs::write(&path, script).expect("write fake server script");
        path
    }
}

fn pid_from_file(path: &Path) -> u32 {
    let raw = std::fs::read_to_string(path).expect("read pid file");
    raw.trim().parse().expect("parse pid")
}

/// True when the process exists and is not a zombie.
fn pid_running(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => {
            // Field 3 (after the parenthesised comm) is the state.
            let after_comm = stat.rsplit(')').next().unwrap_or("");
            !after_comm.trim_start().starts_with('Z')
        }
        Err(_) => false,
    }
}

async fn wait_for_pid_death(pid: u32, within: Duration) -> bool {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if !pid_running(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    !pid_running(pid)
}

fn sh_config(name: &str, script: &Path, namespace: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.into(),
        command: "/bin/sh".into(),
        args: vec![script.display().to_string()],
        namespace: Some(namespace.into()),
        enabled: true,
        env: HashMap::new(),
        env_secrets: HashMap::new(),
    }
}

// --- DS-3: request timeout ------------------------------------------------

/// A server that completes the handshake but never answers a tool call must
/// yield a timeout error instead of hanging the turn forever.
#[tokio::test]
async fn silent_tool_call_times_out_instead_of_hanging() {
    let script = FakeServer {
        call_reply_tag: None,
        ..Default::default()
    }
    .write("silent-call");

    let mut client = McpClient::connect_with_request_timeout(
        "/bin/sh",
        &[script.display().to_string()],
        &HashMap::new(),
        Duration::from_millis(500),
    )
    .await
    .expect("handshake should succeed");

    let result = tokio::time::timeout(
        Duration::from_secs(10),
        client.call_tool("echo", serde_json::json!({})),
    )
    .await
    .expect("call_tool must not hang past its own timeout");

    match result {
        Err(McpError::Timeout { .. }) => {}
        other => panic!("expected McpError::Timeout, got {other:?}"),
    }
    let _ = std::fs::remove_file(&script);
}

/// A server that never answers `initialize` must fail `connect` with a
/// timeout — and the spawned child must be killed, not leaked (DS-2 + DS-3).
#[tokio::test]
async fn silent_initialize_times_out_and_kills_child() {
    let pid_file = temp_path("silent-init.pid");
    let script = FakeServer {
        pid_file: Some(pid_file.clone()),
        reply_initialize: false,
        ..Default::default()
    }
    .write("silent-init");

    let started = Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        McpClient::connect_with_request_timeout(
            "/bin/sh",
            &[script.display().to_string()],
            &HashMap::new(),
            Duration::from_millis(500),
        ),
    )
    .await
    .expect("connect must not hang");
    assert!(
        matches!(result, Err(McpError::Timeout { .. })),
        "expected timeout error, got {result:?}",
    );
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "connect should fail promptly"
    );

    let pid = pid_from_file(&pid_file);
    assert!(
        wait_for_pid_death(pid, Duration::from_secs(3)).await,
        "child process {pid} should be killed after failed connect"
    );
    let _ = std::fs::remove_file(&script);
    let _ = std::fs::remove_file(&pid_file);
}

// --- DS-2: kill on drop ----------------------------------------------------

/// Dropping a connected client without calling `shutdown` (panic, cancelled
/// task) must still kill the child process.
#[tokio::test]
async fn dropped_client_kills_child_process() {
    let pid_file = temp_path("drop.pid");
    let script = FakeServer {
        pid_file: Some(pid_file.clone()),
        ..Default::default()
    }
    .write("drop");

    let client = McpClient::connect(
        "/bin/sh",
        &[script.display().to_string()],
        &HashMap::new(),
    )
    .await
    .expect("connect");

    let pid = pid_from_file(&pid_file);
    assert!(pid_running(pid), "server should be alive while connected");

    drop(client);

    assert!(
        wait_for_pid_death(pid, Duration::from_secs(3)).await,
        "child process {pid} should be killed when the client is dropped"
    );
    let _ = std::fs::remove_file(&script);
    let _ = std::fs::remove_file(&pid_file);
}

// --- DS-4: bounded response lines -------------------------------------------

/// A response line larger than the cap must produce an error rather than
/// buffering it all (and certainly must not hang).
#[tokio::test]
async fn oversized_response_line_is_an_error() {
    let script = FakeServer {
        oversize_call_reply: true,
        ..Default::default()
    }
    .write("oversize");

    let mut client = McpClient::connect(
        "/bin/sh",
        &[script.display().to_string()],
        &HashMap::new(),
    )
    .await
    .expect("connect");

    let result = tokio::time::timeout(
        Duration::from_secs(30),
        client.call_tool("echo", serde_json::json!({})),
    )
    .await
    .expect("oversized reply must not hang");

    match result {
        Err(McpError::UnexpectedResponse(msg)) => {
            assert!(
                msg.contains("exceed"),
                "error should mention the size cap, got: {msg}"
            );
        }
        other => panic!("expected UnexpectedResponse for oversize line, got {other:?}"),
    }
    let _ = std::fs::remove_file(&script);
}

// --- DS-1: per-server locking ------------------------------------------------

/// A slow tool call on server A must not block a tool call on server B.
#[tokio::test]
async fn slow_server_does_not_block_fast_server() {
    let slow_script = FakeServer {
        call_prelude: "sleep 3".into(),
        call_reply_tag: Some("slow".into()),
        ..Default::default()
    }
    .write("slow");
    let fast_script = FakeServer {
        call_reply_tag: Some("fast".into()),
        ..Default::default()
    }
    .write("fast");

    let executor = std::sync::Arc::new(McpToolExecutor::new(vec![
        sh_config("slow", &slow_script, "slow"),
        sh_config("fast", &fast_script, "fast"),
    ]));
    executor.start().await.expect("executor start");

    // Kick off the slow call; it holds server "slow" busy for ~3s.
    let slow_exec = std::sync::Arc::clone(&executor);
    let slow_task = tokio::spawn(async move {
        slow_exec
            .execute_tool("slow__echo", serde_json::json!({}))
            .await
    });
    // Give the slow call time to enter the server.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // The fast server must answer while the slow call is still in flight.
    let started = Instant::now();
    let fast = tokio::time::timeout(
        Duration::from_secs(10),
        executor.execute_tool("fast__echo", serde_json::json!({})),
    )
    .await
    .expect("fast call must not hang")
    .expect("fast call should succeed");
    let elapsed = started.elapsed();

    assert!(fast.contains("done-fast"), "unexpected fast reply: {fast}");
    assert!(
        elapsed < Duration::from_millis(1500),
        "fast server stalled behind slow server: took {elapsed:?}"
    );

    let slow = slow_task.await.expect("join").expect("slow call succeeds");
    assert!(slow.contains("done-slow"), "unexpected slow reply: {slow}");

    executor.shutdown().await;
    let _ = std::fs::remove_file(&slow_script);
    let _ = std::fs::remove_file(&fast_script);
}

// --- Harness sanity ----------------------------------------------------------

/// Happy-path round-trip through the sh fake server, anchoring the harness.
#[tokio::test]
async fn call_tool_roundtrip() {
    let script = FakeServer::default().write("roundtrip");

    let mut client = McpClient::connect(
        "/bin/sh",
        &[script.display().to_string()],
        &HashMap::new(),
    )
    .await
    .expect("connect");

    let tools = client.list_tools().await.expect("list_tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");

    let reply = client
        .call_tool("echo", serde_json::json!({}))
        .await
        .expect("call_tool");
    assert!(reply.contains("done-ok"), "unexpected reply: {reply}");

    client.shutdown().await;
    let _ = std::fs::remove_file(&script);
}
