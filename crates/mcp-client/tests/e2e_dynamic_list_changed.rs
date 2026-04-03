use desktop_assistant_core::ports::tools::ToolExecutor;
use desktop_assistant_mcp_client::executor::{McpServerConfig, McpToolExecutor};
use tokio::time::{Duration, sleep};

fn python3_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn dynamic_mcp_server_script() -> String {
    r#"
import json
import sys
import threading
import time

state = {"version": 0, "changed_emitted": False}


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        req = json.loads(line)
    except Exception:
        continue

    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"listChanged": True},
                    "prompts": {"listChanged": True}
                }
            }
        })
        continue

    if method == "notifications/initialized" and not state["changed_emitted"]:
        state["changed_emitted"] = True

        def emit_changes():
            time.sleep(0.05)
            state["version"] = 1
            send({"jsonrpc": "2.0", "method": "notifications/resources/list_changed"})
            send({"jsonrpc": "2.0", "method": "notifications/prompts/list_changed"})

        threading.Thread(target=emit_changes, daemon=True).start()
        continue

    if method == "tools/list":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "ping",
                        "description": "Ping",
                        "inputSchema": {"type": "object"}
                    }
                ]
            }
        })
        continue

    if method == "resources/list":
        uri = "res://v1" if state["version"] == 1 else "res://v0"
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "resources": [{"uri": uri, "name": "resource"}]
            }
        })
        continue

    if method == "prompts/list":
        name = "prompt-v1" if state["version"] == 1 else "prompt-v0"
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "prompts": [{"name": name}]
            }
        })
        continue

    if method == "tools/call":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": "pong"}]
            }
        })
        continue

    send({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"}
    })
"#
    .to_string()
}

fn first_uri(items: &[serde_json::Value]) -> Option<String> {
    items.first()?.get("uri")?.as_str().map(ToString::to_string)
}

fn first_prompt_name(items: &[serde_json::Value]) -> Option<String> {
    items
        .first()?
        .get("name")?
        .as_str()
        .map(ToString::to_string)
}

#[tokio::test]
async fn executor_refreshes_resources_and_prompts_after_live_list_changed() {
    if !python3_available() {
        eprintln!("SKIP: python3 not found");
        return;
    }

    let script = dynamic_mcp_server_script();
    let configs = vec![McpServerConfig {
        name: "dynamic-mock".into(),
        command: "python3".into(),
        args: vec!["-u".into(), "-c".into(), script],
        namespace: None,
        enabled: true,
        env: std::collections::HashMap::new(),
    }];

    let executor = McpToolExecutor::new(configs);
    executor.start().await.expect("failed to start executor");

    let initial_resources = executor.available_resources().await;
    let initial_prompts = executor.available_prompts().await;
    assert_eq!(first_uri(&initial_resources).as_deref(), Some("res://v0"));
    assert_eq!(
        first_prompt_name(&initial_prompts).as_deref(),
        Some("prompt-v0")
    );

    let mut updated = false;
    for _ in 0..30 {
        sleep(Duration::from_millis(20)).await;

        let _ = executor
            .execute_tool("ping", serde_json::json!({}))
            .await
            .expect("ping tool should succeed");

        let resources = executor.available_resources().await;
        let prompts = executor.available_prompts().await;

        if first_uri(&resources).as_deref() == Some("res://v1")
            && first_prompt_name(&prompts).as_deref() == Some("prompt-v1")
        {
            updated = true;
            break;
        }
    }

    assert!(
        updated,
        "expected resources/prompts caches to refresh to v1 after list_changed notifications"
    );

    executor.shutdown().await;
}
