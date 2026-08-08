//! The trace context this client puts on an outbound MCP call.
//!
//! An operator can see that the daemon spent forty seconds in a tool call and
//! cannot see what the MCP server did during it. Joining the two needs the
//! caller's trace context to reach the server, and neither MCP transport gets
//! that for free:
//!
//! - **stdio** is JSON-RPC over a pipe with no headers, so the context rides
//!   the MCP spec's reserved `_meta` property on `params`.
//! - **Streamable HTTP** has real headers, so it uses `traceparent`, which is
//!   what a server nobody here owns understands.
//!
//! Both tests drive the real client against a real server - a `/bin/sh` script
//! that echoes back the request line it read, and an `httpmock` endpoint that
//! matches on the header - so what is asserted is what went on the wire.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use desktop_assistant_core::ports::turn_telemetry::{resolve_turn_trace, with_turn_trace};
use desktop_assistant_mcp_client::McpClient;
use httpmock::prelude::*;
use serde_json::Value;

/// The turn's correlation id, which is also its trace id.
const REQUEST_ID: &str = "11111111-2222-4333-8444-555555555555";

/// The same value as a W3C trace id: the same sixteen bytes, without the
/// hyphens a uuid spells them with.
const TRACE_ID: &str = "11111111222243338444555555555555";

const CONVERSATION_ID: &str = "conv-1";

/// A tool argument. Content, and the reason this file checks what the
/// injection adds rather than only that it added something.
const ARGUMENT_SENTINEL: &str = "SENTINEL-ARGUMENT-a-path-the-model-chose";

// ---------------------------------------------------------------------------
// stdio: the `_meta` vehicle.
// ---------------------------------------------------------------------------

/// Unique temp path for this test process.
fn temp_path(label: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "mcp-trace-{}-{}-{}",
        std::process::id(),
        n,
        label
    ))
}

/// A fake MCP server that answers `initialize` and `tools/call`, and writes the
/// `tools/call` request line it received to `echo_file`.
fn write_server(label: &str, echo_file: &Path) -> PathBuf {
    let script = format!(
        r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf %s "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"protocolVersion":"2025-11-25","capabilities":{{}},"serverInfo":{{"name":"fake","version":"0.0"}}}}}}\n' "$id" ;;
    *'"method":"tools/call"'*)
      printf '%s\n' "$line" > '{echo}'
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"content":[{{"type":"text","text":"ok"}}]}}}}\n' "$id" ;;
    *) : ;;
  esac
done
"#,
        echo = echo_file.display(),
    );
    let path = temp_path(label);
    std::fs::write(&path, script).expect("write the fake server script");
    path
}

/// Call one tool over stdio and return the request the server actually read.
async fn stdio_call(label: &str) -> Value {
    let echo = temp_path(&format!("{label}-echo"));
    let script = write_server(label, &echo);
    let mut client = McpClient::connect(
        "/bin/sh",
        &[script.to_string_lossy().into_owned()],
        &HashMap::new(),
    )
    .await
    .expect("the fake server must complete the handshake");
    client
        .call_tool(
            "echo",
            serde_json::json!({ "text": ARGUMENT_SENTINEL }),
        )
        .await
        .expect("the fake server must answer the call");
    client.shutdown().await;

    let line = std::fs::read_to_string(&echo).unwrap_or_default();
    let _ = std::fs::remove_file(&script);
    let _ = std::fs::remove_file(&echo);
    serde_json::from_str(line.trim()).unwrap_or_else(|e| {
        panic!("the server must have read one JSON-RPC line, got {line:?}: {e}")
    })
}

#[tokio::test]
async fn stdio_mcp_call_carries_context_in_meta() {
    let request = with_turn_trace(
        Some(resolve_turn_trace(None, REQUEST_ID, CONVERSATION_ID)),
        stdio_call("in-a-turn"),
    )
    .await;

    let traceparent = request["params"]["_meta"]["traceparent"]
        .as_str()
        .unwrap_or_else(|| panic!("no `_meta.traceparent` on the request: {request}"));
    let parsed = adelie_telemetry::extract_traceparent(traceparent)
        .unwrap_or_else(|e| panic!("`{traceparent}` is not a valid traceparent: {e}"));
    assert_eq!(
        parsed.trace_id().to_hex(),
        TRACE_ID,
        "the server must be told the turn's own trace, not another one"
    );
    assert_eq!(
        request["params"]["arguments"]["text"], ARGUMENT_SENTINEL,
        "the call's own arguments must survive the injection"
    );
}

#[tokio::test]
async fn injecting_the_trace_adds_nothing_but_the_trace() {
    // `_meta` is the protocol's own extension point, and a caller may put
    // other keys there. This is the whole of what propagation contributes to
    // it, so a later reader can see that no argument and no prompt was folded
    // in alongside.
    let request = with_turn_trace(
        Some(resolve_turn_trace(None, REQUEST_ID, CONVERSATION_ID)),
        stdio_call("meta-contents"),
    )
    .await;

    let meta = request["params"]["_meta"]
        .as_object()
        .unwrap_or_else(|| panic!("`_meta` must be an object: {request}"));
    assert_eq!(
        meta.keys().collect::<Vec<_>>(),
        vec!["traceparent"],
        "propagation adds exactly one key"
    );
}

#[tokio::test]
async fn a_stdio_call_outside_a_turn_carries_no_meta() {
    // A background handshake or a health probe has no trace. Injecting a
    // placeholder would make the server join a trace nobody started, which is
    // worse than the server starting its own.
    let request = stdio_call("outside-a-turn").await;
    assert!(
        request["params"].get("_meta").is_none(),
        "nothing may be injected outside a turn: {request}"
    );
}

// ---------------------------------------------------------------------------
// Streamable HTTP: the header vehicle.
// ---------------------------------------------------------------------------

/// Register the handshake every connection performs.
async fn mock_handshake(server: &MockServer) {
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"initialize""#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}"#);
        })
        .await;
    server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"notifications/initialized""#);
            then.status(202);
        })
        .await;
}

#[tokio::test]
async fn remote_mcp_call_carries_traceparent_header() {
    let server = MockServer::start_async().await;
    mock_handshake(&server).await;

    // The mock matches on the header, so the assertion is that the header
    // arrived with the right value - not merely that some call was made.
    let expected = format!("00-{TRACE_ID}-");
    let call = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"tools/call""#)
                .header_exists("traceparent")
                .header_matches("traceparent", format!("^{expected}"));
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"ok"}]}}"#);
        })
        .await;

    let url = server.url("/mcp");
    with_turn_trace(
        Some(resolve_turn_trace(None, REQUEST_ID, CONVERSATION_ID)),
        async move {
            let mut client = McpClient::connect_http(&url, None)
                .await
                .expect("the mock must complete the handshake");
            client
                .call_tool("echo", serde_json::json!({ "text": ARGUMENT_SENTINEL }))
                .await
                .expect("the mock must answer the call");
        },
    )
    .await;

    call.assert_async().await;
}

#[tokio::test]
async fn a_remote_call_outside_a_turn_sends_no_traceparent() {
    let server = MockServer::start_async().await;
    mock_handshake(&server).await;

    let call = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/mcp")
                .body_includes(r#""method":"tools/call""#)
                .header_missing("traceparent");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"ok"}]}}"#);
        })
        .await;

    let mut client = McpClient::connect_http(&server.url("/mcp"), None)
        .await
        .expect("the mock must complete the handshake");
    client
        .call_tool("echo", serde_json::json!({ "text": ARGUMENT_SENTINEL }))
        .await
        .expect("the mock must answer the call");

    call.assert_async().await;
}
