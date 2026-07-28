//! Protocol version negotiation for the MCP client (issue #931).
//!
//! The client must request the current spec revision, record what the server
//! actually agreed to, and be conservative about the answer: a known older
//! revision proceeds (every revision from `2024-11-05` on carries the whole
//! surface this client uses), while an unknown or absent version fails the
//! handshake rather than leaving the session in an unreasoned state.
//!
//! Each test drives the real `McpClient` against a small `/bin/sh` fake server,
//! so the stdio transport and the real `initialize` path are exercised.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use desktop_assistant_mcp_client::{McpClient, McpError};

/// Unique temp path for this test process.
fn temp_path(label: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "mcp-protocol-version-{}-{}-{}",
        std::process::id(),
        n,
        label
    ))
}

/// Write a fake MCP server that answers `initialize` with `result_body` and
/// echoes the request line it received to `echo_file`. Returns the script path.
fn write_server(label: &str, result_body: &str, echo_file: &Path) -> PathBuf {
    let script = format!(
        r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf %s "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  case "$line" in
    *'"method":"initialize"'*)
      printf '%s\n' "$line" > '{echo}'
      printf '{{"jsonrpc":"2.0","id":%s,"result":{body}}}\n' "$id" ;;
    *) : ;;
  esac
done
"#,
        echo = echo_file.display(),
        body = result_body,
    );
    let path = temp_path(label);
    std::fs::write(&path, script).expect("write fake server script");
    path
}

/// An `initialize` result carrying `version` as the negotiated `protocolVersion`.
fn result_with_version(version: &str) -> String {
    format!(
        r#"{{"protocolVersion":"{version}","capabilities":{{}},"serverInfo":{{"name":"fake","version":"0.0"}}}}"#
    )
}

/// Connect to a fake server answering with `result_body`, returning the client
/// and the raw `initialize` request line the server saw.
async fn connect_to(label: &str, result_body: &str) -> (Result<McpClient, McpError>, String) {
    let echo = temp_path(&format!("{label}-echo"));
    let script = write_server(label, result_body, &echo);
    let client = McpClient::connect(
        "/bin/sh",
        &[script.to_string_lossy().into_owned()],
        &HashMap::new(),
    )
    .await;
    let request = std::fs::read_to_string(&echo).unwrap_or_default();
    let _ = std::fs::remove_file(&script);
    let _ = std::fs::remove_file(&echo);
    (client, request)
}

#[tokio::test]
async fn initialize_requests_current_protocol_revision() {
    let (client, request) =
        connect_to("requests-current", &result_with_version("2025-11-25")).await;
    client.expect("handshake must succeed");
    assert!(
        request.contains(r#""protocolVersion":"2025-11-25""#),
        "outbound initialize must request the current revision, got: {request}"
    );
    assert!(
        !request.contains("2024-11-05"),
        "the retired hardcoded revision must be gone: {request}"
    );
}

#[tokio::test]
async fn initialize_records_negotiated_version() {
    let (client, _) = connect_to("records", &result_with_version("2025-11-25")).await;
    let client = client.expect("handshake must succeed");
    assert_eq!(client.protocol_version(), Some("2025-11-25"));
}

/// A server on the previous revision is recorded as such, not as what we asked
/// for - otherwise a downgraded session is invisible.
#[tokio::test]
async fn initialize_accepts_previous_revision_downgrade() {
    let (client, _) = connect_to("downgrade", &result_with_version("2025-06-18")).await;
    let client = client.expect("a 2025-06-18 server must still connect");
    assert_eq!(client.protocol_version(), Some("2025-06-18"));
}

/// The don't-break-working-servers guarantee: the oldest revision still
/// connects. Every revision from `2024-11-05` on carries `initialize`,
/// `tools/list`, `tools/call` and `resources/list` identically, so refusing it
/// would break working third-party servers to gain nothing.
#[tokio::test]
async fn initialize_accepts_legacy_revision_with_warning() {
    let (client, _) = connect_to("legacy", &result_with_version("2024-11-05")).await;
    let client = client.expect("a 2024-11-05 server must still connect");
    assert_eq!(client.protocol_version(), Some("2024-11-05"));
}

#[tokio::test]
async fn initialize_rejects_unknown_negotiated_version() {
    let (client, _) = connect_to("unknown", &result_with_version("1999-01-01")).await;
    let err = client
        .err()
        .expect("an unknown revision must fail the handshake");
    assert!(
        err.to_string().contains("1999-01-01"),
        "the error must name what the server sent, got: {err}"
    );
}

/// Absent is not the same as "assume the default": a server that omits the
/// field has not negotiated anything, so there is nothing to reason about.
#[tokio::test]
async fn initialize_rejects_missing_negotiated_version() {
    let body = r#"{"capabilities":{},"serverInfo":{"name":"fake","version":"0.0"}}"#;
    let (client, _) = connect_to("missing", body).await;
    let err = client
        .err()
        .expect("a missing protocolVersion must fail the handshake");
    assert!(
        err.to_string().contains("protocolVersion"),
        "the error must say what was missing, got: {err}"
    );
}

/// A non-string `protocolVersion` is malformed input from an untrusted peer and
/// must be rejected, not coerced.
#[tokio::test]
async fn initialize_rejects_non_string_negotiated_version() {
    let body = r#"{"protocolVersion":20251125,"capabilities":{},"serverInfo":{"name":"fake","version":"0.0"}}"#;
    let (client, _) = connect_to("non-string", body).await;
    assert!(
        client.is_err(),
        "a non-string protocolVersion must fail the handshake"
    );
}
