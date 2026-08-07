//! What a stdio MCP server says on stderr must survive its own death.
//!
//! A server that cannot start says why on stderr — a missing argument, a
//! missing file, a bad credential. `StdioTransport` used to send that stream
//! to `/dev/null` and report only the exit code, so the one sentence naming
//! the cause was destroyed at spawn time and the operator was left to guess.
//!
//! These tests drive the real `McpClient` against small `/bin/sh` fake MCP
//! servers, the same way `robustness.rs` and `env_isolation.rs` do, so what is
//! asserted is observed transport behaviour rather than the shape of the
//! source.
//!
//! Two properties are load-bearing and pull in opposite directions:
//!
//! - The failure message must carry the server's own words.
//! - A piped stderr that nobody reads fills its kernel buffer (64 KB on
//!   Linux) and blocks the writer forever, so a healthy but chatty server
//!   must keep working. `a_healthy_server_writing_past_the_pipe_buffer_still_completes_a_round_trip`
//!   is the test that holds that line.

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;

use desktop_assistant_mcp_client::{McpClient, McpError};

/// This suite's scratch directory: cargo's compile-time per-target tmp dir,
/// not `std::env::temp_dir()` (mirrors `env_isolation.rs`, which explains why).
fn scratch_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
}

/// Unique temp script path for this test process.
fn temp_path(label: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    scratch_dir().join(format!(
        "mcp-stderr-{}-{}-{}.sh",
        std::process::id(),
        n,
        label
    ))
}

/// Write `script` to a fresh temp file and run it as a stdio MCP server,
/// returning whatever `McpClient::connect` made of it. The script is removed
/// before the result is inspected, so a failing assertion never leaves one
/// behind.
async fn connect_to_script(label: &str, script: &str) -> Result<McpClient, McpError> {
    let path = temp_path(label);
    std::fs::write(&path, script).expect("write fake server script");
    let result =
        McpClient::connect("/bin/sh", &[path.display().to_string()], &HashMap::new()).await;
    let _ = std::fs::remove_file(&path);
    result
}

/// The failure message from a handshake that was expected to fail.
///
/// Panics with the successful client's own tools if the handshake succeeded,
/// or with the wrong variant, so a broken fake server never reads as a passing
/// assertion.
async fn handshake_failure(label: &str, script: &str) -> String {
    match connect_to_script(label, script).await {
        Ok(_) => panic!("{label}: expected the handshake to fail, but connect succeeded"),
        Err(McpError::UnexpectedResponse(msg)) => msg,
        Err(other) => panic!("{label}: expected McpError::UnexpectedResponse, got: {other}"),
    }
}

/// Upper bound asserted on a whole handshake failure message.
///
/// The tail is bounded by construction (a fixed number of lines, each capped
/// in bytes), so the message a person reads in a log line or the settings
/// panel has a stated ceiling. 8 KB is comfortably above the implementation's
/// own bound and still far below a runaway server's output.
const MAX_MESSAGE_BYTES: usize = 8 * 1024;

// --- The motivating case --------------------------------------------------

/// A server that explains itself on stderr and exits non-zero must have that
/// explanation in the error, not only its exit code.
///
/// The stderr text is the real one from a server started without a required
/// argument, so this test documents the case that motivated the change.
#[tokio::test]
async fn server_stderr_appears_in_the_handshake_failure_message() {
    let msg = handshake_failure(
        "missing-required-argument",
        r#"#!/bin/sh
# A server whose argument parser rejects its own command line: it says why on
# stderr and exits, exactly as clap does for a missing required argument.
read -r line
printf '%s\n' 'error: the following required arguments were not provided: --config <CONFIG>' >&2
exit 2
"#,
    )
    .await;

    assert!(
        msg.contains(
            "error: the following required arguments were not provided: --config <CONFIG>"
        ),
        "the server's own stderr must reach the error, got: {msg}"
    );
    assert!(
        msg.contains("exited with status 2"),
        "the exit status must still be named, got: {msg}"
    );
}

/// The guess must give way to the evidence: when the server said something,
/// the message must not also tell the operator to go and check an environment
/// variable. That hint was wrong in the motivating case and sent the reader
/// away from the answer that was right in front of them.
#[tokio::test]
async fn stderr_replaces_the_environment_variable_guess() {
    let msg = handshake_failure(
        "stderr-beats-the-guess",
        r#"#!/bin/sh
read -r line
printf '%s\n' 'error: config file /etc/example/absent.toml does not exist' >&2
exit 1
"#,
    )
    .await;

    assert!(
        msg.contains("error: config file /etc/example/absent.toml does not exist"),
        "the server's own stderr must reach the error, got: {msg}"
    );
    assert!(
        !msg.contains("environment variable"),
        "the environment-variable guess must not compete with real stderr, got: {msg}"
    );
}

// --- Bounds ---------------------------------------------------------------

/// A server that floods stderr must surface its *last* lines. The final lines
/// before a crash are the ones that name the cause; the first lines are
/// startup banners.
///
/// Also pins the size ceiling: an unbounded tail would put a whole log into a
/// single log line and a single settings-panel field.
#[tokio::test]
async fn a_flooding_server_surfaces_its_last_stderr_lines_within_a_size_bound() {
    let msg = handshake_failure(
        "flooding-stderr",
        r#"#!/bin/sh
read -r line
i=1
while [ "$i" -le 100 ]; do
  printf 'stderr-line-%03d\n' "$i" >&2
  i=$((i + 1))
done
exit 5
"#,
    )
    .await;

    assert!(
        msg.contains("stderr-line-100") && msg.contains("stderr-line-099"),
        "the last stderr lines must survive, got: {msg}"
    );
    assert!(
        !msg.contains("stderr-line-001") && !msg.contains("stderr-line-050"),
        "an early line must have been evicted by the bounded tail, got: {msg}"
    );
    assert!(
        msg.len() <= MAX_MESSAGE_BYTES,
        "the failure message must stay under {MAX_MESSAGE_BYTES} bytes; it was {} bytes",
        msg.len()
    );
}

/// One enormous line must be cut, not carried whole. A single line has no
/// newline to bound it, so without a per-line cap one `printf` of a megabyte
/// defeats the line-count bound on its own.
#[tokio::test]
async fn an_overlong_stderr_line_is_truncated_rather_than_carried_whole() {
    let msg = handshake_failure(
        "overlong-stderr-line",
        r#"#!/bin/sh
read -r line
# 4096 bytes of padding, built by doubling a shell variable (no fork, no
# external command), then a marker that must NOT survive the cut.
pad=xxxxxxxx
pad=$pad$pad
pad=$pad$pad
pad=$pad$pad
pad=$pad$pad
pad=$pad$pad
pad=$pad$pad
pad=$pad$pad
pad=$pad$pad
pad=$pad$pad
printf '%sEND-OF-OVERLONG-LINE\n' "$pad" >&2
exit 3
"#,
    )
    .await;

    assert!(
        msg.contains("xxxxxxxx"),
        "the head of the long line must survive, got: {msg}"
    );
    assert!(
        !msg.contains("END-OF-OVERLONG-LINE"),
        "the tail of an over-long line must have been cut, got: {msg}"
    );
    assert!(
        msg.len() <= MAX_MESSAGE_BYTES,
        "the failure message must stay under {MAX_MESSAGE_BYTES} bytes; it was {} bytes",
        msg.len()
    );
}

// --- The silent server keeps the hint -------------------------------------

/// A server that dies without a word still reports its exit status, and only
/// then falls back to the environment-variable hint — the guess is for the
/// case where there is no evidence, not a competitor to it.
///
/// `env_isolation.rs::server_exiting_before_handshake_reports_its_exit_status`
/// covers the exit status itself; this one holds the fallback hint in place.
#[tokio::test]
async fn a_silent_server_reports_its_exit_status_and_keeps_the_environment_hint() {
    let msg = handshake_failure(
        "silent-death",
        r#"#!/bin/sh
read -r line
exit 7
"#,
    )
    .await;

    assert!(
        msg.contains("exited with status 7"),
        "the exit status must be named, got: {msg}"
    );
    assert!(
        msg.contains("environment variable"),
        "with no stderr to go on, the environment-variable hint is what is left, got: {msg}"
    );
}

// --- The deadlock guard ---------------------------------------------------

/// A healthy server that writes far more than a pipe buffer holds must still
/// complete its handshake and a tool call.
///
/// This is the guard on the change itself. A piped stderr with no reader fills
/// the kernel's 64 KB pipe buffer and blocks the writing process forever: the
/// server would never get to reply to `initialize`, and every chatty server in
/// the fleet would hang at startup. The fix is a background task that drains
/// the pipe continuously, and this test is what proves it is there.
///
/// The server writes about 400 KB before its `initialize` reply, so it is
/// several pipe buffers deep in the blocking case. That makes the test
/// non-vacuous: against a piped-but-undrained stderr the handshake never
/// completes and this fails on its own bound, not on the client's.
#[tokio::test]
async fn a_healthy_server_writing_past_the_pipe_buffer_still_completes_a_round_trip() {
    let script = r#"#!/bin/sh
# A 256-byte noise line, built by doubling a shell variable.
noise=xxxxxxxxxxxxxxxx
noise=$noise$noise
noise=$noise$noise
noise=$noise$noise
noise=$noise$noise
while IFS= read -r line; do
  id=$(printf %s "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  case "$line" in
    *'"method":"initialize"'*)
      i=0
      while [ "$i" -lt 1600 ]; do
        printf '%s\n' "$noise" >&2
        i=$((i + 1))
      done
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"chatty","version":"0.0"}}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"echo tool","inputSchema":{"type":"object"}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"done-chatty"}]}}\n' "$id"
      ;;
    *) : ;;
  esac
done
"#;

    let path = temp_path("chatty-but-healthy");
    std::fs::write(&path, script).expect("write chatty server script");
    let args = [path.display().to_string()];

    // Bounded well under the client's own 30s handshake cap, so a blocked
    // child fails here as a deadlock rather than 30s later as a timeout.
    let mut client = tokio::time::timeout(
        Duration::from_secs(20),
        McpClient::connect("/bin/sh", &args, &HashMap::new()),
    )
    .await
    .expect("a chatty server must not block on its own stderr pipe")
    .expect("handshake should succeed");

    let tools = tokio::time::timeout(Duration::from_secs(20), client.list_tools())
        .await
        .expect("tools/list must not block on the stderr pipe")
        .expect("tools/list should succeed");
    assert_eq!(
        tools.len(),
        1,
        "the fake server advertises exactly one tool"
    );

    let result = tokio::time::timeout(
        Duration::from_secs(20),
        client.call_tool("echo", serde_json::json!({})),
    )
    .await
    .expect("tools/call must not block on the stderr pipe")
    .expect("tools/call should succeed");
    assert!(
        result.to_string().contains("done-chatty"),
        "the round trip must return the server's reply, got: {result}"
    );

    client.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

// --- Failing twice on one transport ---------------------------------------

/// A transport that reports a dead server once must report it again, not
/// panic.
///
/// `enrich_with_exit_status` runs per failed round trip, so it can run twice
/// against the same transport. Both of its waits have to survive that: the
/// wait on the child is fused and answers from a cached status, and the wait
/// on the stderr drain would panic if its handle were polled a second time
/// after yielding.
///
/// Reaching the second run needs the client's writes to keep succeeding after
/// the server is gone, which is what the backgrounded process in the fixture
/// arranges: it inherits the read end of the pipe the client writes to, so a
/// request still lands somewhere even though nothing will ever answer it. A
/// real server that forks a worker and exits behaves exactly this way.
#[tokio::test]
async fn a_second_failing_round_trip_reports_again_instead_of_panicking() {
    let script = r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf %s "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"forker","version":"0.0"}}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"echo tool","inputSchema":{"type":"object"}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      printf '%s\n' 'fatal: the server gave up' >&2
      # Hand the read end of stdin to a background process, so the client's
      # later writes still succeed. `<&0` is required: an asynchronous list
      # otherwise gets /dev/null for stdin. stdout and stderr are redirected
      # away from it, so the client still sees end-of-file on both.
      sleep 5 <&0 >/dev/null 2>/dev/null &
      exec 1>&-
      exit 6
      ;;
    *) : ;;
  esac
done
"#;

    let path = temp_path("forks-then-dies");
    std::fs::write(&path, script).expect("write forking server script");
    let args = [path.display().to_string()];

    let mut client = McpClient::connect("/bin/sh", &args, &HashMap::new())
        .await
        .expect("handshake should succeed");
    client
        .list_tools()
        .await
        .expect("tools/list should succeed");

    let first = client.call_tool("echo", serde_json::json!({})).await;
    match first {
        Err(McpError::UnexpectedResponse(msg)) => {
            assert!(
                msg.contains("exited with status 6") && msg.contains("fatal: the server gave up"),
                "the first failure must name the exit status and the stderr, got: {msg}"
            );
        }
        other => {
            panic!("expected the first call to fail with the enriched message, got: {other:?}")
        }
    }

    let second = tokio::time::timeout(
        Duration::from_secs(20),
        client.call_tool("echo", serde_json::json!({})),
    )
    .await
    .expect("the second call must return, not hang");
    assert!(
        second.is_err(),
        "the second call to a dead server must fail, got: {second:?}"
    );

    client.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

// --- Bytes that are not text ----------------------------------------------

/// Stderr is a byte stream, not a UTF-8 stream. A server that emits a raw byte
/// sequence — a mis-set locale, a binary dependency's own output — must not
/// panic the daemon, and must not cost the readable lines around it.
#[tokio::test]
async fn non_utf8_stderr_neither_panics_nor_loses_the_surrounding_lines() {
    let msg = handshake_failure(
        "non-utf8-stderr",
        r#"#!/bin/sh
read -r line
printf '%s\n' 'readable-line-before' >&2
printf 'raw-\376\377-bytes\n' >&2
printf '%s\n' 'readable-line-after' >&2
exit 4
"#,
    )
    .await;

    assert!(
        msg.contains("readable-line-before") && msg.contains("readable-line-after"),
        "lines either side of undecodable bytes must survive, got: {msg}"
    );
    assert!(
        msg.contains("exited with status 4"),
        "the exit status must still be named, got: {msg}"
    );
}

/// The message is one line of text that reads left to right, and a server
/// cannot make it otherwise.
///
/// Stderr is whatever the server chose to print, and the message it lands in
/// is rendered by a log reader, by the settings/KCM panel and by the web SPA.
/// Three separate powers have to be taken away, and `char::is_control` covers
/// only the first:
///
/// - **C0/C1 controls (Cc).** A carriage return rewinds the line and paints
///   over what came before it; an ANSI escape clears or recolours it.
/// - **Line and paragraph separators (Zl/Zp).** U+2028 and U+2029 are not
///   control characters. `serde_json` does not escape them either, so one
///   reaches the panel intact and is rendered as a line break — giving the
///   server a second line of its own text, positioned as though the UI had
///   written it.
/// - **Format characters (Cf).** U+202E RIGHT-TO-LEFT OVERRIDE reverses the
///   displayed order of everything after it (Trojan Source, CVE-2021-42574);
///   the zero-width characters hide text outright.
#[tokio::test]
async fn control_characters_in_stderr_cannot_rewrite_the_message_line() {
    // Written as octal byte escapes because POSIX `printf` has no \u form.
    // \342\200\250 = U+2028, \342\200\251 = U+2029, \342\200\256 = U+202E,
    // \342\200\213 = U+200B, \357\273\277 = U+FEFF.
    let msg = handshake_failure(
        "control-characters",
        r#"#!/bin/sh
read -r line
printf 'visible-head\rHIDDEN-BY-CARRIAGE-RETURN\033[2K\033[31mred\n' >&2
printf 'line-sep\342\200\250FORGED-SECOND-LINE\n' >&2
printf 'para-sep\342\200\251FORGED-PARAGRAPH\n' >&2
printf 'bidi\342\200\256REVERSED-FROM-HERE\n' >&2
printf 'zero\342\200\213width\357\273\277marks\n' >&2
exit 8
"#,
    )
    .await;

    assert!(
        msg.contains("visible-head")
            && msg.contains("line-sep")
            && msg.contains("para-sep")
            && msg.contains("bidi")
            && msg.contains("zero"),
        "the readable text must survive, got: {msg}"
    );
    for hazard in [
        '\r', '\n', '\u{1b}', '\u{2028}', '\u{2029}', '\u{202e}', '\u{200b}', '\u{feff}',
    ] {
        assert!(
            !msg.contains(hazard),
            "U+{:04X} must not reach the message, got: {msg:?}",
            hazard as u32
        );
    }
}

// --- Failures that are not an exit code -----------------------------------

/// A server that explains itself and then hangs must have that explanation in
/// the timeout, not only in a stack of silence.
///
/// A hang is the startup failure an operator finds hardest to read: nothing
/// exited, nothing was refused, the daemon simply waits and then gives up.
/// The server said why before it stopped, and the tail already holds it.
#[tokio::test]
async fn a_hanging_server_surfaces_its_stderr_in_the_timeout() {
    let script = r#"#!/bin/sh
# Say why on stderr, then hang forever without answering. TERM is ignored, so
# only the unconditional SIGKILL on client teardown (DS-2) ends this.
read -r line
printf '%s\n' 'fatal: cannot open database' >&2
trap '' TERM
while true; do sleep 1; done
"#;

    let path = temp_path("hangs-after-complaining");
    std::fs::write(&path, script).expect("write hanging server script");
    let args = [path.display().to_string()];

    // A one-second silence budget, so the handshake gives up in a second
    // rather than in the default thirty. Which of the two bounds fires first
    // is a race and does not matter: the property under test is that a
    // timeout carries the tail, whichever bound produced it.
    let result = tokio::time::timeout(
        Duration::from_secs(20),
        McpClient::connect_with_request_timeout(
            "/bin/sh",
            &args,
            &HashMap::new(),
            Duration::from_secs(1),
        ),
    )
    .await
    .expect("connect must return within the test bound");

    let _ = std::fs::remove_file(&path);

    match result {
        Ok(_) => panic!("expected the handshake to time out, but connect succeeded"),
        Err(err @ McpError::Timeout { .. }) => {
            let text = err.to_string();
            assert!(
                text.contains("fatal: cannot open database"),
                "the timeout must carry what the server said before it hung, got: {text}"
            );
            assert!(
                text.contains("timed out"),
                "the timeout must still read as a timeout, got: {text}"
            );
        }
        Err(other) => panic!("expected McpError::Timeout, got: {other}"),
    }
}

/// A server that closes stdout and keeps running has no exit status to
/// report, and must fall back to its stderr rather than to nothing.
///
/// This is the third failure shape, beside a clean exit and a hang. The
/// child is alive, so `child.wait()` runs out its bound and there is no code
/// to name — but the server had already said what was wrong.
#[tokio::test]
async fn a_server_that_closes_stdout_and_stays_alive_still_surfaces_its_stderr() {
    let msg = handshake_failure(
        "closes-stdout-after-complaining",
        r#"#!/bin/sh
read -r line
printf '%s\n' 'fatal: no write access to the state directory' >&2
exec 1>&-
trap '' TERM
while true; do sleep 1; done
"#,
    )
    .await;

    assert!(
        msg.contains("fatal: no write access to the state directory"),
        "a live server's stderr must still be reported, got: {msg}"
    );
    assert!(
        !msg.contains("exited with status"),
        "there is no exit status to name for a child that is still running, got: {msg}"
    );
    assert!(
        !msg.contains("environment variable"),
        "the environment-variable guess must not compete with real stderr, got: {msg}"
    );
}

// --- Buffered, not logged -------------------------------------------------

/// Captured log output, shared between the subscriber and the assertions.
#[derive(Clone, Default)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl CapturedLog {
    fn text(&self) -> String {
        let bytes = self.0.lock().unwrap_or_else(|e| e.into_inner()).clone();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl io::Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Install one process-wide TRACE subscriber writing into a shared buffer, and
/// return a handle on it.
///
/// Process-wide rather than `with_default`: the stderr drain runs on a spawned
/// task, and a thread-local subscriber would not see it — which would make an
/// "it is not logged" assertion pass for the wrong reason.
fn shared_log_capture() -> CapturedLog {
    static INSTALL: Once = Once::new();
    static CAPTURE: Mutex<Option<CapturedLog>> = Mutex::new(None);

    let mut slot = CAPTURE.lock().unwrap_or_else(|e| e.into_inner());
    let capture = slot.get_or_insert_with(CapturedLog::default).clone();
    INSTALL.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_writer(capture.clone())
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
    capture
}

/// A server's stderr is buffered for the failure path and logged nowhere.
///
/// It carries whatever the server chose to print — a credential in an error
/// message, a file path, a fragment of a user's own content. Surfacing it on
/// the one error that a person is already reading is a deliberate, bounded
/// exposure. Streaming it into the daemon's logs at any level is not, and
/// would undo from a second direction the work that kept content out of the
/// default logs.
#[tokio::test]
async fn server_stderr_is_buffered_and_never_logged() {
    const SENTINEL: &str = "SENTINEL-B7F2-STDERR-MUST-NOT-BE-LOGGED";

    let logs = shared_log_capture();

    let msg = handshake_failure(
        "stderr-not-logged",
        &format!(
            r#"#!/bin/sh
read -r line
printf '%s\n' '{SENTINEL}' >&2
exit 9
"#
        ),
    )
    .await;

    assert!(
        msg.contains(SENTINEL),
        "the stderr must reach the error itself, got: {msg}"
    );

    let text = logs.text();
    assert!(
        text.contains("MCP request"),
        "the log capture must itself be working, or the absence check below \
         proves nothing\n--- captured ---\n{text}"
    );
    assert!(
        !text.contains(SENTINEL),
        "server stderr must not reach the logs at any level\n--- captured ---\n{text}"
    );
}
