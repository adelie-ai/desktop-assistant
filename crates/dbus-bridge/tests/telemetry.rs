//! Where the bridge's diagnostics go.
//!
//! Every log line must leave on stderr, never stdout. The rule is the fleet's,
//! not this binary's: an MCP server frames JSON-RPC on stdout and a client
//! writes the model's reply there, so one diagnostic on stdout corrupts what
//! the reader is parsing. The bridge holds the same line so the whole fleet
//! reads the same way.
//!
//! Driven through the real binary rather than through a library call, because
//! the writer is chosen where telemetry is installed and only the binary
//! installs it.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long the binary may take to fail its connection and exit.
const RUN_BUDGET: Duration = Duration::from_secs(60);

/// Run `command` to completion and return `(stdout, stderr)`.
fn run(mut command: Command) -> (String, String) {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the binary under test");

    let deadline = Instant::now() + RUN_BUDGET;
    loop {
        match child.try_wait().expect("poll the child") {
            Some(_) => break,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("the binary did not exit within {RUN_BUDGET:?}");
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }

    let mut stdout = String::new();
    let mut stderr = String::new();
    child
        .stdout
        .take()
        .expect("stdout pipe")
        .read_to_string(&mut stdout)
        .expect("read stdout");
    child
        .stderr
        .take()
        .expect("stderr pipe")
        .read_to_string(&mut stderr)
        .expect("read stderr");
    (stdout, stderr)
}

#[test]
fn logs_go_to_stderr_not_stdout() {
    // A socket path that cannot exist. The bridge logs that it is connecting,
    // fails the connection, and exits - which is all this needs.
    let dir = tempfile::tempdir().expect("a temporary directory");
    let socket = dir.path().join("absent.sock");

    let mut command = Command::new(env!("CARGO_BIN_EXE_adelie-dbus-bridge"));
    command
        .env("RUST_LOG", "info")
        .env("ADELIE_BRIDGE_DAEMON_SOCKET", &socket);
    let (stdout, stderr) = run(command);

    assert!(
        stderr.contains("connecting to daemon UDS"),
        "the connect line must reach stderr\n--- stderr ---\n{stderr}"
    );
    assert!(
        !stdout.contains("connecting to daemon UDS"),
        "no log line may reach stdout\n--- stdout ---\n{stdout}"
    );
}
