//! Where the daemon's diagnostics go.
//!
//! Every log line must leave on stderr. stdout belongs to the process's own
//! output: the daemon prints an operator-facing result there for its two
//! command-line escape hatches, and a diagnostic mixed into that stream is
//! indistinguishable from the result. The same rule keeps the fleet uniform,
//! where an MCP server frames JSON-RPC on stdout and one stray log line
//! corrupts the protocol.
//!
//! Driven through the real binary rather than through a library call, because
//! the writer is chosen where telemetry is installed and only the binary
//! installs it.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long the binary may take to reach its early exit.
const RUN_BUDGET: Duration = Duration::from_secs(60);

/// Run `command` to completion and return `(stdout, stderr)`.
///
/// Killed at [`RUN_BUDGET`] rather than left to block the gate: a daemon that
/// gets past its early exit runs until it is stopped.
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
fn the_default_filter_keeps_a_bare_run_as_quiet_as_before() {
    // The daemon used `EnvFilter::from_default_env()`, which falls back to
    // ERROR when `RUST_LOG` is unset. Adopting a shared crate must not make a
    // desktop daemon start logging where it used to be silent, and only a run
    // of the real binary with the variable genuinely absent proves that the
    // configured fallback is the one the installed subscriber applies.
    let mut command = Command::new(env!("CARGO_BIN_EXE_desktop-assistant-daemon"));
    command.arg("--revoke-token").env_remove("RUST_LOG");
    let (stdout, stderr) = run(command);

    assert!(
        !stderr.contains("desktop-assistant starting"),
        "with RUST_LOG unset the daemon logs nothing below ERROR\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.is_empty(),
        "nothing belongs on stdout either\n--- stdout ---\n{stdout}"
    );
}

#[test]
fn logs_go_to_stderr_not_stdout() {
    // `--revoke-token` with no argument is rejected straight after telemetry
    // is installed, so the process starts, logs, and exits without opening a
    // socket, a database or the session bus.
    let mut command = Command::new(env!("CARGO_BIN_EXE_desktop-assistant-daemon"));
    command.arg("--revoke-token").env("RUST_LOG", "info");
    let (stdout, stderr) = run(command);

    assert!(
        stderr.contains("desktop-assistant starting"),
        "the startup line must reach stderr\n--- stderr ---\n{stderr}"
    );
    assert!(
        !stdout.contains("desktop-assistant starting"),
        "no log line may reach stdout\n--- stdout ---\n{stdout}"
    );
}
