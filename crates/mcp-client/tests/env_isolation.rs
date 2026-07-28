//! Environment isolation for spawned stdio MCP children (#910).
//!
//! Before this fix, `StdioTransport::spawn` only *added* the per-server
//! `env` entries on top of whatever the daemon process already had; the
//! child inherited the daemon's whole environment, including
//! `DESKTOP_ASSISTANT_DATABASE_URL` — the application role's Postgres DSN.
//! #721/#722 confined `builtin_db_query`'s write path to a sandboxed
//! `scratch` schema under an unprivileged role; a stdio child that can open
//! a direct connection with the real DSN makes that sandbox moot.
//!
//! These tests drive the real `StdioTransport` (via `McpClient::connect`)
//! against a tiny POSIX-`sh` fake MCP server that reports back the env
//! values it actually received, so what's asserted is observed child
//! behaviour — never the presence of `env_clear()` in the source.
//!
//! The fake server uses only shell builtins (`read`, `printf`, parameter
//! expansion) for every step, including request-id extraction — never an
//! external command — so its own operation never depends on `PATH`
//! resolving a helper binary. That matters because `PATH` itself is one of
//! the variables this suite probes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use desktop_assistant_mcp_client::{McpClient, McpError};

/// This suite's own scratch directory, resolved once and cached.
///
/// `TMPDIR` is one of the variables under test (`allowlisted_env_tmpdir_reaches_terminal_mcp`),
/// and `std::env::temp_dir()` reads it live on every call. Calling
/// `temp_dir()` fresh from [`temp_path`] would make every *other*
/// concurrently-running test's scratch files resolve against whatever value
/// the TMPDIR test happens to have set at that moment (including a
/// deliberately non-UTF-8 one) — a process-global side effect from a single
/// test, not a race in name only. Resolving once, before any test can have
/// mutated `TMPDIR`, decouples this harness's own file I/O from the value
/// under test.
fn scratch_dir() -> &'static std::path::Path {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(std::env::temp_dir)
}

/// Unique temp file path for this test process (mirrors `robustness.rs`).
fn temp_path(label: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    scratch_dir().join(format!(
        "mcp-env-isolation-{}-{}-{}.sh",
        std::process::id(),
        n,
        label
    ))
}

/// RAII guard: sets an environment variable on the current (test) process
/// and restores its previous value (or removes it) on drop, including on
/// panic.
///
/// Each test in this file gives its variable a name that **no other test in
/// this binary touches** (checked by inspection below), so parallel test
/// threads never race on the same key. This mirrors the existing pattern in
/// `crates/llm-anthropic/src/lib.rs::from_env_missing_key`, which is the
/// precedent this codebase already relies on for the Rust 2024
/// `unsafe`-env-mutation requirement (AGENTS.md: "the only currently
/// acceptable case").
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: `key` is exclusively owned by the calling test within this
        // binary (see struct doc); no other test thread reads or writes it
        // while this guard is alive.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: see `set`.
        match &self.previous {
            Some(v) => unsafe { std::env::set_var(self.key, v) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

/// Render a self-contained POSIX-`sh` fake MCP server. It replies to
/// `initialize` with a fixed handshake, ignores
/// `notifications/initialized`, and replies to any `tools/call` with the
/// *current* value of each name in `probe_vars`, joined as `NAME=value;`
/// (`NAME=<UNSET>;` when the child does not have it).
fn write_env_probe_script(label: &str, probe_vars: &[&str]) -> PathBuf {
    let mut probe_body = String::new();
    for v in probe_vars {
        probe_body.push_str("      body=\"${body}");
        probe_body.push_str(v);
        probe_body.push_str("=${");
        probe_body.push_str(v);
        probe_body.push_str(":-<UNSET>};\"\n");
    }

    let mut script = String::new();
    script.push_str(
        r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*)
      rest=${line#*\"id\":}
      id=${rest%%,*}
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"env-probe","version":"0.0"}}}\n' "$id"
      ;;
    *'"method":"notifications/initialized"'*)
      :
      ;;
    *'"method":"tools/call"'*)
      rest=${line#*\"id\":}
      id=${rest%%,*}
      body=""
"#,
    );
    script.push_str(&probe_body);
    script.push_str(
        r#"      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"%s"}]}}\n' "$id" "$body"
      ;;
  esac
done
"#,
    );

    let path = temp_path(label);
    std::fs::write(&path, script).expect("write fake env-probe server script");
    path
}

/// Spawn the env-probe server (with the given per-server `env` config), call
/// its one tool, and parse the `NAME=value;` response into a map.
async fn probe_env(
    label: &str,
    probe_vars: &[&str],
    configured_env: &HashMap<String, String>,
) -> HashMap<String, String> {
    let script = write_env_probe_script(label, probe_vars);

    let mut client = McpClient::connect("/bin/sh", &[script.display().to_string()], configured_env)
        .await
        .expect("env-probe server should complete the handshake");

    let raw = client
        .call_tool("env_probe", serde_json::json!({}))
        .await
        .expect("env_probe tool call should succeed");

    client.shutdown().await;
    let _ = std::fs::remove_file(&script);

    raw.trim_end_matches(';')
        .split(';')
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let (k, v) = entry
                .split_once('=')
                .unwrap_or_else(|| panic!("probe entry must be key=value, got: {entry}"));
            (k.to_string(), v.to_string())
        })
        .collect()
}

const UNSET: &str = "<UNSET>";

// --- Acceptance criteria (issue #910) --------------------------------------

/// The named exposure is closed: a spawned stdio MCP server must not see the
/// daemon's Postgres DSN.
#[tokio::test]
async fn spawned_server_does_not_inherit_database_url() {
    let _guard = EnvVarGuard::set(
        "DESKTOP_ASSISTANT_DATABASE_URL",
        "postgres://adele:s3cr3t@127.0.0.1:5432/adele",
    );

    let seen = probe_env(
        "db-url",
        &["DESKTOP_ASSISTANT_DATABASE_URL"],
        &HashMap::new(),
    )
    .await;

    assert_eq!(
        seen["DESKTOP_ASSISTANT_DATABASE_URL"], UNSET,
        "the database DSN must not reach a spawned MCP child"
    );
}

/// Not just the named variable: an allowlist closes the door on every
/// variable nobody thought to name, not only the one the issue found.
#[tokio::test]
async fn spawned_server_does_not_inherit_arbitrary_daemon_env() {
    let _guard = EnvVarGuard::set(
        "ADELE_TEST_ARBITRARY_DAEMON_VAR",
        "should-never-reach-a-child",
    );

    let seen = probe_env(
        "arbitrary",
        &["ADELE_TEST_ARBITRARY_DAEMON_VAR"],
        &HashMap::new(),
    )
    .await;

    assert_eq!(seen["ADELE_TEST_ARBITRARY_DAEMON_VAR"], UNSET);
}

/// A server's own configured `env` (`[servers.env]`, or a secret resolved
/// from `env_secrets`) must still arrive — isolation must not also break
/// legitimate per-server configuration.
#[tokio::test]
async fn spawned_server_receives_its_configured_env() {
    let mut configured = HashMap::new();
    configured.insert(
        "HOMEASSISTANT_URL".to_string(),
        "http://ha.example.internal:8123".to_string(),
    );

    let seen = probe_env("configured-env", &["HOMEASSISTANT_URL"], &configured).await;

    assert_eq!(seen["HOMEASSISTANT_URL"], "http://ha.example.internal:8123");
}

/// `PATH` and `HOME` are the baseline every spawned server needs to run at
/// all (resolve its own subprocess dependencies, find its home directory).
#[tokio::test]
async fn spawned_server_receives_allowlisted_passthrough() {
    let expected_path = std::env::var("PATH").expect("PATH must be set in the test environment");
    let expected_home = std::env::var("HOME").expect("HOME must be set in the test environment");

    let seen = probe_env("path-home", &["PATH", "HOME"], &HashMap::new()).await;

    assert_eq!(seen["PATH"], expected_path, "PATH should reach the child");
    assert_eq!(seen["HOME"], expected_home, "HOME should reach the child");
}

// --- One named test per allowlisted variable the shipped fleet needs -------
//
// Each test owns one unique env var name (see `EnvVarGuard`'s doc), so a
// later tightening of `ENV_PASSTHROUGH_ALLOWLIST` fails exactly the test
// that names the server it breaks.

async fn assert_passes_through(var: &'static str, value: &str, label: &str) {
    let _guard = EnvVarGuard::set(var, value);
    let seen = probe_env(label, &[var], &HashMap::new()).await;
    assert_eq!(seen[var], value, "{var} should reach the spawned child");
}

/// weather-forecast-mcp / timeclock-mcp: local-time timestamps rather than
/// bare UTC. Named directly in #910's fix-shape.
#[tokio::test]
async fn allowlisted_env_tz_reaches_child() {
    assert_passes_through("TZ", "America/Denver", "tz").await;
}

/// Locale-dependent output formatting. Named directly in #910's fix-shape.
#[tokio::test]
async fn allowlisted_env_lang_reaches_child() {
    assert_passes_through("LANG", "en_US.UTF-8", "lang").await;
}

/// Outbound HTTP through a proxy: weather-forecast, geocode, openstreetmap,
/// cve, and web all call an external service.
#[tokio::test]
async fn allowlisted_env_http_proxy_uppercase_reaches_child() {
    assert_passes_through(
        "HTTP_PROXY",
        "http://proxy.example.internal:3128",
        "http-proxy-upper",
    )
    .await;
}

#[tokio::test]
async fn allowlisted_env_http_proxy_lowercase_reaches_child() {
    assert_passes_through(
        "http_proxy",
        "http://proxy.example.internal:3128",
        "http-proxy-lower",
    )
    .await;
}

#[tokio::test]
async fn allowlisted_env_https_proxy_uppercase_reaches_child() {
    assert_passes_through(
        "HTTPS_PROXY",
        "http://proxy.example.internal:3128",
        "https-proxy-upper",
    )
    .await;
}

#[tokio::test]
async fn allowlisted_env_https_proxy_lowercase_reaches_child() {
    assert_passes_through(
        "https_proxy",
        "http://proxy.example.internal:3128",
        "https-proxy-lower",
    )
    .await;
}

#[tokio::test]
async fn allowlisted_env_no_proxy_uppercase_reaches_child() {
    assert_passes_through("NO_PROXY", "localhost,127.0.0.1", "no-proxy-upper").await;
}

#[tokio::test]
async fn allowlisted_env_no_proxy_lowercase_reaches_child() {
    assert_passes_through("no_proxy", "localhost,127.0.0.1", "no-proxy-lower").await;
}

/// General XDG config lookup (any server that reads its own config from the
/// XDG base dirs rather than hardcoded defaults).
#[tokio::test]
async fn allowlisted_env_xdg_config_home_reaches_child() {
    assert_passes_through("XDG_CONFIG_HOME", "/test/xdg/config", "xdg-config").await;
}

/// tasks-mcp and timeclock-mcp default their persistent storage under
/// `$XDG_DATA_HOME` (docs/mcp-services.md). The k8s deployment repoints
/// `XDG_DATA_HOME` at the persistent-volume state dir specifically so
/// daemon-side state survives a pod restart (deploy/k8s/base/daemon.yaml).
/// Without pass-through, a spawned server would silently fall back to
/// `$HOME` on the ephemeral container filesystem and lose its data on the
/// next restart.
#[tokio::test]
async fn allowlisted_env_xdg_data_home_reaches_tasks_and_timeclock_mcp() {
    assert_passes_through("XDG_DATA_HOME", "/state/data", "xdg-data").await;
}

#[tokio::test]
async fn allowlisted_env_xdg_cache_home_reaches_child() {
    assert_passes_through("XDG_CACHE_HOME", "/test/xdg/cache", "xdg-cache").await;
}

/// Also covers the override-precedence contract: a server's own configured
/// `env` must win over the ambient allowlisted value for the same key — env
/// isolation adds a floor, not a ceiling.
#[tokio::test]
async fn allowlisted_env_xdg_state_home_reaches_child_and_configured_env_wins_over_it() {
    let _guard = EnvVarGuard::set("XDG_STATE_HOME", "/ambient/state");

    let seen = probe_env("xdg-state-plain", &["XDG_STATE_HOME"], &HashMap::new()).await;
    assert_eq!(seen["XDG_STATE_HOME"], "/ambient/state");

    let mut configured = HashMap::new();
    configured.insert(
        "XDG_STATE_HOME".to_string(),
        "/configured/state".to_string(),
    );
    let seen = probe_env("xdg-state-override", &["XDG_STATE_HOME"], &configured).await;
    assert_eq!(
        seen["XDG_STATE_HOME"], "/configured/state",
        "a server's own configured env must win over the ambient allowlisted value"
    );
}

/// web-mcp: points its bundled headless-Chrome binary (Dockerfile.fleet
/// `ENV WEB_CHROME_PATH=/usr/bin/chromium`; `deploy/mcp/mcp_servers.default.toml`).
#[tokio::test]
async fn allowlisted_env_web_chrome_path_reaches_web_mcp() {
    assert_passes_through("WEB_CHROME_PATH", "/usr/bin/chromium", "web-chrome-path").await;
}

/// skills-mcp: its skill-root search path
/// (`$SKILLS_MCP_ROOTS -> ~/.agents/skills -> ~/.claude/skills`).
#[tokio::test]
async fn allowlisted_env_skills_mcp_roots_reaches_skills_mcp() {
    assert_passes_through(
        "SKILLS_MCP_ROOTS",
        "/opt/adele/skills:/home/assistant/.claude/skills",
        "skills-roots",
    )
    .await;
}

/// skills-mcp (enabled by default): documented override for where NEW skills
/// are written when the default root isn't writable.
#[tokio::test]
async fn allowlisted_env_skills_mcp_write_root_reaches_skills_mcp() {
    assert_passes_through(
        "SKILLS_MCP_WRITE_ROOT",
        "/home/assistant/.local/share/skills",
        "skills-write-root",
    )
    .await;
}

/// internet-radio-mcp spawns `mpv`, which inherits this variable to locate
/// the PipeWire/PulseAudio session socket. This one matters on the
/// client-side MCP host (a real desktop session), not the headless fleet
/// container, since audio playback has nowhere to go there.
#[tokio::test]
async fn allowlisted_env_xdg_runtime_dir_reaches_internet_radio_mcp() {
    assert_passes_through("XDG_RUNTIME_DIR", "/run/user/1000", "xdg-runtime-dir").await;
}

/// tasks-mcp (enabled by default) runs a session-bus signal service so QML
/// widgets refresh when a task changes; it treats a bus failure as
/// non-fatal, so without this pass-through the feature would silently stop
/// working instead of erroring.
#[tokio::test]
async fn allowlisted_env_dbus_session_bus_address_reaches_tasks_mcp() {
    assert_passes_through(
        "DBUS_SESSION_BUS_ADDRESS",
        "unix:path=/run/user/1000/bus",
        "dbus-session-bus",
    )
    .await;
}

/// terminal-mcp's own defence-in-depth env scrub reads exactly PATH, HOME,
/// USER, TMPDIR, TERM, LANG from its process environment before running a
/// command. PATH/HOME/LANG are covered elsewhere; these three cover the
/// rest, so every command it runs sees the full set terminal-mcp expects
/// rather than an arbitrary subset.
#[tokio::test]
async fn allowlisted_env_user_reaches_terminal_mcp() {
    assert_passes_through("USER", "assistant", "user").await;
}

#[tokio::test]
async fn allowlisted_env_term_reaches_terminal_mcp() {
    assert_passes_through("TERM", "xterm-256color", "term").await;
}

/// Also covers the `var_os` fix: `std::env::var` returns `Err` both when a
/// variable is absent and when its value is not valid UTF-8, which would
/// silently drop a well-formed-but-non-UTF-8 value instead of passing it
/// through. Checked here (not a dedicated test) so it exercises a real
/// allowlist entry without a second test contending for the same env var
/// name (see `EnvVarGuard`'s doc on exclusive ownership).
#[tokio::test]
async fn allowlisted_env_tmpdir_reaches_terminal_mcp() {
    assert_passes_through("TMPDIR", "/tmp/terminal-mcp-scratch", "tmpdir").await;
    assert_non_utf8_value_passes_through("TMPDIR").await;
}

/// Spawns the real `StdioTransport` (via `McpClient::connect`) with `var` set
/// to a value that is not valid UTF-8, and confirms the child receives it
/// byte-for-byte. Bypasses the JSON/stdout probe pipeline the other tests
/// use — JSON text cannot carry arbitrary non-UTF-8 bytes — by having the
/// child write the raw bytes straight to a file instead.
async fn assert_non_utf8_value_passes_through(var: &'static str) {
    use std::os::unix::ffi::OsStringExt;

    let raw_bytes: Vec<u8> = vec![b'X', 0xFF, 0xFE, b'Y'];
    let raw_value = std::ffi::OsString::from_vec(raw_bytes.clone());
    let previous = std::env::var_os(var);
    // SAFETY: the caller (a per-variable allowlist test) owns `var`
    // exclusively within this binary; see `EnvVarGuard`'s doc. This runs
    // strictly after that test's own `EnvVarGuard` for the same key has
    // already been set and dropped (sequential, single-threaded within one
    // `#[tokio::test]` function), so there is no overlap.
    unsafe { std::env::set_var(var, &raw_value) };

    let out_path = temp_path("non-utf8-out");
    let script_path = temp_path("non-utf8-script");

    let mut script = String::new();
    script.push_str("#!/bin/sh\nprintf '%s' \"$");
    script.push_str(var);
    script.push_str("\" > '");
    script.push_str(&out_path.display().to_string());
    script.push_str(
        "'\nwhile IFS= read -r line; do\n  case \"$line\" in\n    *'\"method\":\"initialize\"'*)\n      rest=${line#*\\\"id\\\":}\n      id=${rest%%,*}\n",
    );
    script.push_str(
        "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"serverInfo\":{\"name\":\"non-utf8-probe\",\"version\":\"0.0\"}}}\\n' \"$id\"\n      ;;\n    *) : ;;\n  esac\ndone\n",
    );
    std::fs::write(&script_path, script).expect("write non-utf8 probe script");

    let result = McpClient::connect(
        "/bin/sh",
        &[script_path.display().to_string()],
        &HashMap::new(),
    )
    .await;

    // SAFETY: see above.
    match &previous {
        Some(v) => unsafe { std::env::set_var(var, v) },
        None => unsafe { std::env::remove_var(var) },
    }

    let mut client = result.expect("handshake should succeed");
    client.shutdown().await;
    let _ = std::fs::remove_file(&script_path);

    let seen = std::fs::read(&out_path).expect("read raw output file");
    let _ = std::fs::remove_file(&out_path);

    assert_eq!(
        seen, raw_bytes,
        "{var}: a non-UTF-8 allowlisted env value must reach the child byte-for-byte"
    );
}

// --- Diagnosability: a too-tight allowlist must fail loud, not silent ------
//
// If a server genuinely needs a variable this allowlist doesn't carry, it
// may exit immediately instead of completing the handshake. The daemon must
// report *why* (the exit status) rather than a generic protocol error, so
// the honest-state settings/KCM panel can show something a person can act
// on. See `StdioTransport::enrich_with_exit_status`.

/// A server that reads the initialize request and then exits immediately
/// (simulating a crash on a missing dependency/env var) must be reported by
/// its exit status, not the generic "server closed stdout" message.
#[tokio::test]
async fn server_exiting_before_handshake_reports_its_exit_status() {
    let script = temp_path("exits-before-handshake");
    std::fs::write(
        &script,
        r#"#!/bin/sh
# Consume the initialize request, then exit without replying - simulating a
# server that crashes on startup (e.g. a missing environment variable).
read -r line
exit 7
"#,
    )
    .expect("write fake exiting server script");

    let result =
        McpClient::connect("/bin/sh", &[script.display().to_string()], &HashMap::new()).await;

    let _ = std::fs::remove_file(&script);

    match result {
        Ok(_) => panic!("expected the handshake to fail, but connect succeeded"),
        Err(McpError::UnexpectedResponse(msg)) => {
            assert!(
                msg.contains("exited with status 7"),
                "expected the exit status in the error, got: {msg}"
            );
            assert_ne!(
                msg, "MCP server closed stdout",
                "the generic message must have been replaced with the exit status"
            );
        }
        Err(other) => {
            panic!("expected McpError::UnexpectedResponse naming the exit status, got: {other}")
        }
    }
}

/// `enrich_with_exit_status`'s wait for the child's exit status must itself
/// be bounded. `round_trip` (which this runs inside) backs every
/// post-handshake `tools/call` too, and that path has no outer timeout the
/// way the initial handshake does — a *server* that closes stdout but keeps
/// running and ignores `SIGTERM` would hang a live tool call indefinitely
/// against an unbounded wait. Only `EXIT_STATUS_WAIT` stands between "closed
/// stdout" and a hung daemon here.
///
/// Asserts both that the call returns within a bounded window (comfortably
/// above `EXIT_STATUS_WAIT` so this does not flake under load, but well
/// under the outer handshake timeout so it is *this* bound doing the work)
/// and that the result is the original generic message, not a fabricated
/// exit status — proving the fallback path, not just "it eventually
/// returns".
#[tokio::test]
async fn server_closing_stdout_without_exiting_falls_back_within_a_bounded_window() {
    let script = temp_path("closes-stdout-stays-alive");
    std::fs::write(
        &script,
        r#"#!/bin/sh
# Consume the initialize request, close stdout (triggers the "closed
# stdout" error on the client side), then keep running and ignore SIGTERM -
# only SIGKILL (sent unconditionally on client teardown, DS-2) can end this.
read -r line
exec 1>&-
trap '' TERM
while true; do sleep 1; done
"#,
    )
    .expect("write stdout-closing-but-alive probe script");

    let started = std::time::Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(25),
        McpClient::connect("/bin/sh", &[script.display().to_string()], &HashMap::new()),
    )
    .await
    .expect("connect must return within the outer test bound, not hang forever");
    let elapsed = started.elapsed();

    let _ = std::fs::remove_file(&script);

    match result {
        Ok(_) => panic!("expected the handshake to fail, but connect succeeded"),
        Err(McpError::UnexpectedResponse(msg)) => {
            assert_eq!(
                msg, "MCP server closed stdout",
                "a child that never exits must fall back to the original generic \
                 message, not a fabricated exit status"
            );
        }
        Err(other) => panic!("expected McpError::UnexpectedResponse, got: {other}"),
    }
    assert!(
        elapsed < Duration::from_secs(20),
        "the bounded wait for the child's exit status must not run away; took {elapsed:?}"
    );
}
