//! Stable per-machine **system id** for tool-locality co-location (issue #248).
//!
//! Phase 1 (#243/#245) inferred co-location from the connection's
//! [`crate::domain::TransportKind`] (UDS/D-Bus ⇒ same machine, WebSocket ⇒
//! possibly remote). That heuristic is coarse: a WebSocket connection that
//! happens to terminate on the daemon's *own* machine is wrongly treated as
//! remote.
//!
//! Phase 2 replaces the heuristic as the **primary** signal with an exact
//! per-machine id that both the daemon and each client read. When the client's
//! reported id equals the daemon's own id they are the same machine ⇒
//! co-located, **even over WebSocket**. The transport heuristic stays as the
//! fallback for clients that don't report an id (older clients), so Phase-1
//! behaviour is preserved for them.
//!
//! ## What makes a good id
//!
//! - **Machine-local, not user-/network-scoped.** Two hosts must never share
//!   it (that would falsely co-locate them); the same host must keep it stable
//!   across reboots and processes (so co-location is consistent).
//! - Linux **`/etc/machine-id`** fits exactly: a 128-bit hex id assigned at
//!   install/first-boot, host-local, not synced across machines. It's the
//!   preferred source.
//! - When it's absent (non-systemd, containers without it, other platforms) we
//!   **generate a UUID once and persist it** to a machine-local app path
//!   (`$XDG_DATA_HOME/adelie/system-id`, falling back to `~/.local/share`),
//!   reusing it thereafter. That file is per-machine by construction (it lives
//!   on local storage, not a synced/home-roamed location in the common case).
//!
//! ## Security
//!
//! The id is a **co-location/routing hint, not a trust boundary** (#248). The
//! client self-reports it; the daemon never gates privilege or access on it
//! (auth remains the JWT). The worst a spoofed id can do is mislabel a tool's
//! locality or make the daemon prefer its own server-side tools — no
//! escalation, since server tools always run on the daemon and the client can
//! still only call tools it is authorized for.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The canonical Linux per-host machine id.
const ETC_MACHINE_ID: &str = "/etc/machine-id";

/// Process-wide cache of the resolved local system id. Resolved once (the id is
/// stable for the life of the host), then reused — both the client read and the
/// daemon's own read go through [`local_system_id`].
static CACHED_ID: OnceLock<Option<String>> = OnceLock::new();

/// Normalize a raw id read from a file or generated: trim surrounding
/// whitespace/newlines and reject the empty string. `/etc/machine-id` is a
/// single hex line with a trailing newline, so trimming is required.
fn normalize(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Resolve the directory the generated fallback id is persisted in:
/// `$XDG_DATA_HOME/adelie`, falling back to `$HOME/.local/share/adelie`. Kept
/// machine-local on purpose (see module docs). Mirrors the client-side
/// `default_ca_cert_path` resolution so the layout is consistent.
fn fallback_dir() -> Option<PathBuf> {
    let data_home = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
        })?;
    Some(data_home.join("adelie"))
}

/// Read the persisted fallback id from `dir/system-id`, or generate + persist a
/// fresh UUID when the file is absent/empty, reusing it on subsequent reads.
///
/// Split out from [`local_system_id`] with an explicit `dir` so tests can drive
/// both branches (present file / generate-and-persist) against a temp dir
/// without depending on the host's real data directory.
fn read_or_create_fallback_id(dir: &Path) -> io::Result<String> {
    let path = dir.join("system-id");

    // Reuse the persisted id when present and non-empty.
    if let Ok(contents) = std::fs::read_to_string(&path)
        && let Some(id) = normalize(&contents)
    {
        return Ok(id);
    }

    // Generate a fresh UUID and persist it for next time.
    let id = uuid::Uuid::new_v4().to_string();
    std::fs::create_dir_all(dir)?;
    std::fs::write(&path, &id)?;
    Ok(id)
}

/// Resolve the local machine's system id without the process-wide cache.
///
/// Resolution order:
///   1. `/etc/machine-id` (preferred — the canonical per-host id).
///   2. A UUID generated once and persisted to `$XDG_DATA_HOME/adelie/system-id`
///      (or `~/.local/share/adelie/system-id`), reused thereafter.
///   3. `None` when neither source is available (no `/etc/machine-id`, no home
///      directory, and the fallback file can't be written) — the caller then
///      falls back to the transport heuristic, preserving Phase-1 behaviour.
fn resolve_system_id() -> Option<String> {
    if let Ok(contents) = std::fs::read_to_string(ETC_MACHINE_ID)
        && let Some(id) = normalize(&contents)
    {
        return Some(id);
    }

    let dir = fallback_dir()?;
    match read_or_create_fallback_id(&dir) {
        Ok(id) => Some(id),
        Err(e) => {
            tracing::warn!(
                error = %e,
                dir = %dir.display(),
                "failed to read/create the fallback system-id; falling back to the transport \
                 co-location heuristic"
            );
            None
        }
    }
}

/// The local machine's stable system id, resolved once and cached for the life
/// of the process (issue #248).
///
/// Prefers `/etc/machine-id`; otherwise a generated UUID persisted under
/// `$XDG_DATA_HOME/adelie/system-id`. Returns `None` only when neither source
/// is available — co-location then falls back to the transport heuristic. Cheap
/// and safe to call from anywhere (the work happens at most once).
pub fn local_system_id() -> Option<String> {
    CACHED_ID.get_or_init(resolve_system_id).clone()
}

/// Decide co-location from a pair of system ids (issue #248).
///
/// Returns:
///   - `Some(true)`  when both ids are present (after trimming) and equal — the
///     client is on the *same* machine as the daemon, **even over WebSocket**.
///   - `Some(false)` when both are present and differ — distinct machines.
///   - `None`        when either id is absent/blank — there is no authoritative
///     result, so the caller falls back to the transport heuristic (the
///     Phase-1, #243, behaviour for older clients).
///
/// This is a pure comparison (no I/O) so the daemon can call it per connection
/// after reading its own id once. The id is a routing **hint**, not a trust
/// boundary (#248) — see the module docs.
pub fn co_location_from_ids(daemon_id: Option<&str>, client_id: Option<&str>) -> Option<bool> {
    let daemon = daemon_id.map(str::trim).filter(|s| !s.is_empty())?;
    let client = client_id.map(str::trim).filter(|s| !s.is_empty())?;
    Some(daemon == client)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_trims_and_rejects_empty() {
        assert_eq!(normalize("abc123\n").as_deref(), Some("abc123"));
        assert_eq!(normalize("  deadbeef  ").as_deref(), Some("deadbeef"));
        assert_eq!(normalize(""), None);
        assert_eq!(normalize("   \n\t "), None);
    }

    /// Fallback path, branch 1: an existing `system-id` file is read verbatim
    /// (trimmed), not regenerated — so the id is stable across reads.
    #[test]
    fn fallback_reads_existing_file() {
        let dir = std::env::temp_dir().join(format!("adelie-sysid-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("system-id"), "preexisting-id-9999\n").unwrap();

        let id = read_or_create_fallback_id(&dir).unwrap();
        assert_eq!(id, "preexisting-id-9999");

        // A second read returns the same id and does not overwrite the file.
        let again = read_or_create_fallback_id(&dir).unwrap();
        assert_eq!(again, "preexisting-id-9999");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Fallback path, branch 2: no file yet ⇒ generate + persist, and a second
    /// call reuses the persisted id rather than generating a new one.
    #[test]
    fn fallback_generates_and_persists_then_reuses() {
        let dir = std::env::temp_dir().join(format!("adelie-sysid-test-{}", uuid::Uuid::new_v4()));
        // Intentionally do NOT create the dir — the helper must create it.

        let first = read_or_create_fallback_id(&dir).unwrap();
        assert!(!first.is_empty(), "a fresh id must be generated");
        assert!(
            dir.join("system-id").exists(),
            "the generated id must be persisted"
        );

        let second = read_or_create_fallback_id(&dir).unwrap();
        assert_eq!(
            first, second,
            "the persisted id must be reused, not regenerated"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An empty/whitespace-only persisted file is treated as absent and
    /// regenerated, so a truncated write can't yield an empty id.
    #[test]
    fn fallback_regenerates_when_file_is_empty() {
        let dir = std::env::temp_dir().join(format!("adelie-sysid-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("system-id"), "  \n").unwrap();

        let id = read_or_create_fallback_id(&dir).unwrap();
        assert!(
            !id.is_empty(),
            "empty file must be regenerated into a real id"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `local_system_id` resolves *something* on this host (it has
    /// `/etc/machine-id` or a home dir) and is stable across calls (cached).
    #[test]
    fn local_system_id_is_stable() {
        let a = local_system_id();
        let b = local_system_id();
        assert_eq!(a, b, "the cached id must be identical across calls");
    }

    #[test]
    fn co_location_true_on_id_match() {
        assert_eq!(
            co_location_from_ids(Some("abc123"), Some("abc123")),
            Some(true)
        );
        // Trimming makes a newline-suffixed /etc/machine-id match a clean id.
        assert_eq!(
            co_location_from_ids(Some("abc123\n"), Some("  abc123 ")),
            Some(true)
        );
    }

    #[test]
    fn co_location_false_on_id_mismatch() {
        assert_eq!(
            co_location_from_ids(Some("daemon-id"), Some("client-id")),
            Some(false)
        );
    }

    #[test]
    fn co_location_none_when_either_id_missing_or_blank() {
        // Older client (no id) ⇒ no authoritative result ⇒ fall back to transport.
        assert_eq!(co_location_from_ids(Some("daemon-id"), None), None);
        assert_eq!(co_location_from_ids(Some("daemon-id"), Some("   ")), None);
        // Daemon couldn't resolve its own id ⇒ also defer to the heuristic.
        assert_eq!(co_location_from_ids(None, Some("client-id")), None);
        assert_eq!(co_location_from_ids(None, None), None);
    }
}
