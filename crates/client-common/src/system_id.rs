//! Client-side read of the stable per-machine **system id** for tool-locality
//! co-location (issue #248).
//!
//! The [`Connector`](crate::Connector) includes this id in the connect
//! handshake so the daemon can compare it to its own and decide co-location
//! exactly — same machine ⇒ co-located, **even over WebSocket** — instead of
//! relying on the coarse transport heuristic (#243). All clients (voice/gtk/tui)
//! go through the Connector, so they get this for free with no client-repo
//! changes.
//!
//! The resolution strategy is intentionally the **same** as the daemon's
//! `desktop_assistant_core::system_id`: prefer Linux `/etc/machine-id`, else a
//! UUID generated once and persisted under `$XDG_DATA_HOME/adelie/system-id`.
//! Keeping the two in lockstep is what makes the daemon-vs-client comparison
//! meaningful on a single machine. It is duplicated here (rather than depending
//! on `core`) for the same reason `uds_client` re-implements the wire framing:
//! so a client binary doesn't link the daemon/domain stack just to read ~30
//! lines. The logic is small and stable.
//!
//! The id is a **co-location/routing hint, not a trust boundary** (#248): the
//! daemon never gates privilege on it (auth remains the JWT). See the daemon-side
//! module docs for the full security rationale.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The canonical Linux per-host machine id.
const ETC_MACHINE_ID: &str = "/etc/machine-id";

/// Process-wide cache: the id is stable for the life of the host, so resolve it
/// at most once.
static CACHED_ID: OnceLock<Option<String>> = OnceLock::new();

fn normalize(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `$XDG_DATA_HOME/adelie`, falling back to `$HOME/.local/share/adelie` — the
/// machine-local directory the generated fallback id is persisted in.
fn fallback_dir() -> Option<PathBuf> {
    let data_home = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
        })?;
    Some(data_home.join("adelie"))
}

/// Read the persisted fallback id from `dir/system-id`, or generate + persist a
/// fresh UUID when absent/empty, reusing it thereafter. `dir` is explicit so
/// tests can drive both branches against a temp dir.
fn read_or_create_fallback_id(dir: &Path) -> std::io::Result<String> {
    let path = dir.join("system-id");
    if let Ok(contents) = std::fs::read_to_string(&path)
        && let Some(id) = normalize(&contents)
    {
        return Ok(id);
    }
    let id = uuid::Uuid::new_v4().to_string();
    std::fs::create_dir_all(dir)?;
    std::fs::write(&path, &id)?;
    Ok(id)
}

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
                "failed to read/create the fallback system-id; the daemon will fall back to the \
                 transport co-location heuristic"
            );
            None
        }
    }
}

/// The local machine's stable system id, resolved once and cached (issue #248).
/// `None` when neither `/etc/machine-id` nor a writable data dir is available —
/// the handshake then omits the id and the daemon falls back to the transport
/// heuristic.
pub fn local_system_id() -> Option<String> {
    CACHED_ID.get_or_init(resolve_system_id).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_trims_and_rejects_empty() {
        assert_eq!(normalize("abc\n").as_deref(), Some("abc"));
        assert_eq!(normalize("   "), None);
    }

    #[test]
    fn fallback_reads_existing_file() {
        let dir = std::env::temp_dir().join(format!("adelie-cc-sysid-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("system-id"), "fixed-id-42\n").unwrap();
        assert_eq!(read_or_create_fallback_id(&dir).unwrap(), "fixed-id-42");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fallback_generates_persists_and_reuses() {
        let dir = std::env::temp_dir().join(format!("adelie-cc-sysid-{}", uuid::Uuid::new_v4()));
        let first = read_or_create_fallback_id(&dir).unwrap();
        assert!(!first.is_empty());
        assert!(dir.join("system-id").exists());
        let second = read_or_create_fallback_id(&dir).unwrap();
        assert_eq!(first, second, "persisted id must be reused");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn local_system_id_is_stable() {
        assert_eq!(local_system_id(), local_system_id());
    }
}
