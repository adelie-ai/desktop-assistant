//! Peer-credential lookup over a connected `UnixStream`.

use anyhow::{Context, anyhow};
use tokio::net::UnixStream;

/// The OS identity of a connected peer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerIdentity {
    pub uid: u32,
    pub username: String,
}

/// Read the kernel-attested peer credentials from `stream` and resolve
/// the UID to a username via `getpwuid_r`.
pub fn extract_peer_identity(stream: &UnixStream) -> anyhow::Result<PeerIdentity> {
    // TODO(#101): real impl.
    let _ = stream;
    Err(anyhow!("extract_peer_identity: not implemented"))
}

/// Look up the username for `uid` via `getpwuid_r`.
///
/// Returns `Ok(None)` when the UID has no matching entry rather than an
/// error so callers can distinguish "system call failed" from "no such user".
pub fn username_for_uid(uid: u32) -> anyhow::Result<Option<String>> {
    let _ = uid;
    Err(anyhow!("username_for_uid: not implemented"))
}

/// The UID of the current process.
pub fn current_uid() -> u32 {
    // SAFETY: `getuid` is a thread-safe libc call that takes no arguments
    // and cannot fail (per POSIX).
    unsafe { libc::getuid() as u32 }
}

/// Helper for tests that need to confirm we resolved *some* username for
/// the current process.
pub fn _suppress_unused() {
    let _ = username_for_uid;
}

#[allow(dead_code)]
fn _ensure_context_used() -> anyhow::Result<()> {
    Err::<(), _>(anyhow!("x")).context("y")
}
