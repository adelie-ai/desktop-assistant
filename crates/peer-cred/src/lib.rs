//! Kernel-attested peer-credential lookup over a connected `UnixStream`.
//!
//! `SO_PEERCRED` is the kernel's record of whoever opened the other end of a
//! Unix-domain socket — an identity the peer cannot forge. Tokio's
//! `UnixStream::peer_cred` wraps the socket option directly; we only have to map
//! the UID to a username via `getpwuid_r`.
//!
//! This is the authentication primitive for *local* transports: on a UDS the
//! peer UID is the auth, so no bearer token is required (see issue #407). The
//! logic previously lived inside `jwt-minter`; it was relocated here so the
//! UDS server can authenticate by peer-cred and the standalone minter can be
//! retired.

use std::ffi::CStr;

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
    let cred = stream
        .peer_cred()
        .context("failed to read SO_PEERCRED from peer socket")?;
    let uid = cred.uid();
    let username =
        username_for_uid(uid)?.ok_or_else(|| anyhow!("no passwd entry for uid {uid}"))?;
    Ok(PeerIdentity { uid, username })
}

/// Look up the username for `uid` via `getpwuid_r`.
///
/// Returns `Ok(None)` when the UID has no matching entry rather than an
/// error so callers can distinguish "system call failed" from "no such user".
pub fn username_for_uid(uid: u32) -> anyhow::Result<Option<String>> {
    // Start with a comfortable buffer and grow on ERANGE.
    let mut buf_size = sysconf_or(libc::_SC_GETPW_R_SIZE_MAX, 1024).max(1024);
    let mut buf: Vec<libc::c_char> = vec![0; buf_size];
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut result: *mut libc::passwd = std::ptr::null_mut();

    loop {
        // SAFETY: `&mut pwd` is a valid `passwd*` for the lifetime of the
        // call; `buf` is a valid writable buffer of `buf_size` bytes; we
        // pass a non-null `result` out-pointer per the getpwuid_r contract.
        let rc = unsafe {
            libc::getpwuid_r(
                uid as libc::uid_t,
                &mut pwd,
                buf.as_mut_ptr(),
                buf_size,
                &mut result,
            )
        };
        if rc == 0 {
            if result.is_null() {
                return Ok(None);
            }
            // SAFETY: `pwd.pw_name` is a NUL-terminated C string owned by
            // `buf` (filled by `getpwuid_r`); we copy it into an owned
            // `String` before `buf` is dropped.
            let name = unsafe { CStr::from_ptr(pwd.pw_name) }
                .to_str()
                .context("passwd entry username is not valid UTF-8")?
                .to_string();
            return Ok(Some(name));
        }
        if rc == libc::ERANGE {
            buf_size = buf_size.saturating_mul(2);
            if buf_size > 1 << 20 {
                return Err(anyhow!(
                    "getpwuid_r requires implausibly large buffer (>1MiB)"
                ));
            }
            buf.resize(buf_size, 0);
            continue;
        }
        return Err(std::io::Error::from_raw_os_error(rc))
            .with_context(|| format!("getpwuid_r({uid}) failed"));
    }
}

/// The UID of the current process.
pub fn current_uid() -> u32 {
    // SAFETY: `getuid` is documented to always succeed and to have no
    // thread-safety concerns; it takes no arguments.
    unsafe { libc::getuid() as u32 }
}

fn sysconf_or(name: libc::c_int, fallback: usize) -> usize {
    // SAFETY: `sysconf` is a libc query function with no thread-safety
    // concerns; it accepts the integer constant `name`.
    let value = unsafe { libc::sysconf(name) };
    if value <= 0 { fallback } else { value as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_uid_resolves_to_a_username() {
        // The process's own UID must have a passwd entry with a non-empty name.
        let name = username_for_uid(current_uid())
            .expect("getpwuid_r should succeed for the current uid")
            .expect("the current uid must have a passwd entry");
        assert!(!name.is_empty(), "resolved username must be non-empty");
    }

    #[test]
    fn implausible_uid_has_no_entry() {
        // A UID near u32::MAX is reserved/unused on real systems, so the lookup
        // must succeed at the syscall level but report "no such user" (None),
        // not error.
        let resolved = username_for_uid(u32::MAX - 1).expect("syscall should not fail");
        assert!(
            resolved.is_none(),
            "an unused uid should resolve to None, got {resolved:?}"
        );
    }

    #[tokio::test]
    async fn peer_identity_of_a_socketpair_is_the_current_user() {
        // Both ends of a socketpair belong to this process, so the peer cred
        // read off one end must be this process's own uid + username.
        let (a, _b) = UnixStream::pair().expect("socketpair");
        let id = extract_peer_identity(&a).expect("peer identity");
        assert_eq!(id.uid, current_uid());
        assert_eq!(
            id.username,
            username_for_uid(current_uid()).unwrap().unwrap()
        );
    }
}
