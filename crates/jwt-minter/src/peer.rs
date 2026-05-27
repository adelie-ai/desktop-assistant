//! Peer-credential lookup over a connected `UnixStream`.
//!
//! `SO_PEERCRED` is the kernel-attested identity of whoever opened the
//! other end of the socket. Tokio's `UnixStream::peer_cred` wraps the
//! socket option directly; we only have to map the UID to a username via
//! `getpwuid_r` for the `sub` claim.

use std::ffi::CStr;

use anyhow::{anyhow, Context};
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
    let username = username_for_uid(uid)?
        .ok_or_else(|| anyhow!("no passwd entry for uid {uid}"))?;
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
