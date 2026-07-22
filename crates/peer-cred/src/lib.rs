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
    /// Kernel-attested UID of the connecting peer.
    pub uid: u32,
    /// The peer's login name (`pw_name`), resolved from the UID.
    pub username: String,
    /// The peer's real / display name from the GECOS field (`pw_gecos`), when
    /// the passwd entry carries one. Best-effort display hint (#558): it lets a
    /// local UDS client that could not report its own client context (e.g. the
    /// KDE FFI client) still ground the prompt with the user's name. `None` when
    /// the entry has no GECOS name.
    pub real_name: Option<String>,
    /// The peer's home directory (`pw_dir`), when the passwd entry carries one.
    /// Best-effort display hint (#558), same use as `real_name`.
    pub home_dir: Option<String>,
}

/// Read the kernel-attested peer credentials from `stream` and resolve the UID
/// to its passwd entry (login name plus best-effort GECOS name / home dir) via
/// `getpwuid_r`.
pub fn extract_peer_identity(stream: &UnixStream) -> anyhow::Result<PeerIdentity> {
    let cred = stream
        .peer_cred()
        .context("failed to read SO_PEERCRED from peer socket")?;
    let uid = cred.uid();
    let info = passwd_for_uid(uid)?.ok_or_else(|| anyhow!("no passwd entry for uid {uid}"))?;
    Ok(PeerIdentity {
        uid,
        username: info.username,
        real_name: info.real_name,
        home_dir: info.home_dir,
    })
}

/// The subset of a passwd entry a [`PeerIdentity`] carries.
struct PasswdInfo {
    username: String,
    real_name: Option<String>,
    home_dir: Option<String>,
}

/// The real / display-name component of a GECOS string: its first comma-
/// separated field, trimmed, or `None` when that field is blank. GECOS is a
/// comma-separated list (full name, office, phones); the first field is the
/// display name. Pure so it is unit-tested without touching the passwd db.
fn real_name_from_gecos(_gecos: &str) -> Option<String> {
    // Spec stub — see the implementation commit.
    None
}

/// Look up the passwd entry for `uid` via `getpwuid_r`, extracting the login
/// name plus the best-effort GECOS real name and home directory.
///
/// Returns `Ok(None)` when the UID has no matching entry rather than an error so
/// callers can distinguish "system call failed" from "no such user".
fn passwd_for_uid(uid: u32) -> anyhow::Result<Option<PasswdInfo>> {
    // Spec stub — real GECOS / home extraction lands in the implementation
    // commit. For now only the username is resolved.
    Ok(username_for_uid(uid)?.map(|username| PasswdInfo {
        username,
        real_name: real_name_from_gecos(""),
        home_dir: None,
    }))
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

    #[test]
    fn real_name_from_gecos_takes_the_first_comma_field() {
        // GECOS is comma-separated (full name, office, phones); only the first
        // field is the display name, and it is trimmed.
        assert_eq!(
            real_name_from_gecos("Ada Lovelace,Room 1,x1234"),
            Some("Ada Lovelace".to_string())
        );
        assert_eq!(
            real_name_from_gecos("Ada Lovelace"),
            Some("Ada Lovelace".to_string())
        );
        assert_eq!(
            real_name_from_gecos("  Grace Hopper  ,office"),
            Some("Grace Hopper".to_string())
        );
    }

    #[test]
    fn real_name_from_gecos_absent_when_first_field_is_blank() {
        // A blank or all-comma GECOS has no display name.
        assert_eq!(real_name_from_gecos(""), None);
        assert_eq!(real_name_from_gecos(",,,"), None);
        assert_eq!(real_name_from_gecos("   "), None);
        assert_eq!(real_name_from_gecos("  ,x"), None);
    }

    #[tokio::test]
    async fn peer_identity_carries_home_dir_for_current_user() {
        // #558: the peer identity now also reports the passwd home dir (and, when
        // present, the GECOS real name) so a local client that cannot report its
        // own context still grounds the prompt. The current process's account has
        // a home dir (pw_dir); real_name is optional (GECOS may be empty), so we
        // only check it is well-formed when present.
        let (a, _b) = UnixStream::pair().expect("socketpair");
        let id = extract_peer_identity(&a).expect("peer identity");
        let home = id
            .home_dir
            .expect("the current user's passwd entry has a home dir (pw_dir)");
        assert!(!home.is_empty(), "home_dir must be non-empty when reported");
        if let Some(name) = id.real_name {
            assert_eq!(name.trim(), name, "real_name must be pre-trimmed");
            assert!(
                !name.is_empty(),
                "real_name must be non-empty when reported"
            );
        }
    }
}
