//! Optional Unix-group access gate.
//!
//! When the operator sets `--group <name>`, the minter resolves `<name>`
//! to a GID at startup (failing clean if no such group) and at request
//! time checks that the caller's UID is a member of that group via
//! `getgrouplist`.

use std::ffi::{CStr, CString};

use anyhow::{Context, anyhow};

/// A resolved group entry — name + GID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupGate {
    pub name: String,
    pub gid: u32,
}

/// Look up `name` via `getgrnam_r`. Returns `Ok(None)` when no such group
/// exists; `Err` only when the syscall itself fails.
pub fn resolve_group(name: &str) -> anyhow::Result<Option<GroupGate>> {
    let cname = CString::new(name).context("group name contains an embedded NUL")?;
    let mut buf_size = sysconf_or(libc::_SC_GETGR_R_SIZE_MAX, 1024).max(1024);
    let mut buf: Vec<libc::c_char> = vec![0; buf_size];
    let mut grp: libc::group = unsafe { std::mem::zeroed() };
    let mut result: *mut libc::group = std::ptr::null_mut();

    loop {
        // SAFETY: pointers are valid for the call's duration; `result` is
        // a valid out-pointer per the getgrnam_r contract.
        let rc = unsafe {
            libc::getgrnam_r(
                cname.as_ptr(),
                &mut grp,
                buf.as_mut_ptr(),
                buf_size,
                &mut result,
            )
        };
        if rc == 0 {
            if result.is_null() {
                return Ok(None);
            }
            // SAFETY: `grp.gr_name` is a NUL-terminated C string in `buf`;
            // we copy before `buf` is dropped.
            let resolved_name = unsafe { CStr::from_ptr(grp.gr_name) }
                .to_str()
                .context("group entry name is not valid UTF-8")?
                .to_string();
            return Ok(Some(GroupGate {
                name: resolved_name,
                gid: grp.gr_gid as u32,
            }));
        }
        if rc == libc::ERANGE {
            buf_size = buf_size.saturating_mul(2);
            if buf_size > 1 << 20 {
                return Err(anyhow!(
                    "getgrnam_r requires implausibly large buffer (>1MiB)"
                ));
            }
            buf.resize(buf_size, 0);
            continue;
        }
        return Err(std::io::Error::from_raw_os_error(rc))
            .with_context(|| format!("getgrnam_r({name:?}) failed"));
    }
}

/// Return the supplementary group list for `username` (plus `primary_gid`)
/// via `getgrouplist`. Mirrors `id -G <username>`.
pub fn grouplist_for(username: &str, primary_gid: u32) -> anyhow::Result<Vec<u32>> {
    let cname = CString::new(username).context("username contains an embedded NUL")?;
    let mut count: libc::c_int = 32;
    loop {
        let mut groups: Vec<libc::gid_t> = vec![0; count as usize];
        let mut ngroups = count;
        // SAFETY: `cname` is a valid NUL-terminated C string; `groups`
        // points to a writable buffer of `count` `gid_t`; `&mut ngroups`
        // is a valid out-pointer.
        let rc = unsafe {
            libc::getgrouplist(
                cname.as_ptr(),
                primary_gid as libc::gid_t,
                groups.as_mut_ptr(),
                &mut ngroups,
            )
        };
        if rc >= 0 {
            groups.truncate(ngroups.max(0) as usize);
            // `gid_t` is `u32` on Linux/glibc and the rest of the targets
            // we care about, so no cast is needed.
            return Ok(groups);
        }
        // -1 means the buffer was too small; ngroups is set to the
        // required size. Grow and retry, bounded.
        if ngroups <= count {
            return Err(anyhow!(
                "getgrouplist returned -1 without expanding ngroups"
            ));
        }
        if ngroups > 65_536 {
            return Err(anyhow!(
                "getgrouplist requires implausibly large buffer ({ngroups})"
            ));
        }
        count = ngroups;
    }
}

/// Pure predicate: does `target_gid` appear in `groups`?
pub fn uid_in_groups(target_gid: u32, groups: &[u32]) -> bool {
    groups.contains(&target_gid)
}

/// Primary GID of `uid` via `getpwuid_r`.
pub fn primary_gid_for_uid(uid: u32) -> anyhow::Result<Option<u32>> {
    let mut buf_size = sysconf_or(libc::_SC_GETPW_R_SIZE_MAX, 1024).max(1024);
    let mut buf: Vec<libc::c_char> = vec![0; buf_size];
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut result: *mut libc::passwd = std::ptr::null_mut();

    loop {
        // SAFETY: same contract as username_for_uid; see peer.rs.
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
            return Ok(Some(pwd.pw_gid as u32));
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

fn sysconf_or(name: libc::c_int, fallback: usize) -> usize {
    // SAFETY: `sysconf` is a libc query function with no thread-safety
    // concerns.
    let value = unsafe { libc::sysconf(name) };
    if value <= 0 { fallback } else { value as usize }
}
