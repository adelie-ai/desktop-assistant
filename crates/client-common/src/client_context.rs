//! Best-effort resolution of the client's self-reported device/user context
//! (#549) and the gate that decides whether to attach it to the connect
//! handshake.
//!
//! Every field is resolved **independently** and best-effort: whatever the host
//! provides is reported and anything that can't be determined is omitted. This
//! mirrors the wire type's fail-closed posture ([`ClientContext`] is untrusted
//! display data, not a trust boundary) and lets a headless or minimal host
//! degrade to a smaller (possibly empty) context rather than an error.
//!
//! Raw environment / syscall reads live in thin wrappers; the [`ClientContext`]
//! is assembled by the pure [`assemble_client_context`], which is unit-tested by
//! feeding it resolved strings so tests never depend on the host's real
//! `$USER` / timezone / passwd entry.

use std::ffi::CStr;

use desktop_assistant_api_model::ClientContext;

/// Resolve the client's best-effort [`ClientContext`] (#549) from the local
/// environment. Every field is resolved independently and any that cannot be
/// determined is omitted; resolution never fails or panics.
///
/// - `real_name`: the current user's GECOS full name (`getpwuid(getuid())`).
/// - `username`: `$USER`, else `$LOGNAME`.
/// - `home_dir`: `$HOME`.
/// - `hostname`: the kernel hostname (see [`local_hostname`]).
/// - `timezone`: the IANA zone name via `iana-time-zone`.
/// - `os`: the Linux distro `PRETTY_NAME`, else the OS family.
pub fn resolve_client_context() -> ClientContext {
    assemble_client_context(
        resolve_real_name(),
        resolve_username(),
        resolve_home_dir(),
        local_hostname(),
        resolve_timezone(),
        resolve_os(),
    )
}

/// The client context to attach at connect/reconnect. `Some` only when the
/// `share_client_context` setting is on **and** the resolved context has at
/// least one field; an all-absent context collapses to `None` (equivalent to
/// attaching nothing).
///
/// The resolver is a closure so it is not run at all when sharing is disabled —
/// the "off" setting performs no environment or passwd reads.
pub(crate) fn context_to_attach(
    share_client_context: bool,
    resolve: impl FnOnce() -> ClientContext,
) -> Option<ClientContext> {
    share_client_context
        .then(resolve)
        .filter(|ctx| !ctx.is_empty())
}

/// Assemble a [`ClientContext`] from already-resolved raw field values, dropping
/// any that are absent or blank. Pure — no environment or syscalls — so it is
/// unit-testable independent of the host: each value is trimmed and an empty
/// result becomes `None`, so a present-but-blank source is treated as absent.
fn assemble_client_context(
    real_name: Option<String>,
    username: Option<String>,
    home_dir: Option<String>,
    hostname: Option<String>,
    timezone: Option<String>,
    os: Option<String>,
) -> ClientContext {
    fn clean(value: Option<String>) -> Option<String> {
        value
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }
    ClientContext {
        real_name: clean(real_name),
        username: clean(username),
        home_dir: clean(home_dir),
        hostname: clean(hostname),
        timezone: clean(timezone),
        os: clean(os),
    }
}

/// `$USER`, falling back to `$LOGNAME`. A present-but-blank `$USER` is treated as
/// absent so the `$LOGNAME` fallback still applies.
fn resolve_username() -> Option<String> {
    env_nonempty("USER").or_else(|| env_nonempty("LOGNAME"))
}

/// `$HOME`. `dirs` is not in the workspace tree; `$HOME` is the portable source
/// and enough for a best-effort hint.
fn resolve_home_dir() -> Option<String> {
    env_nonempty("HOME")
}

/// The local IANA timezone name (e.g. `"Europe/London"`) via `iana-time-zone`.
fn resolve_timezone() -> Option<String> {
    iana_time_zone::get_timezone()
        .ok()
        .filter(|tz| !tz.trim().is_empty())
}

/// A short, friendly OS description: on Linux the distro's `PRETTY_NAME` from
/// `/etc/os-release` (e.g. `"Ubuntu 24.04 LTS"`), else the compile-time OS
/// family (`std::env::consts::OS`, e.g. `"macos"`).
fn resolve_os() -> Option<String> {
    os_release_pretty_name().or_else(|| {
        let os = std::env::consts::OS;
        (!os.is_empty()).then(|| os.to_string())
    })
}

/// The `PRETTY_NAME` value from `/etc/os-release`, unquoted. `None` when the file
/// is absent (non-Linux, containers without it) or has no such key.
fn os_release_pretty_name() -> Option<String> {
    let contents = std::fs::read_to_string("/etc/os-release").ok()?;
    contents.lines().find_map(|line| {
        let value = line.strip_prefix("PRETTY_NAME=")?;
        let value = value.trim().trim_matches('"').trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

/// Best-effort local hostname (#248 / #549): the Linux kernel hostname, then
/// `/etc/hostname`, then the `HOSTNAME` env var. `None` when none resolve — the
/// value is purely a display hint. Dependency-free, and shared by the #248
/// `host_label` and the #549 client context's `hostname` field.
pub(crate) fn local_hostname() -> Option<String> {
    let from_file = |path: &str| {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    from_file("/proc/sys/kernel/hostname")
        .or_else(|| from_file("/etc/hostname"))
        .or_else(|| env_nonempty("HOSTNAME"))
}

/// The current user's real / display name from the GECOS field of their passwd
/// entry. Best-effort: `None` when the entry or field is unavailable.
fn resolve_real_name() -> Option<String> {
    gecos_full_name(current_uid())
}

/// A trimmed, non-empty environment variable, or `None`.
fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The UID of the current process.
fn current_uid() -> libc::uid_t {
    // SAFETY: `getuid` takes no arguments, always succeeds, and has no
    // thread-safety concerns.
    unsafe { libc::getuid() }
}

/// The full-name (first, comma-separated) component of the GECOS field of the
/// passwd entry for `uid`, via `getpwuid_r`. Best-effort: `None` on any failure
/// or an empty result. Mirrors the daemon's `peer-cred` lookup but reads
/// `pw_gecos` instead of `pw_name` (GECOS is comma-separated: full name, then
/// office / phones — the first field is the display name).
fn gecos_full_name(uid: libc::uid_t) -> Option<String> {
    // Start with a comfortable buffer and grow on ERANGE, capped so a
    // misbehaving libc cannot drive an unbounded allocation.
    let mut buf_size = sysconf_or(libc::_SC_GETPW_R_SIZE_MAX, 1024).max(1024);
    loop {
        let mut buf: Vec<libc::c_char> = vec![0; buf_size];
        let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
        let mut result: *mut libc::passwd = std::ptr::null_mut();
        // SAFETY: `&mut pwd` is a valid `passwd*` for the call; `buf` is a
        // writable buffer of `buf_size` `c_char`s; `result` is a non-null
        // out-pointer, per the `getpwuid_r` contract.
        let rc =
            unsafe { libc::getpwuid_r(uid, &mut pwd, buf.as_mut_ptr(), buf_size, &mut result) };
        if rc == 0 {
            if result.is_null() || pwd.pw_gecos.is_null() {
                return None; // no passwd entry, or no GECOS field
            }
            // SAFETY: `pw_gecos` is a NUL-terminated C string owned by `buf`
            // (filled by `getpwuid_r`); we copy it out before `buf` is dropped.
            let gecos = unsafe { CStr::from_ptr(pwd.pw_gecos) }.to_str().ok()?;
            let name = gecos.split(',').next().unwrap_or_default().trim();
            return (!name.is_empty()).then(|| name.to_string());
        }
        if rc == libc::ERANGE && buf_size < (1 << 20) {
            buf_size = buf_size.saturating_mul(2);
            continue;
        }
        // Any other errno (or an implausibly large buffer request): give up
        // best-effort rather than surfacing an error the caller cannot act on.
        return None;
    }
}

/// `sysconf(name)`, or `fallback` when it is unavailable / non-positive.
fn sysconf_or(name: libc::c_int, fallback: usize) -> usize {
    // SAFETY: `sysconf` is a query function taking an integer name constant; no
    // thread-safety concerns.
    let value = unsafe { libc::sysconf(name) };
    if value <= 0 { fallback } else { value as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_ctx() -> ClientContext {
        ClientContext {
            real_name: Some("Ada Lovelace".into()),
            username: Some("ada".into()),
            home_dir: Some("/home/ada".into()),
            hostname: Some("analytical-engine".into()),
            timezone: Some("Europe/London".into()),
            os: Some("Ubuntu 24.04".into()),
        }
    }

    #[test]
    fn assemble_drops_absent_and_blank_fields_but_keeps_present_ones() {
        // A `None` field and a present-but-blank field are both treated as
        // absent; a present field is kept verbatim.
        let ctx = assemble_client_context(
            Some("Ada Lovelace".to_string()),
            None,
            Some("   ".to_string()), // blank -> dropped
            Some("analytical-engine".to_string()),
            None,
            Some(String::new()), // empty -> dropped
        );
        assert_eq!(ctx.real_name.as_deref(), Some("Ada Lovelace"));
        assert_eq!(ctx.username, None);
        assert_eq!(ctx.home_dir, None);
        assert_eq!(ctx.hostname.as_deref(), Some("analytical-engine"));
        assert_eq!(ctx.timezone, None);
        assert_eq!(ctx.os, None);
    }

    #[test]
    fn assemble_all_absent_yields_empty_context() {
        let ctx = assemble_client_context(None, None, None, None, None, None);
        assert!(ctx.is_empty());
    }

    #[test]
    fn assemble_trims_surrounding_whitespace() {
        let ctx = assemble_client_context(
            None,
            Some("  ada  ".to_string()),
            None,
            None,
            Some(" Europe/London ".to_string()),
            None,
        );
        assert_eq!(ctx.username.as_deref(), Some("ada"));
        assert_eq!(ctx.timezone.as_deref(), Some("Europe/London"));
    }

    #[test]
    fn context_to_attach_is_none_when_sharing_disabled() {
        // With the setting off the resolver must not even run (no env/syscall
        // reads) and nothing is attached.
        let called = std::cell::Cell::new(false);
        let out = context_to_attach(false, || {
            called.set(true);
            full_ctx()
        });
        assert_eq!(out, None);
        assert!(
            !called.get(),
            "resolver must not run when sharing is disabled"
        );
    }

    #[test]
    fn context_to_attach_returns_resolved_when_enabled() {
        assert_eq!(context_to_attach(true, full_ctx), Some(full_ctx()));
    }

    #[test]
    fn context_to_attach_drops_an_empty_resolved_context() {
        // An all-absent resolved context is equivalent to attaching nothing.
        assert_eq!(context_to_attach(true, ClientContext::default), None);
    }

    #[test]
    fn resolve_client_context_never_panics() {
        // Best-effort contract: whatever the host provides, resolution returns a
        // (possibly empty) context and never panics.
        let _ = resolve_client_context();
    }
}
