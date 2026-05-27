//! Optional Unix-group access gate.

use anyhow::anyhow;

/// A resolved group entry — name + GID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupGate {
    pub name: String,
    pub gid: u32,
}

/// Look up `name` via `getgrnam_r`. Returns `Ok(None)` when no such group
/// exists; `Err` only when the syscall itself fails.
pub fn resolve_group(name: &str) -> anyhow::Result<Option<GroupGate>> {
    let _ = name;
    Err(anyhow!("resolve_group: not implemented"))
}

/// Return the supplementary group list for `username` (plus `primary_gid`)
/// via `getgrouplist`.
pub fn grouplist_for(username: &str, primary_gid: u32) -> anyhow::Result<Vec<u32>> {
    let _ = (username, primary_gid);
    Err(anyhow!("grouplist_for: not implemented"))
}

/// Pure predicate: does `target_gid` appear in `groups`?
pub fn uid_in_groups(target_gid: u32, groups: &[u32]) -> bool {
    groups.contains(&target_gid)
}

/// Primary GID of `uid` via `getpwuid_r`.
pub fn primary_gid_for_uid(uid: u32) -> anyhow::Result<Option<u32>> {
    let _ = uid;
    Err(anyhow!("primary_gid_for_uid: not implemented"))
}
