//! Startup scanner for the on-disk skill library (#573).
//!
//! Walks the configured global skill roots (`<root>/<name>/SKILL.md`), parses
//! each skill with the pure `core::domain::skill` helpers, digests its
//! attachments into the integrity hash a blessing pins to, and produces the
//! [`IndexedSkill`] rows the daemon hands to `PgSkillIndexStore::reindex_global`.
//!
//! Unreadable or malformed skills are skipped with a warning — a bad skill must
//! never fail the whole scan (or block daemon startup). Earlier roots take
//! precedence on a name collision.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use desktop_assistant_core::domain::skill::{
    detect_kind, file_sha256_hex, parse_skill_md, skill_content_hash, trust_tier_from_source_type,
    validate_skill_name,
};
use desktop_assistant_core::domain::{AttachmentDigest, IndexedSkill, Locality};

const SKILL_FILE: &str = "SKILL.md";

/// Scan the configured **global** roots into the deduplicated set of global
/// skills (owner-less, `Locality::Daemon`), ready for `reindex_global`. Earlier
/// roots win on a name collision; malformed skills are skipped with a warning.
pub fn scan_global_roots(roots: &[PathBuf]) -> Vec<IndexedSkill> {
    scan_roots(roots, None, Locality::Daemon)
}

/// Scan the configured **user home** roots into the deduplicated set of skills
/// owned by `owner` (`Locality::Client`), ready for `reindex_for_owner`. Used on
/// a co-located single-user daemon so the user's `~/.agents/skills` etc. are
/// indexed as theirs.
pub fn scan_user_roots(roots: &[PathBuf], owner: &str) -> Vec<IndexedSkill> {
    scan_roots(roots, Some(owner), Locality::Client)
}

/// Shared scan: walk each root, stamping every skill with `owner`/`locality`,
/// deduplicating by name (earlier roots win).
fn scan_roots(roots: &[PathBuf], owner: Option<&str>, locality: Locality) -> Vec<IndexedSkill> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for root in roots {
        let lock = load_lock_source_types(root);
        for skill in scan_root(root, &lock, owner, locality) {
            if seen.insert(skill.name.clone()) {
                out.push(skill);
            }
        }
    }
    out
}

/// Scan one root (`<root>/<name>/SKILL.md`). An unreadable root yields nothing
/// (capability-off; the caller logs when no root resolved at all).
fn scan_root(
    root: &Path,
    lock: &HashMap<String, String>,
    owner: Option<&str>,
    locality: Locality,
) -> Vec<IndexedSkill> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() || !dir.join(SKILL_FILE).is_file() {
            continue;
        }
        let Some(name) = dir.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
            continue;
        };
        match scan_skill_dir(
            &dir,
            &name,
            lock.get(&name).map(String::as_str),
            root,
            owner,
            locality,
        ) {
            Ok(skill) => skills.push(skill),
            Err(e) => tracing::warn!(skill = %name, error = %e, "skipping malformed skill"),
        }
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Parse and digest a single skill directory into an [`IndexedSkill`], stamped
/// with the given `owner` and `locality`.
fn scan_skill_dir(
    dir: &Path,
    name: &str,
    source_type: Option<&str>,
    root: &Path,
    owner: Option<&str>,
    locality: Locality,
) -> anyhow::Result<IndexedSkill> {
    validate_skill_name(name).map_err(|e| anyhow::anyhow!("invalid skill name: {e}"))?;
    let md_path = dir.join(SKILL_FILE);
    let raw = std::fs::read_to_string(&md_path)?;
    let parsed = parse_skill_md(&raw).map_err(|e| anyhow::anyhow!("parse: {e}"))?;
    let (digests, attachments) = collect_attachments(dir)?;
    Ok(IndexedSkill {
        name: name.to_string(),
        description: parsed.frontmatter.description,
        kind: detect_kind(&parsed.body),
        disk_path: md_path.to_string_lossy().into_owned(),
        owner_user_id: owner.map(str::to_string),
        locality,
        content_hash: skill_content_hash(raw.as_bytes(), &digests),
        trust_tier: trust_tier_from_source_type(source_type),
        source: Some(root.to_string_lossy().into_owned()),
        tags: parsed.frontmatter.tags,
        attachments,
        body: parsed.body,
        metadata: parsed.frontmatter.metadata,
        // Presence is index state the reconcile pass stamps; a scanner only
        // ever reports what it just read off disk.
        present_on_disk: true,
        last_seen_at: None,
    })
}

/// Collect the sibling files traveling with a `SKILL.md` (excluding it), as
/// integrity digests and the sorted relative-path list.
fn collect_attachments(dir: &Path) -> anyhow::Result<(Vec<AttachmentDigest>, Vec<String>)> {
    let mut digests = Vec::new();
    for entry in walkdir::WalkDir::new(dir)
        .min_depth(1)
        .max_depth(4)
        .sort_by_file_name()
    {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(dir) else {
            continue;
        };
        let rel = rel.to_string_lossy().replace('\\', "/");
        if rel == SKILL_FILE {
            continue;
        }
        let bytes = std::fs::read(entry.path())?;
        digests.push(AttachmentDigest {
            sha256_hex: file_sha256_hex(&bytes),
            mode: file_mode(entry.path()),
            rel_path: rel,
        });
    }
    digests.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    let paths = digests.iter().map(|d| d.rel_path.clone()).collect();
    Ok((digests, paths))
}

#[cfg(unix)]
fn file_mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.permissions().mode())
        .unwrap_or(0)
}

#[cfg(not(unix))]
fn file_mode(_path: &Path) -> u32 {
    0
}

/// Load a `.skill-lock.json` (skill name -> `sourceType`) from beside or within
/// the root, for trust-tier derivation. Returns empty when absent/unparseable.
fn load_lock_source_types(root: &Path) -> HashMap<String, String> {
    let candidates = [
        root.parent().map(|p| p.join(".skill-lock.json")),
        Some(root.join(".skill-lock.json")),
    ];
    for cand in candidates.into_iter().flatten() {
        let Ok(text) = std::fs::read_to_string(&cand) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let mut map = HashMap::new();
        if let Some(skills) = value.get("skills").and_then(|s| s.as_object()) {
            for (name, meta) in skills {
                if let Some(st) = meta.get("sourceType").and_then(|s| s.as_str()) {
                    map.insert(name.clone(), st.to_string());
                }
            }
        }
        return map;
    }
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use desktop_assistant_core::domain::{Locality, SkillKind, TrustTier};
    use std::fs;
    use std::path::Path;

    /// Write `<root>/<name>/SKILL.md` with the given body.
    fn write_skill(root: &Path, name: &str, body: &str) {
        let dir = root.join(name);
        fs::create_dir_all(&dir).expect("mkdir skill");
        let md = format!("---\nname: {name}\ndescription: does {name}\n---\n{body}\n");
        fs::write(dir.join("SKILL.md"), md).expect("write SKILL.md");
    }

    fn write_attachment(root: &Path, skill: &str, rel: &str, contents: &str) {
        let path = root.join(skill).join(rel);
        fs::create_dir_all(path.parent().unwrap()).expect("mkdir attachment dir");
        fs::write(path, contents).expect("write attachment");
    }

    #[test]
    fn scans_skill_with_attachment() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_skill(root, "gh-stack", "# gh-stack\n\nprose about stacks");
        write_attachment(root, "gh-stack", "scripts/run.sh", "echo hi");

        let skills = scan_global_roots(&[root.to_path_buf()]);
        assert_eq!(skills.len(), 1);
        let s = &skills[0];
        assert_eq!(s.name, "gh-stack");
        assert_eq!(s.kind, SkillKind::Skill);
        assert_eq!(s.locality, Locality::Daemon);
        assert!(s.owner_user_id.is_none());
        assert_eq!(s.attachments, vec!["scripts/run.sh"]);
        assert_eq!(s.content_hash.len(), 64);
        assert!(s.disk_path.ends_with("gh-stack/SKILL.md"));
        assert_eq!(s.trust_tier, TrustTier::Local);
    }

    #[test]
    fn detects_workflow_kind_from_steps() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_skill(
            root,
            "invoicing",
            "Intro\n\n## Steps\n1. preview\n2. finalize\n",
        );
        let skills = scan_global_roots(&[root.to_path_buf()]);
        assert_eq!(skills[0].kind, SkillKind::Workflow);
    }

    #[test]
    fn content_hash_changes_when_attachment_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_skill(root, "deploy", "# deploy");
        write_attachment(root, "deploy", "scripts/run.sh", "echo one");
        let before = scan_global_roots(&[root.to_path_buf()])[0]
            .content_hash
            .clone();

        write_attachment(root, "deploy", "scripts/run.sh", "echo two (swapped)");
        let after = scan_global_roots(&[root.to_path_buf()])[0]
            .content_hash
            .clone();
        assert_ne!(
            before, after,
            "a swapped script must change the content hash"
        );
    }

    #[test]
    fn skips_malformed_and_missing_skill_md() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_skill(root, "good", "# good");
        // A directory with a malformed SKILL.md (no frontmatter).
        let bad = root.join("bad");
        fs::create_dir_all(&bad).unwrap();
        fs::write(bad.join("SKILL.md"), "no frontmatter here").unwrap();
        // A directory with no SKILL.md at all.
        fs::create_dir_all(root.join("empty")).unwrap();

        let skills = scan_global_roots(&[root.to_path_buf()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "good");
    }

    #[test]
    fn dedups_by_name_earlier_root_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let r1 = tmp.path().join("r1");
        let r2 = tmp.path().join("r2");
        write_skill(&r1, "dup", "# from r1");
        write_skill(&r2, "dup", "# from r2");

        let skills = scan_global_roots(&[r1.clone(), r2.clone()]);
        assert_eq!(skills.len(), 1);
        assert!(skills[0].disk_path.contains("r1"), "earlier root wins");
    }

    #[test]
    fn reads_trust_tier_from_lockfile() {
        let tmp = tempfile::tempdir().unwrap();
        // Layout mirrors ~/.agents/{skills, .skill-lock.json}.
        let agents = tmp.path().join("agents");
        let root = agents.join("skills");
        write_skill(&root, "gh-stack", "# gh-stack");
        fs::write(
            agents.join(".skill-lock.json"),
            r#"{"version":3,"skills":{"gh-stack":{"sourceType":"github"}}}"#,
        )
        .unwrap();

        let skills = scan_global_roots(&[root]);
        assert_eq!(skills[0].trust_tier, TrustTier::Github);
    }

    #[test]
    fn scan_user_roots_stamps_owner_and_client_locality() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_skill(root, "mine", "# mine\n\nprose");
        write_attachment(root, "mine", "scripts/run.sh", "echo hi");

        let skills = scan_user_roots(&[root.to_path_buf()], "dave");
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "mine");
        assert_eq!(skills[0].owner_user_id.as_deref(), Some("dave"));
        assert_eq!(skills[0].locality, Locality::Client);
        // The attachment-covering hash is computed the same way regardless of scope.
        assert_eq!(skills[0].content_hash.len(), 64);
        assert_eq!(skills[0].attachments, vec!["scripts/run.sh"]);
    }
}
