//! Domain types and pure logic for the on-disk skill library (#573).
//!
//! A *skill* is an on-disk `SKILL.md` playbook (frontmatter + markdown body)
//! that the daemon indexes into a searchable catalog. A *workflow* is a skill
//! whose body carries a `## Steps` section. Everything in this module is pure
//! and adapter-independent: parsing a `SKILL.md` string, deriving the workflow
//! kind and trust tier, validating a skill name against path traversal, and
//! computing the integrity hash a blessing pins to. Filesystem walking and DB
//! persistence live in adapters (the daemon scanner and the storage crates).
//!
//! Why the hash covers attachments: a skill's bundled scripts are executed, so
//! the blessing must be invalidated if any of them changes. Hashing only
//! `SKILL.md` would let a swapped `scripts/run.sh` keep a stale blessing valid.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Whether an indexed skill is a plain playbook or a runnable workflow.
///
/// `Workflow` is derived structurally from a `## Steps` section in the body
/// (see [`detect_kind`]) rather than a frontmatter tag, so skills authored for
/// the shared agent directories stay format-compatible with other tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillKind {
    /// A prose playbook with no structured step sequence.
    Skill,
    /// A skill whose body defines a numbered `## Steps` procedure.
    Workflow,
}

impl SkillKind {
    /// Stable lowercase token for the `kind` DB column.
    pub fn as_str(&self) -> &'static str {
        match self {
            SkillKind::Skill => "skill",
            SkillKind::Workflow => "workflow",
        }
    }

    /// Parse a `kind` DB token, failing closed to [`SkillKind::Skill`] for any
    /// unrecognized value (an unknown kind must never grant workflow behavior).
    pub fn from_db(s: &str) -> Self {
        match s {
            "workflow" => SkillKind::Workflow,
            _ => SkillKind::Skill,
        }
    }
}

/// Provenance trust tier, derived from the source that installed a skill.
///
/// Remote/third-party sources are materially higher risk than a locally
/// authored skill and get stronger disclosure and gating downstream, so the
/// tier is a first-class field rather than an afterthought.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustTier {
    /// Authored locally by the user or Adele.
    Local,
    /// Installed from a GitHub source.
    Github,
    /// Fetched from a `.well-known` HTTP source.
    WellKnown,
    /// Source could not be classified.
    Unknown,
}

impl TrustTier {
    /// Stable lowercase token for the `trust_tier` DB column.
    pub fn as_str(&self) -> &'static str {
        match self {
            TrustTier::Local => "local",
            TrustTier::Github => "github",
            TrustTier::WellKnown => "well_known",
            TrustTier::Unknown => "unknown",
        }
    }

    /// Parse a `trust_tier` DB token, failing closed to [`TrustTier::Unknown`].
    pub fn from_db(s: &str) -> Self {
        match s {
            "local" => TrustTier::Local,
            "github" => TrustTier::Github,
            "well_known" => TrustTier::WellKnown,
            _ => TrustTier::Unknown,
        }
    }
}

/// Where a skill's files live and, consequently, where its scripts execute.
///
/// Global skills scanned from a system root are `Daemon`; user-scoped skills a
/// client scans from a home directory and registers are `Client`. Attachment
/// execution is gated at whichever runner hosts the skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    /// Hosted by the daemon (global system roots).
    Daemon,
    /// Hosted by a client (user home roots).
    Client,
}

impl Locality {
    /// Stable lowercase token for the `locality` DB column.
    pub fn as_str(&self) -> &'static str {
        match self {
            Locality::Daemon => "daemon",
            Locality::Client => "client",
        }
    }

    /// Parse a `locality` DB token, failing closed to [`Locality::Daemon`].
    pub fn from_db(s: &str) -> Self {
        match s {
            "client" => Locality::Client,
            _ => Locality::Daemon,
        }
    }
}

/// YAML frontmatter parsed from a `SKILL.md`.
///
/// Lenient by design: `name` and `description` are required, `tags` defaults to
/// empty, and any other keys (e.g. `metadata: {author, version}`) are captured
/// into [`SkillFrontmatter::metadata`] so the shared cross-product format is
/// preserved rather than rejected.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillFrontmatter {
    /// Human-facing skill name; must match the directory name.
    pub name: String,
    /// One or two sentence "when to use" trigger.
    pub description: String,
    /// Optional free-form tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Any additional frontmatter keys, preserved verbatim.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// The result of parsing a `SKILL.md` string: its frontmatter plus the markdown
/// body that follows the closing fence.
#[derive(Debug, Clone)]
pub struct ParsedSkill {
    /// The parsed YAML frontmatter.
    pub frontmatter: SkillFrontmatter,
    /// The markdown body after the frontmatter fence.
    pub body: String,
}

/// A single attachment's digest, the unit the integrity hash is built from.
///
/// The scanner produces one per sibling file traveling with a `SKILL.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentDigest {
    /// Path relative to the skill directory (e.g. `scripts/run.sh`).
    pub rel_path: String,
    /// Lowercase hex SHA-256 of the file's bytes.
    pub sha256_hex: String,
    /// Unix mode bits (execute bit matters for scripts).
    pub mode: u32,
}

/// Errors from parsing or validating a skill.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SkillError {
    /// No leading `---` frontmatter fence.
    #[error("missing YAML frontmatter (expected a leading `---` fence)")]
    MissingFrontmatter,
    /// A leading fence with no matching closing `---`.
    #[error("unterminated YAML frontmatter (no closing `---`)")]
    UnterminatedFrontmatter,
    /// The frontmatter block was not valid YAML or lacked required fields.
    #[error("invalid YAML frontmatter: {0}")]
    InvalidFrontmatter(String),
    /// The skill name is empty or attempts path traversal.
    #[error("invalid skill name: {0}")]
    InvalidName(String),
}

/// Parse a `SKILL.md` string into frontmatter and body.
///
/// Tolerates a leading BOM and leading whitespace, then requires a `---` fenced
/// YAML block followed by the markdown body. `name` and `description` are
/// required frontmatter fields; unknown keys are preserved in
/// [`SkillFrontmatter::metadata`].
pub fn parse_skill_md(raw: &str) -> Result<ParsedSkill, SkillError> {
    // Tolerate a leading BOM and any leading whitespace before the fence.
    let trimmed = raw.strip_prefix('\u{feff}').unwrap_or(raw).trim_start();
    let after_open = trimmed
        .strip_prefix("---\n")
        .or_else(|| trimmed.strip_prefix("---\r\n"))
        .ok_or(SkillError::MissingFrontmatter)?;
    let (yaml, body) =
        split_at_close_fence(after_open).ok_or(SkillError::UnterminatedFrontmatter)?;
    let frontmatter: SkillFrontmatter =
        serde_yaml_ng::from_str(yaml).map_err(|e| SkillError::InvalidFrontmatter(e.to_string()))?;
    // Drop the single newline the closing fence leaves at the head of the body.
    let body = body
        .strip_prefix("\r\n")
        .or_else(|| body.strip_prefix('\n'))
        .unwrap_or(body);
    Ok(ParsedSkill {
        frontmatter,
        body: body.to_string(),
    })
}

/// Split the text following the opening fence at the next line that is exactly
/// `---`, returning `(yaml_before, remainder_after_fence_line)`.
fn split_at_close_fence(s: &str) -> Option<(&str, &str)> {
    let mut offset = 0;
    for line in s.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            return Some((&s[..offset], &s[offset + line.len()..]));
        }
        offset += line.len();
    }
    None
}

/// Derive the [`SkillKind`] from a markdown body.
///
/// Returns [`SkillKind::Workflow`] iff the body contains a level-2 heading whose
/// text is exactly `Steps` (case-insensitive), e.g. `## Steps`. A deeper heading
/// (`### Steps`) or a longer title (`## Steps to reproduce`) does not qualify.
pub fn detect_kind(body: &str) -> SkillKind {
    for line in body.lines() {
        if let Some(rest) = line.trim().strip_prefix("## ")
            && rest.trim().eq_ignore_ascii_case("steps")
        {
            return SkillKind::Workflow;
        }
    }
    SkillKind::Skill
}

/// Validate a skill name against path traversal.
///
/// Rejects empty names, `.`/`..`, path separators, and absolute paths so a name
/// can never escape its root. Mirrors the guard in `skills-mcp`.
pub fn validate_skill_name(name: &str) -> Result<(), SkillError> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0')
    {
        return Err(SkillError::InvalidName(name.to_string()));
    }
    Ok(())
}

/// Map a `.skill-lock.json` `sourceType` to a [`TrustTier`].
///
/// `None` (no lockfile entry) is treated as locally authored.
pub fn trust_tier_from_source_type(source_type: Option<&str>) -> TrustTier {
    match source_type {
        None | Some("local") => TrustTier::Local,
        Some("github") => TrustTier::Github,
        Some("well-known") => TrustTier::WellKnown,
        Some(_) => TrustTier::Unknown,
    }
}

/// Lowercase hex SHA-256 of an arbitrary byte slice (one attachment's digest).
pub fn file_sha256_hex(bytes: &[u8]) -> String {
    to_hex(&Sha256::digest(bytes))
}

/// Lowercase hex-encode a byte slice.
fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Compute the integrity hash a blessing pins to.
///
/// Covers the `SKILL.md` bytes **and** every attachment's path, content digest,
/// and mode, so any change to a bundled script invalidates the hash. The result
/// is deterministic and independent of the order attachments are supplied in
/// (they are sorted internally). Returned as lowercase hex.
pub fn skill_content_hash(skill_md: &[u8], attachments: &[AttachmentDigest]) -> String {
    let mut sorted: Vec<&AttachmentDigest> = attachments.iter().collect();
    sorted.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    let mut hasher = Sha256::new();
    // Domain separator + version so the scheme can evolve without silent clashes.
    hasher.update(b"adelie-skill-v1\n");
    update_len_prefixed(&mut hasher, skill_md);
    hasher.update((sorted.len() as u64).to_be_bytes());
    for att in sorted {
        update_len_prefixed(&mut hasher, att.rel_path.as_bytes());
        update_len_prefixed(&mut hasher, att.sha256_hex.as_bytes());
        hasher.update(att.mode.to_be_bytes());
    }
    to_hex(&hasher.finalize())
}

/// Feed a length-prefixed field into the hasher so adjacent fields cannot be
/// ambiguously re-partitioned (which would let two distinct inputs collide).
fn update_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

/// The indexed representation of a skill: the currency of the `SkillIndexStore`
/// port and what search/get return. Storage-only concerns (embeddings) are not
/// part of this domain type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedSkill {
    /// Skill name (unique within its scope).
    pub name: String,
    /// The "when to use" description.
    pub description: String,
    /// Whether this is a plain skill or a workflow.
    pub kind: SkillKind,
    /// Absolute on-disk path to the `SKILL.md` (the backlink).
    pub disk_path: String,
    /// `None` for a global skill; the owner's id for a user-scoped skill.
    pub owner_user_id: Option<String>,
    /// Which runner hosts the skill (and runs its scripts).
    pub locality: Locality,
    /// The attachment-covering integrity hash a blessing pins to.
    pub content_hash: String,
    /// Provenance trust tier.
    pub trust_tier: TrustTier,
    /// Free-form source label (e.g. lockfile `sourceUrl` or root path).
    #[serde(default)]
    pub source: Option<String>,
    /// Frontmatter tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Attachment paths relative to the skill directory.
    #[serde(default)]
    pub attachments: Vec<String>,
    /// The markdown body (used for search and `skill_get`).
    #[serde(default)]
    pub body: String,
    /// Extra frontmatter preserved verbatim.
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Whether the skill's files were on disk at the last scan of its scope.
    ///
    /// `false` means the indexed copy is all that survives: the body still
    /// reads, but `disk_path` and [`Self::attachments`] no longer resolve, so
    /// bundled scripts cannot be run.
    ///
    /// Why a flag and not a delete: the catalog is cumulative, so a skill
    /// disappearing from disk is never a reason to forget the procedure. This
    /// is index state rather than scan output -- a reconcile pass sets it, and
    /// whatever a scanner puts here is ignored on write.
    #[serde(default = "present_on_disk_default")]
    pub present_on_disk: bool,
    /// When a scan last saw this skill on disk; `None` for a row no
    /// presence-tracking scan has covered yet. Index state, like
    /// [`Self::present_on_disk`].
    #[serde(default)]
    pub last_seen_at: Option<DateTime<Utc>>,
}

/// A row with no recorded presence predates presence tracking, and the skill it
/// describes was on disk when it was indexed -- so absent evidence means
/// present, not missing.
fn present_on_disk_default() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "---\nname: gh-stack\ndescription: Manage stacked PRs.\ntags: [git, github]\n---\n\n# gh-stack\n\nBody text here.\n";

    fn att(path: &str, sha: &str, mode: u32) -> AttachmentDigest {
        AttachmentDigest {
            rel_path: path.to_string(),
            sha256_hex: sha.to_string(),
            mode,
        }
    }

    // --- parse_skill_md ------------------------------------------------------

    #[test]
    fn parse_extracts_frontmatter_and_body() {
        let parsed = parse_skill_md(VALID).expect("valid SKILL.md parses");
        assert_eq!(parsed.frontmatter.name, "gh-stack");
        assert_eq!(parsed.frontmatter.description, "Manage stacked PRs.");
        assert_eq!(parsed.frontmatter.tags, vec!["git", "github"]);
        assert!(parsed.body.starts_with("# gh-stack"));
        assert!(parsed.body.contains("Body text here."));
    }

    #[test]
    fn parse_captures_unknown_metadata_keys() {
        let raw = "---\nname: x\ndescription: y\nmetadata:\n  author: github\n  version: \"0.0.1\"\n---\nbody\n";
        let parsed = parse_skill_md(raw).expect("parses");
        assert_eq!(parsed.frontmatter.metadata["author"], "github");
        assert_eq!(parsed.frontmatter.metadata["version"], "0.0.1");
    }

    #[test]
    fn parse_defaults_tags_to_empty() {
        let raw = "---\nname: x\ndescription: y\n---\nbody\n";
        let parsed = parse_skill_md(raw).expect("parses");
        assert!(parsed.frontmatter.tags.is_empty());
    }

    #[test]
    fn parse_tolerates_bom_and_leading_whitespace() {
        let raw = "\u{feff}\n  ---\nname: x\ndescription: y\n---\nbody\n";
        let parsed = parse_skill_md(raw).expect("parses despite BOM/whitespace");
        assert_eq!(parsed.frontmatter.name, "x");
    }

    #[test]
    fn parse_rejects_missing_frontmatter() {
        let err = parse_skill_md("no frontmatter here").unwrap_err();
        assert_eq!(err, SkillError::MissingFrontmatter);
    }

    #[test]
    fn parse_rejects_unterminated_frontmatter() {
        let err = parse_skill_md("---\nname: x\ndescription: y\nbody with no close\n").unwrap_err();
        assert_eq!(err, SkillError::UnterminatedFrontmatter);
    }

    #[test]
    fn parse_rejects_missing_required_field() {
        // No `description` -> serde error surfaced as InvalidFrontmatter.
        let err = parse_skill_md("---\nname: x\n---\nbody\n").unwrap_err();
        assert!(matches!(err, SkillError::InvalidFrontmatter(_)));
    }

    // --- detect_kind ---------------------------------------------------------

    #[test]
    fn detect_kind_workflow_on_steps_heading() {
        assert_eq!(
            detect_kind("intro\n\n## Steps\n1. do it\n"),
            SkillKind::Workflow
        );
    }

    #[test]
    fn detect_kind_is_case_insensitive() {
        assert_eq!(detect_kind("## steps\n"), SkillKind::Workflow);
    }

    #[test]
    fn detect_kind_skill_without_steps() {
        assert_eq!(detect_kind("# Title\n\njust prose\n"), SkillKind::Skill);
    }

    #[test]
    fn detect_kind_ignores_deeper_heading() {
        assert_eq!(detect_kind("### Steps\n"), SkillKind::Skill);
    }

    #[test]
    fn detect_kind_ignores_longer_title() {
        assert_eq!(detect_kind("## Steps to reproduce\n"), SkillKind::Skill);
    }

    // --- validate_skill_name -------------------------------------------------

    #[test]
    fn validate_accepts_plain_name() {
        assert!(validate_skill_name("gh-stack").is_ok());
    }

    #[test]
    fn validate_rejects_empty() {
        assert!(matches!(
            validate_skill_name(""),
            Err(SkillError::InvalidName(_))
        ));
    }

    #[test]
    fn validate_rejects_traversal_and_separators() {
        for bad in ["..", "../etc", "a/b", "a\\b", "/abs", "."] {
            assert!(
                matches!(validate_skill_name(bad), Err(SkillError::InvalidName(_))),
                "expected {bad:?} to be rejected"
            );
        }
    }

    // --- trust_tier_from_source_type -----------------------------------------

    #[test]
    fn trust_tier_mapping() {
        assert_eq!(
            trust_tier_from_source_type(Some("github")),
            TrustTier::Github
        );
        assert_eq!(
            trust_tier_from_source_type(Some("well-known")),
            TrustTier::WellKnown
        );
        assert_eq!(trust_tier_from_source_type(Some("local")), TrustTier::Local);
        assert_eq!(trust_tier_from_source_type(None), TrustTier::Local);
        assert_eq!(
            trust_tier_from_source_type(Some("mystery")),
            TrustTier::Unknown
        );
    }

    // --- file_sha256_hex -----------------------------------------------------

    #[test]
    fn file_sha256_matches_known_vector() {
        // SHA-256 of the empty input.
        assert_eq!(
            file_sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // SHA-256 of "abc".
        assert_eq!(
            file_sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // --- skill_content_hash --------------------------------------------------

    #[test]
    fn content_hash_is_64_hex_chars() {
        let h = skill_content_hash(b"body", &[]);
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn content_hash_is_deterministic() {
        let a = [att("scripts/run.sh", "aa", 0o755)];
        assert_eq!(
            skill_content_hash(b"body", &a),
            skill_content_hash(b"body", &a)
        );
    }

    #[test]
    fn content_hash_is_order_independent() {
        let ordered = [att("a.txt", "11", 0o644), att("b.txt", "22", 0o644)];
        let reversed = [att("b.txt", "22", 0o644), att("a.txt", "11", 0o644)];
        assert_eq!(
            skill_content_hash(b"body", &ordered),
            skill_content_hash(b"body", &reversed)
        );
    }

    #[test]
    fn content_hash_changes_with_skill_md() {
        assert_ne!(
            skill_content_hash(b"body one", &[]),
            skill_content_hash(b"body two", &[])
        );
    }

    #[test]
    fn content_hash_changes_when_attachment_bytes_change() {
        // The security-critical property: swapping a script's content (its
        // digest) invalidates the hash even if SKILL.md is byte-identical.
        let before = [att("scripts/run.sh", "aaaa", 0o755)];
        let after = [att("scripts/run.sh", "bbbb", 0o755)];
        assert_ne!(
            skill_content_hash(b"body", &before),
            skill_content_hash(b"body", &after)
        );
    }

    #[test]
    fn content_hash_changes_when_attachment_mode_changes() {
        let non_exec = [att("scripts/run.sh", "aaaa", 0o644)];
        let exec = [att("scripts/run.sh", "aaaa", 0o755)];
        assert_ne!(
            skill_content_hash(b"body", &non_exec),
            skill_content_hash(b"body", &exec)
        );
    }

    #[test]
    fn content_hash_changes_when_attachment_path_changes() {
        let a = [att("scripts/run.sh", "aaaa", 0o755)];
        let b = [att("scripts/other.sh", "aaaa", 0o755)];
        assert_ne!(
            skill_content_hash(b"body", &a),
            skill_content_hash(b"body", &b)
        );
    }

    #[test]
    fn content_hash_distinguishes_present_vs_absent_attachment() {
        assert_ne!(
            skill_content_hash(b"body", &[]),
            skill_content_hash(b"body", &[att("x", "00", 0o644)])
        );
    }
}
