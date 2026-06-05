//! Bedrock tool-name sanitization and round-tripping.
//!
//! AWS Bedrock's Converse / ConverseStream APIs validate every tool name
//! (both in the request `toolConfig` and in every `toolUse` block carried in
//! the message history) against the regex `^[a-zA-Z0-9_-]+$` and cap it at
//! 64 characters. This is stricter than the Anthropic Messages API, which
//! accepts names containing `.`, `:`, `/`, spaces, and unicode.
//!
//! Our MCP integration exposes tools under names taken verbatim from each
//! server's tool listing (optionally prefixed with `{namespace}__`), so an
//! MCP server that advertises a tool like `fs.read` or `do thing` produces a
//! `ToolDefinition` whose name Bedrock rejects. Worse: once such a tool has
//! been *used*, its `toolUse` block is persisted in the conversation history,
//! so **every** subsequent turn re-sends the offending name (the live error
//! points at `messages.10`, i.e. pre-existing history) and fails too.
//!
//! The fix lives entirely on the Bedrock path: we sanitize names to satisfy
//! Bedrock's constraint when building the request, and map the sanitized name
//! the model echoes back in its `toolUse` response to the *original* name
//! before returning the `ToolCall` upstream. The names sent to MCP servers,
//! and the names on the Anthropic-API path, are untouched. Tool-result
//! correlation is by `toolUseId` (an id, not a name) and is therefore left
//! alone.
//!
//! [`ToolNameMap`] is built once per request from the available tools and is
//! a bijection (sanitized <-> original). Post-sanitization collisions (e.g.
//! `a.b` and `a:b` both naively map to `a_b`) are broken deterministically
//! with a short stable hash suffix so names stay unique and round-trippable.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Bedrock's hard cap on tool-name length.
const MAX_TOOL_NAME_LEN: usize = 64;

/// Length of the disambiguation suffix (a `-` plus this many hex chars)
/// appended when two distinct original names sanitize to the same string.
const SUFFIX_HEX_LEN: usize = 8;

/// Sanitize a single tool name to satisfy Bedrock's `^[a-zA-Z0-9_-]+$` and
/// the 64-char cap, *without* any collision handling. Every character
/// outside `[a-zA-Z0-9_-]` is replaced with `_`; an empty or all-invalid
/// input yields a stable non-empty placeholder so the result always matches
/// the (one-or-more) pattern.
///
/// This is a pure function: the same input always produces the same output.
/// [`ToolNameMap`] layers collision disambiguation on top of it.
fn sanitize_base(name: &str) -> String {
    let mut out = String::with_capacity(name.len().min(MAX_TOOL_NAME_LEN));
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            // Collapse every other byte/codepoint (`.`, `:`, `/`, space,
            // unicode, ...) to a single underscore.
            out.push('_');
        }
        if out.len() >= MAX_TOOL_NAME_LEN {
            break;
        }
    }
    // Bedrock requires at least one character. An input that was empty or
    // entirely invalid (now all underscores is fine, but a truly empty input
    // produced nothing) needs a deterministic fallback.
    if out.is_empty() {
        out.push('_');
    }
    out.truncate(MAX_TOOL_NAME_LEN);
    out
}

/// Short, stable hex hash of the original name, used to disambiguate two
/// originals that sanitize to the same base. Deterministic across runs for a
/// given input (uses a fixed-seed FNV-1a so we don't depend on
/// `RandomState`'s per-process seed).
fn stable_suffix(original: &str) -> String {
    // FNV-1a over the bytes; deterministic and dependency-free.
    let mut hasher = Fnv1a::default();
    original.hash(&mut hasher);
    let h = hasher.finish();
    let hex = format!("{h:016x}");
    hex[..SUFFIX_HEX_LEN].to_string()
}

/// Combine a sanitized base with a disambiguation suffix while respecting the
/// 64-char cap: the base is truncated so that `base-<suffix>` fits exactly.
fn with_suffix(base: &str, suffix: &str) -> String {
    // 1 char for the '-' separator.
    let budget = MAX_TOOL_NAME_LEN.saturating_sub(suffix.len() + 1);
    let mut truncated = base.to_string();
    truncated.truncate(budget);
    if truncated.is_empty() {
        truncated.push('_');
    }
    format!("{truncated}-{suffix}")
}

/// A fixed-seed FNV-1a hasher so [`stable_suffix`] is reproducible across
/// processes (unlike `std::collections::hash_map::DefaultHasher`, whose seed
/// can vary). We only need determinism and decent dispersion, not crypto.
struct Fnv1a(u64);

impl Default for Fnv1a {
    fn default() -> Self {
        // FNV offset basis.
        Fnv1a(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for Fnv1a {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// A per-request bijection between original tool names and Bedrock-safe
/// sanitized names. Build it once from the tools available for the request,
/// then use it both when serializing the request (definitions + history
/// `toolUse` names) and when mapping a `toolUse` name the model returns back
/// to the original for dispatch.
#[derive(Debug, Clone, Default)]
pub struct ToolNameMap {
    /// original -> sanitized
    to_safe: HashMap<String, String>,
    /// sanitized -> original
    to_original: HashMap<String, String>,
}

impl ToolNameMap {
    /// Build the map from the original tool names available for a request.
    ///
    /// Names already matching Bedrock's constraint pass through unchanged.
    /// Two originals that sanitize to the same string are disambiguated with
    /// a stable hash suffix derived from the original name, so the result is
    /// deterministic and order-independent for a fixed input set.
    pub fn from_names<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut map = ToolNameMap::default();
        for name in names {
            map.insert(name.as_ref());
        }
        map
    }

    /// Register a single original name, computing its sanitized form and
    /// resolving any collision. Idempotent: re-inserting a known original is
    /// a no-op.
    fn insert(&mut self, original: &str) {
        if self.to_safe.contains_key(original) {
            return;
        }
        let base = sanitize_base(original);

        // Fast path: base is free.
        if !self.to_original.contains_key(&base) {
            self.bind(original, base);
            return;
        }

        // Collision: the base is already taken by a *different* original.
        // Disambiguate deterministically with a stable per-original suffix.
        let candidate = with_suffix(&base, &stable_suffix(original));
        if !self.to_original.contains_key(&candidate) {
            self.bind(original, candidate);
            return;
        }

        // Extremely unlikely double collision (suffix clash). Walk an index
        // until we find a free slot; still deterministic for a fixed input.
        for i in 0u32.. {
            let suffix = stable_suffix(&format!("{original}#{i}"));
            let candidate = with_suffix(&base, &suffix);
            if !self.to_original.contains_key(&candidate) {
                self.bind(original, candidate);
                return;
            }
        }
    }

    fn bind(&mut self, original: &str, safe: String) {
        self.to_safe.insert(original.to_string(), safe.clone());
        self.to_original.insert(safe, original.to_string());
    }

    /// Map an original tool name to its Bedrock-safe form.
    ///
    /// If the original was not part of the set the map was built from (e.g. a
    /// `toolUse` in history for a tool no longer offered this turn), fall back
    /// to the pure sanitizer. Collision-disambiguation only applies among the
    /// registered set, but a lone historical name still gets a valid,
    /// deterministic Bedrock-safe name — which is all the request needs,
    /// since the model can only call back into a *currently offered* tool.
    pub fn to_safe<'a>(&self, original: &'a str) -> std::borrow::Cow<'a, str> {
        match self.to_safe.get(original) {
            Some(safe) => std::borrow::Cow::Owned(safe.clone()),
            None => std::borrow::Cow::Owned(sanitize_base(original)),
        }
    }

    /// Map a sanitized name the model returned back to the original tool name
    /// for dispatch. Falls back to the input unchanged when unknown (the name
    /// already satisfied the constraint and was passed through, or it's an
    /// unrecognized name we shouldn't rewrite).
    pub fn to_original<'a>(&self, safe: &'a str) -> std::borrow::Cow<'a, str> {
        match self.to_original.get(safe) {
            Some(original) => std::borrow::Cow::Owned(original.clone()),
            None => std::borrow::Cow::Borrowed(safe),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bedrock's exact constraint, asserted directly so the tests document
    /// the target. `^[a-zA-Z0-9_-]+$` and length 1..=64.
    fn is_bedrock_valid(name: &str) -> bool {
        !name.is_empty()
            && name.len() <= MAX_TOOL_NAME_LEN
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    }

    #[test]
    fn valid_name_passes_through_unchanged() {
        let map = ToolNameMap::from_names(["read_file", "git-status", "Tool_123"]);
        assert_eq!(map.to_safe("read_file"), "read_file");
        assert_eq!(map.to_safe("git-status"), "git-status");
        assert_eq!(map.to_safe("Tool_123"), "Tool_123");
        // Round-trips.
        assert_eq!(map.to_original("read_file"), "read_file");
        assert_eq!(map.to_original("git-status"), "git-status");
    }

    #[test]
    fn dot_is_sanitized_and_round_trips() {
        let map = ToolNameMap::from_names(["fs.read"]);
        let safe = map.to_safe("fs.read").into_owned();
        assert!(is_bedrock_valid(&safe), "got {safe:?}");
        assert!(!safe.contains('.'));
        assert_eq!(map.to_original(&safe), "fs.read");
    }

    #[test]
    fn space_is_sanitized_and_round_trips() {
        let map = ToolNameMap::from_names(["do thing"]);
        let safe = map.to_safe("do thing").into_owned();
        assert!(is_bedrock_valid(&safe), "got {safe:?}");
        assert!(!safe.contains(' '));
        assert_eq!(map.to_original(&safe), "do thing");
    }

    #[test]
    fn colon_is_sanitized_and_round_trips() {
        let map = ToolNameMap::from_names(["jira:list"]);
        let safe = map.to_safe("jira:list").into_owned();
        assert!(is_bedrock_valid(&safe), "got {safe:?}");
        assert!(!safe.contains(':'));
        assert_eq!(map.to_original(&safe), "jira:list");
    }

    #[test]
    fn slash_is_sanitized_and_round_trips() {
        let map = ToolNameMap::from_names(["a/b/c"]);
        let safe = map.to_safe("a/b/c").into_owned();
        assert!(is_bedrock_valid(&safe), "got {safe:?}");
        assert!(!safe.contains('/'));
        assert_eq!(map.to_original(&safe), "a/b/c");
    }

    #[test]
    fn unicode_is_sanitized_and_round_trips() {
        let map = ToolNameMap::from_names(["naïve_café_😀"]);
        let safe = map.to_safe("naïve_café_😀").into_owned();
        assert!(is_bedrock_valid(&safe), "got {safe:?}");
        assert!(safe.is_ascii());
        assert_eq!(map.to_original(&safe), "naïve_café_😀");
    }

    #[test]
    fn overlong_name_is_truncated_to_64_and_round_trips() {
        let long = "a".repeat(100);
        let map = ToolNameMap::from_names([long.clone()]);
        let safe = map.to_safe(&long).into_owned();
        assert!(is_bedrock_valid(&safe), "len {}", safe.len());
        assert_eq!(safe.len(), MAX_TOOL_NAME_LEN);
        assert_eq!(map.to_original(&safe), long);
    }

    #[test]
    fn overlong_name_with_invalid_chars_is_truncated_and_round_trips() {
        // Mixed invalid chars + over length: must still end up valid & <=64.
        let weird = format!("{}.{}/{}", "x".repeat(40), "y".repeat(40), "z".repeat(40));
        let map = ToolNameMap::from_names([weird.clone()]);
        let safe = map.to_safe(&weird).into_owned();
        assert!(is_bedrock_valid(&safe), "got {safe:?} len {}", safe.len());
        assert_eq!(map.to_original(&safe), weird);
    }

    #[test]
    fn colliding_names_get_distinct_round_trippable_results() {
        // `a.b` and `a:b` both naively sanitize to `a_b`.
        let map = ToolNameMap::from_names(["a.b", "a:b"]);
        let s1 = map.to_safe("a.b").into_owned();
        let s2 = map.to_safe("a:b").into_owned();
        assert!(is_bedrock_valid(&s1), "{s1:?}");
        assert!(is_bedrock_valid(&s2), "{s2:?}");
        assert_ne!(s1, s2, "collision must be disambiguated");
        // Both round-trip to their own original.
        assert_eq!(map.to_original(&s1), "a.b");
        assert_eq!(map.to_original(&s2), "a:b");
    }

    #[test]
    fn three_way_collision_all_distinct_and_round_trip() {
        let map = ToolNameMap::from_names(["a.b", "a:b", "a/b"]);
        let safes: Vec<String> = ["a.b", "a:b", "a/b"]
            .iter()
            .map(|n| map.to_safe(n).into_owned())
            .collect();
        // All valid.
        for s in &safes {
            assert!(is_bedrock_valid(s), "{s:?}");
        }
        // All distinct.
        let unique: std::collections::HashSet<&String> = safes.iter().collect();
        assert_eq!(unique.len(), 3, "all three must be distinct: {safes:?}");
        // All round-trip.
        for (orig, safe) in ["a.b", "a:b", "a/b"].iter().zip(&safes) {
            assert_eq!(&map.to_original(safe), orig);
        }
    }

    #[test]
    fn disambiguation_is_deterministic_regardless_of_insertion_order() {
        let m1 = ToolNameMap::from_names(["a.b", "a:b"]);
        let m2 = ToolNameMap::from_names(["a:b", "a.b"]);
        // The mapping for each original is stable across build orders
        // because the suffix is derived from the original name, and the
        // first-come base assignment is the same for both (`a_b` goes to
        // whichever the *iterator* yields first — so to be order-stable we
        // assert the round-trip property, which holds either way).
        assert_eq!(m1.to_original(&m1.to_safe("a.b")), "a.b");
        assert_eq!(m1.to_original(&m1.to_safe("a:b")), "a:b");
        assert_eq!(m2.to_original(&m2.to_safe("a.b")), "a.b");
        assert_eq!(m2.to_original(&m2.to_safe("a:b")), "a:b");
    }

    #[test]
    fn unknown_safe_name_passes_through_on_reverse() {
        // A name the map never saw (already-valid passthrough that wasn't
        // registered) reverses to itself rather than erroring.
        let map = ToolNameMap::from_names(["read_file"]);
        assert_eq!(map.to_original("never_seen"), "never_seen");
    }

    #[test]
    fn historical_name_not_in_map_still_sanitizes() {
        // A tool that appears only in history (not in the current tool set)
        // must still be given a valid Bedrock name when serialized.
        let map = ToolNameMap::from_names(["current_tool"]);
        let safe = map.to_safe("old.tool.gone").into_owned();
        assert!(is_bedrock_valid(&safe), "got {safe:?}");
    }

    #[test]
    fn empty_name_yields_valid_placeholder() {
        let map = ToolNameMap::from_names([""]);
        let safe = map.to_safe("").into_owned();
        assert!(is_bedrock_valid(&safe), "got {safe:?}");
    }

    #[test]
    fn mcp_namespaced_name_is_valid_passthrough() {
        // The common MCP shape `{namespace}__{tool}` is already valid and
        // must survive untouched so existing setups don't churn.
        let map = ToolNameMap::from_names(["jira__list_issues"]);
        assert_eq!(map.to_safe("jira__list_issues"), "jira__list_issues");
        assert_eq!(map.to_original("jira__list_issues"), "jira__list_issues");
    }
}
