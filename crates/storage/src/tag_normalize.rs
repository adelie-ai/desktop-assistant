//! Facet-preserving normalization for knowledge-base tags.
//!
//! KB reads filter tags by Postgres array-overlap (`tags && $2`), an exact,
//! case-sensitive match. Without normalization `Preference`, `preference ` and
//! `preference` are three distinct tags that never overlap, so the same intent
//! fragments across variants and filters silently miss. This collapses that
//! drift on the write path.
//!
//! Unlike [`crate::tag_registry::normalize_tag_name`] — which strips every
//! non-alphanumeric character and so mangles a facet tag `project:adelie-ai`
//! into `projectadelie-ai` — this preserves a single `facet:value` colon by
//! normalizing the facet and value halves independently. Facet tags carry
//! meaning in that shape (`project:<name>`, `topic:<subject>`); losing the
//! separator would break the whole facet scheme.

use std::collections::HashSet;

/// Normalize a single knowledge-base tag, preserving a `facet:value` colon.
///
/// Lowercases, trims, and collapses internal whitespace runs to a single `-`.
/// A tag in `facet:value` shape (a non-empty facet name, then the first colon)
/// keeps its separator: the facet and value halves are normalized
/// independently and rejoined with `:`. A leading colon (empty facet) is not a
/// facet and is normalized as a plain token.
pub fn normalize_tag(raw: &str) -> String {
    match raw.split_once(':') {
        Some((facet, value)) if !facet.trim().is_empty() => {
            format!("{}:{}", normalize_token(facet), normalize_token(value))
        }
        _ => normalize_token(raw),
    }
}

/// Lowercase, trim, and collapse internal whitespace runs to single dashes.
/// Existing dashes are preserved; an all-whitespace/empty input yields `""`.
fn normalize_token(raw: &str) -> String {
    raw.split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .to_lowercase()
}

/// Normalize a list of tags, dropping empties and duplicates that collapse
/// together while preserving first-seen order.
pub fn normalize_tags<I, S>(tags: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for tag in tags {
        let norm = normalize_tag(tag.as_ref());
        if norm.is_empty() {
            continue;
        }
        if seen.insert(norm.clone()) {
            out.push(norm);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_and_trims_kind_tags() {
        assert_eq!(normalize_tag("Preference"), "preference");
        assert_eq!(normalize_tag("  Memory  "), "memory");
        assert_eq!(normalize_tag("INSTRUCTION"), "instruction");
    }

    #[test]
    fn collapses_internal_whitespace_to_dash() {
        assert_eq!(normalize_tag("multi word tag"), "multi-word-tag");
        assert_eq!(normalize_tag("multi   word\ttag"), "multi-word-tag");
    }

    #[test]
    fn preserves_facet_colon_and_normalizes_both_halves() {
        // The colon separator MUST survive: turning project:adelie-ai into
        // project-adelie-ai would break every facet filter.
        assert_eq!(normalize_tag("project:Adelie-AI"), "project:adelie-ai");
        assert_eq!(normalize_tag("Project: Adelie AI"), "project:adelie-ai");
        assert_eq!(normalize_tag("topic:Deploy"), "topic:deploy");
        assert_ne!(normalize_tag("project:Adelie-AI"), "project-adelie-ai");
    }

    #[test]
    fn splits_on_first_colon_only() {
        // Only the first colon is the facet separator; any further colons ride
        // along in the value untouched, so a value may itself contain a colon.
        assert_eq!(normalize_tag("Topic:Release:2026"), "topic:release:2026");
    }

    #[test]
    fn leading_colon_is_not_split_into_an_empty_facet() {
        // A leading colon has no facet name, so it is normalized as a plain
        // token (the stray colon is kept literally) rather than producing a
        // `:value`-shaped facet with an empty key.
        assert_eq!(normalize_tag(":deploy"), ":deploy");
    }

    #[test]
    fn normalize_tags_dedups_preserving_order() {
        assert_eq!(
            normalize_tags(["Preference", " Memory "]),
            vec!["preference".to_string(), "memory".to_string()]
        );
        // Case/whitespace variants collapse to one entry, first-seen order kept.
        assert_eq!(
            normalize_tags(["instruction", "Instruction", "project:X", "PROJECT:x"]),
            vec!["instruction".to_string(), "project:x".to_string()]
        );
    }

    #[test]
    fn normalize_tags_drops_empty() {
        assert_eq!(normalize_tags(["", "   ", "ok"]), vec!["ok".to_string()]);
    }
}
