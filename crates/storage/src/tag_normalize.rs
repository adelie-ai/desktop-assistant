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

// STUB (red commit): identity passthrough so the spec tests compile and fail
// for the real reason before the implementation lands.

/// Normalize a single knowledge-base tag, preserving a `facet:value` colon.
pub fn normalize_tag(raw: &str) -> String {
    raw.to_string()
}

/// Normalize a list of tags, dropping empties and duplicates that collapse
/// together while preserving first-seen order.
pub fn normalize_tags<I, S>(tags: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    tags.into_iter().map(|t| t.as_ref().to_string()).collect()
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
    fn empty_facet_is_not_treated_as_a_facet() {
        // A leading colon has no facet name; fall back to whole-token
        // normalization rather than emitting a `:value` tag.
        assert_eq!(normalize_tag(":Deploy"), "deploy");
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
        assert_eq!(
            normalize_tags(["", "   ", "ok"]),
            vec!["ok".to_string()]
        );
    }
}
