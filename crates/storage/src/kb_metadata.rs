//! Typed shape for `knowledge_base.metadata`.
//!
//! Metadata is stored as JSONB and carries two load-bearing fields for the
//! redesigned dream cycle (issue #108):
//!
//! - `scope`: a categorical disambiguator. `None` means the fact is universal
//!   (e.g. "the user prefers dark mode"). `Some(scope)` means the fact is
//!   conditional on the keys in the scope (e.g. `{project: "adelie-ai"}`).
//!   Consolidation never merges entries with differing scopes.
//! - `source_conversation_id`: the conversation that produced the fact.
//!   Consolidation may pull the source transcript on low-confidence calls
//!   to disambiguate. May reference a row that has been hard-deleted; readers
//!   must handle that gracefully.
//!
//! Unknown fields round-trip through `extra` so older writers don't lose data.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Categorical scope under which a fact is true.
///
/// Keys are dimension names (`project`, `tool`, ...); values are the specific
/// instance the fact applies to. An empty map on read is treated the same as
/// `None` (universal); writers should prefer `None` for universal facts.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KbScope(pub BTreeMap<String, String>);

impl KbScope {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Insert a scope dimension (e.g. `("project", "adelie-ai")`).
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.0.insert(key.into(), value.into());
        self
    }
}

/// Structured view of a KB entry's metadata.
///
/// All fields are optional so partially-populated rows (e.g. legacy entries
/// written before this redesign) round-trip without data loss.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KbMetadata {
    /// `None` => universal fact. `Some(empty)` is normalized to `None` on read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<KbScope>,

    /// Conversation that produced this fact, if known. May point at a row
    /// that has been hard-deleted by archival.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_conversation_id: Option<String>,

    /// Any additional fields written by future code paths.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl KbMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse from JSONB. Returns `KbMetadata::default()` if the value is not
    /// a JSON object — old rows have `{}` and that's fine.
    pub fn from_json(value: &serde_json::Value) -> Self {
        serde_json::from_value(value.clone()).unwrap_or_default()
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_else(|_| serde_json::json!({}))
    }

    /// Returns `Some` only when the scope is set and non-empty.
    pub fn effective_scope(&self) -> Option<&KbScope> {
        self.scope.as_ref().filter(|s| !s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_universal_fact() {
        let m = KbMetadata {
            source_conversation_id: Some("conv-1".into()),
            ..Default::default()
        };
        let json = m.to_json();
        let back = KbMetadata::from_json(&json);
        assert!(back.effective_scope().is_none());
        assert_eq!(back.source_conversation_id.as_deref(), Some("conv-1"));
    }

    #[test]
    fn roundtrip_scoped_fact() {
        let m = KbMetadata {
            scope: Some(KbScope::new().with("project", "adelie-ai")),
            source_conversation_id: Some("conv-2".into()),
            ..Default::default()
        };
        let json = m.to_json();
        let back = KbMetadata::from_json(&json);
        let scope = back.effective_scope().expect("scope should be present");
        assert_eq!(scope.0.get("project").map(String::as_str), Some("adelie-ai"));
    }

    #[test]
    fn empty_scope_treated_as_universal() {
        let m = KbMetadata {
            scope: Some(KbScope::new()),
            ..Default::default()
        };
        assert!(m.effective_scope().is_none());
    }

    #[test]
    fn legacy_empty_object_parses_cleanly() {
        let m = KbMetadata::from_json(&serde_json::json!({}));
        assert!(m.effective_scope().is_none());
        assert!(m.source_conversation_id.is_none());
    }

    #[test]
    fn unknown_fields_round_trip_through_extra() {
        let original = serde_json::json!({
            "scope": {"project": "adelie-ai"},
            "future_field": {"x": 1}
        });
        let m = KbMetadata::from_json(&original);
        assert_eq!(
            m.extra.get("future_field"),
            Some(&serde_json::json!({"x": 1}))
        );
        let back = m.to_json();
        assert_eq!(back.get("future_field"), Some(&serde_json::json!({"x": 1})));
    }
}
