use serde::{Deserialize, Serialize};

/// A unified knowledge base entry, replacing separate preferences and memory stores.
/// Each entry is prose content with tags and optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub created_at: String,
    pub updated_at: String,
    /// First-class provenance: `extraction` | `consolidation` | `explicit`,
    /// or `None` when unknown (legacy rows, or read paths that don't select
    /// it). On write, `None` preserves any existing value rather than clearing
    /// it.
    #[serde(default)]
    pub source: Option<String>,
}

impl KnowledgeEntry {
    pub fn new(id: impl Into<String>, content: impl Into<String>, tags: Vec<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            tags,
            metadata: serde_json::json!({}),
            created_at: String::new(),
            updated_at: String::new(),
            source: None,
        }
    }

    /// Builder-style setter for provenance.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knowledge_entry_creation() {
        let entry = KnowledgeEntry::new(
            "kb-1",
            "User prefers dark mode",
            vec!["preference".to_string()],
        );
        assert_eq!(entry.id, "kb-1");
        assert_eq!(entry.content, "User prefers dark mode");
        assert_eq!(entry.tags, vec!["preference"]);
        assert_eq!(entry.metadata, serde_json::json!({}));
    }

    #[test]
    fn knowledge_entry_serialization_roundtrip() {
        let mut entry = KnowledgeEntry::new("kb-1", "test content", vec!["tag1".to_string()]);
        entry.metadata = serde_json::json!({"key": "editor", "scope": "global"});
        entry.created_at = "2024-01-01 00:00:00".to_string();
        entry.updated_at = "2024-01-01 00:00:00".to_string();

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: KnowledgeEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, entry.id);
        assert_eq!(deserialized.content, entry.content);
        assert_eq!(deserialized.tags, entry.tags);
        assert_eq!(deserialized.metadata, entry.metadata);
    }
}
