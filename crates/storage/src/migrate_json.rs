//! JSON-to-Postgres data migration for conversations, preferences, and factual memory.
//!
//! Reads existing JSON files and inserts records into Postgres tables.
//! Intended for first-startup migration when JSON files exist and DB tables are empty.

use std::path::Path;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Conversation, KnowledgeEntry};
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
use desktop_assistant_core::ports::store::ConversationStore;
use sqlx::PgPool;

use crate::PgConversationStore;
use crate::PgKnowledgeBaseStore;

/// Legacy preference entry matching the old `preferences.json` format.
#[derive(Debug, serde::Deserialize)]
struct LegacyPreference {
    key: String,
    value: String,
    scope: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    updated_at: u64,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

/// Legacy preference store wrapper.
#[derive(Debug, Default, serde::Deserialize)]
struct LegacyPreferenceStore {
    items: Vec<LegacyPreference>,
}

/// Legacy memory entry matching the old `factual_memory.json` format.
#[derive(Debug, serde::Deserialize)]
struct LegacyMemory {
    id: String,
    fact: String,
    #[serde(default)]
    tags: Vec<String>,
    source: Option<String>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    created_at: u64,
    #[serde(default)]
    #[allow(dead_code)]
    updated_at: u64,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

/// Legacy memory store wrapper.
#[derive(Debug, Default, serde::Deserialize)]
struct LegacyMemoryStore {
    items: Vec<LegacyMemory>,
}

/// Migrate conversations from a JSON file into Postgres.
///
/// Reads the JSON array of `Conversation` objects and inserts each one via
/// `PgConversationStore::create`. Skips conversations with no messages.
///
/// Returns the number of conversations migrated.
pub async fn migrate_conversations(
    json_path: &Path,
    pool: &PgPool,
) -> Result<usize, CoreError> {
    if !json_path.exists() {
        tracing::info!(
            "no conversations JSON file at {}; skipping migration",
            json_path.display()
        );
        return Ok(0);
    }

    let content = std::fs::read_to_string(json_path).map_err(|e| {
        CoreError::Storage(format!(
            "failed reading conversations file {}: {e}",
            json_path.display()
        ))
    })?;

    if content.trim().is_empty() {
        return Ok(0);
    }

    let conversations: Vec<Conversation> = serde_json::from_str(&content).map_err(|e| {
        CoreError::Storage(format!(
            "failed parsing conversations file {}: {e}",
            json_path.display()
        ))
    })?;

    let store = PgConversationStore::new(pool.clone());
    let mut count = 0;

    for conv in conversations {
        if conv.messages.is_empty() {
            continue;
        }
        if let Err(e) = store.create(conv).await {
            tracing::warn!("failed to migrate conversation: {e}");
        } else {
            count += 1;
        }
    }

    tracing::info!("migrated {count} conversation(s) from JSON to Postgres");
    Ok(count)
}

/// Migrate preferences and factual memory from JSON files into the unified knowledge base.
///
/// - Preferences get tag `["preference"]`; key/value are combined into prose content.
/// - Memories get tag `["memory"]` plus any existing tags.
/// - Existing embeddings from JSON are carried over if present.
///
/// Returns the number of entries migrated.
pub async fn migrate_knowledge(
    preferences_path: &Path,
    memory_path: &Path,
    pool: &PgPool,
) -> Result<usize, CoreError> {
    let kb_store = PgKnowledgeBaseStore::new(pool.clone());
    let mut count = 0;

    // --- Preferences ---
    if preferences_path.exists() {
        let content = std::fs::read_to_string(preferences_path).map_err(|e| {
            CoreError::Storage(format!(
                "failed reading preferences file {}: {e}",
                preferences_path.display()
            ))
        })?;

        if !content.trim().is_empty() {
            let prefs: LegacyPreferenceStore =
                serde_json::from_str(&content).map_err(|e| {
                    CoreError::Storage(format!(
                        "failed parsing preferences file {}: {e}",
                        preferences_path.display()
                    ))
                })?;

            for pref in prefs.items {
                let id = format!("pref_{}", slug(&pref.key));
                let prose = if let Some(scope) = &pref.scope {
                    format!("[{}] {}: {}", scope, pref.key, pref.value)
                } else {
                    format!("{}: {}", pref.key, pref.value)
                };

                let mut tags = vec!["preference".to_string()];
                if let Some(scope) = pref.scope {
                    tags.push(format!("project:{scope}"));
                }

                let entry = KnowledgeEntry::new(&id, &prose, tags);
                if let Err(e) = kb_store.write(entry, pref.embedding).await {
                    tracing::warn!("failed to migrate preference '{}': {e}", pref.key);
                } else {
                    count += 1;
                }
            }
        }
    } else {
        tracing::info!(
            "no preferences file at {}; skipping",
            preferences_path.display()
        );
    }

    // --- Factual memory ---
    if memory_path.exists() {
        let content = std::fs::read_to_string(memory_path).map_err(|e| {
            CoreError::Storage(format!(
                "failed reading memory file {}: {e}",
                memory_path.display()
            ))
        })?;

        if !content.trim().is_empty() {
            let memories: LegacyMemoryStore =
                serde_json::from_str(&content).map_err(|e| {
                    CoreError::Storage(format!(
                        "failed parsing memory file {}: {e}",
                        memory_path.display()
                    ))
                })?;

            for mem in memories.items {
                let mut tags: Vec<String> = vec!["memory".to_string()];
                tags.extend(mem.tags);

                let mut metadata = serde_json::Map::new();
                if let Some(source) = mem.source {
                    metadata.insert("source".to_string(), serde_json::Value::String(source));
                }
                if let Some(confidence) = mem.confidence {
                    metadata.insert(
                        "confidence".to_string(),
                        serde_json::Value::from(confidence),
                    );
                }

                let mut entry = KnowledgeEntry::new(&mem.id, &mem.fact, tags);
                entry.metadata = serde_json::Value::Object(metadata);

                if let Err(e) = kb_store.write(entry, mem.embedding).await {
                    tracing::warn!("failed to migrate memory '{}': {e}", mem.id);
                } else {
                    count += 1;
                }
            }
        }
    } else {
        tracing::info!("no memory file at {}; skipping", memory_path.display());
    }

    tracing::info!("migrated {count} knowledge base entry/entries from JSON to Postgres");
    Ok(count)
}

/// Check if the conversations table is empty.
pub async fn is_conversations_table_empty(pool: &PgPool) -> bool {
    let result: Result<(i64,), _> =
        sqlx::query_as("SELECT COUNT(*) FROM conversations")
            .fetch_one(pool)
            .await;
    matches!(result, Ok((0,)))
}

/// Check if the knowledge_base table is empty.
pub async fn is_knowledge_base_table_empty(pool: &PgPool) -> bool {
    let result: Result<(i64,), _> =
        sqlx::query_as("SELECT COUNT(*) FROM knowledge_base")
            .fetch_one(pool)
            .await;
    matches!(result, Ok((0,)))
}

/// Create a simple slug from a key string for use as an ID.
fn slug(key: &str) -> String {
    key.chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_basic() {
        assert_eq!(slug("Dark Mode"), "dark_mode");
        assert_eq!(slug("editor/theme"), "editor_theme");
        assert_eq!(slug("A-B_C"), "a_b_c");
    }

    #[test]
    fn legacy_preference_deserializes() {
        let json = r#"{"items": [{"key": "theme", "value": "dark", "updated_at": 123}]}"#;
        let store: LegacyPreferenceStore = serde_json::from_str(json).unwrap();
        assert_eq!(store.items.len(), 1);
        assert_eq!(store.items[0].key, "theme");
        assert_eq!(store.items[0].value, "dark");
    }

    #[test]
    fn legacy_memory_deserializes() {
        let json = r#"{"items": [{"id": "m1", "fact": "User likes Rust", "tags": ["lang"], "created_at": 100, "updated_at": 200}]}"#;
        let store: LegacyMemoryStore = serde_json::from_str(json).unwrap();
        assert_eq!(store.items.len(), 1);
        assert_eq!(store.items[0].fact, "User likes Rust");
        assert_eq!(store.items[0].tags, vec!["lang"]);
    }

    #[test]
    fn legacy_preference_with_scope() {
        let json = r#"{"items": [{"key": "build_cmd", "value": "cargo build", "scope": "adelie", "updated_at": 0}]}"#;
        let store: LegacyPreferenceStore = serde_json::from_str(json).unwrap();
        assert_eq!(store.items[0].scope.as_deref(), Some("adelie"));
    }
}
