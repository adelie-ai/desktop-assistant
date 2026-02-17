use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::embedding::EmbedFn;

const TOOL_PREF_REMEMBER: &str = "builtin_preferences_remember";
const TOOL_PREF_SEARCH: &str = "builtin_preferences_search";
const TOOL_PREF_RETRIEVE: &str = "builtin_preferences_retrieve";
const TOOL_PREF_DELETE: &str = "builtin_preferences_delete";
const TOOL_MEM_REMEMBER: &str = "builtin_memory_remember";
const TOOL_MEM_SEARCH: &str = "builtin_memory_search";
const TOOL_MEM_RETRIEVE: &str = "builtin_memory_retrieve";
const TOOL_MEM_UPDATE: &str = "builtin_memory_update";
const TOOL_MEM_DELETE: &str = "builtin_memory_delete";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PreferenceEntry {
    key: String,
    value: String,
    scope: Option<String>,
    updated_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding_model: Option<String>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
struct PreferenceStoreData {
    items: Vec<PreferenceEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MemoryEntry {
    id: String,
    fact: String,
    tags: Vec<String>,
    source: Option<String>,
    confidence: Option<f64>,
    supersedes: Option<String>,
    created_at: u64,
    updated_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding_model: Option<String>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
struct MemoryStoreData {
    items: Vec<MemoryEntry>,
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn normalize(text: &str) -> String {
    text.trim().to_lowercase()
}

fn normalized_eq(left: &str, right: &str) -> bool {
    normalize(left) == normalize(right)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

fn score_match(haystack: &str, needle: &str) -> i64 {
    let haystack = normalize(haystack);
    let needle = normalize(needle);
    if needle.is_empty() {
        return 0;
    }
    if haystack == needle {
        return 100;
    }
    if haystack.contains(&needle) {
        return 50;
    }
    let tokens: Vec<&str> = needle.split_whitespace().collect();
    let mut score = 0;
    for token in tokens {
        if haystack.contains(token) {
            score += 10;
        }
    }
    score
}

fn default_data_home() -> PathBuf {
    std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".local").join("share")
        })
}

fn default_preferences_path() -> PathBuf {
    default_data_home()
        .join("desktop-assistant")
        .join("preferences.json")
}

fn default_memory_path() -> PathBuf {
    default_data_home()
        .join("desktop-assistant")
        .join("factual_memory.json")
}

pub struct BuiltinToolService {
    preferences_path: PathBuf,
    memory_path: PathBuf,
    preferences: Mutex<PreferenceStoreData>,
    memory: Mutex<MemoryStoreData>,
    embed_fn: Option<EmbedFn>,
    embedding_model: String,
}

impl BuiltinToolService {
    pub fn from_default_paths() -> Self {
        Self::new(default_preferences_path(), default_memory_path())
    }

    pub fn new(preferences_path: PathBuf, memory_path: PathBuf) -> Self {
        let preferences = load_json::<PreferenceStoreData>(&preferences_path).unwrap_or_else(|e| {
            tracing::warn!(
                "failed to load preferences from {}: {e}; starting empty",
                preferences_path.display()
            );
            PreferenceStoreData::default()
        });

        let memory = load_json::<MemoryStoreData>(&memory_path).unwrap_or_else(|e| {
            tracing::warn!(
                "failed to load memory from {}: {e}; starting empty",
                memory_path.display()
            );
            MemoryStoreData::default()
        });

        Self {
            preferences_path,
            memory_path,
            preferences: Mutex::new(preferences),
            memory: Mutex::new(memory),
            embed_fn: None,
            embedding_model: String::new(),
        }
    }

    /// Configure the embedding function and model name for vector indexing.
    pub fn with_embedding(mut self, embed_fn: EmbedFn, model: String) -> Self {
        self.embed_fn = Some(embed_fn);
        self.embedding_model = model;
        self
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                TOOL_PREF_REMEMBER,
                "Remember or update a long-term user preference",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                        "scope": {"type": "string"},
                        "overwrite": {"type": "boolean"}
                    },
                    "required": ["key", "value"]
                }),
            ),
            ToolDefinition::new(
                TOOL_PREF_SEARCH,
                "Search stored user preferences",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                        "scope": {"type": "string"}
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_PREF_RETRIEVE,
                "Retrieve a user preference by key",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "scope": {"type": "string"}
                    },
                    "required": ["key"]
                }),
            ),
            ToolDefinition::new(
                TOOL_PREF_DELETE,
                "Delete a specific user preference by key and optional scope",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "scope": {"type": "string"}
                    },
                    "required": ["key"]
                }),
            ),
            ToolDefinition::new(
                TOOL_MEM_REMEMBER,
                "Store a factual memory item",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "source": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["fact"]
                }),
            ),
            ToolDefinition::new(
                TOOL_MEM_SEARCH,
                "Search factual memory",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "limit": {"type": "integer", "minimum": 1}
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_MEM_RETRIEVE,
                "Retrieve factual memory by id",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    },
                    "required": ["id"]
                }),
            ),
            ToolDefinition::new(
                TOOL_MEM_UPDATE,
                "Update factual memory by id",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "fact": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "source": {"type": "string"},
                        "confidence": {"type": "number"},
                        "supersedes": {"type": "string"}
                    },
                    "required": ["id"]
                }),
            ),
            ToolDefinition::new(
                TOOL_MEM_DELETE,
                "Delete a specific factual memory by id",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    },
                    "required": ["id"]
                }),
            ),
        ]
    }

    pub fn supports_tool(name: &str) -> bool {
        matches!(
            name,
            TOOL_PREF_REMEMBER
                | TOOL_PREF_SEARCH
                | TOOL_PREF_RETRIEVE
                | TOOL_PREF_DELETE
                | TOOL_MEM_REMEMBER
                | TOOL_MEM_SEARCH
                | TOOL_MEM_RETRIEVE
                | TOOL_MEM_UPDATE
                | TOOL_MEM_DELETE
        )
    }

    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        match name {
            TOOL_PREF_REMEMBER => self.preferences_remember(arguments).await,
            TOOL_PREF_SEARCH => self.preferences_search(arguments).await,
            TOOL_PREF_RETRIEVE => self.preferences_retrieve(arguments),
            TOOL_PREF_DELETE => self.preferences_delete(arguments),
            TOOL_MEM_REMEMBER => self.memory_remember(arguments).await,
            TOOL_MEM_SEARCH => self.memory_search(arguments).await,
            TOOL_MEM_RETRIEVE => self.memory_retrieve(arguments),
            TOOL_MEM_UPDATE => self.memory_update(arguments).await,
            TOOL_MEM_DELETE => self.memory_delete(arguments),
            _ => Err(CoreError::ToolExecution(format!(
                "unknown built-in tool: {name}"
            ))),
        }
    }

    async fn preferences_remember(
        &self,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        let key = normalize(&required_string(&arguments, "key")?);
        let value = required_string(&arguments, "value")?;
        let scope = optional_string(&arguments, "scope");
        let overwrite = arguments
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        let now = now_ts();

        // Phase 1: mutate data under lock, then release
        let (response_json, embed_text) = {
            let mut prefs = self.preferences.lock().unwrap();

            if let Some(existing_idx) = prefs
                .items
                .iter()
                .position(|item| normalized_eq(&item.key, &key) && item.scope == scope)
            {
                let updated_at;
                prefs.items[existing_idx].key = key.clone();
                if overwrite {
                    prefs.items[existing_idx].value = value.clone();
                    prefs.items[existing_idx].updated_at = now;
                }
                updated_at = prefs.items[existing_idx].updated_at;
                let embed_text = pref_embed_text(
                    &prefs.items[existing_idx].key,
                    &prefs.items[existing_idx].value,
                );
                self.persist_preferences(&prefs)?;
                (
                    serde_json::json!({
                        "ok": true,
                        "key": key,
                        "scope": scope,
                        "stored": overwrite,
                        "updated_at": updated_at,
                    }),
                    embed_text,
                )
            } else {
                let embed_text = pref_embed_text(&key, &value);
                prefs.items.push(PreferenceEntry {
                    key: key.clone(),
                    value,
                    scope: scope.clone(),
                    updated_at: now,
                    embedding: None,
                    embedding_model: None,
                });
                self.persist_preferences(&prefs)?;
                (
                    serde_json::json!({
                        "ok": true,
                        "key": key,
                        "scope": scope,
                        "stored": true,
                        "updated_at": now,
                    }),
                    embed_text,
                )
            }
        };

        // Phase 2: generate embedding without holding lock
        self.embed_preference(&key, &scope, &embed_text).await;

        Ok(response_json.to_string())
    }

    async fn preferences_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let query = required_string(&arguments, "query")?;
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;
        let scope_filter = optional_string(&arguments, "scope");

        // Try to embed the query for vector search
        let query_embedding = self.embed_query(&query).await;

        let prefs = self.preferences.lock().unwrap();
        let mut results: Vec<serde_json::Value> = prefs
            .items
            .iter()
            .filter(|item| {
                if let Some(scope) = &scope_filter {
                    item.scope.as_deref() == Some(scope.as_str())
                } else {
                    true
                }
            })
            .filter_map(|item| {
                let text_score = score_match(&format!("{} {}", item.key, item.value), &query);
                let vector_score = match (&query_embedding, &item.embedding) {
                    (Some(qe), Some(ie)) => (cosine_similarity(qe, ie) * 100.0) as i64,
                    _ => 0,
                };
                let score = text_score.max(vector_score);
                if score <= 0 {
                    return None;
                }
                Some(serde_json::json!({
                    "key": item.key,
                    "value": item.value,
                    "scope": item.scope,
                    "score": score,
                    "updated_at": item.updated_at,
                }))
            })
            .collect();

        results.sort_by(|a, b| {
            let sa = a
                .get("score")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            let sb = b
                .get("score")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            sb.cmp(&sa)
        });
        results.truncate(limit);

        Ok(serde_json::json!({
            "ok": true,
            "results": results,
        })
        .to_string())
    }

    fn preferences_retrieve(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let key = required_string(&arguments, "key")?;
        let scope = optional_string(&arguments, "scope");

        let prefs = self.preferences.lock().unwrap();
        let found = prefs
            .items
            .iter()
            .find(|item| normalized_eq(&item.key, &key) && item.scope == scope);

        if let Some(item) = found {
            Ok(serde_json::json!({
                "ok": true,
                "found": true,
                "item": {
                    "key": item.key,
                    "value": item.value,
                    "scope": item.scope,
                    "updated_at": item.updated_at,
                }
            })
            .to_string())
        } else {
            Ok(serde_json::json!({
                "ok": true,
                "found": false,
            })
            .to_string())
        }
    }

    fn preferences_delete(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let key = required_string(&arguments, "key")?;
        let scope = optional_string(&arguments, "scope");

        let mut prefs = self.preferences.lock().unwrap();
        let before = prefs.items.len();
        prefs
            .items
            .retain(|item| !(normalized_eq(&item.key, &key) && item.scope == scope));
        let after = prefs.items.len();
        let deleted = before.saturating_sub(after);

        if deleted > 0 {
            self.persist_preferences(&prefs)?;
        }

        Ok(serde_json::json!({
            "ok": true,
            "deleted": deleted,
            "found": deleted > 0,
            "key": normalize(&key),
            "scope": scope,
            "remaining": after,
        })
        .to_string())
    }

    async fn memory_remember(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let fact = required_string(&arguments, "fact")?;
        let tags = optional_string_array(&arguments, "tags");
        let source = optional_string(&arguments, "source");
        let confidence = arguments
            .get("confidence")
            .and_then(serde_json::Value::as_f64);

        let now = now_ts();
        let id = format!("mem-{}", now_ts_nanos());

        let embed_text = mem_embed_text(&fact, &tags);

        {
            let mut memory = self.memory.lock().unwrap();
            memory.items.push(MemoryEntry {
                id: id.clone(),
                fact,
                tags,
                source,
                confidence,
                supersedes: None,
                created_at: now,
                updated_at: now,
                embedding: None,
                embedding_model: None,
            });
            self.persist_memory(&memory)?;
        }

        self.embed_memory(&id, &embed_text).await;

        Ok(serde_json::json!({
            "ok": true,
            "id": id,
            "created_at": now,
        })
        .to_string())
    }

    async fn memory_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let query = required_string(&arguments, "query")?;
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;
        let tags_filter = optional_string_array(&arguments, "tags");

        let query_embedding = self.embed_query(&query).await;

        let memory = self.memory.lock().unwrap();
        let mut results: Vec<serde_json::Value> = memory
            .items
            .iter()
            .filter(|item| {
                if tags_filter.is_empty() {
                    return true;
                }
                tags_filter
                    .iter()
                    .all(|tag| item.tags.iter().any(|it| normalize(it) == normalize(tag)))
            })
            .filter_map(|item| {
                let searchable = format!("{} {}", item.fact, item.tags.join(" "));
                let text_score = score_match(&searchable, &query);
                let vector_score = match (&query_embedding, &item.embedding) {
                    (Some(qe), Some(ie)) => (cosine_similarity(qe, ie) * 100.0) as i64,
                    _ => 0,
                };
                let score = text_score.max(vector_score);
                if score <= 0 {
                    return None;
                }
                Some(serde_json::json!({
                    "id": item.id,
                    "fact": item.fact,
                    "tags": item.tags,
                    "source": item.source,
                    "confidence": item.confidence,
                    "score": score,
                    "updated_at": item.updated_at,
                }))
            })
            .collect();

        results.sort_by(|a, b| {
            let sa = a
                .get("score")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            let sb = b
                .get("score")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            sb.cmp(&sa)
        });
        results.truncate(limit);

        Ok(serde_json::json!({
            "ok": true,
            "results": results,
        })
        .to_string())
    }

    fn memory_retrieve(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let id = required_string(&arguments, "id")?;

        let memory = self.memory.lock().unwrap();
        if let Some(item) = memory
            .items
            .iter()
            .find(|entry| normalized_eq(&entry.id, &id))
        {
            Ok(serde_json::json!({
                "ok": true,
                "found": true,
                "item": item,
            })
            .to_string())
        } else {
            Ok(serde_json::json!({
                "ok": true,
                "found": false,
            })
            .to_string())
        }
    }

    async fn memory_update(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let id = required_string(&arguments, "id")?;
        let fact = optional_string(&arguments, "fact");
        let tags = optional_string_array_nonempty(&arguments, "tags");
        let source = optional_string(&arguments, "source");
        let confidence = arguments
            .get("confidence")
            .and_then(serde_json::Value::as_f64);
        let supersedes = optional_string(&arguments, "supersedes");

        let (response_json, embed_text, matched_id) = {
            let mut memory = self.memory.lock().unwrap();
            if let Some(item_idx) = memory
                .items
                .iter()
                .position(|entry| normalized_eq(&entry.id, &id))
            {
                if let Some(new_fact) = fact {
                    memory.items[item_idx].fact = new_fact;
                }
                if let Some(new_tags) = tags {
                    memory.items[item_idx].tags = new_tags;
                }
                if let Some(new_source) = source {
                    memory.items[item_idx].source = Some(new_source);
                }
                if let Some(new_confidence) = confidence {
                    memory.items[item_idx].confidence = Some(new_confidence);
                }
                if let Some(new_supersedes) = supersedes {
                    memory.items[item_idx].supersedes = Some(new_supersedes);
                }
                memory.items[item_idx].updated_at = now_ts();

                let updated_id = memory.items[item_idx].id.clone();
                let updated_at = memory.items[item_idx].updated_at;
                let embed_text =
                    mem_embed_text(&memory.items[item_idx].fact, &memory.items[item_idx].tags);

                self.persist_memory(&memory)?;
                (
                    serde_json::json!({
                        "ok": true,
                        "id": updated_id,
                        "updated_at": updated_at,
                    }),
                    embed_text,
                    updated_id,
                )
            } else {
                return Err(CoreError::ToolExecution(format!(
                    "memory id not found: {id}"
                )));
            }
        };

        self.embed_memory(&matched_id, &embed_text).await;

        Ok(response_json.to_string())
    }

    fn memory_delete(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let id = required_string(&arguments, "id")?;

        let mut memory = self.memory.lock().unwrap();
        let before = memory.items.len();
        memory.items.retain(|item| !normalized_eq(&item.id, &id));
        let after = memory.items.len();
        let deleted = before.saturating_sub(after);

        if deleted > 0 {
            self.persist_memory(&memory)?;
        }

        Ok(serde_json::json!({
            "ok": true,
            "deleted": deleted,
            "found": deleted > 0,
            "id": id,
            "remaining": after,
        })
        .to_string())
    }

    /// Embed a single query string, returning None if embeddings are unavailable.
    async fn embed_query(&self, text: &str) -> Option<Vec<f32>> {
        let embed_fn = self.embed_fn.as_ref()?;
        match embed_fn(vec![text.to_string()]).await {
            Ok(mut vecs) => vecs.pop(),
            Err(e) => {
                tracing::warn!("failed to embed query: {e}");
                None
            }
        }
    }

    /// Generate and store an embedding for a preference entry.
    async fn embed_preference(&self, key: &str, scope: &Option<String>, text: &str) {
        let Some(embed_fn) = &self.embed_fn else {
            return;
        };
        let embedding = match embed_fn(vec![text.to_string()]).await {
            Ok(mut vecs) => vecs.pop(),
            Err(e) => {
                tracing::warn!("failed to embed preference '{key}': {e}");
                return;
            }
        };
        let Some(embedding) = embedding else { return };

        let mut prefs = self.preferences.lock().unwrap();
        if let Some(entry) = prefs
            .items
            .iter_mut()
            .find(|item| normalized_eq(&item.key, key) && item.scope == *scope)
        {
            entry.embedding = Some(embedding);
            entry.embedding_model = Some(self.embedding_model.clone());
            if let Err(e) = self.persist_preferences(&prefs) {
                tracing::warn!("failed to persist preference embedding: {e}");
            }
        }
    }

    /// Generate and store an embedding for a memory entry.
    async fn embed_memory(&self, id: &str, text: &str) {
        let Some(embed_fn) = &self.embed_fn else {
            return;
        };
        let embedding = match embed_fn(vec![text.to_string()]).await {
            Ok(mut vecs) => vecs.pop(),
            Err(e) => {
                tracing::warn!("failed to embed memory '{id}': {e}");
                return;
            }
        };
        let Some(embedding) = embedding else { return };

        let mut memory = self.memory.lock().unwrap();
        if let Some(entry) = memory
            .items
            .iter_mut()
            .find(|item| normalized_eq(&item.id, id))
        {
            entry.embedding = Some(embedding);
            entry.embedding_model = Some(self.embedding_model.clone());
            if let Err(e) = self.persist_memory(&memory) {
                tracing::warn!("failed to persist memory embedding: {e}");
            }
        }
    }

    fn persist_preferences(&self, data: &PreferenceStoreData) -> Result<(), CoreError> {
        persist_json(&self.preferences_path, data)
    }

    fn persist_memory(&self, data: &MemoryStoreData) -> Result<(), CoreError> {
        persist_json(&self.memory_path, data)
    }
}

fn pref_embed_text(key: &str, value: &str) -> String {
    format!("{key} {value}")
}

fn mem_embed_text(fact: &str, tags: &[String]) -> String {
    if tags.is_empty() {
        fact.to_string()
    } else {
        format!("{} {}", fact, tags.join(" "))
    }
}

fn load_json<T>(path: &PathBuf) -> Result<T, CoreError>
where
    T: serde::de::DeserializeOwned + Default,
{
    if !path.exists() {
        return Ok(T::default());
    }

    let content = fs::read_to_string(path)
        .map_err(|e| CoreError::Storage(format!("failed reading {}: {e}", path.display())))?;

    if content.trim().is_empty() {
        return Ok(T::default());
    }

    serde_json::from_str(&content)
        .map_err(|e| CoreError::Storage(format!("failed parsing {}: {e}", path.display())))
}

fn persist_json<T>(path: &PathBuf, data: &T) -> Result<(), CoreError>
where
    T: serde::Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            CoreError::Storage(format!("failed creating {}: {e}", parent.display()))
        })?;
    }

    let content = serde_json::to_string_pretty(data)
        .map_err(|e| CoreError::Storage(format!("failed serializing store data: {e}")))?;

    let tmp_path = path.with_extension("json.tmp");
    fs::write(&tmp_path, content)
        .map_err(|e| CoreError::Storage(format!("failed writing {}: {e}", tmp_path.display())))?;
    fs::rename(&tmp_path, path)
        .map_err(|e| CoreError::Storage(format!("failed replacing {}: {e}", path.display())))?;

    Ok(())
}

fn required_string(args: &serde_json::Value, key: &str) -> Result<String, CoreError> {
    args.get(key)
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CoreError::ToolExecution(format!("missing required string argument: {key}")))
}

fn optional_string(args: &serde_json::Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn optional_string_array(args: &serde_json::Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn optional_string_array_nonempty(args: &serde_json::Value, key: &str) -> Option<Vec<String>> {
    let values = optional_string_array(args, key);
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn now_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn temp_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "desktop-assistant-{}-{}.json",
            name,
            now_ts_nanos()
        ))
    }

    fn semantic_embed_fn() -> EmbedFn {
        Arc::new(|texts: Vec<String>| {
            Box::pin(async move {
                let vectors = texts
                    .into_iter()
                    .map(|text| {
                        let t = normalize(&text);
                        let editor_like =
                            if t.contains("neovim") || t.contains("vim") || t.contains("modal") {
                                1.0
                            } else {
                                0.0
                            };
                        let ide_like = if t.contains("vscode")
                            || t.contains("visual studio code")
                            || t.contains("ide")
                        {
                            1.0
                        } else {
                            0.0
                        };
                        vec![editor_like, ide_like]
                    })
                    .collect();
                Ok(vectors)
            })
        })
    }

    #[test]
    fn builtins_expose_expected_tools() {
        let service = BuiltinToolService::new(temp_file("pref"), temp_file("mem"));
        let names: Vec<String> = service
            .tool_definitions()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(names.contains(&TOOL_PREF_REMEMBER.to_string()));
        assert!(names.contains(&TOOL_PREF_DELETE.to_string()));
        assert!(names.contains(&TOOL_MEM_UPDATE.to_string()));
        assert!(names.contains(&TOOL_MEM_DELETE.to_string()));
    }

    #[tokio::test]
    async fn preferences_remember_and_retrieve() {
        let pref_path = temp_file("pref");
        let mem_path = temp_file("mem");
        let service = BuiltinToolService::new(pref_path.clone(), mem_path.clone());

        let _ = service
            .execute_tool(
                TOOL_PREF_REMEMBER,
                serde_json::json!({
                    "key": "theme",
                    "value": "dark"
                }),
            )
            .await
            .unwrap();

        let retrieved = service
            .execute_tool(
                TOOL_PREF_RETRIEVE,
                serde_json::json!({
                    "key": "theme"
                }),
            )
            .await
            .unwrap();

        assert!(retrieved.contains("\"found\":true"));
        assert!(retrieved.contains("dark"));

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[tokio::test]
    async fn preferences_lookup_is_case_insensitive() {
        let pref_path = temp_file("pref");
        let mem_path = temp_file("mem");
        let service = BuiltinToolService::new(pref_path.clone(), mem_path.clone());

        let _ = service
            .execute_tool(
                TOOL_PREF_REMEMBER,
                serde_json::json!({
                    "key": "Project.MyApp.Path",
                    "value": "/home/dave/projects/my-app"
                }),
            )
            .await
            .unwrap();

        let retrieved = service
            .execute_tool(
                TOOL_PREF_RETRIEVE,
                serde_json::json!({
                    "key": "project.myapp.path"
                }),
            )
            .await
            .unwrap();

        let retrieved_json: serde_json::Value = serde_json::from_str(&retrieved).unwrap();
        assert_eq!(
            retrieved_json["item"]["key"],
            serde_json::Value::String("project.myapp.path".to_string())
        );
        assert!(retrieved.contains("\"found\":true"));
        assert!(retrieved.contains("/home/dave/projects/my-app"));

        let _ = service
            .execute_tool(
                TOOL_PREF_REMEMBER,
                serde_json::json!({
                    "key": "PROJECT.MYAPP.PATH",
                    "value": "/tmp/my-app"
                }),
            )
            .await
            .unwrap();

        let search = service
            .execute_tool(
                TOOL_PREF_SEARCH,
                serde_json::json!({
                    "query": "project.myapp.path",
                    "limit": 10
                }),
            )
            .await
            .unwrap();

        let search_json: serde_json::Value = serde_json::from_str(&search).unwrap();
        let results_len = search_json
            .get("results")
            .and_then(serde_json::Value::as_array)
            .map(|a| a.len())
            .unwrap_or(0);
        assert_eq!(results_len, 1);
        assert!(search.contains("/tmp/my-app"));

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[tokio::test]
    async fn preferences_search_uses_embeddings_and_persists_vectors() {
        let pref_path = temp_file("pref");
        let mem_path = temp_file("mem");
        let service = BuiltinToolService::new(pref_path.clone(), mem_path.clone())
            .with_embedding(semantic_embed_fn(), "test-embed-model".to_string());

        service
            .execute_tool(
                TOOL_PREF_REMEMBER,
                serde_json::json!({
                    "key": "global.editor",
                    "value": "neovim"
                }),
            )
            .await
            .unwrap();

        service
            .execute_tool(
                TOOL_PREF_REMEMBER,
                serde_json::json!({
                    "key": "global.ide",
                    "value": "vscode"
                }),
            )
            .await
            .unwrap();

        let search = service
            .execute_tool(
                TOOL_PREF_SEARCH,
                serde_json::json!({
                    "query": "best modal editor",
                    "limit": 1
                }),
            )
            .await
            .unwrap();
        let search_json: serde_json::Value = serde_json::from_str(&search).unwrap();
        let top_key = search_json["results"][0]["key"]
            .as_str()
            .unwrap_or_default();
        assert_eq!(top_key, "global.editor");

        let stored: PreferenceStoreData = load_json(&pref_path).unwrap();
        let editor = stored
            .items
            .iter()
            .find(|item| item.key == "global.editor")
            .unwrap();
        assert_eq!(editor.embedding_model.as_deref(), Some("test-embed-model"));
        assert!(editor.embedding.as_ref().is_some_and(|v| !v.is_empty()));

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[tokio::test]
    async fn memory_remember_update_retrieve() {
        let pref_path = temp_file("pref");
        let mem_path = temp_file("mem");
        let service = BuiltinToolService::new(pref_path.clone(), mem_path.clone());

        let created = service
            .execute_tool(
                TOOL_MEM_REMEMBER,
                serde_json::json!({
                    "fact": "User lives in Raleigh",
                    "tags": ["location", "profile"]
                }),
            )
            .await
            .unwrap();

        let created_json: serde_json::Value = serde_json::from_str(&created).unwrap();
        let id = created_json
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap()
            .to_string();

        let _ = service
            .execute_tool(
                TOOL_MEM_UPDATE,
                serde_json::json!({
                    "id": id,
                    "fact": "User lives in Holly Springs"
                }),
            )
            .await
            .unwrap();

        let retrieved = service
            .execute_tool(
                TOOL_MEM_RETRIEVE,
                serde_json::json!({
                    "id": created_json["id"]
                }),
            )
            .await
            .unwrap();

        assert!(retrieved.contains("Holly Springs"));

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[tokio::test]
    async fn memory_id_lookup_is_case_insensitive() {
        let pref_path = temp_file("pref");
        let mem_path = temp_file("mem");
        let service = BuiltinToolService::new(pref_path.clone(), mem_path.clone());

        let created = service
            .execute_tool(
                TOOL_MEM_REMEMBER,
                serde_json::json!({
                    "fact": "Primary email is dave@example.com",
                    "tags": ["profile"]
                }),
            )
            .await
            .unwrap();

        let created_json: serde_json::Value = serde_json::from_str(&created).unwrap();
        let id = created_json
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap()
            .to_string();
        let upper_id = id.to_uppercase();

        let _ = service
            .execute_tool(
                TOOL_MEM_UPDATE,
                serde_json::json!({
                    "id": upper_id,
                    "fact": "Primary email is dave+work@example.com"
                }),
            )
            .await
            .unwrap();

        let retrieved = service
            .execute_tool(
                TOOL_MEM_RETRIEVE,
                serde_json::json!({
                    "id": id.to_uppercase()
                }),
            )
            .await
            .unwrap();

        assert!(retrieved.contains("\"found\":true"));
        assert!(retrieved.contains("dave+work@example.com"));

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[tokio::test]
    async fn memory_search_uses_embeddings_and_persists_vectors() {
        let pref_path = temp_file("pref");
        let mem_path = temp_file("mem");
        let service = BuiltinToolService::new(pref_path.clone(), mem_path.clone())
            .with_embedding(semantic_embed_fn(), "test-embed-model".to_string());

        service
            .execute_tool(
                TOOL_MEM_REMEMBER,
                serde_json::json!({
                    "fact": "User prefers neovim",
                    "tags": ["editor"]
                }),
            )
            .await
            .unwrap();

        service
            .execute_tool(
                TOOL_MEM_REMEMBER,
                serde_json::json!({
                    "fact": "User prefers vscode",
                    "tags": ["editor"]
                }),
            )
            .await
            .unwrap();

        let search = service
            .execute_tool(
                TOOL_MEM_SEARCH,
                serde_json::json!({
                    "query": "modal editing workflow",
                    "limit": 1
                }),
            )
            .await
            .unwrap();
        let search_json: serde_json::Value = serde_json::from_str(&search).unwrap();
        let top_fact = search_json["results"][0]["fact"]
            .as_str()
            .unwrap_or_default();
        assert!(top_fact.contains("neovim"));

        let stored: MemoryStoreData = load_json(&mem_path).unwrap();
        let any_embedded = stored.items.iter().any(|item| {
            item.embedding_model.as_deref() == Some("test-embed-model")
                && item.embedding.as_ref().is_some_and(|v| !v.is_empty())
        });
        assert!(any_embedded);

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[test]
    fn preferences_delete_removes_exact_key_and_scope() {
        let pref_path = temp_file("pref");
        let mem_path = temp_file("mem");

        let initial = PreferenceStoreData {
            items: vec![
                PreferenceEntry {
                    key: "project.editor".to_string(),
                    value: "nvim".to_string(),
                    scope: Some("project.alpha".to_string()),
                    updated_at: now_ts(),
                    embedding: None,
                    embedding_model: None,
                },
                PreferenceEntry {
                    key: "project.editor".to_string(),
                    value: "code".to_string(),
                    scope: Some("project.beta".to_string()),
                    updated_at: now_ts(),
                    embedding: None,
                    embedding_model: None,
                },
            ],
        };

        persist_json(&pref_path, &initial).unwrap();
        let service = BuiltinToolService::new(pref_path.clone(), mem_path.clone());

        let deleted = service
            .preferences_delete(serde_json::json!({
                "key": "PROJECT.EDITOR",
                "scope": "project.alpha"
            }))
            .unwrap();
        let deleted_json: serde_json::Value = serde_json::from_str(&deleted).unwrap();
        assert_eq!(deleted_json["deleted"], serde_json::json!(1));
        assert_eq!(deleted_json["found"], serde_json::json!(true));

        let stored: PreferenceStoreData = load_json(&pref_path).unwrap();
        assert_eq!(stored.items.len(), 1);
        assert_eq!(stored.items[0].scope.as_deref(), Some("project.beta"));

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[test]
    fn memory_delete_removes_exact_id_case_insensitive() {
        let pref_path = temp_file("pref");
        let mem_path = temp_file("mem");

        let initial = MemoryStoreData {
            items: vec![
                MemoryEntry {
                    id: "mem-abc".to_string(),
                    fact: "first".to_string(),
                    tags: vec!["a".to_string()],
                    source: None,
                    confidence: None,
                    supersedes: None,
                    created_at: now_ts(),
                    updated_at: now_ts(),
                    embedding: None,
                    embedding_model: None,
                },
                MemoryEntry {
                    id: "mem-def".to_string(),
                    fact: "second".to_string(),
                    tags: vec!["b".to_string()],
                    source: None,
                    confidence: None,
                    supersedes: None,
                    created_at: now_ts(),
                    updated_at: now_ts(),
                    embedding: None,
                    embedding_model: None,
                },
            ],
        };

        persist_json(&mem_path, &initial).unwrap();
        let service = BuiltinToolService::new(pref_path.clone(), mem_path.clone());

        let deleted = service
            .memory_delete(serde_json::json!({
                "id": "MEM-ABC"
            }))
            .unwrap();
        let deleted_json: serde_json::Value = serde_json::from_str(&deleted).unwrap();
        assert_eq!(deleted_json["deleted"], serde_json::json!(1));
        assert_eq!(deleted_json["found"], serde_json::json!(true));

        let stored: MemoryStoreData = load_json(&mem_path).unwrap();
        assert_eq!(stored.items.len(), 1);
        assert_eq!(stored.items[0].id, "mem-def");

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }
}
