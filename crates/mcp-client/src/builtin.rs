use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;

const TOOL_PREF_REMEMBER: &str = "builtin_preferences_remember";
const TOOL_PREF_SEARCH: &str = "builtin_preferences_search";
const TOOL_PREF_RETRIEVE: &str = "builtin_preferences_retrieve";
const TOOL_MEM_REMEMBER: &str = "builtin_memory_remember";
const TOOL_MEM_SEARCH: &str = "builtin_memory_search";
const TOOL_MEM_RETRIEVE: &str = "builtin_memory_retrieve";
const TOOL_MEM_UPDATE: &str = "builtin_memory_update";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PreferenceEntry {
    key: String,
    value: String,
    scope: Option<String>,
    updated_at: u64,
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
        }
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
        ]
    }

    pub fn supports_tool(name: &str) -> bool {
        matches!(
            name,
            TOOL_PREF_REMEMBER
                | TOOL_PREF_SEARCH
                | TOOL_PREF_RETRIEVE
                | TOOL_MEM_REMEMBER
                | TOOL_MEM_SEARCH
                | TOOL_MEM_RETRIEVE
                | TOOL_MEM_UPDATE
        )
    }

    pub fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        match name {
            TOOL_PREF_REMEMBER => self.preferences_remember(arguments),
            TOOL_PREF_SEARCH => self.preferences_search(arguments),
            TOOL_PREF_RETRIEVE => self.preferences_retrieve(arguments),
            TOOL_MEM_REMEMBER => self.memory_remember(arguments),
            TOOL_MEM_SEARCH => self.memory_search(arguments),
            TOOL_MEM_RETRIEVE => self.memory_retrieve(arguments),
            TOOL_MEM_UPDATE => self.memory_update(arguments),
            _ => Err(CoreError::ToolExecution(format!(
                "unknown built-in tool: {name}"
            ))),
        }
    }

    fn preferences_remember(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let key = required_string(&arguments, "key")?;
        let value = required_string(&arguments, "value")?;
        let scope = optional_string(&arguments, "scope");
        let overwrite = arguments
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        let now = now_ts();
        let mut prefs = self.preferences.lock().unwrap();

        if let Some(existing_idx) = prefs
            .items
            .iter()
            .position(|item| item.key == key && item.scope == scope)
        {
            let updated_at;
            if overwrite {
                prefs.items[existing_idx].value = value.clone();
                prefs.items[existing_idx].updated_at = now;
            }
            updated_at = prefs.items[existing_idx].updated_at;
            self.persist_preferences(&prefs)?;
            return Ok(serde_json::json!({
                "ok": true,
                "key": key,
                "scope": scope,
                "stored": overwrite,
                "updated_at": updated_at,
            })
            .to_string());
        }

        prefs.items.push(PreferenceEntry {
            key: key.clone(),
            value,
            scope: scope.clone(),
            updated_at: now,
        });

        self.persist_preferences(&prefs)?;
        Ok(serde_json::json!({
            "ok": true,
            "key": key,
            "scope": scope,
            "stored": true,
            "updated_at": now,
        })
        .to_string())
    }

    fn preferences_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let query = required_string(&arguments, "query")?;
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;
        let scope_filter = optional_string(&arguments, "scope");

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
                let score = score_match(&format!("{} {}", item.key, item.value), &query);
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
            .find(|item| item.key == key && item.scope == scope);

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

    fn memory_remember(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let fact = required_string(&arguments, "fact")?;
        let tags = optional_string_array(&arguments, "tags");
        let source = optional_string(&arguments, "source");
        let confidence = arguments
            .get("confidence")
            .and_then(serde_json::Value::as_f64);

        let now = now_ts();
        let id = format!("mem-{}", now_ts_nanos());

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
        });

        self.persist_memory(&memory)?;

        Ok(serde_json::json!({
            "ok": true,
            "id": id,
            "created_at": now,
        })
        .to_string())
    }

    fn memory_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let query = required_string(&arguments, "query")?;
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;
        let tags_filter = optional_string_array(&arguments, "tags");

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
                let score = score_match(&searchable, &query);
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
        if let Some(item) = memory.items.iter().find(|entry| entry.id == id) {
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

    fn memory_update(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let id = required_string(&arguments, "id")?;
        let fact = optional_string(&arguments, "fact");
        let tags = optional_string_array_nonempty(&arguments, "tags");
        let source = optional_string(&arguments, "source");
        let confidence = arguments
            .get("confidence")
            .and_then(serde_json::Value::as_f64);
        let supersedes = optional_string(&arguments, "supersedes");

        let mut memory = self.memory.lock().unwrap();
        if let Some(item_idx) = memory.items.iter().position(|entry| entry.id == id) {
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

            self.persist_memory(&memory)?;
            return Ok(serde_json::json!({
                "ok": true,
                "id": updated_id,
                "updated_at": updated_at,
            })
            .to_string());
        }

        Err(CoreError::ToolExecution(format!(
            "memory id not found: {id}"
        )))
    }

    fn persist_preferences(&self, data: &PreferenceStoreData) -> Result<(), CoreError> {
        persist_json(&self.preferences_path, data)
    }

    fn persist_memory(&self, data: &MemoryStoreData) -> Result<(), CoreError> {
        persist_json(&self.memory_path, data)
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

    fn temp_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "desktop-assistant-{}-{}.json",
            name,
            now_ts_nanos()
        ))
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
        assert!(names.contains(&TOOL_MEM_UPDATE.to_string()));
    }

    #[test]
    fn preferences_remember_and_retrieve() {
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
            .unwrap();

        let retrieved = service
            .execute_tool(
                TOOL_PREF_RETRIEVE,
                serde_json::json!({
                    "key": "theme"
                }),
            )
            .unwrap();

        assert!(retrieved.contains("\"found\":true"));
        assert!(retrieved.contains("dark"));

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }

    #[test]
    fn memory_remember_update_retrieve() {
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
            .unwrap();

        let retrieved = service
            .execute_tool(
                TOOL_MEM_RETRIEVE,
                serde_json::json!({
                    "id": created_json["id"]
                }),
            )
            .unwrap();

        assert!(retrieved.contains("Holly Springs"));

        let _ = fs::remove_file(pref_path);
        let _ = fs::remove_file(mem_path);
    }
}
