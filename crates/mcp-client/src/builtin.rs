use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Local, SecondsFormat, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::embedding::EmbedFn;
use desktop_assistant_core::ports::knowledge::{KnowledgeDeleteFn, KnowledgeSearchFn, KnowledgeWriteFn};
use desktop_assistant_core::ports::tool_registry::{ToolDefinitionFn, ToolSearchFn};

const TOOL_KB_WRITE: &str = "builtin_knowledge_base_write";
const TOOL_KB_SEARCH: &str = "builtin_knowledge_base_search";
const TOOL_KB_DELETE: &str = "builtin_knowledge_base_delete";
const TOOL_SEARCH: &str = "builtin_tool_search";
const TOOL_SYS_PROPS: &str = "builtin_sys_props";

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub struct BuiltinToolService {
    embed_fn: Option<EmbedFn>,
    kb_write_fn: Option<KnowledgeWriteFn>,
    kb_search_fn: Option<KnowledgeSearchFn>,
    kb_delete_fn: Option<KnowledgeDeleteFn>,
    tool_search_fn: Option<ToolSearchFn>,
    #[allow(dead_code)]
    tool_definition_fn: Option<ToolDefinitionFn>,
}

impl BuiltinToolService {
    /// Create a minimal BuiltinToolService with no backing stores.
    /// KB and tool_search calls will return errors until closures are configured.
    pub fn new() -> Self {
        Self {
            embed_fn: None,
            kb_write_fn: None,
            kb_search_fn: None,
            kb_delete_fn: None,
            tool_search_fn: None,
            tool_definition_fn: None,
        }
    }

    /// Configure the embedding function for generating query vectors.
    pub fn with_embedding(mut self, embed_fn: EmbedFn) -> Self {
        self.embed_fn = Some(embed_fn);
        self
    }

    /// Configure knowledge base store closures.
    pub fn with_knowledge_base(
        mut self,
        write_fn: KnowledgeWriteFn,
        search_fn: KnowledgeSearchFn,
        delete_fn: KnowledgeDeleteFn,
    ) -> Self {
        self.kb_write_fn = Some(write_fn);
        self.kb_search_fn = Some(search_fn);
        self.kb_delete_fn = Some(delete_fn);
        self
    }

    /// Configure tool registry closures.
    pub fn with_tool_registry(
        mut self,
        search_fn: ToolSearchFn,
        definition_fn: ToolDefinitionFn,
    ) -> Self {
        self.tool_search_fn = Some(search_fn);
        self.tool_definition_fn = Some(definition_fn);
        self
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                TOOL_KB_WRITE,
                "Write or update a knowledge base entry. Use for storing preferences, facts, \
                 instructions, project context, or any durable information the user wants remembered. \
                 Content should be self-contained prose that describes both the context (when/why \
                 this information is useful) and the information itself.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Self-contained prose describing the context and information. \
                                            Write naturally, e.g. 'The user lives at 123 Main St, Springfield. \
                                            Use this as their default location for weather, directions, and local searches.' \
                                            Do not use key-value format."
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization (e.g. 'preference', 'memory', 'instruction', 'project:myapp')"
                        },
                        "id": {
                            "type": "string",
                            "description": "Optional ID for updates. Omit to create a new entry."
                        }
                    },
                    "required": ["content"]
                }),
            ),
            ToolDefinition::new(
                TOOL_KB_SEARCH,
                "Search the knowledge base for preferences, memories, and stored context. \
                 Uses hybrid vector + full-text search.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter results by tags"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Max results (default 10)"
                        }
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_KB_DELETE,
                "Delete a knowledge base entry by ID",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "ID of the entry to delete"
                        }
                    },
                    "required": ["id"]
                }),
            ),
            ToolDefinition::new(
                TOOL_SEARCH,
                "Search for available tools by description. Use this when the user's request \
                 might require a tool that isn't in your current set. Returns tool names and \
                 descriptions; matched tools become available automatically.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What kind of tool are you looking for?"
                        }
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_SYS_PROPS,
                "Return a compact property sheet with basic runtime/system context",
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            ),
        ]
    }

    pub fn supports_tool(name: &str) -> bool {
        matches!(
            name,
            TOOL_KB_WRITE
                | TOOL_KB_SEARCH
                | TOOL_KB_DELETE
                | TOOL_SEARCH
                | TOOL_SYS_PROPS
        )
    }

    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, CoreError> {
        match name {
            TOOL_KB_WRITE => self.kb_write(arguments).await,
            TOOL_KB_SEARCH => self.kb_search(arguments).await,
            TOOL_KB_DELETE => self.kb_delete(arguments).await,
            TOOL_SEARCH => self.tool_search(arguments).await,
            TOOL_SYS_PROPS => Ok(self.sys_props()),
            _ => Err(CoreError::ToolExecution(format!(
                "unknown built-in tool: {name}"
            ))),
        }
    }

    fn sys_props(&self) -> String {
        let local_now = Local::now();
        serde_json::json!({
            "ok": true,
            "props": {
                "note": "Relative paths are interpreted from daemon_cwd unless a tool specifies otherwise.",
                "generated_at_epoch": now_ts(),
                "generated_at_utc": Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
                "generated_at_local": local_now.to_rfc3339_opts(SecondsFormat::Secs, false),
                "timezone": format!("{} ({})", local_now.format("%:z"), local_now.format("%Z")),
                "username": detect_username(),
                "home_dir": detect_home_dir(),
                "daemon_cwd": detect_daemon_cwd(),
                "xdg_dirs": detect_xdg_dirs(),
                "shell": detect_shell(),
                "locale": detect_locale(),
                "session_type": detect_session_type(),
                "hostname": detect_hostname(),
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "os_version": detect_os_version(),
            },
        })
        .to_string()
    }

    async fn kb_write(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let write_fn = self.kb_write_fn.as_ref().ok_or_else(|| {
            CoreError::ToolExecution("knowledge base not configured".to_string())
        })?;

        let content = required_string(&arguments, "content")?;
        let tags = optional_string_array(&arguments, "tags");
        let metadata = arguments
            .get("metadata")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let id = optional_string(&arguments, "id")
            .unwrap_or_else(|| uuid::Uuid::now_v7().to_string());

        let entry = desktop_assistant_core::domain::KnowledgeEntry {
            id,
            content: content.clone(),
            tags,
            metadata,
            created_at: String::new(),
            updated_at: String::new(),
        };

        // Generate embedding for the content
        let embedding = self.embed_text(&content).await;

        let saved = write_fn(entry, embedding).await?;

        Ok(serde_json::json!({
            "ok": true,
            "id": saved.id,
            "created_at": saved.created_at,
            "updated_at": saved.updated_at,
        })
        .to_string())
    }

    async fn kb_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self.kb_search_fn.as_ref().ok_or_else(|| {
            CoreError::ToolExecution("knowledge base not configured".to_string())
        })?;

        let query = required_string(&arguments, "query")?;
        let tags = optional_string_array_nonempty(&arguments, "tags");
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;

        tracing::info!(query = %query, ?tags, limit, "knowledge base search");

        let query_embedding = self.embed_text(&query).await.unwrap_or_default();

        let results = search_fn(query, query_embedding, tags, limit).await?;

        let items: Vec<serde_json::Value> = results
            .into_iter()
            .map(|entry| {
                serde_json::json!({
                    "id": entry.id,
                    "content": entry.content,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "updated_at": entry.updated_at,
                })
            })
            .collect();

        tracing::info!(result_count = items.len(), "knowledge base search results");
        tracing::debug!(results = %serde_json::to_string(&items).unwrap_or_default(), "knowledge base search response");

        Ok(serde_json::json!({
            "ok": true,
            "results": items,
        })
        .to_string())
    }

    async fn kb_delete(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let delete_fn = self.kb_delete_fn.as_ref().ok_or_else(|| {
            CoreError::ToolExecution("knowledge base not configured".to_string())
        })?;

        let id = required_string(&arguments, "id")?;
        delete_fn(id.clone()).await?;

        Ok(serde_json::json!({
            "ok": true,
            "deleted": id,
        })
        .to_string())
    }

    async fn tool_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self.tool_search_fn.as_ref().ok_or_else(|| {
            CoreError::ToolExecution("tool registry not configured".to_string())
        })?;

        let query = required_string(&arguments, "query")?;
        tracing::info!(query = %query, "tool search");

        let query_embedding = self.embed_text(&query).await.unwrap_or_default();

        let results = search_fn(query, query_embedding, 10).await?;

        let tools: Vec<serde_json::Value> = results
            .into_iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                })
            })
            .collect();

        let tool_names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        tracing::info!(result_count = tools.len(), ?tool_names, "tool search results");

        Ok(serde_json::json!({
            "ok": true,
            "tools": tools,
        })
        .to_string())
    }

    /// Embed a single text string, returning None if embeddings are unavailable.
    async fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        let embed_fn = self.embed_fn.as_ref()?;
        match embed_fn(vec![text.to_string()]).await {
            Ok(mut vecs) => vecs.pop(),
            Err(e) => {
                tracing::warn!("failed to embed text: {e}");
                None
            }
        }
    }
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

fn detect_username() -> Option<String> {
    ["USER", "LOGNAME", "USERNAME"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

fn detect_home_dir() -> Option<String> {
    ["HOME", "USERPROFILE"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

fn detect_daemon_cwd() -> Option<String> {
    std::env::current_dir()
        .ok()
        .map(|p| p.display().to_string())
        .filter(|s| !s.is_empty())
}

fn detect_xdg_dirs() -> serde_json::Value {
    let home = detect_home_dir();
    let fallback_base = home
        .as_ref()
        .map(|h| PathBuf::from(h).join(".local"))
        .unwrap_or_else(|| PathBuf::from(".local"));

    let config = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_base.join("config").display().to_string());
    let data = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_base.join("share").display().to_string());
    let state = std::env::var("XDG_STATE_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_base.join("state").display().to_string());
    let cache = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| fallback_base.join("cache").display().to_string());
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty());

    serde_json::json!({
        "config": config,
        "data": data,
        "state": state,
        "cache": cache,
        "runtime": runtime,
    })
}

fn detect_shell() -> Option<String> {
    ["SHELL", "COMSPEC"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

fn detect_locale() -> Option<String> {
    ["LC_ALL", "LANG"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

fn detect_session_type() -> Option<String> {
    std::env::var("XDG_SESSION_TYPE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn detect_hostname() -> Option<String> {
    if let Ok(hostname) = std::env::var("HOSTNAME") {
        let trimmed = hostname.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    if let Ok(contents) = fs::read_to_string("/etc/hostname") {
        let trimmed = contents.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    None
}

fn detect_os_version() -> Option<String> {
    if std::env::consts::OS != "linux" {
        return None;
    }

    let contents = fs::read_to_string("/etc/os-release").ok()?;
    parse_os_release_field(&contents, "PRETTY_NAME")
        .or_else(|| parse_os_release_field(&contents, "VERSION"))
        .or_else(|| parse_os_release_field(&contents, "VERSION_ID"))
}

fn parse_os_release_field(contents: &str, key: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        let (line_key, raw_value) = line.split_once('=')?;
        if line_key.trim() != key {
            return None;
        }
        let value = raw_value.trim().trim_matches('"').trim_matches('\'');
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_expose_expected_tools() {
        let service = BuiltinToolService::new();
        let names: Vec<String> = service
            .tool_definitions()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(names.contains(&TOOL_KB_WRITE.to_string()));
        assert!(names.contains(&TOOL_KB_SEARCH.to_string()));
        assert!(names.contains(&TOOL_KB_DELETE.to_string()));
        assert!(names.contains(&TOOL_SEARCH.to_string()));
        assert!(names.contains(&TOOL_SYS_PROPS.to_string()));
    }

    #[tokio::test]
    async fn sys_props_returns_compact_property_sheet() {
        let service = BuiltinToolService::new();

        let response = service
            .execute_tool("builtin_sys_props", serde_json::json!({}))
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(
            json.get("ok").and_then(serde_json::Value::as_bool),
            Some(true)
        );
        let props = json
            .get("props")
            .and_then(serde_json::Value::as_object)
            .expect("props object");
        assert!(
            props
                .get("generated_at_epoch")
                .and_then(serde_json::Value::as_u64)
                .is_some()
        );
        assert!(
            props
                .get("os")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|s| !s.is_empty())
        );
    }

    #[tokio::test]
    async fn kb_write_without_store_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(
                TOOL_KB_WRITE,
                serde_json::json!({"content": "test"}),
            )
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn kb_search_without_store_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(
                TOOL_KB_SEARCH,
                serde_json::json!({"query": "test"}),
            )
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn tool_search_without_registry_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(
                TOOL_SEARCH,
                serde_json::json!({"query": "file operations"}),
            )
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn kb_write_and_search_with_closures() {
        use std::sync::{Arc, Mutex};
        use desktop_assistant_core::domain::KnowledgeEntry;

        let store: Arc<Mutex<Vec<KnowledgeEntry>>> = Arc::new(Mutex::new(Vec::new()));

        let write_store = Arc::clone(&store);
        let write_fn: KnowledgeWriteFn = Arc::new(move |mut entry, _embedding| {
            let s = Arc::clone(&write_store);
            Box::pin(async move {
                entry.created_at = "2024-01-01".to_string();
                entry.updated_at = "2024-01-01".to_string();
                s.lock().unwrap().push(entry.clone());
                Ok(entry)
            })
        });

        let search_store = Arc::clone(&store);
        let search_fn: KnowledgeSearchFn = Arc::new(move |_query, _emb, _tags, limit| {
            let s = Arc::clone(&search_store);
            Box::pin(async move {
                let entries = s.lock().unwrap();
                Ok(entries.iter().take(limit).cloned().collect())
            })
        });

        let delete_fn: KnowledgeDeleteFn = Arc::new(|_id| {
            Box::pin(async { Ok(()) })
        });

        let service = BuiltinToolService::new()
            .with_knowledge_base(write_fn, search_fn, delete_fn);

        // Write
        let write_result = service
            .execute_tool(
                TOOL_KB_WRITE,
                serde_json::json!({
                    "content": "User prefers dark mode",
                    "tags": ["preference"]
                }),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&write_result).unwrap();
        assert_eq!(json["ok"], true);
        assert!(json["id"].as_str().is_some());

        // Search
        let search_result = service
            .execute_tool(
                TOOL_KB_SEARCH,
                serde_json::json!({"query": "dark mode"}),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&search_result).unwrap();
        assert_eq!(json["ok"], true);
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0]["content"].as_str().unwrap().contains("dark mode"));
    }

    #[tokio::test]
    async fn tool_search_with_closure() {
        use std::sync::Arc;
        use desktop_assistant_core::domain::ToolDefinition;

        let search_fn: ToolSearchFn = Arc::new(|_query, _emb, _limit| {
            Box::pin(async {
                Ok(vec![
                    ToolDefinition::new("jira__create_issue", "Create a Jira issue", serde_json::json!({})),
                ])
            })
        });

        let def_fn: ToolDefinitionFn = Arc::new(|_name| {
            Box::pin(async { Ok(None) })
        });

        let service = BuiltinToolService::new().with_tool_registry(search_fn, def_fn);

        let result = service
            .execute_tool(
                TOOL_SEARCH,
                serde_json::json!({"query": "create ticket"}),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["ok"], true);
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "jira__create_issue");
    }
}
