use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Local, SecondsFormat, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{Role, ToolDefinition};
use desktop_assistant_core::ports::conversation_search::ConversationSearchFn;
use desktop_assistant_core::ports::database::DbQueryFn;
use desktop_assistant_core::ports::embedding::EmbedFn;
use desktop_assistant_core::ports::knowledge::{
    KnowledgeDeleteFn, KnowledgeSearchFn, KnowledgeWriteFn,
};
use desktop_assistant_core::ports::tool_registry::{ToolDefinitionFn, ToolSearchFn};

use crate::executor::McpControlHandle;

const TOOL_KB_WRITE: &str = "builtin_knowledge_base_write";
const TOOL_KB_SEARCH: &str = "builtin_knowledge_base_search";
const TOOL_KB_DELETE: &str = "builtin_knowledge_base_delete";
const TOOL_SEARCH: &str = "builtin_tool_search";
const TOOL_SYS_PROPS: &str = "builtin_sys_props";
const TOOL_DB_QUERY: &str = "builtin_db_query";
const TOOL_MCP_CONTROL: &str = "builtin_mcp_control";
const TOOL_CONV_SEARCH: &str = "builtin_conversation_search";

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
    db_query_fn: Option<DbQueryFn>,
    mcp_handle: Option<McpControlHandle>,
    conversation_search_fn: Option<ConversationSearchFn>,
}

impl Default for BuiltinToolService {
    fn default() -> Self {
        Self::new()
    }
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
            db_query_fn: None,
            mcp_handle: None,
            conversation_search_fn: None,
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

    /// Configure database query closure for read-only SQL access.
    pub fn with_database(mut self, query_fn: DbQueryFn) -> Self {
        self.db_query_fn = Some(query_fn);
        self
    }

    /// Configure the past-conversation full-text search closure (#71).
    /// When unset, `builtin_conversation_search` returns a clear error
    /// rather than silently no-op-ing.
    pub fn with_conversation_search(mut self, search_fn: ConversationSearchFn) -> Self {
        self.conversation_search_fn = Some(search_fn);
        self
    }

    /// Set the MCP control handle (used by builtin_mcp_control tool).
    pub fn set_mcp_control(&mut self, handle: McpControlHandle) {
        self.mcp_handle = Some(handle);
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
            ToolDefinition::new(
                TOOL_DB_QUERY,
                "Execute a SQL query against the assistant's PostgreSQL database. \
                 Use this to inspect your own conversations, messages, knowledge base \
                 entries, tool definitions, and other stored data. You can also modify \
                 data directly — use this to debug issues, fix inconsistencies, or \
                 rework entries that lack a dedicated tool.\n\n\
                 A `scratch` schema is available for temporary relational work (staging \
                 tables, intermediate joins, materialized views, etc.). Write queries \
                 default to the scratch schema via search_path; the main data in the \
                 `public` schema is always readable. To modify public tables directly, \
                 use fully-qualified names (e.g. `UPDATE public.knowledge_base ...`).\n\n\
                 SELECT/WITH/TABLE/VALUES/EXPLAIN run in a read-only transaction. \
                 Other statements (CREATE, INSERT, UPDATE, DELETE, etc.) run in a \
                 normal transaction and are committed.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "description": "Maximum rows to return for SELECT queries (default 100). Ignored for write queries."
                        }
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_CONV_SEARCH,
                "Search past conversations by full-text query. Useful for \
                 recalling what was discussed, what decisions were made, or \
                 finding a specific exchange. Returns matching messages \
                 with conversation title, ordinal, role, content, a \
                 highlighted snippet around the match, and a relevance \
                 rank. Hits where the conversation title or summary \
                 matches surface even if no individual message text does. \
                 Use this when the user asks about prior conversations \
                 (\"what did we discuss about X\", \"find where we talked \
                 about Y\").",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Full-text search query (English tsvector). Multi-word phrases are AND-ed."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Max hits to return (default 10)."
                        },
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant"],
                            "description": "Restrict matches to a specific role (omit to search all)."
                        }
                    },
                    "required": ["query"]
                }),
            ),
            ToolDefinition::new(
                TOOL_MCP_CONTROL,
                "Check status, start, stop, or restart MCP (Model Context Protocol) \
                 servers. Use this when a tool call fails because an MCP server is \
                 disconnected, or to inspect what servers are available.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "start", "stop", "restart"],
                            "description": "Action to perform"
                        },
                        "server": {
                            "type": "string",
                            "description": "Server name (omit for all servers)"
                        }
                    },
                    "required": ["action"]
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
                | TOOL_DB_QUERY
                | TOOL_MCP_CONTROL
                | TOOL_CONV_SEARCH
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
            TOOL_DB_QUERY => self.db_query(arguments).await,
            TOOL_MCP_CONTROL => self.mcp_control(arguments).await,
            TOOL_CONV_SEARCH => self.conversation_search(arguments).await,
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
        let write_fn = self
            .kb_write_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        let content = required_string(&arguments, "content")?;
        let tags = optional_string_array(&arguments, "tags");
        let metadata = arguments
            .get("metadata")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let id =
            optional_string(&arguments, "id").unwrap_or_else(|| uuid::Uuid::now_v7().to_string());

        let entry = desktop_assistant_core::domain::KnowledgeEntry {
            id,
            content: content.clone(),
            tags,
            metadata,
            created_at: String::new(),
            updated_at: String::new(),
        };

        // Generate chunked embeddings for the content.
        // If embedding fails, save the entry anyway with a NULL embedding so
        // the background backfill/dreaming cycle re-embeds it later.
        let embedding = self.embed_chunks(&content).await;
        let embedded = embedding.is_some();
        if self.embed_fn.is_some() && !embedded {
            tracing::warn!(
                "embedding failed for knowledge entry; saving without embedding (backfill will retry)"
            );
        }

        let saved = write_fn(entry, embedding).await?;

        Ok(serde_json::json!({
            "ok": true,
            "id": saved.id,
            "embedded": embedded,
            "created_at": saved.created_at,
            "updated_at": saved.updated_at,
        })
        .to_string())
    }

    async fn kb_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self
            .kb_search_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

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

    async fn conversation_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self.conversation_search_fn.as_ref().ok_or_else(|| {
            CoreError::ToolExecution("conversation search not configured".to_string())
        })?;

        let query = required_string(&arguments, "query")?;
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;
        let role_filter = arguments
            .get("role")
            .and_then(serde_json::Value::as_str)
            .and_then(|s| match s {
                "user" => Some(Role::User),
                "assistant" => Some(Role::Assistant),
                // Reject other roles at the boundary so the SQL layer
                // doesn't have to defend against arbitrary text.
                _ => None,
            });

        tracing::info!(query = %query, limit, ?role_filter, "conversation search");

        let hits = search_fn(query, limit, role_filter).await?;

        let items: Vec<serde_json::Value> = hits
            .into_iter()
            .map(|h| {
                serde_json::json!({
                    "conversation_id": h.conversation_id,
                    "conversation_title": h.conversation_title,
                    "ordinal": h.ordinal,
                    "role": match h.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::System => "system",
                        Role::Tool => "tool",
                    },
                    "snippet": h.snippet,
                    "content": h.content,
                    "rank": h.rank,
                    "updated_at": h.updated_at,
                })
            })
            .collect();

        tracing::info!(result_count = items.len(), "conversation search results");

        Ok(serde_json::json!({
            "ok": true,
            "results": items,
        })
        .to_string())
    }

    async fn kb_delete(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let delete_fn = self
            .kb_delete_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        let id = required_string(&arguments, "id")?;
        delete_fn(id.clone()).await?;

        Ok(serde_json::json!({
            "ok": true,
            "deleted": id,
        })
        .to_string())
    }

    async fn tool_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self
            .tool_search_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("tool registry not configured".to_string()))?;

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
        tracing::info!(
            result_count = tools.len(),
            ?tool_names,
            "tool search results"
        );

        Ok(serde_json::json!({
            "ok": true,
            "tools": tools,
        })
        .to_string())
    }

    async fn db_query(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let query_fn = self
            .db_query_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("database query not configured".to_string()))?;

        let query = required_string(&arguments, "query")?;
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(100) as usize;

        tracing::info!(limit, "executing db query");
        tracing::debug!(sql = %query, "db query SQL");

        let result = query_fn(query, limit).await?;

        Ok(serde_json::json!({
            "ok": true,
            "result": result,
        })
        .to_string())
    }

    async fn mcp_control(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let handle = self
            .mcp_handle
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("MCP control not configured".to_string()))?;

        let action = required_string(&arguments, "action")?;
        let server = optional_string(&arguments, "server");
        let server_ref = server.as_deref();

        match action.as_str() {
            "status" => {
                let statuses = handle.status(server_ref).await;
                Ok(serde_json::json!({
                    "ok": true,
                    "servers": statuses,
                })
                .to_string())
            }
            "start" => {
                let result = handle
                    .start_server(server_ref)
                    .await
                    .map_err(|e| CoreError::ToolExecution(format!("start failed: {e}")))?;
                let statuses = handle.status(server_ref).await;
                Ok(serde_json::json!({
                    "ok": true,
                    "message": result,
                    "servers": statuses,
                })
                .to_string())
            }
            "stop" => {
                let result = handle
                    .stop_server(server_ref)
                    .await
                    .map_err(|e| CoreError::ToolExecution(format!("stop failed: {e}")))?;
                let statuses = handle.status(server_ref).await;
                Ok(serde_json::json!({
                    "ok": true,
                    "message": result,
                    "servers": statuses,
                })
                .to_string())
            }
            "restart" => {
                let result = handle
                    .restart_server(server_ref)
                    .await
                    .map_err(|e| CoreError::ToolExecution(format!("restart failed: {e}")))?;
                let statuses = handle.status(server_ref).await;
                Ok(serde_json::json!({
                    "ok": true,
                    "message": result,
                    "servers": statuses,
                })
                .to_string())
            }
            _ => Err(CoreError::ToolExecution(format!(
                "unknown MCP control action: {action}"
            ))),
        }
    }

    /// Embed a single text string, returning None if embeddings are unavailable.
    /// Used for search queries which are always short and don't need chunking.
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

    /// Chunk text and embed each chunk, returning None if embeddings are unavailable.
    /// Used for KB writes where content may exceed the model's context window.
    async fn embed_chunks(&self, text: &str) -> Option<Vec<Vec<f32>>> {
        use desktop_assistant_core::chunking::{CHUNK_MAX_CHARS, CHUNK_OVERLAP, chunk_text};

        let embed_fn = self.embed_fn.as_ref()?;
        let chunks = chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        match embed_fn(chunks).await {
            Ok(vecs) if !vecs.is_empty() => Some(vecs),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!("failed to embed chunks: {e}");
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
        assert!(names.contains(&TOOL_DB_QUERY.to_string()));
        assert!(names.contains(&TOOL_MCP_CONTROL.to_string()));
        assert!(names.contains(&TOOL_CONV_SEARCH.to_string()));
    }

    #[tokio::test]
    async fn conversation_search_without_store_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_CONV_SEARCH, serde_json::json!({"query": "test"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn conversation_search_with_closure_returns_results() {
        use desktop_assistant_core::ports::conversation_search::{
            ConversationSearchFn, MessageHit,
        };
        use std::sync::Arc;

        let search_fn: ConversationSearchFn = Arc::new(move |query, limit, role_filter| {
            let q = query.clone();
            Box::pin(async move {
                assert_eq!(q, "deploy");
                assert_eq!(limit, 5);
                assert!(matches!(role_filter, Some(Role::Assistant)));
                Ok(vec![MessageHit {
                    conversation_id: "c-1".into(),
                    conversation_title: "Deploy timeline".into(),
                    ordinal: 4,
                    role: Role::Assistant,
                    content: "We can deploy on Friday".into(),
                    snippet: "We can <mark>deploy</mark> on Friday".into(),
                    rank: 0.42,
                    updated_at: "2026-05-02T13:00:00+00:00".into(),
                }])
            })
        });

        let service = BuiltinToolService::new().with_conversation_search(search_fn);
        let response = service
            .execute_tool(
                TOOL_CONV_SEARCH,
                serde_json::json!({"query": "deploy", "limit": 5, "role": "assistant"}),
            )
            .await
            .expect("search succeeds");

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(json["ok"], serde_json::json!(true));
        let results = json["results"].as_array().expect("results array");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["conversation_id"], "c-1");
        assert_eq!(results[0]["ordinal"], 4);
        assert_eq!(results[0]["role"], "assistant");
        assert!(results[0]["snippet"].as_str().unwrap().contains("<mark>"));
    }

    #[tokio::test]
    async fn conversation_search_rejects_unknown_role() {
        // Unknown roles must not reach the search closure: the boundary
        // strips them rather than passing through arbitrary text.
        use desktop_assistant_core::ports::conversation_search::ConversationSearchFn;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let saw_role_filter = Arc::new(AtomicBool::new(false));
        let saw_clone = Arc::clone(&saw_role_filter);
        let search_fn: ConversationSearchFn = Arc::new(move |_q, _l, role_filter| {
            if role_filter.is_some() {
                saw_clone.store(true, Ordering::SeqCst);
            }
            Box::pin(async { Ok(Vec::new()) })
        });

        let service = BuiltinToolService::new().with_conversation_search(search_fn);
        let _ = service
            .execute_tool(
                TOOL_CONV_SEARCH,
                serde_json::json!({"query": "x", "role": "robot"}),
            )
            .await
            .unwrap();
        assert!(
            !saw_role_filter.load(Ordering::SeqCst),
            "unknown role values must not propagate to the search closure"
        );
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
            .execute_tool(TOOL_KB_WRITE, serde_json::json!({"content": "test"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn kb_search_without_store_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_KB_SEARCH, serde_json::json!({"query": "test"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn db_query_without_database_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_DB_QUERY, serde_json::json!({"query": "SELECT 1"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn db_query_with_closure() {
        use desktop_assistant_core::ports::database::DbQueryFn;
        use std::sync::Arc;

        let query_fn: DbQueryFn = Arc::new(|_sql, _limit| {
            Box::pin(async {
                Ok(serde_json::json!({
                    "columns": ["count"],
                    "rows": [[42]],
                    "row_count": 1
                }))
            })
        });

        let service = BuiltinToolService::new().with_database(query_fn);

        let result = service
            .execute_tool(
                TOOL_DB_QUERY,
                serde_json::json!({"query": "SELECT count(*) FROM conversations"}),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["result"]["row_count"], 1);
        assert_eq!(json["result"]["rows"][0][0], 42);
    }

    #[tokio::test]
    async fn tool_search_without_registry_returns_error() {
        let service = BuiltinToolService::new();
        let result = service
            .execute_tool(TOOL_SEARCH, serde_json::json!({"query": "file operations"}))
            .await;
        assert!(matches!(result, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn kb_write_and_search_with_closures() {
        use desktop_assistant_core::domain::KnowledgeEntry;
        use std::sync::{Arc, Mutex};

        let store: Arc<Mutex<Vec<KnowledgeEntry>>> = Arc::new(Mutex::new(Vec::new()));

        let write_store = Arc::clone(&store);
        let write_fn: KnowledgeWriteFn =
            Arc::new(move |mut entry, _embedding: Option<Vec<Vec<f32>>>| {
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

        let delete_fn: KnowledgeDeleteFn = Arc::new(|_id| Box::pin(async { Ok(()) }));

        let service = BuiltinToolService::new().with_knowledge_base(write_fn, search_fn, delete_fn);

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
            .execute_tool(TOOL_KB_SEARCH, serde_json::json!({"query": "dark mode"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&search_result).unwrap();
        assert_eq!(json["ok"], true);
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0]["content"]
                .as_str()
                .unwrap()
                .contains("dark mode")
        );
    }

    #[tokio::test]
    async fn tool_search_with_closure() {
        use desktop_assistant_core::domain::ToolDefinition;
        use std::sync::Arc;

        let search_fn: ToolSearchFn = Arc::new(|_query, _emb, _limit| {
            Box::pin(async {
                Ok(vec![ToolDefinition::new(
                    "jira__create_issue",
                    "Create a Jira issue",
                    serde_json::json!({}),
                )])
            })
        });

        let def_fn: ToolDefinitionFn = Arc::new(|_name| Box::pin(async { Ok(None) }));

        let service = BuiltinToolService::new().with_tool_registry(search_fn, def_fn);

        let result = service
            .execute_tool(TOOL_SEARCH, serde_json::json!({"query": "create ticket"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["ok"], true);
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "jira__create_issue");
    }
}
