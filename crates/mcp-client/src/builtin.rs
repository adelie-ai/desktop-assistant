use std::fs;
use std::path::PathBuf;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::clock::NowSnapshot;
use desktop_assistant_core::domain::{Role, ToolDefinition};
use desktop_assistant_core::ports::conversation_ctx::current_conversation_id;
use desktop_assistant_core::ports::conversation_search::ConversationSearchFn;
use desktop_assistant_core::ports::database::DbQueryFn;
use desktop_assistant_core::ports::embedding::EmbedFn;
use desktop_assistant_core::ports::knowledge::{
    KnowledgeDeleteFn, KnowledgeGetFn, KnowledgeListFn, KnowledgeListQuery, KnowledgeSearchFn,
    KnowledgeWriteFn, ListOrder, ListOrderOpt,
};
use desktop_assistant_core::ports::notify::{NotifyFn, NotifyUrgency};
use desktop_assistant_core::ports::scratchpad::{
    MAX_KEYS_PER_CALL, MAX_NOTE_BYTES, MAX_NOTES_PER_WRITE, MAX_RESULTS_CEILING, NewScratchpadNote,
    RESPONSE_BYTE_BUDGET, ScratchpadClearFn, ScratchpadDeleteManyFn, ScratchpadGetManyFn,
    ScratchpadListFn, ScratchpadSearchFn, ScratchpadWriteFn,
};
use desktop_assistant_core::ports::skill_index::{SkillGetFn, SkillSearchFn};
use desktop_assistant_core::ports::tool_registry::{ToolDefinitionFn, ToolSearchFn};
use desktop_assistant_core::ports::transport::current_client_context;

use crate::executor::McpControlHandle;

const TOOL_KB_WRITE: &str = "builtin_knowledge_base_write";
const TOOL_KB_SEARCH: &str = "builtin_knowledge_base_search";
const TOOL_KB_DELETE: &str = "builtin_knowledge_base_delete";
const TOOL_KB_LIST: &str = "builtin_knowledge_base_list";
const TOOL_SEARCH: &str = "builtin_tool_search";
const TOOL_NOTIFY: &str = "builtin_notify";
const TOOL_SYS_PROPS: &str = "builtin_sys_props";
const TOOL_DB_QUERY: &str = "builtin_db_query";
const TOOL_MCP_CONTROL: &str = "builtin_mcp_control";
const TOOL_CONV_SEARCH: &str = "builtin_conversation_search";
const TOOL_SCRATCHPAD_WRITE: &str = "builtin_scratchpad_write";
const TOOL_SCRATCHPAD_SEARCH: &str = "builtin_scratchpad_search";
const TOOL_SCRATCHPAD_DELETE: &str = "builtin_scratchpad_delete";
const TOOL_SKILL_SEARCH: &str = "builtin_skill_search";
const TOOL_SKILL_GET: &str = "builtin_skill_get";

/// Hard cap on how long an embedding call may block a real-time request. A
/// slow/wedged embedding backend (e.g. a stuck Ollama) must not hang the turn:
/// on timeout we return no embedding, so semantic search falls back to FTS and
/// KB writes persist without an embedding for the background dreaming/backfill
/// cycle to fill in later.
const EMBED_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

pub struct BuiltinToolService {
    embed_fn: Option<EmbedFn>,
    kb_write_fn: Option<KnowledgeWriteFn>,
    kb_search_fn: Option<KnowledgeSearchFn>,
    kb_delete_fn: Option<KnowledgeDeleteFn>,
    kb_list_fn: Option<KnowledgeListFn>,
    kb_get_fn: Option<KnowledgeGetFn>,
    tool_search_fn: Option<ToolSearchFn>,
    #[allow(dead_code)]
    tool_definition_fn: Option<ToolDefinitionFn>,
    db_query_fn: Option<DbQueryFn>,
    mcp_handle: Option<McpControlHandle>,
    conversation_search_fn: Option<ConversationSearchFn>,
    scratchpad_write_fn: Option<ScratchpadWriteFn>,
    scratchpad_get_many_fn: Option<ScratchpadGetManyFn>,
    scratchpad_list_fn: Option<ScratchpadListFn>,
    scratchpad_search_fn: Option<ScratchpadSearchFn>,
    scratchpad_delete_many_fn: Option<ScratchpadDeleteManyFn>,
    scratchpad_clear_fn: Option<ScratchpadClearFn>,
    notify_fn: Option<NotifyFn>,
    skill_search_fn: Option<SkillSearchFn>,
    skill_get_fn: Option<SkillGetFn>,
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
            kb_list_fn: None,
            kb_get_fn: None,
            tool_search_fn: None,
            tool_definition_fn: None,
            db_query_fn: None,
            mcp_handle: None,
            conversation_search_fn: None,
            scratchpad_write_fn: None,
            scratchpad_get_many_fn: None,
            scratchpad_list_fn: None,
            scratchpad_search_fn: None,
            scratchpad_delete_many_fn: None,
            scratchpad_clear_fn: None,
            notify_fn: None,
            skill_search_fn: None,
            skill_get_fn: None,
        }
    }

    /// Configure the embedding function for generating query vectors.
    pub fn with_embedding(mut self, embed_fn: EmbedFn) -> Self {
        self.embed_fn = Some(embed_fn);
        self
    }

    /// Configure the desktop-notification closure (#`builtin_notify`).
    ///
    /// Capability-gated: the daemon only calls this when a notification
    /// service is present on the session bus, so the tool is simply absent on a
    /// headless host rather than failing at call time.
    pub fn with_notify(mut self, notify_fn: NotifyFn) -> Self {
        self.notify_fn = Some(notify_fn);
        self
    }

    /// Configure the skill-library closures (`builtin_skill_search` /
    /// `builtin_skill_get`). Wired only when a skill index is available, so the
    /// tools are simply absent otherwise (capability-gated like `builtin_notify`).
    pub fn with_skills(mut self, search_fn: SkillSearchFn, get_fn: SkillGetFn) -> Self {
        self.skill_search_fn = Some(search_fn);
        self.skill_get_fn = Some(get_fn);
        self
    }

    /// Configure knowledge base store closures.
    pub fn with_knowledge_base(
        mut self,
        write_fn: KnowledgeWriteFn,
        search_fn: KnowledgeSearchFn,
        delete_fn: KnowledgeDeleteFn,
        list_fn: KnowledgeListFn,
        get_fn: KnowledgeGetFn,
    ) -> Self {
        self.kb_write_fn = Some(write_fn);
        self.kb_search_fn = Some(search_fn);
        self.kb_delete_fn = Some(delete_fn);
        self.kb_list_fn = Some(list_fn);
        self.kb_get_fn = Some(get_fn);
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

    /// Configure the database-query closure for the `builtin_db_query`
    /// tool.
    ///
    /// ## Security posture (issue #141)
    ///
    /// The closure runs *arbitrary* LLM-supplied SQL. The implementation
    /// behind it (see `desktop_assistant_storage::execute_database_query`)
    /// enforces the following invariants before any text reaches the
    /// pool, so it is safe to wire the tool against the same pool used
    /// for ordinary application traffic:
    ///
    /// - **SELECT-only on the read path.** Only single-statement
    ///   `SELECT` / `WITH` / `TABLE` / `VALUES` / `EXPLAIN` queries
    ///   are accepted; everything else is parsed-and-rejected.
    /// - **Per-user (`user_id`) scoping by AST rewrite.** Every
    ///   reference to a personal-data table (`conversations`,
    ///   `messages`, `knowledge_base`, etc.) has a
    ///   `<table>.user_id = $N` predicate grafted into its `WHERE`
    ///   clause, bound to the caller's task-local `UserId`. An
    ///   LLM-supplied predicate naming a different user_id is AND'd
    ///   with the grafted one, so the intersection is empty.
    /// - **Compound statements rejected.** `SELECT 1; DROP TABLE …`
    ///   produces two statements at parse time and is refused.
    /// - **Writes confined to scratch.** DDL/DML that names a
    ///   personal-data table (qualified or otherwise) is rejected; the
    ///   write path's `search_path TO scratch, public` then carries
    ///   unqualified writes into the per-database `scratch` schema
    ///   only, so the LLM can still set up staging tables and
    ///   intermediate joins.
    ///
    /// Pre-#141 this docstring contained a single-line "read-only"
    /// claim — which the implementation did not enforce. The audit
    /// test `comment_in_builtin_rs_matches_actual_security_posture`
    /// in this file pins the wording against that regression.
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

    /// Configure the per-conversation scratchpad store closures (#184). The
    /// builtin tools resolve the active conversation from the task-local
    /// installed by the service dispatch loop; these closures forward to the
    /// store. When unset, the scratchpad tools return a clear error.
    #[allow(clippy::too_many_arguments)]
    pub fn with_scratchpad(
        mut self,
        write_fn: ScratchpadWriteFn,
        get_many_fn: ScratchpadGetManyFn,
        list_fn: ScratchpadListFn,
        search_fn: ScratchpadSearchFn,
        delete_many_fn: ScratchpadDeleteManyFn,
        clear_fn: ScratchpadClearFn,
    ) -> Self {
        self.scratchpad_write_fn = Some(write_fn);
        self.scratchpad_get_many_fn = Some(get_many_fn);
        self.scratchpad_list_fn = Some(list_fn);
        self.scratchpad_search_fn = Some(search_fn);
        self.scratchpad_delete_many_fn = Some(delete_many_fn);
        self.scratchpad_clear_fn = Some(clear_fn);
        self
    }

    /// Set the MCP control handle (used by builtin_mcp_control tool).
    pub fn set_mcp_control(&mut self, handle: McpControlHandle) {
        self.mcp_handle = Some(handle);
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = vec![
            ToolDefinition::new(
                TOOL_KB_WRITE,
                "Write or update knowledge base entries. Use for storing preferences, facts, \
                 instructions, project context, or any durable information the user wants remembered. \
                 Content should be self-contained prose that describes both the context (when/why \
                 this information is useful) and the information itself. Provide either a single \
                 entry (top-level `content`/`tags`/`id`) or a batch via `entries`. To update only \
                 the tags of an existing entry, pass its `id` and omit `content` — the existing \
                 content is preserved.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Self-contained prose describing the context and information. \
                                            Write naturally, e.g. 'The user lives at 123 Main St, Springfield. \
                                            Use this as their default location for weather, directions, and local searches.' \
                                            Do not use key-value format. Optional when `id` is given (tags-only update)."
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Two-level tags. Give a coarse KIND ('preference', 'memory', or 'instruction') PLUS at least one SPECIFIC facet: 'project:<name>', 'tool:<name>', 'topic:<subject>', or 'person:<name>'. Prefer specific over generic. Good: ['instruction', 'project:adelie-ai', 'topic:deploy']. Too generic: ['instruction']."
                        },
                        "id": {
                            "type": "string",
                            "description": "Optional ID for updates. Omit to create a new entry."
                        },
                        "entries": {
                            "type": "array",
                            "description": "Batch form: a list of {content?, tags?, id?} objects. When present, the top-level content/tags/id are ignored.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Two-level tags: a coarse KIND ('preference'/'memory'/'instruction') PLUS at least one SPECIFIC facet ('project:<name>', 'tool:<name>', 'topic:<subject>', 'person:<name>'). Prefer specific over generic."
                                    },
                                    "id": {"type": "string"}
                                }
                            }
                        }
                    }
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
                            "description": "Only return entries carrying at least one of these tags"
                        },
                        "exclude_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude entries carrying any of these tags"
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
                "Delete knowledge base entries by ID. Accepts a single `id` or a list of `ids`.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "ID of a single entry to delete"
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of multiple entries to delete in one call"
                        }
                    }
                }),
            ),
            ToolDefinition::new(
                TOOL_KB_LIST,
                "List knowledge base entries without a search query — a straight paginated \
                 enumeration for audits and review. Returns entries plus a `next_cursor`; pass it \
                 back as `cursor` to fetch the next page (null when there are no more).",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "description": "Max entries per page (default 50)"
                        },
                        "cursor": {
                            "type": "string",
                            "description": "Opaque pagination cursor from a previous page's next_cursor. Omit for the first page."
                        },
                        "order": {
                            "type": "string",
                            "enum": ["newest_first", "oldest_first"],
                            "description": "Sort direction by creation time (default newest_first)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only include entries carrying at least one of these tags"
                        },
                        "exclude_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude entries carrying any of these tags"
                        },
                        "source": {
                            "type": "string",
                            "description": "Only include entries with this provenance: 'extraction', 'consolidation', or 'explicit'"
                        }
                    }
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
                 You may also `CREATE SCHEMA` your own named schemas for durable \
                 tracking, and define tables, views, functions, and procedures in \
                 them; helper scripts that load or maintain data are fine too — see \
                 the database design section of your system prompt for conventions \
                 (naming, COMMENT ON, what not to touch in public).\n\n\
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
            ToolDefinition::new(
                TOOL_SCRATCHPAD_WRITE,
                "Add or update notes in this conversation's scratchpad — an ephemeral, \
                 per-conversation working store for facts you want to keep high in context \
                 right now (an evolving plan, open questions, a working set of IDs). Use it \
                 SELECTIVELY: only when you need to carry information forward across a large or \
                 multi-step task (a multi-step plan, investigation notes you'll reference later, \
                 intermediate results held across many turns). For small one-shot tasks — a \
                 single question, quick lookup, or one-line action — don't write here; just \
                 answer or act. Notes are \
                 keyed; writing the same key again replaces it. Pass `notes` to upsert several \
                 at once. Use the reserved key 'goal' for the current objective: it is \
                 auto-surfaced as your task anchor every turn (so it survives compaction), and \
                 you should evolve it as the goal shifts and delete it when done. Each note has \
                 a `type` (default \"note\") and an optional integer `sequence` (same-type notes \
                 sort by it). For working a multi-step task, prefer the begin_step / complete_step \
                 tools — they record and number your plan as todos for you and compact each finished \
                 step's raw work into a note — rather than hand-managing `todo` notes here. The \
                 scratchpad is discarded when the conversation is deleted and is NOT durable across \
                 conversations — promote anything worth keeping to the knowledge base with \
                 builtin_knowledge_base_write, then delete the note here.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string", "description": "Short handle for the note; upserts by key."},
                                    "content": {"type": "string", "description": "The note body (keep it small and high-signal)."},
                                    "type": {"type": "string", "description": "Category, e.g. \"todo\"/\"note\"/\"other\". Defaults to \"note\". Used for filtering/grouping; same-type notes sort by `sequence`."},
                                    "sequence": {"type": "integer", "description": "Optional ordering hint within the type (ascending). Use for ordered todos."},
                                    "done": {"type": "boolean", "description": "Whether this note (e.g. a todo) is checked off. Defaults to false."}
                                },
                                "required": ["key", "content"]
                            },
                            "description": "One or more notes to add/update in a single call."
                        },
                        "key": {"type": "string", "description": "Single-note convenience: the note key (use with `content`)."},
                        "content": {"type": "string", "description": "Single-note convenience: the note body (use with `key`)."},
                        "type": {"type": "string", "description": "Single-note convenience: the note type (default \"note\")."},
                        "sequence": {"type": "integer", "description": "Single-note convenience: ordering hint within the type."},
                        "done": {"type": "boolean", "description": "Single-note convenience: checked-off flag."}
                    }
                }),
            ),
            ToolDefinition::new(
                TOOL_SCRATCHPAD_SEARCH,
                "Read this conversation's scratchpad. Omit `query` and `keys` to list all notes \
                 (ordered by type, then `sequence`); pass `query` for a full-text search over \
                 note keys and content; pass `keys` to fetch specific notes. Pass `type` to \
                 restrict a list/search to one category, e.g. `type: \"todo\"` for just your \
                 plan. Each returned note includes its `type`, `sequence`, and `done`. \
                 `max_results` is required. Results are bounded — if the response is truncated \
                 you'll get `truncated: true` and should narrow with a `query`, a `type`, or a \
                 smaller key set.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Full-text query over note keys + content. Omit to list all notes."},
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fetch specific notes by key. Takes precedence over `query`."
                        },
                        "type": {"type": "string", "description": "Restrict a list/search to one note type, e.g. \"todo\". Ignored when `keys` is given."},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum notes to return (required; clamped to 100)."
                        }
                    },
                    "required": ["max_results"]
                }),
            ),
            ToolDefinition::new(
                TOOL_SCRATCHPAD_DELETE,
                "Delete notes from this conversation's scratchpad. Pass `keys` to delete \
                 specific notes, or `all: true` to clear the whole pad. Exactly one of the two \
                 must be supplied.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys of notes to delete."
                        },
                        "all": {
                            "type": "boolean",
                            "description": "Delete every note in this scratchpad. Mutually exclusive with `keys`."
                        }
                    }
                }),
            ),
        ];

        // Capability-gated: only advertise the notification tool when a
        // notification service was wired (present on the session bus).
        if self.notify_fn.is_some() {
            defs.push(ToolDefinition::new(
                TOOL_NOTIFY,
                "Show a desktop notification to the user via the system notification service. \
                 Use to surface something the user should see now — e.g. a long-running task \
                 finished, or a time-sensitive finding worth interrupting for. Prefer the normal \
                 reply for ordinary output; reserve notifications for things that warrant the \
                 user's attention away from the chat. Only available when a desktop notification \
                 service is present.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Short title line (a few words)."
                        },
                        "body": {
                            "type": "string",
                            "description": "Optional longer detail shown under the title."
                        },
                        "urgency": {
                            "type": "string",
                            "enum": ["low", "normal", "critical"],
                            "description": "Urgency (default normal). 'critical' stays on screen until dismissed; use sparingly."
                        }
                    },
                    "required": ["summary"]
                }),
            ));
        }

        // Capability-gated: only advertise the skill tools when a skill index
        // was wired (a Postgres pool + configured roots).
        if self.skill_search_fn.is_some() {
            defs.push(ToolDefinition::new(
                TOOL_SKILL_SEARCH,
                "Search the on-disk skill library — reusable how-to playbooks and workflows — by \
                 meaning. Call this before a recurring or procedural task to check whether an \
                 established skill already covers it, then read the full body with \
                 builtin_skill_get before following one.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What you are trying to do."},
                        "kind": {
                            "type": "string",
                            "enum": ["skill", "workflow"],
                            "description": "Optional filter: only plain skills or only workflows."
                        },
                        "limit": {"type": "integer", "description": "Max results (default 5)."}
                    },
                    "required": ["query"]
                }),
            ));
            defs.push(ToolDefinition::new(
                TOOL_SKILL_GET,
                "Fetch one skill by name: its full markdown body, on-disk path, attachment \
                 filenames, kind, and trust tier. Use after builtin_skill_search to read a skill \
                 before following it.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The skill name."},
                        "owner": {
                            "type": "string",
                            "description": "Omit for a global skill; a user id for a user-scoped one."
                        }
                    },
                    "required": ["name"]
                }),
            ));
        }

        defs
    }

    pub fn supports_tool(name: &str) -> bool {
        matches!(
            name,
            TOOL_KB_WRITE
                | TOOL_KB_SEARCH
                | TOOL_KB_DELETE
                | TOOL_KB_LIST
                | TOOL_SEARCH
                | TOOL_NOTIFY
                | TOOL_SYS_PROPS
                | TOOL_DB_QUERY
                | TOOL_MCP_CONTROL
                | TOOL_CONV_SEARCH
                | TOOL_SCRATCHPAD_WRITE
                | TOOL_SCRATCHPAD_SEARCH
                | TOOL_SCRATCHPAD_DELETE
                | TOOL_SKILL_SEARCH
                | TOOL_SKILL_GET
        )
    }

    /// The builtin provider groups (Phase 1): a stable group id plus the authored
    /// blurb that seeds the group's synthetic `provider:<id>` row. Built-ins are
    /// surfaced to tool-search by the SAME provider mechanism as external MCP
    /// servers — this classification is what unifies them.
    pub const PROVIDER_GROUPS: &'static [(&'static str, &'static str)] = &[
        (
            "knowledge",
            "Long-term memory: store and recall the user's preferences, facts, \
             instructions, and project context as durable tagged entries, via hybrid \
             vector + full-text search.",
        ),
        (
            "scratchpad",
            "Ephemeral per-conversation working notes: hold a plan, findings, and \
             intermediate results across a multi-step task; discarded when the \
             conversation ends.",
        ),
        (
            "database",
            "Run SQL against the assistant's own PostgreSQL database to inspect or \
             modify its conversations, messages, knowledge, and tool data, and to \
             build your own schemas/views.",
        ),
        (
            "recall",
            "Search past conversations by full-text query to recall what was \
             discussed or decided.",
        ),
        (
            "system",
            "System and desktop touchpoints: read runtime/system context and raise \
             desktop notifications for things that need attention now.",
        ),
        (
            "tool-meta",
            "Discover additional tools by description and manage the MCP servers that \
             provide them (status/start/stop/restart).",
        ),
        (
            "skills",
            "Reusable how-to playbooks and workflows on disk: find an established skill \
             for a recurring or procedural task and read its steps before acting.",
        ),
    ];

    /// Every builtin tool name, including the capability-gated `builtin_notify`
    /// (absent at runtime when no notifier is wired). The exhaustiveness guard
    /// walks this so a NEW builtin without a provider mapping fails the build.
    pub const ALL_TOOL_NAMES: &'static [&'static str] = &[
        TOOL_KB_WRITE,
        TOOL_KB_SEARCH,
        TOOL_KB_DELETE,
        TOOL_KB_LIST,
        TOOL_SEARCH,
        TOOL_NOTIFY,
        TOOL_SYS_PROPS,
        TOOL_DB_QUERY,
        TOOL_MCP_CONTROL,
        TOOL_CONV_SEARCH,
        TOOL_SCRATCHPAD_WRITE,
        TOOL_SCRATCHPAD_SEARCH,
        TOOL_SCRATCHPAD_DELETE,
        TOOL_SKILL_SEARCH,
        TOOL_SKILL_GET,
    ];

    /// Classify a builtin tool name into its provider group, or `None` when the
    /// name is not a known builtin. Callers that must register every builtin
    /// (never drop one) fall back to a generic group on `None`; the
    /// `builtin_provider_map_is_exhaustive` test ensures no known builtin relies
    /// on that fallback.
    pub fn provider_group(tool_name: &str) -> Option<&'static str> {
        match tool_name {
            TOOL_KB_WRITE | TOOL_KB_SEARCH | TOOL_KB_DELETE | TOOL_KB_LIST => Some("knowledge"),
            TOOL_SCRATCHPAD_WRITE | TOOL_SCRATCHPAD_SEARCH | TOOL_SCRATCHPAD_DELETE => {
                Some("scratchpad")
            }
            TOOL_DB_QUERY => Some("database"),
            TOOL_CONV_SEARCH => Some("recall"),
            TOOL_SYS_PROPS | TOOL_NOTIFY => Some("system"),
            TOOL_SEARCH | TOOL_MCP_CONTROL => Some("tool-meta"),
            TOOL_SKILL_SEARCH | TOOL_SKILL_GET => Some("skills"),
            _ => None,
        }
    }

    /// The authored blurb for a provider group id, or `None` if unknown.
    pub fn provider_blurb(provider: &str) -> Option<&'static str> {
        Self::PROVIDER_GROUPS
            .iter()
            .find(|(id, _)| *id == provider)
            .map(|(_, blurb)| *blurb)
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
            TOOL_KB_LIST => self.kb_list(arguments).await,
            TOOL_SEARCH => self.tool_search(arguments).await,
            TOOL_NOTIFY => self.notify(arguments).await,
            TOOL_SYS_PROPS => Ok(self.sys_props()),
            TOOL_DB_QUERY => self.db_query(arguments).await,
            TOOL_MCP_CONTROL => self.mcp_control(arguments).await,
            TOOL_CONV_SEARCH => self.conversation_search(arguments).await,
            TOOL_SCRATCHPAD_WRITE => self.scratchpad_write(arguments).await,
            TOOL_SCRATCHPAD_SEARCH => self.scratchpad_search(arguments).await,
            TOOL_SCRATCHPAD_DELETE => self.scratchpad_delete(arguments).await,
            TOOL_SKILL_SEARCH => self.skill_search(arguments).await,
            TOOL_SKILL_GET => self.skill_get(arguments).await,
            _ => Err(CoreError::ToolExecution(format!(
                "unknown built-in tool: {name}"
            ))),
        }
    }

    fn sys_props(&self) -> String {
        let now = NowSnapshot::now();

        // Prefer the CONNECTING CLIENT's self-reported context (#549/#558) for
        // the user + device identity fields. The daemon may be remote or
        // containerized, so its own host env is NOT the user's environment —
        // reporting the daemon host AS the user (the pre-#558 bug) is wrong.
        // We fall back to daemon-host detection only when the client sent no
        // context, and label the source with `identity_source` so daemon-host
        // values are never mistaken for the client's. An empty context counts
        // as absent (fail-closed), and fields the client omitted stay null
        // rather than borrowing the daemon's — a partial client context never
        // leaks a daemon-host value as if it were the user's.
        let client = current_client_context().filter(|c| !c.is_empty());

        let identity = match &client {
            Some(c) => IdentityFields {
                source: "client",
                real_name: c.real_name.clone(),
                username: c.username.clone(),
                home_dir: c.home_dir.clone(),
                hostname: c.hostname.clone(),
                os: c.os.clone(),
                timezone: c.timezone.clone(),
            },
            None => IdentityFields {
                source: "daemon_host_fallback",
                // The daemon host has no notion of the user's real name.
                real_name: None,
                username: detect_username(),
                home_dir: detect_home_dir(),
                hostname: detect_hostname(),
                os: Some(std::env::consts::OS.to_string()),
                timezone: Some(now.timezone()),
            },
        };

        serde_json::json!({
            "ok": true,
            "props": {
                "note": "`identity_source` says where the user/device fields came \
                         from: \"client\" = the connecting client reported them; \
                         \"daemon_host_fallback\" = the client sent none, so these \
                         are the daemon host's own values and may not be the \
                         user's. Server-side tools (file, terminal) run on the \
                         daemon host: relative paths resolve from `daemon_host.cwd`, \
                         not the client's home.",
                "generated_at_epoch": now.epoch_secs(),
                "generated_at_utc": now.utc_rfc3339(),
                "identity_source": identity.source,
                "real_name": identity.real_name,
                "username": identity.username,
                "home_dir": identity.home_dir,
                "hostname": identity.hostname,
                "os": identity.os,
                "timezone": identity.timezone,
                "daemon_host": {
                    "cwd": detect_daemon_cwd(),
                    "generated_at_local": now.local_rfc3339(),
                    "timezone": now.timezone(),
                    "hostname": detect_hostname(),
                    "os": std::env::consts::OS,
                    "arch": std::env::consts::ARCH,
                    "os_version": detect_os_version(),
                    "username": detect_username(),
                    "home_dir": detect_home_dir(),
                    "xdg_dirs": detect_xdg_dirs(),
                    "shell": detect_shell(),
                    "locale": detect_locale(),
                    "session_type": detect_session_type(),
                },
            },
        })
        .to_string()
    }

    async fn kb_write(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let write_fn = self
            .kb_write_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        // Batch form (`entries`) takes precedence over the single top-level
        // form. Each spec is one {content?, tags?, id?} object.
        let specs: Vec<serde_json::Value> = match arguments.get("entries") {
            Some(serde_json::Value::Array(items)) => items.clone(),
            _ => vec![arguments.clone()],
        };

        let mut saved_out = Vec::with_capacity(specs.len());
        for spec in &specs {
            let entry = self.build_write_entry(spec).await?;
            // Embedding generation is decoupled from the write: the entry lands
            // immediately (NULL embedding on create, stale embedding left in
            // place on update) and the background embedding-backfill task
            // generates the vector within its next pass. The row is
            // keyword-searchable (FTS) right away; semantic recall follows.
            let saved = write_fn(entry).await?;
            saved_out.push(serde_json::json!({
                "id": saved.id,
                "created_at": saved.created_at,
                "updated_at": saved.updated_at,
            }));
        }

        Ok(serde_json::json!({
            "ok": true,
            "count": saved_out.len(),
            "entries": saved_out,
        })
        .to_string())
    }

    /// Build a [`KnowledgeEntry`] from one write spec. When `content` is
    /// omitted and an `id` is given, the existing entry is fetched and its
    /// content (and, if `tags` is also omitted, its tags) are preserved — a
    /// tags-only / re-tag / promote-to-explicit update. Tool-authored writes
    /// always carry `source = "explicit"`.
    async fn build_write_entry(
        &self,
        spec: &serde_json::Value,
    ) -> Result<desktop_assistant_core::domain::KnowledgeEntry, CoreError> {
        use desktop_assistant_core::domain::KnowledgeEntry;

        let content_opt = optional_string(spec, "content");
        let id_opt = optional_string(spec, "id");
        let tags_present = spec.get("tags").is_some();
        let tags = optional_string_array(spec, "tags");

        // Partial update: no content, but an id to look up.
        let existing = if content_opt.is_none() {
            let id = id_opt.clone().ok_or_else(|| {
                CoreError::ToolExecution(
                    "knowledge_base write requires `content`, or an `id` of an existing entry to \
                     update its tags"
                        .to_string(),
                )
            })?;
            let get_fn = self.kb_get_fn.as_ref().ok_or_else(|| {
                CoreError::ToolExecution("knowledge base not configured".to_string())
            })?;
            Some(get_fn(id.clone()).await?.ok_or_else(|| {
                CoreError::ToolExecution(format!("no knowledge entry with id {id}"))
            })?)
        } else {
            None
        };

        let id = id_opt.unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
        let content = content_opt
            .or_else(|| existing.as_ref().map(|e| e.content.clone()))
            .ok_or_else(|| {
                CoreError::ToolExecution("knowledge_base write requires content".into())
            })?;
        let tags = if tags_present {
            tags
        } else {
            existing
                .as_ref()
                .map(|e| e.tags.clone())
                .unwrap_or_default()
        };
        let mut metadata = existing
            .as_ref()
            .map(|e| e.metadata.clone())
            .unwrap_or_else(|| serde_json::json!({}));

        // Provenance (#240): stamp the originating conversation so a tool-saved
        // finding is traceable back to where it was learned. Only when a
        // conversation scope is active and it isn't already set.
        if let Some(conv) = current_conversation_id()
            && let Some(obj) = metadata.as_object_mut()
            && !obj.contains_key("source_conversation_id")
        {
            obj.insert(
                "source_conversation_id".to_string(),
                serde_json::Value::String(conv.0),
            );
        }

        Ok(KnowledgeEntry {
            id,
            content,
            tags,
            metadata,
            created_at: String::new(),
            updated_at: String::new(),
            source: Some("explicit".to_string()),
        })
    }

    async fn kb_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self
            .kb_search_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        let query = required_string(&arguments, "query")?;
        let tags = optional_string_array_nonempty(&arguments, "tags");
        let exclude_tags = optional_string_array_nonempty(&arguments, "exclude_tags");
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10) as usize;

        tracing::info!(query = %query, ?tags, ?exclude_tags, limit, "knowledge base search");

        let query_embedding = self.embed_text(&query).await.unwrap_or_default();

        let results = search_fn(query, query_embedding, tags, exclude_tags, limit).await?;

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

    async fn skill_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let search_fn = self
            .skill_search_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("skill library not configured".to_string()))?;

        let query = required_string(&arguments, "query")?;
        let kind_filter = arguments
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(5) as usize;

        tracing::info!(query = %query, ?kind_filter, limit, "skill search");

        let query_embedding = self.embed_text(&query).await.unwrap_or_default();
        // Over-fetch when filtering by kind, then trim to the requested limit.
        let fetch = if kind_filter.is_some() {
            limit.saturating_mul(3)
        } else {
            limit
        };
        let mut results = search_fn(query, query_embedding, fetch).await?;
        if let Some(kind) = &kind_filter {
            results.retain(|s| s.kind.as_str() == kind);
        }
        results.truncate(limit);

        let items: Vec<serde_json::Value> = results
            .into_iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "description": s.description,
                    "kind": s.kind.as_str(),
                    "trust_tier": s.trust_tier.as_str(),
                    "disk_path": s.disk_path,
                    "attachments": s.attachments,
                })
            })
            .collect();

        Ok(serde_json::json!({ "ok": true, "results": items }).to_string())
    }

    async fn skill_get(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let get_fn = self
            .skill_get_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("skill library not configured".to_string()))?;

        let name = required_string(&arguments, "name")?;
        let owner = arguments
            .get("owner")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);

        match get_fn(name.clone(), owner).await? {
            Some(s) => Ok(serde_json::json!({
                "ok": true,
                "name": s.name,
                "description": s.description,
                "kind": s.kind.as_str(),
                "trust_tier": s.trust_tier.as_str(),
                "disk_path": s.disk_path,
                "attachments": s.attachments,
                "tags": s.tags,
                "body": s.body,
            })
            .to_string()),
            None => Ok(
                serde_json::json!({ "ok": false, "reason": format!("no skill named {name}") })
                    .to_string(),
            ),
        }
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

        // Accept either a single `id` or a list of `ids`.
        let mut ids = optional_string_array(&arguments, "ids");
        if let Some(id) = optional_string(&arguments, "id") {
            ids.push(id);
        }
        if ids.is_empty() {
            return Err(CoreError::ToolExecution(
                "knowledge_base delete requires `id` or `ids`".to_string(),
            ));
        }

        let deleted = delete_fn(ids.clone()).await?;

        Ok(serde_json::json!({
            "ok": true,
            "deleted": deleted,
            "ids": ids,
        })
        .to_string())
    }

    async fn kb_list(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let list_fn = self
            .kb_list_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("knowledge base not configured".to_string()))?;

        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(50) as usize;
        let order = match arguments.get("order").and_then(serde_json::Value::as_str) {
            Some("oldest_first") => ListOrder::OldestFirst,
            _ => ListOrder::NewestFirst,
        };
        let query = KnowledgeListQuery {
            limit,
            after: optional_string(&arguments, "cursor"),
            order: ListOrderOpt(order),
            tags: optional_string_array_nonempty(&arguments, "tags"),
            exclude_tags: optional_string_array_nonempty(&arguments, "exclude_tags"),
            source: optional_string(&arguments, "source"),
        };

        let page = list_fn(query).await?;

        let items: Vec<serde_json::Value> = page
            .entries
            .into_iter()
            .map(|entry| {
                serde_json::json!({
                    "id": entry.id,
                    "content": entry.content,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "source": entry.source,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "ok": true,
            "count": items.len(),
            "entries": items,
            "next_cursor": page.next_cursor,
        })
        .to_string())
    }

    async fn notify(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let notify_fn = self.notify_fn.as_ref().ok_or_else(|| {
            CoreError::ToolExecution("desktop notifications are not available".to_string())
        })?;

        let summary = required_string(&arguments, "summary")?;
        let body = optional_string(&arguments, "body").unwrap_or_default();
        let urgency =
            NotifyUrgency::parse(arguments.get("urgency").and_then(serde_json::Value::as_str));

        match notify_fn(summary, body, urgency).await? {
            Some(id) => Ok(serde_json::json!({ "ok": true, "shown": true, "id": id }).to_string()),
            // Suppressed by rate-limiting (e.g. an identical notification just
            // fired) — report it without making it an error.
            None => Ok(serde_json::json!({
                "ok": true,
                "shown": false,
                "reason": "suppressed (duplicate of a recent notification)"
            })
            .to_string()),
        }
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

    /// Resolve the conversation the scratchpad tools operate on from the
    /// task-local installed by the service dispatch loop. Errors clearly when
    /// no conversation scope is active (e.g. a non-conversation tool call).
    fn scratchpad_conversation() -> Result<String, CoreError> {
        current_conversation_id().map(|c| c.0).ok_or_else(|| {
            CoreError::ToolExecution("scratchpad requires an active conversation".to_string())
        })
    }

    async fn scratchpad_write(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let conversation_id = Self::scratchpad_conversation()?;
        let write_fn = self
            .scratchpad_write_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("scratchpad not configured".to_string()))?;

        // Accept either a `notes` array or a single `key`+`content`. Each note
        // may carry an optional `type` (default "note"), `sequence`, and `done`.
        let raw: Vec<NewScratchpadNote> =
            if let Some(arr) = arguments.get("notes").and_then(serde_json::Value::as_array) {
                arr.iter().filter_map(parse_new_note).collect()
            } else if arguments.get("key").is_some() || arguments.get("content").is_some() {
                match parse_new_note(&arguments) {
                    Some(note) => vec![note],
                    None => Vec::new(),
                }
            } else {
                return Err(CoreError::ToolExecution(
                "scratchpad_write requires `notes: [{key, content}]` or a single `key` + `content`"
                    .to_string(),
            ));
            };

        if raw.is_empty() {
            return Err(CoreError::ToolExecution(
                "scratchpad_write: no notes provided".to_string(),
            ));
        }

        // Validate each note, then dedupe repeated keys last-wins (a single
        // INSERT can't carry a duplicate ON CONFLICT target). Invalid notes
        // are reported individually rather than failing the whole call.
        let mut rejected: Vec<serde_json::Value> = Vec::new();
        let mut accepted: Vec<NewScratchpadNote> = Vec::new();
        for note in raw {
            if note.key.is_empty() {
                rejected.push(serde_json::json!({"key": note.key, "reason": "empty key"}));
                continue;
            }
            if note.content.len() > MAX_NOTE_BYTES {
                rejected.push(serde_json::json!({
                    "key": note.key,
                    "reason": format!("content exceeds {MAX_NOTE_BYTES} bytes")
                }));
                continue;
            }
            if let Some(existing) = accepted.iter_mut().find(|n| n.key == note.key) {
                *existing = note;
            } else {
                accepted.push(note);
            }
        }

        // Bound the batch: anything past the per-call cap is reported as skipped.
        let mut truncated = false;
        let mut skipped: Vec<String> = Vec::new();
        if accepted.len() > MAX_NOTES_PER_WRITE {
            truncated = true;
            skipped = accepted
                .split_off(MAX_NOTES_PER_WRITE)
                .into_iter()
                .map(|n| n.key)
                .collect();
        }

        let saved = if accepted.is_empty() {
            Vec::new()
        } else {
            write_fn(conversation_id, accepted).await?
        };

        let written: Vec<serde_json::Value> = saved
            .iter()
            .map(|n| serde_json::json!({"key": n.key, "id": n.id, "updated_at": n.updated_at}))
            .collect();

        let mut response = serde_json::json!({"ok": true, "written": written});
        if !rejected.is_empty() {
            response["rejected"] = serde_json::Value::Array(rejected);
        }
        if truncated {
            response["truncated"] = serde_json::Value::Bool(true);
            response["skipped"] = serde_json::json!(skipped);
            response["message"] = serde_json::json!(format!(
                "only the first {MAX_NOTES_PER_WRITE} notes were written; call again with the rest"
            ));
        }
        Ok(response.to_string())
    }

    async fn scratchpad_search(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let conversation_id = Self::scratchpad_conversation()?;

        // `max_results` is required and clamped so a single read is bounded.
        let max_results = arguments
            .get("max_results")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                CoreError::ToolExecution("scratchpad_search requires `max_results`".to_string())
            })? as usize;
        let limit = max_results.clamp(1, MAX_RESULTS_CEILING);

        let keys = optional_string_array(&arguments, "keys");
        let query = optional_string(&arguments, "query");
        // Optional structured filter restricting list/search to one note_type
        // (e.g. only `todo`s). Ignored on the by-keys path (keys are explicit).
        let note_type = optional_string(&arguments, "type");

        // Mode precedence: keys -> query -> list-all. Each path is bounded.
        let mut keys_truncated = false;
        let results =
            if !keys.is_empty() {
                let get_many = self.scratchpad_get_many_fn.as_ref().ok_or_else(|| {
                    CoreError::ToolExecution("scratchpad not configured".to_string())
                })?;
                let mut keys = keys;
                if keys.len() > MAX_KEYS_PER_CALL {
                    keys_truncated = true;
                    keys.truncate(MAX_KEYS_PER_CALL);
                }
                get_many(conversation_id, keys, limit).await?
            } else if let Some(query) = query {
                let search = self.scratchpad_search_fn.as_ref().ok_or_else(|| {
                    CoreError::ToolExecution("scratchpad not configured".to_string())
                })?;
                search(conversation_id, query, note_type, limit).await?
            } else {
                let list = self.scratchpad_list_fn.as_ref().ok_or_else(|| {
                    CoreError::ToolExecution("scratchpad not configured".to_string())
                })?;
                list(conversation_id, note_type, limit).await?
            };

        let hit_limit = results.len() >= limit;

        // Enforce the response byte budget so one read can't blow out context.
        // Always include at least one entry even if it alone is large.
        let mut items: Vec<serde_json::Value> = Vec::new();
        let mut bytes = 0usize;
        let mut budget_truncated = false;
        for note in &results {
            let entry = serde_json::json!({
                "key": note.key,
                "content": note.content,
                "type": note.note_type,
                "sequence": note.sequence,
                "done": note.done,
                "updated_at": note.updated_at,
            });
            let size = entry.to_string().len();
            if !items.is_empty() && bytes + size > RESPONSE_BYTE_BUDGET {
                budget_truncated = true;
                break;
            }
            bytes += size;
            items.push(entry);
        }

        let truncated = keys_truncated || budget_truncated || hit_limit;
        let mut response =
            serde_json::json!({"ok": true, "results": items.clone(), "returned": items.len()});
        if truncated {
            response["truncated"] = serde_json::Value::Bool(true);
            response["message"] = serde_json::json!(
                "results were truncated; narrow with a `query`, fewer `keys`, or a smaller scope"
            );
        }
        Ok(response.to_string())
    }

    async fn scratchpad_delete(&self, arguments: serde_json::Value) -> Result<String, CoreError> {
        let conversation_id = Self::scratchpad_conversation()?;

        let all = arguments
            .get("all")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let keys = optional_string_array(&arguments, "keys");

        // Exactly one mode: refuse both/neither so a stray arg can't mass-delete.
        if all && !keys.is_empty() {
            return Err(CoreError::ToolExecution(
                "scratchpad_delete: pass either `keys` or `all`, not both".to_string(),
            ));
        }
        if !all && keys.is_empty() {
            return Err(CoreError::ToolExecution(
                "scratchpad_delete requires `keys: [...]` or `all: true`".to_string(),
            ));
        }

        if all {
            let clear = self
                .scratchpad_clear_fn
                .as_ref()
                .ok_or_else(|| CoreError::ToolExecution("scratchpad not configured".to_string()))?;
            let deleted = clear(conversation_id).await?;
            return Ok(serde_json::json!({"ok": true, "deleted": deleted}).to_string());
        }

        let delete_many = self
            .scratchpad_delete_many_fn
            .as_ref()
            .ok_or_else(|| CoreError::ToolExecution("scratchpad not configured".to_string()))?;
        let requested = keys.len();
        let mut keys = keys;
        let mut truncated = false;
        if keys.len() > MAX_KEYS_PER_CALL {
            truncated = true;
            keys.truncate(MAX_KEYS_PER_CALL);
        }
        let deleted = delete_many(conversation_id, keys).await?;

        let mut response =
            serde_json::json!({"ok": true, "deleted": deleted, "requested": requested});
        if truncated {
            response["truncated"] = serde_json::Value::Bool(true);
            response["message"] = serde_json::json!(format!(
                "only the first {MAX_KEYS_PER_CALL} keys were processed; call again for the rest"
            ));
        }
        Ok(response.to_string())
    }

    /// Embed a single text string, returning None if embeddings are unavailable.
    /// Used for search queries which are always short and don't need chunking.
    async fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        let embed_fn = self.embed_fn.as_ref()?;
        match tokio::time::timeout(EMBED_TIMEOUT, embed_fn(vec![text.to_string()])).await {
            Ok(Ok(mut vecs)) => vecs.pop(),
            Ok(Err(e)) => {
                tracing::warn!("failed to embed text: {e}");
                None
            }
            Err(_) => {
                tracing::warn!(
                    timeout = ?EMBED_TIMEOUT,
                    "embedding timed out; falling back to full-text search"
                );
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

/// Parse one scratchpad note object (`{key, content, type?, sequence?, done?}`)
/// into a [`NewScratchpadNote`]. Returns `None` when `key` or `content` is
/// absent (the caller treats that as a malformed note). `type` defaults to
/// [`DEFAULT_NOTE_TYPE`]; the key is trimmed (emptiness is validated upstream).
fn parse_new_note(obj: &serde_json::Value) -> Option<NewScratchpadNote> {
    let key = obj.get("key").and_then(serde_json::Value::as_str)?;
    let content = obj.get("content").and_then(serde_json::Value::as_str)?;
    let note_type = obj
        .get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(desktop_assistant_core::domain::DEFAULT_NOTE_TYPE)
        .to_string();
    let sequence = obj
        .get("sequence")
        .and_then(serde_json::Value::as_i64)
        .map(|v| v as i32);
    let done = obj
        .get("done")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Some(NewScratchpadNote {
        key: key.trim().to_string(),
        content: content.to_string(),
        note_type,
        sequence,
        done,
    })
}

/// The user/device identity fields of `builtin_sys_props`, resolved once from
/// either the connecting client's self-reported context or the daemon-host
/// fallback. A named struct (rather than a bare tuple) keeps the two resolution
/// arms and the JSON assembly legible about which value is which (#558).
struct IdentityFields {
    /// Where the fields came from: `"client"` or `"daemon_host_fallback"`.
    source: &'static str,
    real_name: Option<String>,
    username: Option<String>,
    home_dir: Option<String>,
    hostname: Option<String>,
    os: Option<String>,
    timezone: Option<String>,
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
    fn builtin_provider_map_is_exhaustive() {
        // Every known builtin tool must classify into one of the authored
        // PROVIDER_GROUPS — so a NEW builtin added without a mapping fails here
        // instead of silently registering unclassified (spec requirement).
        let group_ids: Vec<&str> = BuiltinToolService::PROVIDER_GROUPS
            .iter()
            .map(|(id, _)| *id)
            .collect();
        for name in BuiltinToolService::ALL_TOOL_NAMES {
            let group = BuiltinToolService::provider_group(name).unwrap_or_else(|| {
                panic!("builtin '{name}' has no provider group — classify it in provider_group()")
            });
            assert!(
                group_ids.contains(&group),
                "builtin '{name}' maps to '{group}', which is not an authored PROVIDER_GROUP"
            );
        }
        // ALL_TOOL_NAMES must also cover everything supports_tool accepts and
        // everything the default service actually emits (notify aside) — a
        // classified name that is not a real builtin, or a real builtin missing
        // from the list, is a drift bug.
        for name in BuiltinToolService::ALL_TOOL_NAMES {
            assert!(
                BuiltinToolService::supports_tool(name),
                "ALL_TOOL_NAMES lists '{name}', which supports_tool rejects"
            );
        }
        for def in BuiltinToolService::new().tool_definitions() {
            assert!(
                BuiltinToolService::provider_group(&def.name).is_some(),
                "runtime builtin '{}' is unclassified",
                def.name
            );
        }
    }

    /// The pre-#141 docstring on `with_database` claimed "read-only SQL
    /// access" — which the implementation did not enforce. Comment-vs-
    /// behaviour drift on a security-relevant surface is a real bug;
    /// the audit pass in #141 surfaced exactly this kind of drift on
    /// the `execute_database_query` tool.
    ///
    /// This test pins the docstring against the post-#141 contract.
    /// If you change the wording, update this test in the same commit
    /// so the assertion still describes what the code actually does.
    ///
    /// The check reads the source file at compile time via
    /// `include_str!` so we're asserting against the *literal* text
    /// the reviewer will see, not against something the compiler
    /// could fold away.
    #[test]
    fn comment_in_builtin_rs_matches_actual_security_posture() {
        const SRC: &str = include_str!("builtin.rs");

        // Locate the doc-comment block immediately preceding
        // `pub fn with_database(`. The block is the contiguous run of
        // `///` lines above the function signature.
        let fn_pos = SRC
            .find("pub fn with_database(")
            .expect("with_database fn declaration must exist");
        let preceding = &SRC[..fn_pos];
        let doc_block: String = preceding
            .lines()
            .rev()
            .take_while(|l| {
                let t = l.trim_start();
                t.starts_with("///") || t.is_empty()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
            .to_ascii_lowercase();

        // Forbidden: the misleading "read-only" claim from before
        // #141. It's misleading in two ways — the tool *did* allow
        // writes (to the scratch namespace and, footgun, to qualified
        // public tables), and even the "read-only" reads were
        // unscoped across tenants.
        assert!(
            !doc_block.contains("read-only sql access"),
            "with_database docstring still claims `read-only SQL access`; \
             pre-#141 wording is back. Current block:\n---\n{doc_block}\n---"
        );

        // Required: the doc must surface the two facts the LLM-
        // exposed tool actually enforces post-#141 — SELECT-only and
        // per-user scoping. Word choice is flexible (`scoped` /
        // `tenant` / `user_id` all read as the same thing); the test
        // just refuses an empty mention.
        assert!(
            doc_block.contains("select"),
            "with_database docstring must mention SELECT-only enforcement. \
             Current block:\n---\n{doc_block}\n---"
        );
        assert!(
            doc_block.contains("user_id")
                || doc_block.contains("per-user")
                || doc_block.contains("tenant"),
            "with_database docstring must mention per-user / user_id / tenant scoping. \
             Current block:\n---\n{doc_block}\n---"
        );
    }

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
        assert!(names.contains(&TOOL_SCRATCHPAD_WRITE.to_string()));
        assert!(names.contains(&TOOL_SCRATCHPAD_SEARCH.to_string()));
        assert!(names.contains(&TOOL_SCRATCHPAD_DELETE.to_string()));
    }

    #[test]
    fn kb_write_tags_description_urges_specific_facets() {
        // Generic tags ("instruction", "memory") make KB entries fragment and
        // over-surface. Both the single-write and the batch `tags` schema
        // descriptions must push the two-level rule (a specific facet, not just
        // a bare kind) so the in-schema hint matches the system-prompt guidance.
        let service = BuiltinToolService::new();
        let def = service
            .tool_definitions()
            .into_iter()
            .find(|t| t.name == TOOL_KB_WRITE)
            .expect("kb_write tool is advertised");
        let props = &def.parameters["properties"];

        let single = props["tags"]["description"]
            .as_str()
            .expect("single-write tags has a description");
        assert!(
            single.to_lowercase().contains("specific"),
            "single-write tags description must urge a specific facet: {single}"
        );
        assert!(
            single.contains("topic:") && single.contains("tool:"),
            "single-write tags description must list facet examples: {single}"
        );

        let batch = props["entries"]["items"]["properties"]["tags"]["description"]
            .as_str()
            .expect("batch tags must carry a description too");
        assert!(
            batch.to_lowercase().contains("specific"),
            "batch tags description must urge a specific facet: {batch}"
        );
    }

    // --- Scratchpad tools (#184) ---

    use std::sync::Arc;

    use desktop_assistant_core::domain::{ConversationId, ScratchpadNote};
    use desktop_assistant_core::ports::conversation_ctx::with_conversation_id;

    /// Build a BuiltinToolService whose scratchpad closures share one
    /// in-memory note store, so write/search/delete round-trips are testable
    /// without Postgres. Returns the service and a handle to the store.
    fn scratchpad_service() -> (
        BuiltinToolService,
        Arc<std::sync::Mutex<Vec<ScratchpadNote>>>,
    ) {
        use std::pin::Pin;
        let store: Arc<std::sync::Mutex<Vec<ScratchpadNote>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        let w = Arc::clone(&store);
        let write_fn: ScratchpadWriteFn =
            Arc::new(move |conv: String, notes: Vec<NewScratchpadNote>| {
                let store = Arc::clone(&w);
                Box::pin(async move {
                    let mut guard = store.lock().unwrap();
                    let mut saved = Vec::new();
                    for (i, note) in notes.into_iter().enumerate() {
                        if let Some(existing) = guard
                            .iter_mut()
                            .find(|n| n.conversation_id == conv && n.key == note.key)
                        {
                            existing.content = note.content;
                            existing.note_type = note.note_type;
                            existing.sequence = note.sequence;
                            existing.done = note.done;
                            existing.updated_at = "t1".into();
                            saved.push(existing.clone());
                        } else {
                            let mut n = ScratchpadNote::new(
                                format!("id-{i}-{}", note.key),
                                &conv,
                                &note.key,
                                &note.content,
                            );
                            n.note_type = note.note_type;
                            n.sequence = note.sequence;
                            n.done = note.done;
                            n.updated_at = "t0".into();
                            guard.push(n.clone());
                            saved.push(n);
                        }
                    }
                    Ok(saved)
                })
                    as Pin<
                        Box<
                            dyn std::future::Future<Output = Result<Vec<ScratchpadNote>, CoreError>>
                                + Send,
                        >,
                    >
            });

        let g = Arc::clone(&store);
        let get_many_fn: ScratchpadGetManyFn =
            Arc::new(move |conv: String, keys: Vec<String>, limit: usize| {
                let store = Arc::clone(&g);
                Box::pin(async move {
                    let guard = store.lock().unwrap();
                    Ok(guard
                        .iter()
                        .filter(|n| n.conversation_id == conv && keys.contains(&n.key))
                        .take(limit)
                        .cloned()
                        .collect())
                })
            });

        let l = Arc::clone(&store);
        let list_fn: ScratchpadListFn = Arc::new(
            move |conv: String, note_type: Option<String>, limit: usize| {
                let store = Arc::clone(&l);
                Box::pin(async move {
                    let guard = store.lock().unwrap();
                    let mut notes: Vec<ScratchpadNote> = guard
                        .iter()
                        .filter(|n| n.conversation_id == conv)
                        .filter(|n| note_type.as_deref().is_none_or(|t| n.note_type == t))
                        .cloned()
                        .collect();
                    // Mirror the store ordering: type, then sequence ascending
                    // (nulls last), then recency (timestamps omitted here).
                    notes.sort_by(|a, b| {
                        a.note_type
                            .cmp(&b.note_type)
                            .then_with(|| match (a.sequence, b.sequence) {
                                (Some(x), Some(y)) => x.cmp(&y),
                                (Some(_), None) => std::cmp::Ordering::Less,
                                (None, Some(_)) => std::cmp::Ordering::Greater,
                                (None, None) => std::cmp::Ordering::Equal,
                            })
                    });
                    notes.truncate(limit);
                    Ok(notes)
                })
            },
        );

        let s = Arc::clone(&store);
        let search_fn: ScratchpadSearchFn = Arc::new(
            move |conv: String, query: String, note_type: Option<String>, limit: usize| {
                let store = Arc::clone(&s);
                Box::pin(async move {
                    let guard = store.lock().unwrap();
                    Ok(guard
                        .iter()
                        .filter(|n| {
                            n.conversation_id == conv
                                && (n.content.contains(&query) || n.key.contains(&query))
                                && note_type.as_deref().is_none_or(|t| n.note_type == t)
                        })
                        .take(limit)
                        .cloned()
                        .collect())
                })
            },
        );

        let d = Arc::clone(&store);
        let delete_many_fn: ScratchpadDeleteManyFn =
            Arc::new(move |conv: String, keys: Vec<String>| {
                let store = Arc::clone(&d);
                Box::pin(async move {
                    let mut guard = store.lock().unwrap();
                    let before = guard.len();
                    guard.retain(|n| !(n.conversation_id == conv && keys.contains(&n.key)));
                    Ok((before - guard.len()) as u64)
                })
            });

        let c = Arc::clone(&store);
        let clear_fn: ScratchpadClearFn = Arc::new(move |conv: String| {
            let store = Arc::clone(&c);
            Box::pin(async move {
                let mut guard = store.lock().unwrap();
                let before = guard.len();
                guard.retain(|n| n.conversation_id != conv);
                Ok((before - guard.len()) as u64)
            })
        });

        let service = BuiltinToolService::new().with_scratchpad(
            write_fn,
            get_many_fn,
            list_fn,
            search_fn,
            delete_many_fn,
            clear_fn,
        );
        (service, store)
    }

    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    #[tokio::test]
    async fn scratchpad_requires_active_conversation() {
        // Closures configured, but no conversation scope installed.
        let (service, _store) = scratchpad_service();
        for (tool, args) in [
            (
                TOOL_SCRATCHPAD_WRITE,
                serde_json::json!({"key": "k", "content": "v"}),
            ),
            (
                TOOL_SCRATCHPAD_SEARCH,
                serde_json::json!({"max_results": 10}),
            ),
            (TOOL_SCRATCHPAD_DELETE, serde_json::json!({"all": true})),
        ] {
            let result = service.execute_tool(tool, args).await;
            assert!(
                matches!(&result, Err(CoreError::ToolExecution(m)) if m.contains("active conversation")),
                "{tool} must require an active conversation, got {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn scratchpad_write_search_delete_roundtrip() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            // Batch write two notes.
            let written = service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"notes": [
                        {"key": "goal", "content": "ship the scratchpad"},
                        {"key": "q", "content": "which database to use"}
                    ]}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&written)["written"].as_array().unwrap().len(), 2);

            // List (no query) returns both.
            let listed = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"max_results": 10}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&listed)["results"].as_array().unwrap().len(), 2);

            // Search by query matches one.
            let hit = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"query": "database", "max_results": 10}),
                )
                .await
                .unwrap();
            let results = parse(&hit);
            assert_eq!(results["results"].as_array().unwrap().len(), 1);
            assert_eq!(results["results"][0]["key"], "q");

            // Fetch by keys.
            let by_key = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"keys": ["goal"], "max_results": 10}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&by_key)["results"][0]["key"], "goal");

            // Upsert by key updates content, not count.
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"key": "goal", "content": "ship it well"}),
                )
                .await
                .unwrap();
            let after = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"max_results": 10}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&after)["results"].as_array().unwrap().len(), 2);

            // Delete one key.
            let del = service
                .execute_tool(TOOL_SCRATCHPAD_DELETE, serde_json::json!({"keys": ["q"]}))
                .await
                .unwrap();
            assert_eq!(parse(&del)["deleted"], 1);

            // Delete all.
            let cleared = service
                .execute_tool(TOOL_SCRATCHPAD_DELETE, serde_json::json!({"all": true}))
                .await
                .unwrap();
            assert_eq!(parse(&cleared)["deleted"], 1);
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_write_rejects_empty_key_and_oversize_content() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            let huge = "x".repeat(MAX_NOTE_BYTES + 1);
            let result = service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"notes": [
                        {"key": "", "content": "no key"},
                        {"key": "big", "content": huge},
                        {"key": "ok", "content": "fine"}
                    ]}),
                )
                .await
                .unwrap();
            let json = parse(&result);
            assert_eq!(
                json["written"].as_array().unwrap().len(),
                1,
                "only the valid note is written"
            );
            assert_eq!(json["written"][0]["key"], "ok");
            assert_eq!(json["rejected"].as_array().unwrap().len(), 2);
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_write_truncates_over_cap() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            let notes: Vec<serde_json::Value> = (0..MAX_NOTES_PER_WRITE + 5)
                .map(|i| serde_json::json!({"key": format!("k{i}"), "content": "v"}))
                .collect();
            let result = service
                .execute_tool(TOOL_SCRATCHPAD_WRITE, serde_json::json!({"notes": notes}))
                .await
                .unwrap();
            let json = parse(&result);
            assert_eq!(json["truncated"], true);
            assert_eq!(
                json["written"].as_array().unwrap().len(),
                MAX_NOTES_PER_WRITE
            );
            assert_eq!(json["skipped"].as_array().unwrap().len(), 5);
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_search_requires_max_results() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            let result = service
                .execute_tool(TOOL_SCRATCHPAD_SEARCH, serde_json::json!({"query": "x"}))
                .await;
            assert!(
                matches!(&result, Err(CoreError::ToolExecution(m)) if m.contains("max_results")),
                "search must require max_results, got {result:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_delete_requires_exactly_one_mode() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            // Neither.
            let neither = service
                .execute_tool(TOOL_SCRATCHPAD_DELETE, serde_json::json!({}))
                .await;
            assert!(matches!(neither, Err(CoreError::ToolExecution(_))));
            // Both.
            let both = service
                .execute_tool(
                    TOOL_SCRATCHPAD_DELETE,
                    serde_json::json!({"keys": ["a"], "all": true}),
                )
                .await;
            assert!(matches!(both, Err(CoreError::ToolExecution(_))));
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_search_byte_budget_truncates() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            // Write enough near-max notes that the serialized list exceeds the
            // response byte budget, forcing truncation.
            let big = "y".repeat(MAX_NOTE_BYTES - 100);
            let count = (RESPONSE_BYTE_BUDGET / MAX_NOTE_BYTES) + 3;
            let notes: Vec<serde_json::Value> = (0..count)
                .map(|i| serde_json::json!({"key": format!("k{i}"), "content": big}))
                .collect();
            // Cap is MAX_NOTES_PER_WRITE; write in chunks if needed. count is
            // small (< cap for 20KB/8KB), so a single call suffices.
            service
                .execute_tool(TOOL_SCRATCHPAD_WRITE, serde_json::json!({"notes": notes}))
                .await
                .unwrap();

            let listed = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"max_results": 100}),
                )
                .await
                .unwrap();
            let json = parse(&listed);
            assert_eq!(
                json["truncated"], true,
                "oversized list must signal truncation"
            );
            let returned = json["results"].as_array().unwrap().len();
            assert!(
                returned < count,
                "fewer than all notes are returned under the byte budget"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_write_persists_type_sequence_done() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({
                        "key": "t1", "content": "wire the migration",
                        "type": "todo", "sequence": 2, "done": false
                    }),
                )
                .await
                .unwrap();

            let by_key = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"keys": ["t1"], "max_results": 10}),
                )
                .await
                .unwrap();
            let note = &parse(&by_key)["results"][0];
            assert_eq!(note["type"], "todo");
            assert_eq!(note["sequence"], 2);
            assert_eq!(note["done"], false);

            // Re-writing the same key flips `done` (the check-off path).
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({
                        "key": "t1", "content": "wire the migration",
                        "type": "todo", "sequence": 2, "done": true
                    }),
                )
                .await
                .unwrap();
            let after = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"keys": ["t1"], "max_results": 10}),
                )
                .await
                .unwrap();
            assert_eq!(parse(&after)["results"][0]["done"], true);
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_search_filters_by_type() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"notes": [
                        {"key": "t1", "content": "do a thing", "type": "todo", "sequence": 1},
                        {"key": "n1", "content": "a plain note", "type": "note"}
                    ]}),
                )
                .await
                .unwrap();

            let todos = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"type": "todo", "max_results": 10}),
                )
                .await
                .unwrap();
            let results = parse(&todos);
            assert_eq!(results["results"].as_array().unwrap().len(), 1);
            assert_eq!(results["results"][0]["key"], "t1");
        })
        .await;
    }

    #[tokio::test]
    async fn scratchpad_list_orders_todos_by_sequence() {
        let (service, _store) = scratchpad_service();
        with_conversation_id(ConversationId::from("c1"), async {
            // Written out of order; expect list to return them sorted by `seq`.
            service
                .execute_tool(
                    TOOL_SCRATCHPAD_WRITE,
                    serde_json::json!({"notes": [
                        {"key": "c", "content": "third",  "type": "todo", "sequence": 3},
                        {"key": "a", "content": "first",  "type": "todo", "sequence": 1},
                        {"key": "b", "content": "second", "type": "todo", "sequence": 2}
                    ]}),
                )
                .await
                .unwrap();

            let listed = service
                .execute_tool(
                    TOOL_SCRATCHPAD_SEARCH,
                    serde_json::json!({"type": "todo", "max_results": 10}),
                )
                .await
                .unwrap();
            let results = parse(&listed);
            let keys: Vec<String> = results["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n["key"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(keys, vec!["a", "b", "c"], "todos sort by sequence");
        })
        .await;
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
    async fn sys_props_prefers_client_context_over_daemon_env() {
        // #558: the daemon may be remote/containerized, so its host env is NOT
        // the user's. When the connecting client reported a context, the
        // identity fields must be the CLIENT's, and the source is labeled
        // `client`.
        use desktop_assistant_core::ports::transport::{ClientContext, with_client_context};

        let service = BuiltinToolService::new();
        let ctx = ClientContext {
            real_name: Some("Ada Lovelace".into()),
            username: Some("ada-client".into()),
            home_dir: Some("/home/ada-client".into()),
            hostname: Some("analytical-engine".into()),
            timezone: Some("Europe/London".into()),
            os: Some("TestOS 9000".into()),
        };
        let response = with_client_context(Some(ctx), async {
            service
                .execute_tool("builtin_sys_props", serde_json::json!({}))
                .await
        })
        .await
        .unwrap();

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        let props = json
            .get("props")
            .and_then(serde_json::Value::as_object)
            .expect("props object");

        let s = |k: &str| props.get(k).and_then(serde_json::Value::as_str);
        assert_eq!(s("identity_source"), Some("client"));
        assert_eq!(s("real_name"), Some("Ada Lovelace"));
        assert_eq!(s("username"), Some("ada-client"));
        assert_eq!(s("home_dir"), Some("/home/ada-client"));
        assert_eq!(s("hostname"), Some("analytical-engine"));
        assert_eq!(s("timezone"), Some("Europe/London"));
        assert_eq!(s("os"), Some("TestOS 9000"));

        // The daemon host is still reported, but under a clearly-labeled block —
        // never AS the client's identity. Its username is the real daemon env
        // user (or absent), which is never our synthetic client username.
        let daemon = props
            .get("daemon_host")
            .and_then(serde_json::Value::as_object)
            .expect("daemon_host object");
        assert!(daemon.contains_key("cwd"), "daemon working dir is labeled");
        assert_ne!(
            daemon.get("username").and_then(serde_json::Value::as_str),
            Some("ada-client"),
            "daemon-host username must never be the client's value"
        );
    }

    #[tokio::test]
    async fn sys_props_partial_client_context_does_not_borrow_daemon_identity() {
        // #558: a client that reports only its timezone must not have the OTHER
        // identity fields silently filled from the daemon host — that would
        // present daemon-host values AS the client's. Absent client fields stay
        // null under the `client` source.
        use desktop_assistant_core::ports::transport::{ClientContext, with_client_context};

        let service = BuiltinToolService::new();
        let ctx = ClientContext {
            timezone: Some("America/New_York".into()),
            ..ClientContext::default()
        };
        let response = with_client_context(Some(ctx), async {
            service
                .execute_tool("builtin_sys_props", serde_json::json!({}))
                .await
        })
        .await
        .unwrap();

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        let props = json
            .get("props")
            .and_then(serde_json::Value::as_object)
            .expect("props object");

        assert_eq!(
            props
                .get("identity_source")
                .and_then(serde_json::Value::as_str),
            Some("client")
        );
        assert_eq!(
            props.get("timezone").and_then(serde_json::Value::as_str),
            Some("America/New_York")
        );
        assert!(
            props
                .get("username")
                .expect("username key present")
                .is_null(),
            "absent client username must stay null, not borrow the daemon's"
        );
        assert!(
            props
                .get("home_dir")
                .expect("home_dir key present")
                .is_null(),
            "absent client home_dir must stay null, not borrow the daemon's"
        );
        assert!(
            props
                .get("hostname")
                .expect("hostname key present")
                .is_null(),
            "absent client hostname must stay null, not borrow the daemon's"
        );
    }

    #[tokio::test]
    async fn sys_props_without_client_context_labels_daemon_host_fallback() {
        // #558: with no client context installed (the common unset case) the
        // identity fields fall back to the daemon host, explicitly labeled so a
        // reader never mistakes them for the connecting client's values.
        let service = BuiltinToolService::new();
        let response = service
            .execute_tool("builtin_sys_props", serde_json::json!({}))
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        let props = json
            .get("props")
            .and_then(serde_json::Value::as_object)
            .expect("props object");

        assert_eq!(
            props
                .get("identity_source")
                .and_then(serde_json::Value::as_str),
            Some("daemon_host_fallback")
        );
        // `real_name` has no daemon-host equivalent, so it stays null in fallback.
        assert!(
            props
                .get("real_name")
                .expect("real_name key present")
                .is_null(),
            "daemon host has no real_name to report"
        );
        // The daemon host block is present with the daemon's own os.
        let daemon = props
            .get("daemon_host")
            .and_then(serde_json::Value::as_object)
            .expect("daemon_host object");
        assert!(
            daemon
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
        let write_fn: KnowledgeWriteFn = Arc::new(move |mut entry| {
            let s = Arc::clone(&write_store);
            Box::pin(async move {
                entry.created_at = "2024-01-01".to_string();
                entry.updated_at = "2024-01-01".to_string();
                // Upsert by id, mirroring the store's ON CONFLICT semantics.
                let mut g = s.lock().unwrap();
                g.retain(|e| e.id != entry.id);
                g.push(entry.clone());
                Ok(entry)
            })
        });

        let search_store = Arc::clone(&store);
        let search_fn: KnowledgeSearchFn =
            Arc::new(move |_query, _emb, _tags, _exclude_tags, limit| {
                let s = Arc::clone(&search_store);
                Box::pin(async move {
                    let entries = s.lock().unwrap();
                    Ok(entries.iter().take(limit).cloned().collect())
                })
            });

        let delete_store = Arc::clone(&store);
        let delete_fn: KnowledgeDeleteFn = Arc::new(move |ids| {
            let s = Arc::clone(&delete_store);
            Box::pin(async move {
                let mut g = s.lock().unwrap();
                let before = g.len();
                g.retain(|e| !ids.contains(&e.id));
                Ok(before - g.len())
            })
        });

        let list_store = Arc::clone(&store);
        let list_fn: KnowledgeListFn = Arc::new(move |q| {
            let s = Arc::clone(&list_store);
            Box::pin(async move {
                let g = s.lock().unwrap();
                let entries = g.iter().take(q.limit.max(1)).cloned().collect();
                Ok(
                    desktop_assistant_core::ports::knowledge::KnowledgeListPage {
                        entries,
                        next_cursor: None,
                    },
                )
            })
        });

        let get_store = Arc::clone(&store);
        let get_fn: KnowledgeGetFn = Arc::new(move |id| {
            let s = Arc::clone(&get_store);
            Box::pin(async move { Ok(s.lock().unwrap().iter().find(|e| e.id == id).cloned()) })
        });

        let service = BuiltinToolService::new()
            .with_knowledge_base(write_fn, search_fn, delete_fn, list_fn, get_fn);

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
        assert_eq!(json["count"], 1);
        assert!(json["entries"][0]["id"].as_str().is_some());

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

        // List surfaces the entry with its provenance ('explicit' for a
        // tool-authored write) and an id we can operate on.
        let list_result = service
            .execute_tool(TOOL_KB_LIST, serde_json::json!({"limit": 10}))
            .await
            .unwrap();
        let lj: serde_json::Value = serde_json::from_str(&list_result).unwrap();
        assert_eq!(lj["count"], 1);
        assert_eq!(lj["entries"][0]["source"], "explicit");
        let id = lj["entries"][0]["id"].as_str().unwrap().to_string();

        // Partial update: tags only, `content` omitted — existing content is
        // preserved.
        service
            .execute_tool(
                TOOL_KB_WRITE,
                serde_json::json!({"id": id, "tags": ["preference", "retagged"]}),
            )
            .await
            .unwrap();
        {
            let g = store.lock().unwrap();
            assert_eq!(g.len(), 1);
            assert_eq!(g[0].content, "User prefers dark mode");
            assert!(g[0].tags.iter().any(|t| t == "retagged"));
        }

        // Bulk delete by ids.
        let del = service
            .execute_tool(TOOL_KB_DELETE, serde_json::json!({"ids": [id]}))
            .await
            .unwrap();
        let dj: serde_json::Value = serde_json::from_str(&del).unwrap();
        assert_eq!(dj["deleted"], 1);
        assert!(store.lock().unwrap().is_empty());
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

    #[tokio::test]
    async fn skill_search_and_get_with_closures() {
        use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, TrustTier};
        use desktop_assistant_core::ports::skill_index::{SkillGetFn, SkillSearchFn};
        use std::sync::Arc;

        fn sample(name: &str, kind: SkillKind) -> IndexedSkill {
            IndexedSkill {
                name: name.to_string(),
                description: format!("does {name}"),
                kind,
                disk_path: format!("/skills/{name}/SKILL.md"),
                owner_user_id: None,
                locality: Locality::Daemon,
                content_hash: "h".to_string(),
                trust_tier: TrustTier::Local,
                source: None,
                tags: vec!["ops".to_string()],
                attachments: vec!["scripts/run.sh".to_string()],
                body: "# body\n\n## Steps\n1. go".to_string(),
                metadata: serde_json::Value::Null,
            }
        }

        let search_fn: SkillSearchFn = Arc::new(|_q, _emb, _limit| {
            Box::pin(async {
                Ok(vec![
                    sample("invoicing", SkillKind::Workflow),
                    sample("notes", SkillKind::Skill),
                ])
            })
        });
        let get_fn: SkillGetFn = Arc::new(|name, _owner| {
            Box::pin(async move {
                Ok((name == "invoicing").then(|| sample("invoicing", SkillKind::Workflow)))
            })
        });
        let service = BuiltinToolService::new().with_skills(search_fn, get_fn);

        // The `kind` filter keeps only workflows.
        let out = service
            .execute_tool(
                TOOL_SKILL_SEARCH,
                serde_json::json!({"query": "invoice", "kind": "workflow"}),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(json["ok"], true);
        let results = json["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "invoicing");
        assert_eq!(results[0]["kind"], "workflow");

        // `get` returns the full body for a hit and `ok:false` for a miss.
        let hit = service
            .execute_tool(TOOL_SKILL_GET, serde_json::json!({"name": "invoicing"}))
            .await
            .unwrap();
        let hit_json: serde_json::Value = serde_json::from_str(&hit).unwrap();
        assert_eq!(hit_json["ok"], true);
        assert!(hit_json["body"].as_str().unwrap().contains("## Steps"));
        assert_eq!(hit_json["attachments"][0], "scripts/run.sh");

        let miss = service
            .execute_tool(TOOL_SKILL_GET, serde_json::json!({"name": "nope"}))
            .await
            .unwrap();
        let miss_json: serde_json::Value = serde_json::from_str(&miss).unwrap();
        assert_eq!(miss_json["ok"], false);
    }

    #[test]
    fn every_advertised_builtin_is_routable() {
        // Regression guard: a tool that `tool_definitions()` advertises but
        // `supports_tool()` doesn't recognize gets routed to MCP at execution
        // and fails with "unknown tool" (this bit builtin_knowledge_base_list
        // and builtin_notify). Wiring notify makes the full builtin set appear.
        use std::sync::Arc;
        let notify_fn: NotifyFn = Arc::new(|_, _, _| Box::pin(async { Ok(Some(1u32)) }));
        let service = BuiltinToolService::new().with_notify(notify_fn);
        for def in service.tool_definitions() {
            assert!(
                BuiltinToolService::supports_tool(&def.name),
                "tool '{}' is advertised by tool_definitions() but supports_tool() rejects it — \
                 it would fail to route at execution time",
                def.name
            );
        }
    }

    #[tokio::test]
    async fn notify_absent_and_errors_without_capability() {
        let service = BuiltinToolService::new();
        // Not advertised when no notification capability is wired.
        assert!(
            !service
                .tool_definitions()
                .iter()
                .any(|t| t.name == TOOL_NOTIFY)
        );
        // Calling it anyway is a clean error, not a panic.
        let err = service
            .execute_tool(TOOL_NOTIFY, serde_json::json!({"summary": "hi"}))
            .await;
        assert!(matches!(err, Err(CoreError::ToolExecution(_))));
    }

    #[tokio::test]
    async fn notify_with_closure_reports_shown_and_suppressed() {
        use std::sync::Arc;

        // Returns an id for "show me", None for "duplicate" — keyed off summary.
        let notify_fn: NotifyFn = Arc::new(|summary, _body, _urgency| {
            Box::pin(async move {
                if summary == "dup" {
                    Ok(None)
                } else {
                    Ok(Some(42u32))
                }
            })
        });
        let service = BuiltinToolService::new().with_notify(notify_fn);

        // Advertised once wired.
        assert!(
            service
                .tool_definitions()
                .iter()
                .any(|t| t.name == TOOL_NOTIFY)
        );

        // summary is required.
        assert!(
            service
                .execute_tool(TOOL_NOTIFY, serde_json::json!({"body": "no summary"}))
                .await
                .is_err()
        );

        let shown = service
            .execute_tool(
                TOOL_NOTIFY,
                serde_json::json!({"summary": "Build done", "urgency": "low"}),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&shown).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["shown"], true);
        assert_eq!(json["id"], 42);

        let suppressed = service
            .execute_tool(TOOL_NOTIFY, serde_json::json!({"summary": "dup"}))
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&suppressed).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["shown"], false);
    }

    #[tokio::test(start_paused = true)]
    async fn embedding_timeout_falls_back_to_empty_embedding() {
        // A wedged embedding backend (a never-completing future, like a stuck
        // Ollama) must not hang the search: `embed_text` times out after
        // EMBED_TIMEOUT and the search runs with an empty embedding, which the
        // store turns into an FTS-only query. With the clock paused, the 5s
        // timeout elapses immediately so the test is instant.
        use desktop_assistant_core::domain::ToolDefinition;
        use desktop_assistant_core::ports::embedding::EmbedFn;
        use std::sync::{Arc, Mutex};

        let embed_fn: EmbedFn = Arc::new(|_texts| Box::pin(std::future::pending()));

        // Capture the embedding the search closure is handed.
        let seen: Arc<Mutex<Option<Vec<f32>>>> = Arc::new(Mutex::new(None));
        let seen_w = Arc::clone(&seen);
        let search_fn: ToolSearchFn = Arc::new(move |_query, emb, _limit| {
            *seen_w.lock().unwrap() = Some(emb);
            Box::pin(async {
                Ok(vec![ToolDefinition::new(
                    "weather__forecast",
                    "Get the forecast",
                    serde_json::json!({}),
                )])
            })
        });
        let def_fn: ToolDefinitionFn = Arc::new(|_name| Box::pin(async { Ok(None) }));

        let service = BuiltinToolService::new()
            .with_embedding(embed_fn)
            .with_tool_registry(search_fn, def_fn);

        let result = service
            .execute_tool(
                TOOL_SEARCH,
                serde_json::json!({"query": "weather forecast"}),
            )
            .await
            .expect("tool search must return, not hang, when embedding times out");
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["tools"].as_array().unwrap().len(), 1);
        assert_eq!(
            seen.lock().unwrap().as_ref().expect("search ran").len(),
            0,
            "a timed-out embedding must yield an empty vector so the store falls back to FTS"
        );
    }
}
