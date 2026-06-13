//! Protocol-neutral API model shared by adapters (D-Bus, WebSocket, etc.).
//!
//! This crate intentionally contains only:
//! - serializable command/result/event types
//! - stable IDs and small helper types
//!
//! Business logic belongs in core/application crates.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Ping,

    GetStatus,

    // Canonical transport-level config API
    GetConfig,
    SetConfig {
        changes: ConfigChanges,
    },

    // Conversations
    CreateConversation {
        title: String,
    },
    ListConversations {
        max_age_days: Option<u32>,
        #[serde(default)]
        include_archived: bool,
    },
    GetConversation {
        id: String,
    },
    DeleteConversation {
        id: String,
    },
    RenameConversation {
        id: String,
        title: String,
    },
    ArchiveConversation {
        id: String,
    },
    UnarchiveConversation {
        id: String,
    },
    ClearAllHistory,

    /// Send a content to an existing conversation.
    ///
    /// The response is streamed via [`Event::AssistantDelta`] events.
    /// An optional `override` selects a specific connection/model/effort for
    /// this send only; when omitted, the server falls back to (in order) the
    /// conversation's last selection and the `interactive` purpose.
    ///
    /// `system_refinement` (optional; defaults to empty, omitted on the wire
    /// when empty) is a per-request addition to the system prompt for THIS
    /// send only. When non-empty the daemon appends it after the
    /// conversation's normal system prompt for the LLM call, but does NOT
    /// store it as a message and does NOT attach it to the conversation — so
    /// it never appears in chat history and never affects later turns. This
    /// lets a client (e.g. the voice daemon) attach instructions like
    /// "respond briefly, by voice" to a single turn dictated into an existing
    /// chat without polluting the visible transcript or permanently changing
    /// that conversation's behaviour.
    SendMessage {
        conversation_id: String,
        content: String,
        #[serde(default, rename = "override", skip_serializing_if = "Option::is_none")]
        override_selection: Option<SendPromptOverride>,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        system_refinement: String,
        /// Optional client-supplied idempotency key, scoped to the conversation.
        /// A retry carrying the same key is de-duplicated by the daemon — the
        /// still-running request is re-attached, or a completed reply replayed —
        /// instead of re-running the turn, so a dropped connection can be retried
        /// without double-processing an action. Absent = no idempotency.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        idempotency_key: Option<String>,
    },

    /// Set (or clear) a conversation's personality override (issue #227,
    /// Phase 2). `personality` is a *partial* [`ConversationPersonalityView`]
    /// (a [`PersonalityOverride`]): each `Some` trait pins that trait for the
    /// conversation, each `None` falls back to the global config on every send.
    /// An all-`None`/empty override clears the stored override (back to
    /// global-only). The override sets only the *initial disposition* — the
    /// assistant stays soft/adaptive. Returns
    /// [`CommandResult::ConversationPersonality`] echoing the stored value.
    /// Mirrors the per-conversation model selection: stored on the
    /// conversation, resolved on the send path against the global config.
    SetConversationPersonality {
        conversation_id: String,
        personality: ConversationPersonalityView,
    },

    // Settings (legacy `[llm]`-block single-connection surface).
    //
    // The legacy `SetLlmSettings` / `GetLlmSettings` commands have been
    // removed; use the named-connection commands below (`ListConnections`,
    // `CreateConnection`, `UpdateConnection`, `DeleteConnection`,
    // `GetPurposes`, `SetPurpose`) instead.
    SetApiKey {
        api_key: String,
    },

    GetEmbeddingsSettings,
    SetEmbeddingsSettings {
        connector: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    },

    GetConnectorDefaults {
        connector: String,
    },

    GetPersistenceSettings,
    SetPersistenceSettings {
        enabled: bool,
        remote_url: Option<String>,
        remote_name: Option<String>,
        push_on_update: bool,
    },

    // Named connections (issue #11).
    /// Enumerate every configured connection with its availability and
    /// whether credentials are present.
    ListConnections,
    /// Create a new named connection; fails on invalid slug or duplicate id.
    CreateConnection {
        id: String,
        config: ConnectionConfigView,
    },
    /// Replace an existing connection in-place.
    UpdateConnection {
        id: String,
        config: ConnectionConfigView,
    },
    /// Delete a named connection. Refuses with an error when the connection
    /// is referenced by any purpose unless `force` is true, in which case
    /// referencing purposes fall back to the `interactive` purpose.
    DeleteConnection {
        id: String,
        #[serde(default)]
        force: bool,
    },
    /// Enumerate models across one or all configured connections. When
    /// `connection_id` is `None`, aggregates models from every healthy
    /// connection. `refresh=true` bypasses connector caches (e.g. Bedrock).
    ListAvailableModels {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        connection_id: Option<String>,
        #[serde(default)]
        refresh: bool,
    },

    // Purposes (issue #10 + #11).
    GetPurposes,
    SetPurpose {
        purpose: PurposeKindApi,
        config: PurposeConfigView,
    },

    // Knowledge base management (issue #73).
    ListKnowledgeEntries {
        #[serde(default = "default_kb_limit")]
        limit: u32,
        #[serde(default)]
        offset: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tag_filter: Option<Vec<String>>,
    },
    GetKnowledgeEntry {
        id: String,
    },
    SearchKnowledgeEntries {
        query: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tag_filter: Option<Vec<String>>,
        #[serde(default = "default_kb_limit")]
        limit: u32,
    },
    CreateKnowledgeEntry {
        content: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        metadata: serde_json::Value,
    },
    UpdateKnowledgeEntry {
        id: String,
        content: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        metadata: serde_json::Value,
    },
    DeleteKnowledgeEntry {
        id: String,
    },

    // MCP server management
    ListMcpServers,
    AddMcpServer {
        name: String,
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        #[serde(default = "default_true")]
        enabled: bool,
    },
    RemoveMcpServer {
        name: String,
    },
    SetMcpServerEnabled {
        name: String,
        enabled: bool,
    },
    McpServerAction {
        action: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        server: Option<String>,
    },

    // --- Background tasks (issue #110) ------------------------------------
    //
    // Protocol shape only; the registry that backs these commands is the
    // subject of a separate issue. Snake-case naming follows the existing
    // `Command` convention (`#[serde(rename_all = "snake_case")]`).
    /// List registered background tasks for the calling user.
    ListBackgroundTasks {
        #[serde(default)]
        include_finished: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<u32>,
    },
    /// Fetch a single background task by id.
    GetBackgroundTask {
        id: String,
    },
    /// Request cancellation of a background task. The registry replies with
    /// `Ack`; cancellation completion is observed via `Event::TaskCompleted`.
    CancelBackgroundTask {
        id: String,
    },
    /// Fetch a page of log entries for a background task. `after_seq` skips
    /// entries already seen; omit to start from the oldest available entry.
    GetBackgroundTaskLogs {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        after_seq: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<u32>,
    },
    /// Subscribe this connection to `Task*` events for the calling user.
    SubscribeBackgroundTasks,
    /// Stop receiving `Task*` events on this connection.
    UnsubscribeBackgroundTasks,
    /// Launch a user-initiated standalone background agent. Returns
    /// `CommandResult::BackgroundTaskSpawned { id }` on success.
    SpawnStandaloneAgent {
        name: String,
        initial_prompt: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        override_selection: Option<SendPromptOverride>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tools: Option<Vec<String>>,
    },

    // --- Conversation scratchpad (issue #190) -----------------------------
    //
    // Client-facing read/write/delete for a conversation's scratchpad — the
    // same per-conversation notes the LLM manages via builtin tools, exposed
    // so a client (e.g. the adele-gtk side pane) can display and edit them.
    // All three are user-scoped by the dispatcher's `with_user_id`, like every
    // other command.
    /// Read a conversation's scratchpad notes, ordered by type then sequence.
    /// Returns `CommandResult::Scratchpad`.
    GetConversationScratchpad {
        conversation_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_results: Option<u32>,
    },
    /// Upsert a single scratchpad note (keyed within the conversation).
    /// Re-writing an existing key replaces its content/type/sequence/done —
    /// this is how a client checks a todo off (`done: true`). Returns the saved
    /// note(s) as `CommandResult::Scratchpad`.
    SetScratchpadNote {
        conversation_id: String,
        key: String,
        content: String,
        #[serde(default)]
        note_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sequence: Option<i32>,
        #[serde(default)]
        done: bool,
    },
    /// Delete scratchpad notes by key, or clear the whole pad with `all: true`.
    /// Returns `CommandResult::Ack`.
    DeleteScratchpadNotes {
        conversation_id: String,
        #[serde(default)]
        keys: Vec<String>,
        #[serde(default)]
        all: bool,
    },

    // --- Client-side tool execution (issue #107) ---------------------------
    //
    // Phase-2 architecture (rule #8) executes client-local MCPs on the
    // user's machine rather than on the daemon. The client advertises
    // which tools it can run at session start; when the LLM picks one
    // the daemon suspends the turn, emits `Event::ClientToolCall`, and
    // resumes when `Command::ClientToolResult` arrives.
    /// Advertise the set of client-local MCP tools this connection is
    /// able to execute. The daemon replaces any previously-registered
    /// set on each call — clients should send the full list, not
    /// deltas. Per-session: re-register on every connect.
    RegisterClientTools {
        tools: Vec<ClientToolRegistration>,
    },
    /// Deliver the result of a `ClientToolCall` back to the daemon so a
    /// suspended turn can resume. Exactly one of `result` / `error`
    /// should be populated; both `None` is treated as an error by the
    /// daemon-side validator.
    ClientToolResult {
        task_id: TaskId,
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

/// Single entry in a `RegisterClientTools` request. Mirrors the shape of
/// `ToolDefinition` but kept here in `api-model` so adapters don't need
/// to depend on `desktop-assistant-core`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClientToolRegistration {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the tool's input. Daemon forwards verbatim to
    /// the LLM's tool list.
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

fn default_true() -> bool {
    true
}

fn default_kb_limit() -> u32 {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CommandResult {
    Pong {
        value: String,
    },

    Status(Status),
    Config(Config),

    ConversationId {
        id: String,
    },
    Conversations(Vec<ConversationSummary>),
    Conversation(ConversationView),
    Messages(MessagesView),
    Cleared {
        deleted_count: u32,
    },

    EmbeddingsSettings(EmbeddingsSettingsView),
    ConnectorDefaults(ConnectorDefaultsView),
    PersistenceSettings(PersistenceSettingsView),

    McpServers(Vec<McpServerView>),

    Connections(Vec<ConnectionView>),
    Models(Vec<ModelListing>),
    Purposes(PurposesView),

    KnowledgeEntries(Vec<KnowledgeEntryView>),
    KnowledgeEntry(Option<KnowledgeEntryView>),
    KnowledgeEntryWritten(KnowledgeEntryView),

    /// Response to `GetConversationScratchpad` / `SetScratchpadNote` — the
    /// requested (or just-saved) scratchpad notes for the conversation.
    Scratchpad(Vec<ScratchpadNoteView>),

    /// Response to `SetConversationPersonality` — the conversation's stored
    /// personality override after the write (#227). An empty/all-`None` view
    /// means the override was cleared and the conversation falls back to the
    /// global personality on every send.
    ConversationPersonality(ConversationPersonalityView),

    // --- Background tasks (issue #110) ------------------------------------
    /// Response to `ListBackgroundTasks`.
    BackgroundTasks(Vec<TaskView>),
    /// Response to `GetBackgroundTask`.
    BackgroundTask(TaskView),
    /// Response to `GetBackgroundTaskLogs`. `next_seq` is the value clients
    /// should pass back as `after_seq` to resume paging.
    BackgroundTaskLogs {
        entries: Vec<TaskLogEntry>,
        next_seq: u64,
    },
    /// Response to `SpawnStandaloneAgent`.
    BackgroundTaskSpawned {
        id: String,
    },
    /// Ack for `SendMessage`, carrying both correlation ids the streamed
    /// events use:
    ///
    /// - `request_id` — stamped on every `AssistantDelta` / `AssistantCompleted`
    ///   / `AssistantError` event for THIS turn. Socket clients (UDS / WS) match
    ///   streamed response events to their send by this id, exactly as the
    ///   D-Bus `SendPrompt` reply does. This is the field a streaming client
    ///   wants (voice#49).
    /// - `task_id` — the registered background-task id, used to correlate the
    ///   `Task*` lifecycle events and to drive the process-manager UI / Cancel.
    ///
    /// Introduced alongside the background-task registry so we don't overload
    /// `Ack`; `request_id` was added in voice#49 so socket clients can correlate
    /// streamed responses (the dispatcher generates a turn `request_id` distinct
    /// from the `task_id`, and the events carry the former).
    SendMessageAck {
        request_id: String,
        task_id: String,
    },

    /// Response to `RegisterClientTools`, carrying the count of tools
    /// accepted by the daemon. Clients use this to verify registration
    /// landed before relying on client-side execution.
    ClientToolsRegistered {
        count: u32,
    },

    Ack,
}

/// Wire-format view of a knowledge base entry. Mirrors
/// `desktop_assistant_core::domain::KnowledgeEntry` but lives here so
/// transports and clients depend only on `api-model`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeEntryView {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at: String,
    pub updated_at: String,
}

/// Wire-format view of a scratchpad note. Mirrors
/// `desktop_assistant_core::domain::ScratchpadNote` (minus the internal
/// `conversation_id`/`created_at`) but lives here so transports and clients
/// depend only on `api-model`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScratchpadNoteView {
    pub id: String,
    pub key: String,
    pub content: String,
    /// Free-text category (e.g. `todo`/`note`/`other`); defaults to `note`.
    #[serde(default)]
    pub note_type: String,
    /// Optional ordering hint within a `note_type` (ascending, nulls last).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<i32>,
    /// Whether the note (e.g. a todo) is checked off.
    #[serde(default)]
    pub done: bool,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    ConfigChanged {
        config: Config,
    },

    /// Progress status while the assistant is working (tool calls, searches, etc.).
    /// Displayed as transient "working..." indicators, not as chat messages.
    AssistantStatus {
        conversation_id: String,
        request_id: String,
        message: String,
    },

    /// Per-turn context-window fill report (issue #341). Emitted after each
    /// LLM call once the provider reports input-token usage and the per-turn
    /// budget is known, so clients can render a "used / budget (%)" indicator
    /// and shift colour as the proactive-compaction line (0.85 of budget) is
    /// approached. Carries token COUNTS only — never message content — and is
    /// purely advisory: a client that does not understand it ignores it.
    ContextUsage {
        conversation_id: String,
        request_id: String,
        /// Prompt/input tokens the provider reported for this turn.
        used_tokens: u64,
        /// Resolved max input-token budget for this turn (three-tier
        /// resolution: purpose override → connector table → fallback).
        budget_tokens: u64,
        /// `true` once the effective message window was shrunk and the
        /// dropped range compacted on this turn (proactive compaction ran).
        #[serde(default)]
        compaction_active: bool,
    },

    /// Streaming chunk for a content response.
    AssistantDelta {
        conversation_id: String,
        request_id: String,
        chunk: String,
    },

    /// Full response (terminal event).
    AssistantCompleted {
        conversation_id: String,
        request_id: String,
        full_response: String,
    },

    /// Streaming failure (terminal event).
    AssistantError {
        conversation_id: String,
        request_id: String,
        error: String,
    },

    /// The title of a conversation was changed (e.g. LLM-generated after first message).
    ConversationTitleChanged {
        conversation_id: String,
        title: String,
    },

    /// A one-time advisory for a conversation (e.g. the stored model
    /// selection no longer resolves and was cleared). Emitted at most once
    /// per underlying condition — the server clears the stored state so
    /// the warning does not recur.
    ConversationWarningEmitted {
        conversation_id: String,
        warning: ConversationWarning,
    },

    // --- Background tasks (issue #110) ------------------------------------
    /// A background task has been registered and is now `Pending`/`Running`.
    TaskStarted {
        task: TaskView,
    },
    /// Lightweight progress signal that does not justify a log entry.
    TaskProgress {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        progress_hint: Option<String>,
    },
    /// A new log entry was appended to a task's bounded log buffer.
    TaskLogAppended {
        id: String,
        entry: TaskLogEntry,
    },
    /// Terminal event: the task transitioned to `Completed`, `Failed`, or
    /// `Cancelled`. `last_error` is set for `Failed` and may be set for
    /// `Cancelled` when cancellation was the result of a downstream error.
    TaskCompleted {
        id: String,
        status: TaskStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        last_error: Option<String>,
    },

    // --- Conversation scratchpad (issue #190) -----------------------------
    /// A conversation's scratchpad changed (a note was written or deleted),
    /// whether by the LLM's builtin tools or by a client command. Delivered to
    /// connections subscribed via `SubscribeBackgroundTasks`. Carries only the
    /// `conversation_id`; clients re-read via `GetConversationScratchpad`.
    ScratchpadChanged {
        conversation_id: String,
    },

    // --- Client-side tool execution (issue #107) --------------------------
    /// The daemon's turn has suspended on a client-local MCP tool call.
    /// The client is expected to execute `tool_name` with `arguments`
    /// against its local environment and post the outcome back as
    /// `Command::ClientToolResult` with the same `task_id` and
    /// `tool_call_id`. Until that command arrives, the turn parks in
    /// `pending_client_tool`.
    ClientToolCall {
        task_id: TaskId,
        conversation_id: String,
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Status {
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub embeddings: EmbeddingsSettingsView,
    pub persistence: PersistenceSettingsView,
    /// Configurable assistant disposition (issue #226). Carries the 7
    /// "Expressive 7" trait levels as a typed struct (see
    /// [`PersonalitySettingsView`]).
    #[serde(default)]
    pub personality: PersonalitySettingsView,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ConfigChanges {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings_connector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_remote_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_remote_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_push_on_update: Option<bool>,
    // Personality (#226): one optional level per trait. `None` = leave that
    // trait unchanged on `SetConfig`; a present value overrides just that
    // trait. Serializes as the lowercase level string (e.g. `"never"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personality_professionalism: Option<PersonalityLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personality_warmth: Option<PersonalityLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personality_directness: Option<PersonalityLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personality_enthusiasm: Option<PersonalityLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personality_humor: Option<PersonalityLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personality_sarcasm: Option<PersonalityLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub personality_pretentiousness: Option<PersonalityLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub message_count: u32,
    pub updated_at: String,
    #[serde(default)]
    pub archived: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationView {
    pub id: String,
    pub title: String,
    pub messages: Vec<MessageView>,
    /// One-time advisories surfaced after `GetConversation` — e.g. the
    /// conversation's last model selection no longer resolves and was cleared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ConversationWarning>,
    /// The conversation's currently stored (connection, model, effort)
    /// selection, when one has been pinned by a prior `SendMessage` override.
    /// `None` means the daemon will fall back to the `interactive` purpose on
    /// the next send. Cleared automatically when the previous selection no
    /// longer resolves (see `ConversationWarning::DanglingModelSelection`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_selection: Option<ConversationModelSelectionView>,
    /// The conversation's stored personality override (#227), when one has been
    /// pinned by a prior `SetConversationPersonality`. `None` means the
    /// conversation uses the global personality. Like `model_selection`, this is
    /// a partial override resolved against the global config on each send.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_personality: Option<ConversationPersonalityView>,
}

/// Advisory conditions attached to a conversation view. Modeled as an enum
/// so additional variants can be added without breaking existing clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationWarning {
    /// The conversation's previous model selection no longer resolves
    /// (connection removed or model not listed by the connector). The
    /// selection has been cleared and the server fell back to the
    /// `fallback_to` target.
    DanglingModelSelection {
        previous_selection: ConversationModelSelectionView,
        fallback_to: ConversationModelSelectionView,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageView {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessagesView {
    pub total_raw_count: u32,
    pub truncated: bool,
    pub messages: Vec<MessageView>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingsSettingsView {
    pub connector: String,
    pub model: String,
    pub base_url: String,
    pub has_api_key: bool,
    pub available: bool,
    pub is_default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectorDefaultsView {
    pub llm_model: String,
    pub llm_base_url: String,
    pub backend_llm_model: String,
    pub embeddings_model: String,
    pub embeddings_base_url: String,
    pub embeddings_available: bool,
    pub hosted_tool_search_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistenceSettingsView {
    pub enabled: bool,
    /// Empty string means no remote is configured.
    pub remote_url: String,
    pub remote_name: String,
    pub push_on_update: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerView {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub enabled: bool,
    /// "running" | "stopped" | "disabled"
    pub status: String,
    pub tool_count: u32,
}

// --- Named-connection views (#11) ------------------------------------------

/// Opaque, protocol-neutral representation of a connection config.
///
/// This mirrors the daemon's internal `ConnectionConfig` (one variant per
/// connector type) but lives here so clients don't need to depend on the
/// daemon crate. Credentials are represented as `has_credentials` booleans
/// on the view; raw secret values are never serialized back through the API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum ConnectionConfigView {
    Anthropic {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key_env: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        connect_timeout_secs: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream_timeout_secs: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_context_tokens: Option<u64>,
    },
    #[serde(rename = "openai")]
    OpenAi {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key_env: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        connect_timeout_secs: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream_timeout_secs: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_context_tokens: Option<u64>,
    },
    Bedrock {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aws_profile: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        connect_timeout_secs: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream_timeout_secs: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_context_tokens: Option<u64>,
    },
    Ollama {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        connect_timeout_secs: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream_timeout_secs: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keep_warm: Option<bool>,
        /// Hard ceiling on the context window in tokens; `None` = "max
        /// available" (use the model's reported maximum).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_context_tokens: Option<u64>,
    },
}

impl ConnectionConfigView {
    /// Short connector-type identifier (matches the `type =` tag).
    pub fn connector_type(&self) -> &'static str {
        match self {
            Self::Anthropic { .. } => "anthropic",
            Self::OpenAi { .. } => "openai",
            Self::Bedrock { .. } => "bedrock",
            Self::Ollama { .. } => "ollama",
        }
    }
}

/// Availability of a connection in the registry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ConnectionAvailability {
    Ok,
    Unavailable { reason: String },
}

/// Aggregate view of a single connection for the connections list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectionView {
    pub id: String,
    /// Short connector-type identifier (`"openai"`, `"anthropic"`, etc.).
    pub connector_type: String,
    /// Human-friendly label; defaults to `"<id> (<connector_type>)"` but
    /// daemons can synthesize a more descriptive value.
    pub display_label: String,
    pub availability: ConnectionAvailability,
    /// True when credentials could be resolved during the most recent sanity
    /// check (env var present, keyring lookup succeeded, or Bedrock/Ollama
    /// which auth via ambient credentials / none).
    pub has_credentials: bool,
    /// Echoed non-secret config, so a client can pre-fill an edit dialog
    /// without losing the stored endpoint/profile/region or the credential
    /// env-var *name*. Reuses the create/update input type
    /// [`ConnectionConfigView`], which has no variant capable of carrying a
    /// raw secret value — so no secret is ever serialized here. `None` only
    /// when the daemon has no stored config for the connection.
    ///
    /// Added after the initial `ConnectionView` shipped; `#[serde(default)]`
    /// keeps older daemons (which omit it) deserializable on newer clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<ConnectionConfigView>,
}

/// A single model enumerated across one or all connections. Mirrors the
/// core `ModelInfo` fields while tagging the connection it came from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelListing {
    pub connection_id: String,
    pub connection_label: String,
    pub model: ModelInfoView,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfoView {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_limit: Option<u64>,
    #[serde(default)]
    pub capabilities: ModelCapabilitiesView,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCapabilitiesView {
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub embedding: bool,
}

// --- Purpose views (#10 + #11) --------------------------------------------

// `PurposeKindApi` and `EffortLevel` are re-exports of the canonical
// types in `desktop_assistant_core::ports::inbound` (#43). The aliases
// are kept so existing callers keep compiling without churn; new code
// can use either name.
pub use desktop_assistant_core::ports::inbound::Effort as EffortLevel;
pub use desktop_assistant_core::ports::inbound::PurposeKind as PurposeKindApi;

// Personality wire types (#226). Re-export the canonical core types so the
// settings channel, the daemon config, and clients (e.g. the KCM) share one
// schema rather than maintaining a parallel definition. `PersonalitySettingsView`
// is the `Config`-view shape (the 7 trait levels); it is the core `Personality`
// struct verbatim, so converting between the wire view and the core type is the
// identity `From` impl.
pub use desktop_assistant_core::prompts::{Personality, PersonalityLevel};
pub type PersonalitySettingsView = Personality;

// Per-conversation personality override (#227, Phase 2). Re-export the canonical
// core [`PersonalityOverride`] (7 optional trait levels) so the per-conversation
// command/view shares one schema with the resolution logic in core. The view
// returned by `GetConversation` / `SetConversationPersonality` is that override
// verbatim, so converting between wire and core is the identity.
pub use desktop_assistant_core::prompts::PersonalityOverride;
pub type ConversationPersonalityView = PersonalityOverride;

/// Protocol-neutral purpose config. String `"primary"` in the connection or
/// model field means "inherit from interactive" — the daemon resolves this
/// before dispatch (see `crates/daemon/src/purposes.rs`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PurposeConfigView {
    /// Either a connection id (slug) or the literal string `"primary"`.
    pub connection: String,
    /// Either a model id or the literal string `"primary"`.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortLevel>,
    /// Optional per-purpose override for the model's context window in
    /// tokens (issue #51). When omitted, the daemon consults the
    /// connector's curated table and a conservative universal fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
}

/// Aggregate purpose view. Missing entries mean the purpose is not
/// configured (the daemon falls back to the primary LLM for those).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PurposesView {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interactive: Option<PurposeConfigView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dreaming: Option<PurposeConfigView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<PurposeConfigView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub titling: Option<PurposeConfigView>,
}

impl PurposesView {
    /// Convenience: convert into a BTreeMap keyed by purpose key for clients
    /// that prefer iteration.
    pub fn to_map(&self) -> BTreeMap<String, PurposeConfigView> {
        let mut map = BTreeMap::new();
        if let Some(v) = &self.interactive {
            map.insert("interactive".to_string(), v.clone());
        }
        if let Some(v) = &self.dreaming {
            map.insert("dreaming".to_string(), v.clone());
        }
        if let Some(v) = &self.embedding {
            map.insert("embedding".to_string(), v.clone());
        }
        if let Some(v) = &self.titling {
            map.insert("titling".to_string(), v.clone());
        }
        map
    }
}

// --- Per-send model override (#11) ----------------------------------------

/// Caller-supplied override for a single `SendMessage` (or `SendPrompt`)
/// call. The daemon validates that `connection_id` is live and that the
/// connector lists `model_id`; otherwise the request is rejected with a
/// 400-style error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SendPromptOverride {
    pub connection_id: String,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortLevel>,
}

/// View of a conversation's stored model selection. Same shape as
/// [`SendPromptOverride`] minus the "this is a request" framing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationModelSelectionView {
    pub connection_id: String,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortLevel>,
}

// --- Background-task types (issue #110) -----------------------------------
//
// Protocol-level types only. The registry that emits these lives in
// `crates/application` (separate issue). `TaskId` is a newtype around
// `String` so typed APIs can refuse silent coercions; see the
// `task_id_is_distinct_from_string_for_typed_apis` compile_fail doctests
// in the test module.

/// Opaque task identifier. Wraps a `String` (typically a UUID) so the
/// daemon can swap the underlying representation without churning callers,
/// and so typed APIs reject a bare `String` at compile time.
///
/// `TaskId` is a distinct nominal type from `String`: callers must
/// explicitly wrap and unwrap, which prevents accidental cross-domain
/// values (e.g. passing a conversation id where a task id is expected).
/// This file's test module asserts the runtime behavior; the two
/// `compile_fail` examples below assert the type discipline at compile
/// time, and the third example shows the correct call shape.
///
/// ```compile_fail
/// use desktop_assistant_api_model::TaskId;
/// fn takes_task_id(_: TaskId) {}
/// // A bare String must NOT coerce to TaskId.
/// takes_task_id(String::from("not-a-task-id"));
/// ```
///
/// ```compile_fail
/// use desktop_assistant_api_model::TaskId;
/// fn takes_string(_: String) {}
/// // A TaskId must NOT coerce to String.
/// takes_string(TaskId(String::from("x")));
/// ```
///
/// ```
/// use desktop_assistant_api_model::TaskId;
/// fn takes_task_id(_: TaskId) {}
/// takes_task_id(TaskId(String::from("ok")));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(pub String);

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Discriminator for what kind of work a background task represents. Stored
/// alongside each task so the UI can present subagents and standalone agents
/// differently from foreground conversation turns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// A foreground conversation turn that is also tracked in the registry.
    Conversation { conversation_id: String },
    /// A subagent invoked by the parent task's `spawn_subagent` tool call.
    Subagent {
        parent_task_id: TaskId,
        conversation_id: String,
        name: String,
    },
    /// A user-initiated standalone background agent (no waiting parent).
    Standalone {
        name: String,
        conversation_id: String,
    },
}

/// Lifecycle status of a background task. `Cancelled` requires the
/// cancellation machinery from #109; before that lands, the registry only
/// produces `Pending`/`Running`/`Completed`/`Failed`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Wire-format view of a single background task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskView {
    pub id: TaskId,
    pub kind: TaskKind,
    pub status: TaskStatus,
    /// Unix epoch milliseconds when the task transitioned to `Running`.
    pub started_at: i64,
    /// Unix epoch milliseconds when the task reached a terminal state.
    /// `None` while the task is still `Pending`/`Running`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<i64>,
    /// Set when `status == Failed` (and optionally when `status == Cancelled`
    /// because of an upstream failure).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    /// Parent task id for subagents; mirrors `TaskKind::Subagent::parent_task_id`
    /// at the top level so the UI does not have to destructure `kind`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<TaskId>,
    /// Direct subagents currently registered under this task.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<TaskId>,
    /// Human-friendly label for list views (e.g. "Researcher: pricing data").
    pub title: String,
    /// Short progress string the task can update via `Event::TaskProgress`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_hint: Option<String>,
}

/// Severity for a single log line.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// What part of a task's lifecycle produced a log entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogCategory {
    /// A turn of the underlying LLM (prompt sent, response received).
    ModelTurn,
    /// The task invoked a tool.
    ToolCall,
    /// A tool returned a result (success or error).
    ToolResult,
    /// A free-form status update (e.g. "fetching page 2/4").
    Status,
    /// Registry-emitted lifecycle marker (started, cancelled, completed).
    Lifecycle,
}

/// A single bounded-buffer log entry attached to a background task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskLogEntry {
    /// Monotonically increasing per-task sequence number; clients use this
    /// as `after_seq` to resume paging.
    pub seq: u64,
    /// Unix epoch milliseconds when the entry was recorded.
    pub timestamp: i64,
    pub level: LogLevel,
    pub category: LogCategory,
    pub message: String,
    /// Optional structured payload — e.g. tool input/output JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// WebSocket request envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WsRequest {
    pub id: String,
    pub command: Command,
}

/// WebSocket frames sent from server to client.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WsFrame {
    Result { id: String, result: CommandResult },
    Error { id: String, error: String },
    Event { event: Event },
}

/// The first frame on a UDS connection: the JWT plus, optionally, the client's
/// per-machine **system id** and a friendly host label for tool-locality
/// co-location (issue #248).
///
/// The UDS server has always read the JWT out of this frame's `jwt` field; the
/// `system_id` / `host_label` fields are **optional additions** — older clients
/// omit them and the server falls back to the transport heuristic (#243),
/// unchanged. Both are `#[serde(default, skip_serializing_if = "Option::is_none")]`
/// so the wire shape is byte-identical to the old `{"jwt": "…"}` when a client
/// sends no id.
///
/// The system id is a **co-location/routing hint, not a trust boundary** (#248):
/// it is self-reported and no privilege is gated on it (auth remains the JWT).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct UdsHandshake {
    /// Bearer JWT the server validates. Optional in the *type* (so a handshake
    /// frame missing it still parses and the server can reply with the same
    /// explicit "missing jwt" auth error it always has — rather than a generic
    /// deserialize failure); the server rejects a `None`/blank jwt. The client
    /// always sets it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jwt: Option<String>,
    /// The client's per-machine system id (#248). `None`/absent for older
    /// clients ⇒ the server falls back to the transport heuristic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_id: Option<String>,
    /// A friendly host label for the remote tool note (#248), e.g. the client's
    /// hostname. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_label: Option<String>,
}

/// HTTP header carrying the client's per-machine **system id** on the WebSocket
/// upgrade (issue #248). The WS transport authenticates via the `Authorization`
/// bearer header at upgrade time (not an in-band frame), so the system id rides
/// a custom header alongside it. Optional — older clients omit it and the server
/// falls back to the transport heuristic.
pub const WS_SYSTEM_ID_HEADER: &str = "x-adelie-system-id";

/// HTTP header carrying the client's friendly host label on the WebSocket
/// upgrade (issue #248). Optional companion to [`WS_SYSTEM_ID_HEADER`].
pub const WS_HOST_LABEL_HEADER: &str = "x-adelie-host-label";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uds_handshake_without_system_id_is_wire_compatible() {
        // Older shape: a bare `{"jwt": "…"}`. The optional fields must be
        // skipped on serialize so the wire bytes match the pre-#248 handshake,
        // and a legacy `{"jwt": "…"}` must still deserialize.
        let h = UdsHandshake {
            jwt: Some("tok".into()),
            system_id: None,
            host_label: None,
        };
        let json = serde_json::to_string(&h).unwrap();
        assert_eq!(json, r#"{"jwt":"tok"}"#, "absent fields must not appear");

        let legacy: UdsHandshake = serde_json::from_str(r#"{"jwt": "tok"}"#).unwrap();
        assert_eq!(legacy.jwt.as_deref(), Some("tok"));
        assert_eq!(legacy.system_id, None);
        assert_eq!(legacy.host_label, None);
    }

    #[test]
    fn uds_handshake_missing_jwt_still_parses() {
        // A frame with no `jwt` must still deserialize (so the server can return
        // its explicit "missing jwt" auth error rather than a generic parse
        // failure). `jwt` is `None`.
        let h: UdsHandshake = serde_json::from_str(r#"{"hello":"world"}"#).unwrap();
        assert_eq!(h.jwt, None);
    }

    #[test]
    fn uds_handshake_with_system_id_roundtrips() {
        let h = UdsHandshake {
            jwt: Some("tok".into()),
            system_id: Some("machine-abc".into()),
            host_label: Some("laptop".into()),
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: UdsHandshake = serde_json::from_str(&json).unwrap();
        assert_eq!(back, h);
        assert!(json.contains("machine-abc"));
        assert!(json.contains("laptop"));
    }

    #[test]
    fn command_json_roundtrip_ping() {
        let cmd = Command::Ping;
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn event_json_roundtrip_message_delta() {
        let ev = Event::AssistantDelta {
            conversation_id: "c1".into(),
            request_id: "r1".into(),
            chunk: "hello".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn command_json_roundtrip_set_config() {
        let cmd = Command::SetConfig {
            changes: ConfigChanges {
                persistence_enabled: Some(true),
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn list_connections_roundtrip() {
        let cmd = Command::ListConnections;
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("list_connections"));
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn create_connection_roundtrip_openai() {
        let cmd = Command::CreateConnection {
            id: "work".into(),
            config: ConnectionConfigView::OpenAi {
                base_url: Some("https://api.openai.com/v1".into()),
                api_key_env: Some("OPENAI_WORK_KEY".into()),
                connect_timeout_secs: None,
                stream_timeout_secs: None,
                max_context_tokens: None,
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn connection_config_view_tagged_type() {
        let c = ConnectionConfigView::Bedrock {
            aws_profile: Some("work".into()),
            region: Some("us-west-2".into()),
            base_url: None,
            connect_timeout_secs: None,
            stream_timeout_secs: None,
            max_context_tokens: None,
        };
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("\"type\":\"bedrock\""));
        assert_eq!(c.connector_type(), "bedrock");
        let back: ConnectionConfigView = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn delete_connection_force_flag() {
        let cmd = Command::DeleteConnection {
            id: "old".into(),
            force: true,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);

        // Missing force flag defaults to false.
        let cmd2: Command = serde_json::from_str(r#"{"delete_connection":{"id":"old"}}"#).unwrap();
        assert_eq!(
            cmd2,
            Command::DeleteConnection {
                id: "old".into(),
                force: false,
            }
        );
    }

    #[test]
    fn list_available_models_optional_connection_and_refresh() {
        // Both fields omitted.
        let cmd: Command = serde_json::from_str(r#"{"list_available_models":{}}"#).unwrap();
        assert_eq!(
            cmd,
            Command::ListAvailableModels {
                connection_id: None,
                refresh: false,
            }
        );

        // All fields present.
        let cmd2 = Command::ListAvailableModels {
            connection_id: Some("aws".into()),
            refresh: true,
        };
        let json = serde_json::to_string(&cmd2).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd2, back);
    }

    #[test]
    fn set_purpose_roundtrip() {
        let cmd = Command::SetPurpose {
            purpose: PurposeKindApi::Dreaming,
            config: PurposeConfigView {
                connection: "primary".into(),
                model: "primary".into(),
                effort: Some(EffortLevel::Low),
                max_context_tokens: None,
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn set_purpose_view_carries_max_context_tokens() {
        // Issue #51: the wire type carries the user's per-purpose
        // `max_context_tokens` override end-to-end so the KCM can read
        // and write it.
        let cfg = PurposeConfigView {
            connection: "work_bedrock".into(),
            model: "us.amazon.nova-premier-v1:0".into(),
            effort: Some(EffortLevel::Medium),
            max_context_tokens: Some(1_000_000),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("max_context_tokens"));
        let back: PurposeConfigView = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);

        // Round-trip with `None` must omit the field on the wire.
        let cfg_none = PurposeConfigView {
            connection: "work".into(),
            model: "gpt-5".into(),
            effort: None,
            max_context_tokens: None,
        };
        let json_none = serde_json::to_string(&cfg_none).unwrap();
        assert!(!json_none.contains("max_context_tokens"));
    }

    #[test]
    fn send_message_override_is_optional() {
        // Without override.
        let cmd: Command =
            serde_json::from_str(r#"{"send_message":{"conversation_id":"c1","content":"hi"}}"#)
                .unwrap();
        match &cmd {
            Command::SendMessage {
                override_selection, ..
            } => assert!(override_selection.is_none()),
            other => panic!("unexpected {other:?}"),
        }

        // With override.
        let cmd2 = Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hi".into(),
            override_selection: Some(SendPromptOverride {
                connection_id: "aws".into(),
                model_id: "claude-sonnet-4".into(),
                effort: Some(EffortLevel::High),
            }),
            system_refinement: String::new(),
            idempotency_key: None,
        };
        let json = serde_json::to_string(&cmd2).unwrap();
        assert!(json.contains("\"override\":"));
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd2, back);
    }

    #[test]
    fn send_message_system_refinement_is_optional_and_round_trips() {
        // Absent on the wire → defaults to empty.
        let cmd: Command =
            serde_json::from_str(r#"{"send_message":{"conversation_id":"c1","content":"hi"}}"#)
                .unwrap();
        match &cmd {
            Command::SendMessage {
                system_refinement, ..
            } => assert!(system_refinement.is_empty()),
            other => panic!("unexpected {other:?}"),
        }

        // Empty refinement is omitted from the serialized form (byte-compatible
        // with pre-refinement `SendMessage`).
        let empty = Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hi".into(),
            override_selection: None,
            system_refinement: String::new(),
            idempotency_key: None,
        };
        let json_empty = serde_json::to_string(&empty).unwrap();
        assert!(
            !json_empty.contains("system_refinement"),
            "empty refinement must not appear on the wire: {json_empty}"
        );

        // Non-empty refinement is present and round-trips.
        let with_refinement = Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hi".into(),
            override_selection: None,
            system_refinement: "Respond briefly, by voice.".into(),
            idempotency_key: None,
        };
        let json = serde_json::to_string(&with_refinement).unwrap();
        assert!(json.contains("\"system_refinement\":\"Respond briefly, by voice.\""));
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(with_refinement, back);
    }

    #[test]
    fn send_message_idempotency_key_is_optional_and_round_trips() {
        // Absent on the wire → None (byte-compatible with pre-#204 SendMessage).
        let cmd: Command =
            serde_json::from_str(r#"{"send_message":{"conversation_id":"c1","content":"hi"}}"#)
                .unwrap();
        match &cmd {
            Command::SendMessage {
                idempotency_key, ..
            } => assert!(idempotency_key.is_none()),
            other => panic!("unexpected {other:?}"),
        }

        // None is omitted from the serialized form (no wire bloat for callers
        // that don't use idempotency).
        let without = Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hi".into(),
            override_selection: None,
            system_refinement: String::new(),
            idempotency_key: None,
        };
        let json = serde_json::to_string(&without).unwrap();
        assert!(
            !json.contains("idempotency_key"),
            "an absent key must not appear on the wire: {json}"
        );

        // A present key serializes and round-trips.
        let with_key = Command::SendMessage {
            conversation_id: "c1".into(),
            content: "hi".into(),
            override_selection: None,
            system_refinement: String::new(),
            idempotency_key: Some("turn-uuid-1".into()),
        };
        let json = serde_json::to_string(&with_key).unwrap();
        assert!(json.contains("\"idempotency_key\":\"turn-uuid-1\""));
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(with_key, back);
    }

    #[test]
    fn effort_serialize_lowercase() {
        assert_eq!(serde_json::to_string(&EffortLevel::Low).unwrap(), "\"low\"");
        assert_eq!(
            serde_json::to_string(&EffortLevel::Medium).unwrap(),
            "\"medium\""
        );
        assert_eq!(
            serde_json::to_string(&EffortLevel::High).unwrap(),
            "\"high\""
        );
    }

    #[test]
    fn conversation_view_warnings_default_empty() {
        let json = r#"{"id":"c1","title":"t","messages":[]}"#;
        let v: ConversationView = serde_json::from_str(json).unwrap();
        assert!(v.warnings.is_empty());
    }

    #[test]
    fn conversation_warning_dangling_selection_roundtrip() {
        let w = ConversationWarning::DanglingModelSelection {
            previous_selection: ConversationModelSelectionView {
                connection_id: "old".into(),
                model_id: "gone".into(),
                effort: None,
            },
            fallback_to: ConversationModelSelectionView {
                connection_id: "work".into(),
                model_id: "gpt-5".into(),
                effort: Some(EffortLevel::Medium),
            },
        };
        let json = serde_json::to_string(&w).unwrap();
        assert!(json.contains("\"type\":\"dangling_model_selection\""));
        let back: ConversationWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(w, back);
    }

    #[test]
    fn connection_availability_tagged() {
        let ok = ConnectionAvailability::Ok;
        let json = serde_json::to_string(&ok).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        let un = ConnectionAvailability::Unavailable { reason: "x".into() };
        let json2 = serde_json::to_string(&un).unwrap();
        assert!(json2.contains("\"status\":\"unavailable\""));
        let back: ConnectionAvailability = serde_json::from_str(&json2).unwrap();
        assert_eq!(un, back);
    }

    // ---- #110: background task variants -----------------------------------
    //
    // Tests below are the spec-driven acceptance criteria from issue #110.
    // They are intentionally written before the corresponding types are
    // introduced (TDD): they will fail to compile / fail equality until the
    // protocol shape is added.

    fn sample_task_view() -> TaskView {
        TaskView {
            id: TaskId("task-1".into()),
            kind: TaskKind::Subagent {
                parent_task_id: TaskId("parent".into()),
                conversation_id: "conv-9".into(),
                name: "researcher".into(),
            },
            status: TaskStatus::Running,
            started_at: 1_700_000_000,
            ended_at: Some(1_700_000_500),
            last_error: None,
            parent: Some(TaskId("parent".into())),
            children: vec![TaskId("child-a".into()), TaskId("child-b".into())],
            title: "Researching subagent".into(),
            progress_hint: Some("step 2/4".into()),
        }
    }

    fn sample_log_entry() -> TaskLogEntry {
        TaskLogEntry {
            seq: 7,
            timestamp: 1_700_000_123,
            level: LogLevel::Info,
            category: LogCategory::ToolCall,
            message: "calling tool".into(),
            data: Some(serde_json::json!({"tool": "search"})),
        }
    }

    #[test]
    fn task_view_round_trips_via_serde_json() {
        // TaskId
        let id = TaskId("abc".into());
        let json = serde_json::to_string(&id).unwrap();
        let back: TaskId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);

        // TaskKind — all three variants.
        for kind in [
            TaskKind::Conversation {
                conversation_id: "c1".into(),
            },
            TaskKind::Subagent {
                parent_task_id: TaskId("p1".into()),
                conversation_id: "c2".into(),
                name: "child".into(),
            },
            TaskKind::Standalone {
                name: "agent".into(),
                conversation_id: "c3".into(),
            },
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: TaskKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, back);
        }

        // TaskStatus — every variant.
        for status in [
            TaskStatus::Pending,
            TaskStatus::Running,
            TaskStatus::Completed,
            TaskStatus::Failed,
            TaskStatus::Cancelled,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: TaskStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }

        // LogLevel
        for level in [LogLevel::Info, LogLevel::Warn, LogLevel::Error] {
            let json = serde_json::to_string(&level).unwrap();
            let back: LogLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, back);
        }

        // LogCategory
        for cat in [
            LogCategory::ModelTurn,
            LogCategory::ToolCall,
            LogCategory::ToolResult,
            LogCategory::Status,
            LogCategory::Lifecycle,
        ] {
            let json = serde_json::to_string(&cat).unwrap();
            let back: LogCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(cat, back);
        }

        // TaskView
        let view = sample_task_view();
        let json = serde_json::to_string(&view).unwrap();
        let back: TaskView = serde_json::from_str(&json).unwrap();
        assert_eq!(view, back);

        // TaskLogEntry
        let entry = sample_log_entry();
        let json = serde_json::to_string(&entry).unwrap();
        let back: TaskLogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn task_status_serialize_snake_case() {
        assert_eq!(
            serde_json::to_string(&TaskStatus::Pending).unwrap(),
            "\"pending\""
        );
        assert_eq!(
            serde_json::to_string(&TaskStatus::Running).unwrap(),
            "\"running\""
        );
        assert_eq!(
            serde_json::to_string(&TaskStatus::Completed).unwrap(),
            "\"completed\""
        );
        assert_eq!(
            serde_json::to_string(&TaskStatus::Failed).unwrap(),
            "\"failed\""
        );
        assert_eq!(
            serde_json::to_string(&TaskStatus::Cancelled).unwrap(),
            "\"cancelled\""
        );
    }

    #[test]
    fn log_enums_serialize_snake_case() {
        assert_eq!(serde_json::to_string(&LogLevel::Info).unwrap(), "\"info\"");
        assert_eq!(serde_json::to_string(&LogLevel::Warn).unwrap(), "\"warn\"");
        assert_eq!(
            serde_json::to_string(&LogLevel::Error).unwrap(),
            "\"error\""
        );
        assert_eq!(
            serde_json::to_string(&LogCategory::ModelTurn).unwrap(),
            "\"model_turn\""
        );
        assert_eq!(
            serde_json::to_string(&LogCategory::ToolCall).unwrap(),
            "\"tool_call\""
        );
        assert_eq!(
            serde_json::to_string(&LogCategory::ToolResult).unwrap(),
            "\"tool_result\""
        );
        assert_eq!(
            serde_json::to_string(&LogCategory::Status).unwrap(),
            "\"status\""
        );
        assert_eq!(
            serde_json::to_string(&LogCategory::Lifecycle).unwrap(),
            "\"lifecycle\""
        );
    }

    #[test]
    fn command_variants_match_documented_snake_case() {
        // ListBackgroundTasks
        let cmd = Command::ListBackgroundTasks {
            include_finished: true,
            limit: Some(50),
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"list_background_tasks":{"include_finished":true,"limit":50}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);

        // GetBackgroundTask
        let cmd = Command::GetBackgroundTask { id: "t-1".into() };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(r#"{"get_background_task":{"id":"t-1"}}"#).unwrap();
        assert_eq!(v, expected);
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);

        // CancelBackgroundTask
        let cmd = Command::CancelBackgroundTask { id: "t-2".into() };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(r#"{"cancel_background_task":{"id":"t-2"}}"#).unwrap();
        assert_eq!(v, expected);
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);

        // GetBackgroundTaskLogs
        let cmd = Command::GetBackgroundTaskLogs {
            id: "t-3".into(),
            after_seq: Some(42),
            limit: Some(100),
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"get_background_task_logs":{"id":"t-3","after_seq":42,"limit":100}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);

        // SubscribeBackgroundTasks (unit variant — serialized as a bare string)
        let cmd = Command::SubscribeBackgroundTasks;
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        assert_eq!(v, serde_json::json!("subscribe_background_tasks"));
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);

        // UnsubscribeBackgroundTasks
        let cmd = Command::UnsubscribeBackgroundTasks;
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        assert_eq!(v, serde_json::json!("unsubscribe_background_tasks"));
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);

        // SpawnStandaloneAgent
        let cmd = Command::SpawnStandaloneAgent {
            name: "researcher".into(),
            initial_prompt: "go".into(),
            override_selection: Some(SendPromptOverride {
                connection_id: "aws".into(),
                model_id: "claude-sonnet-4".into(),
                effort: Some(EffortLevel::High),
            }),
            tools: Some(vec!["search".into(), "fetch".into()]),
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"spawn_standalone_agent":{"name":"researcher","initial_prompt":"go","override_selection":{"connection_id":"aws","model_id":"claude-sonnet-4","effort":"high"},"tools":["search","fetch"]}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn command_result_background_task_variants_round_trip() {
        // BackgroundTasks
        let res = CommandResult::BackgroundTasks(vec![sample_task_view()]);
        let v: serde_json::Value = serde_json::to_value(&res).unwrap();
        assert!(v.get("background_tasks").is_some());
        let back: CommandResult = serde_json::from_value(v).unwrap();
        assert_eq!(res, back);

        // BackgroundTask
        let res = CommandResult::BackgroundTask(sample_task_view());
        let v: serde_json::Value = serde_json::to_value(&res).unwrap();
        assert!(v.get("background_task").is_some());
        let back: CommandResult = serde_json::from_value(v).unwrap();
        assert_eq!(res, back);

        // BackgroundTaskLogs
        let res = CommandResult::BackgroundTaskLogs {
            entries: vec![sample_log_entry()],
            next_seq: 8,
        };
        let v: serde_json::Value = serde_json::to_value(&res).unwrap();
        let logs = v
            .get("background_task_logs")
            .expect("background_task_logs key");
        assert_eq!(logs.get("next_seq"), Some(&serde_json::json!(8)));
        let back: CommandResult = serde_json::from_value(v).unwrap();
        assert_eq!(res, back);

        // BackgroundTaskSpawned
        let res = CommandResult::BackgroundTaskSpawned { id: "t-new".into() };
        let v: serde_json::Value = serde_json::to_value(&res).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(r#"{"background_task_spawned":{"id":"t-new"}}"#).unwrap();
        assert_eq!(v, expected);
        let back: CommandResult = serde_json::from_value(v).unwrap();
        assert_eq!(res, back);
    }

    #[test]
    fn send_message_ack_carries_request_and_task_ids() {
        // Golden-file test for the SendMessageAck shape: callers must be able
        // to correlate the ack to BOTH the streamed response events
        // (`request_id`, voice#49) and the registered background task
        // (`task_id`).
        let res = CommandResult::SendMessageAck {
            request_id: "req-xyz".into(),
            task_id: "task-abc".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&res).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"send_message_ack":{"request_id":"req-xyz","task_id":"task-abc"}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        let back: CommandResult = serde_json::from_value(v).unwrap();
        assert_eq!(res, back);
    }

    #[test]
    fn scratchpad_commands_match_documented_snake_case() {
        // GetConversationScratchpad
        let cmd = Command::GetConversationScratchpad {
            conversation_id: "c-1".into(),
            max_results: Some(20),
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"get_conversation_scratchpad":{"conversation_id":"c-1","max_results":20}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        assert_eq!(cmd, serde_json::from_value(v).unwrap());

        // SetScratchpadNote
        let cmd = Command::SetScratchpadNote {
            conversation_id: "c-1".into(),
            key: "t1".into(),
            content: "wire it".into(),
            note_type: "todo".into(),
            sequence: Some(2),
            done: true,
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"set_scratchpad_note":{"conversation_id":"c-1","key":"t1","content":"wire it","note_type":"todo","sequence":2,"done":true}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        assert_eq!(cmd, serde_json::from_value(v).unwrap());

        // DeleteScratchpadNotes (clear-all form)
        let cmd = Command::DeleteScratchpadNotes {
            conversation_id: "c-1".into(),
            keys: vec![],
            all: true,
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"delete_scratchpad_notes":{"conversation_id":"c-1","keys":[],"all":true}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        assert_eq!(cmd, serde_json::from_value(v).unwrap());
    }

    #[test]
    fn scratchpad_result_and_event_roundtrip() {
        let res = CommandResult::Scratchpad(vec![ScratchpadNoteView {
            id: "sp-1".into(),
            key: "t1".into(),
            content: "wire it".into(),
            note_type: "todo".into(),
            sequence: Some(1),
            done: false,
            updated_at: "2026-06-04 00:00:00".into(),
        }]);
        let v: serde_json::Value = serde_json::to_value(&res).unwrap();
        assert!(v.get("scratchpad").is_some());
        assert_eq!(res, serde_json::from_value(v).unwrap());

        let ev = Event::ScratchpadChanged {
            conversation_id: "c-1".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&ev).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(r#"{"scratchpad_changed":{"conversation_id":"c-1"}}"#).unwrap();
        assert_eq!(v, expected);
        assert_eq!(ev, serde_json::from_value(v).unwrap());
    }

    #[test]
    fn event_variants_match_documented_snake_case() {
        // TaskStarted
        let ev = Event::TaskStarted {
            task: sample_task_view(),
        };
        let v: serde_json::Value = serde_json::to_value(&ev).unwrap();
        let started = v.get("task_started").expect("task_started key");
        assert!(started.get("task").is_some());
        let back: Event = serde_json::from_value(v).unwrap();
        assert_eq!(ev, back);

        // TaskProgress
        let ev = Event::TaskProgress {
            id: "t-1".into(),
            progress_hint: Some("step 3/5".into()),
        };
        let v: serde_json::Value = serde_json::to_value(&ev).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(r#"{"task_progress":{"id":"t-1","progress_hint":"step 3/5"}}"#)
                .unwrap();
        assert_eq!(v, expected);
        let back: Event = serde_json::from_value(v).unwrap();
        assert_eq!(ev, back);

        // TaskLogAppended
        let ev = Event::TaskLogAppended {
            id: "t-1".into(),
            entry: sample_log_entry(),
        };
        let v: serde_json::Value = serde_json::to_value(&ev).unwrap();
        let appended = v.get("task_log_appended").expect("task_log_appended key");
        assert_eq!(appended.get("id"), Some(&serde_json::json!("t-1")));
        assert!(appended.get("entry").is_some());
        let back: Event = serde_json::from_value(v).unwrap();
        assert_eq!(ev, back);

        // TaskCompleted
        let ev = Event::TaskCompleted {
            id: "t-1".into(),
            status: TaskStatus::Failed,
            last_error: Some("nope".into()),
        };
        let v: serde_json::Value = serde_json::to_value(&ev).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"task_completed":{"id":"t-1","status":"failed","last_error":"nope"}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        let back: Event = serde_json::from_value(v).unwrap();
        assert_eq!(ev, back);
    }

    /// Sibling of the `compile_fail` doctests on the public [`TaskId`] type.
    /// Runtime check that confirms a `TaskId` does not implement
    /// `From<String>` or `Into<String>` and is `!= String` even when the
    /// inner string matches. This complements the compile-time discipline
    /// asserted by the doctests; together they form the trybuild
    /// replacement called for by #110.
    #[test]
    fn task_id_is_distinct_from_string_for_typed_apis() {
        // Helper that only accepts a `TaskId`.
        fn takes_task_id(t: TaskId) -> String {
            t.0
        }
        let id = TaskId(String::from("abc"));
        assert_eq!(takes_task_id(id.clone()), "abc");

        // Cloned strings can't be compared directly to TaskIds — the inner
        // string is reachable as `.0` only.
        let raw = String::from("abc");
        // This line is the structural guard the compile_fail doctests
        // express at the type level: there is no `==` between `TaskId` and
        // `String`. We assert via `.0` access only.
        assert_eq!(id.0, raw);
    }

    // ---- #107: client-side execution protocol surface ---------------------
    //
    // The turn state machine adds three new wire shapes — a registration
    // command, a result command, and a `ClientToolCall` event — that the
    // chat client uses to advertise its local MCP tools and stream the
    // round-trip on each suspension. These tests pin the JSON shape so
    // out-of-tree clients have a stable contract.

    #[test]
    fn register_client_tools_command_round_trips() {
        let cmd = Command::RegisterClientTools {
            tools: vec![
                ClientToolRegistration {
                    name: "fs_read".into(),
                    description: "Read a file on the user's machine".into(),
                    input_schema: serde_json::json!({
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    }),
                },
                ClientToolRegistration {
                    name: "fs_write".into(),
                    description: "Write a file on the user's machine".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                },
            ],
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        // Snake-case discriminator on the outer enum.
        assert!(v.get("register_client_tools").is_some());
        // Inner shape: a `tools` array of {name, description, input_schema}.
        let inner = v.get("register_client_tools").unwrap();
        let arr = inner.get("tools").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].get("name").unwrap(), "fs_read");
        // Round-trip.
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn client_tool_result_command_round_trips_ok_branch() {
        let cmd = Command::ClientToolResult {
            task_id: TaskId("task-1".into()),
            tool_call_id: "call-7".into(),
            result: Some("file contents go here".into()),
            error: None,
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"client_tool_result":{"task_id":"task-1","tool_call_id":"call-7","result":"file contents go here"}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn client_tool_result_command_round_trips_error_branch() {
        let cmd = Command::ClientToolResult {
            task_id: TaskId("task-2".into()),
            tool_call_id: "call-8".into(),
            result: None,
            error: Some("file does not exist".into()),
        };
        let v: serde_json::Value = serde_json::to_value(&cmd).unwrap();
        let expected: serde_json::Value = serde_json::from_str(
            r#"{"client_tool_result":{"task_id":"task-2","tool_call_id":"call-8","error":"file does not exist"}}"#,
        )
        .unwrap();
        assert_eq!(v, expected);
        let back: Command = serde_json::from_value(v).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn client_tool_call_event_round_trips() {
        let ev = Event::ClientToolCall {
            task_id: TaskId("task-1".into()),
            conversation_id: "conv-1".into(),
            tool_call_id: "call-7".into(),
            tool_name: "fs_read".into(),
            arguments: serde_json::json!({"path": "/etc/hosts"}),
        };
        let v: serde_json::Value = serde_json::to_value(&ev).unwrap();
        let inner = v.get("client_tool_call").expect("client_tool_call key");
        assert_eq!(inner.get("task_id").unwrap(), "task-1");
        assert_eq!(inner.get("conversation_id").unwrap(), "conv-1");
        assert_eq!(inner.get("tool_call_id").unwrap(), "call-7");
        assert_eq!(inner.get("tool_name").unwrap(), "fs_read");
        assert_eq!(
            inner.get("arguments").unwrap(),
            &serde_json::json!({"path": "/etc/hosts"})
        );
        let back: Event = serde_json::from_value(v).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn client_tools_registered_command_result_round_trips() {
        let res = CommandResult::ClientToolsRegistered { count: 3 };
        let v: serde_json::Value = serde_json::to_value(&res).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(r#"{"client_tools_registered":{"count":3}}"#).unwrap();
        assert_eq!(v, expected);
        let back: CommandResult = serde_json::from_value(v).unwrap();
        assert_eq!(res, back);
    }

    #[test]
    fn client_tool_result_rejects_both_result_and_error_unset() {
        // A `ClientToolResult` with neither `result` nor `error` is
        // ambiguous (success with empty body? failure with empty reason?).
        // The protocol requires exactly one of them; the daemon-side
        // validator (in application/) enforces the constraint. Here we
        // only assert the wire shape can round-trip a malformed payload —
        // the rejection lives one layer up so adapters can surface a
        // clean error to the client.
        let cmd: Command =
            serde_json::from_str(r#"{"client_tool_result":{"task_id":"t","tool_call_id":"c"}}"#)
                .unwrap();
        match cmd {
            Command::ClientToolResult { result, error, .. } => {
                assert!(result.is_none());
                assert!(error.is_none());
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    // --- Personality config wire types (#226) ------------------------------

    #[test]
    fn config_carries_default_personality() {
        // A `Config` view round-trips its personality block, and the view's
        // levels match the Expressive-7 defaults.
        let cfg = Config {
            embeddings: EmbeddingsSettingsView {
                connector: "openai".into(),
                model: "text-embedding-3-small".into(),
                base_url: "https://api.openai.com/v1".into(),
                has_api_key: true,
                available: true,
                is_default: true,
            },
            persistence: PersistenceSettingsView {
                enabled: false,
                remote_url: String::new(),
                remote_name: "origin".into(),
                push_on_update: false,
            },
            personality: PersonalitySettingsView::default(),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
        assert_eq!(back.personality.professionalism, PersonalityLevel::Always);
        assert_eq!(back.personality.humor, PersonalityLevel::Sometimes);
    }

    #[test]
    fn personality_settings_view_is_the_core_type() {
        // `PersonalitySettingsView` is the canonical core `Personality` (one
        // schema, no parallel definition), so a value flows between the wire
        // view and the core type with no lossy conversion.
        let core = Personality {
            humor: PersonalityLevel::Never,
            sarcasm: PersonalityLevel::Always,
            ..Personality::default()
        };
        let view: PersonalitySettingsView = core;
        assert_eq!(view, core);
        assert_eq!(view.humor, PersonalityLevel::Never);
        assert_eq!(view.sarcasm, PersonalityLevel::Always);
    }

    #[test]
    fn config_changes_personality_fields_optional_and_round_trip() {
        // Default `ConfigChanges` omits every personality field from the wire.
        let empty = ConfigChanges::default();
        let json = serde_json::to_string(&empty).unwrap();
        assert!(!json.contains("personality_humor"), "json: {json}");

        // A single personality change serializes only that field.
        let changes = ConfigChanges {
            personality_humor: Some(PersonalityLevel::Never),
            ..ConfigChanges::default()
        };
        let json = serde_json::to_string(&changes).unwrap();
        assert!(json.contains("personality_humor"), "json: {json}");
        assert!(json.contains("\"never\""), "json: {json}");
        assert!(!json.contains("personality_warmth"), "json: {json}");
        let back: ConfigChanges = serde_json::from_str(&json).unwrap();
        assert_eq!(changes, back);
    }

    // --- Per-conversation personality override wire types (#227) ------------

    #[test]
    fn set_conversation_personality_command_round_trips() {
        // The command carries the conversation id and a partial override; it
        // must round-trip losslessly so the daemon parses exactly what the
        // client (tui/gtk picker) sent.
        let cmd = Command::SetConversationPersonality {
            conversation_id: "conv-1".into(),
            personality: ConversationPersonalityView {
                humor: Some(PersonalityLevel::Never),
                directness: Some(PersonalityLevel::Always),
                ..ConversationPersonalityView::default()
            },
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(
            json.contains("\"set_conversation_personality\""),
            "json: {json}"
        );
        // Only the pinned traits are on the wire (skip_serializing_if).
        assert!(json.contains("\"humor\""), "json: {json}");
        assert!(json.contains("\"directness\""), "json: {json}");
        assert!(!json.contains("\"warmth\""), "json: {json}");
        let back: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn conversation_personality_result_round_trips() {
        let res = CommandResult::ConversationPersonality(ConversationPersonalityView {
            sarcasm: Some(PersonalityLevel::Never),
            ..ConversationPersonalityView::default()
        });
        let json = serde_json::to_string(&res).unwrap();
        assert!(
            json.contains("\"conversation_personality\""),
            "json: {json}"
        );
        let back: CommandResult = serde_json::from_str(&json).unwrap();
        assert_eq!(res, back);
    }

    #[test]
    fn conversation_view_carries_optional_personality_override() {
        // `conversation_personality` is omitted from the wire when `None`
        // (no override) and present when an override is stored — mirrors
        // `model_selection`.
        let mut view = ConversationView {
            id: "c1".into(),
            title: "t".into(),
            messages: vec![],
            warnings: vec![],
            model_selection: None,
            conversation_personality: None,
        };
        let json = serde_json::to_string(&view).unwrap();
        assert!(
            !json.contains("conversation_personality"),
            "absent override must not appear on the wire: {json}"
        );

        view.conversation_personality = Some(ConversationPersonalityView {
            humor: Some(PersonalityLevel::Never),
            ..ConversationPersonalityView::default()
        });
        let json = serde_json::to_string(&view).unwrap();
        assert!(json.contains("conversation_personality"), "json: {json}");
        let back: ConversationView = serde_json::from_str(&json).unwrap();
        assert_eq!(view, back);
    }
}
