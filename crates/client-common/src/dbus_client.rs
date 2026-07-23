use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use desktop_assistant_api_model as api;
use futures::StreamExt;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use zbus::Connection;
use zbus::zvariant::{OwnedValue, Value};

use crate::commands::AssistantCommands;
use crate::signal::SignalEvent;
use crate::types::{ChatMessage, ConversationDetail, ConversationSummary};

/// Encode a tag filter for the JSON-string D-Bus argument. `None` and
/// `Some(empty)` both serialise to `"null"`, matching the parsing on
/// the server side (#73).
fn tag_filter_to_json(filter: &Option<Vec<String>>) -> String {
    match filter {
        Some(tags) if !tags.is_empty() => {
            serde_json::to_string(tags).unwrap_or_else(|_| "null".to_string())
        }
        _ => "null".to_string(),
    }
}

fn decode_entries(raw: &str) -> Result<Vec<api::KnowledgeEntryView>> {
    let envelope: api::CommandResult =
        serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("decoding entries: {e}"))?;
    match envelope {
        api::CommandResult::KnowledgeEntries(items) => Ok(items),
        other => Err(anyhow::anyhow!(
            "unexpected dbus response for knowledge entries: {other:?}"
        )),
    }
}

fn decode_entry_written(raw: &str) -> Result<api::KnowledgeEntryView> {
    let envelope: api::CommandResult =
        serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("decoding entry: {e}"))?;
    match envelope {
        api::CommandResult::KnowledgeEntryWritten(entry) => Ok(entry),
        other => Err(anyhow::anyhow!(
            "unexpected dbus response for knowledge entry write: {other:?}"
        )),
    }
}

// --- Background-task `a{sv}` decoding (#116/#367) -----------------------------
//
// The bridge encodes `TaskView`/`TaskLogEntry` by serialising them to a
// `serde_json::Value` and walking it into a D-Bus `a{sv}` dictionary (see
// `dbus-bridge`'s `task_view_to_dict`). These helpers are the exact inverse:
// rebuild a `serde_json::Value` from the dict, then `serde_json::from_value`
// into the typed view — so the D-Bus client decodes the SAME JSON shape the
// UDS/WS clients get from `map_event_to_signal`. All wire input is untrusted
// (the D-Bus trust boundary), so every path returns `Err` rather than panicking.

/// Recursively project a D-Bus [`Value`] back onto a [`serde_json::Value`],
/// inverting the bridge's `json_value_to_owned`. Variant-wrapped elements
/// (`Value::Value`) are unwrapped — the bridge wraps array elements and nested
/// dict values in a variant so heterogeneous JSON arrays/objects round-trip.
/// Returns `None` for D-Bus types that have no JSON projection (file
/// descriptors, signatures) so the caller can drop the key rather than guess.
fn value_to_json(value: &Value<'_>) -> Option<JsonValue> {
    match value {
        Value::Bool(b) => Some(JsonValue::Bool(*b)),
        // Integers narrow to the JSON number space; the bridge only ever emits
        // u64/i64/f64, but accept the smaller widths defensively.
        Value::U8(n) => Some(JsonValue::from(*n)),
        Value::I16(n) => Some(JsonValue::from(*n)),
        Value::U16(n) => Some(JsonValue::from(*n)),
        Value::I32(n) => Some(JsonValue::from(*n)),
        Value::U32(n) => Some(JsonValue::from(*n)),
        Value::I64(n) => Some(JsonValue::from(*n)),
        Value::U64(n) => Some(JsonValue::from(*n)),
        // A non-finite f64 has no JSON representation (`serde_json::Number`
        // rejects NaN/±Inf); drop it rather than emit something lossy.
        Value::F64(n) => serde_json::Number::from_f64(*n).map(JsonValue::Number),
        Value::Str(s) => Some(JsonValue::String(s.as_str().to_string())),
        // Unwrap a boxed variant to its inner value (array/dict element shape).
        Value::Value(inner) => value_to_json(inner),
        Value::Array(arr) => {
            let items = arr.inner().iter().filter_map(value_to_json).collect();
            Some(JsonValue::Array(items))
        }
        Value::Dict(dict) => Some(dict_value_to_json(dict)),
        // ObjectPath/Signature/Structure/Fd/Maybe don't appear in the
        // JSON-derived task dicts; no faithful projection, so skip.
        _ => None,
    }
}

/// Project a D-Bus dictionary (`a{sv}` or `a{ss}` …) onto a JSON object. Only
/// string keys are kept (the task dicts are JSON-object-derived, so every key is
/// a string); a non-string key is skipped rather than coerced.
fn dict_value_to_json(dict: &zbus::zvariant::Dict<'_, '_>) -> JsonValue {
    let mut map = serde_json::Map::new();
    for (k, v) in dict.iter() {
        let Value::Str(key) = k else {
            continue;
        };
        if let Some(jv) = value_to_json(v) {
            map.insert(key.as_str().to_string(), jv);
        }
    }
    JsonValue::Object(map)
}

/// Project a signal's `a{sv}` argument (`HashMap<String, OwnedValue>`) onto a
/// JSON object — the shared first half of decoding `TaskView`/`TaskLogEntry`.
/// The caller `serde_json::from_value`s the result into the concrete view (kept
/// at the call site so this helper needs no `serde` trait bound).
fn dict_to_json_object(dict: &HashMap<String, OwnedValue>) -> JsonValue {
    let mut map = serde_json::Map::with_capacity(dict.len());
    for (key, owned) in dict {
        // `OwnedValue` derefs to `Value`; project each entry to JSON.
        if let Some(jv) = value_to_json(owned) {
            map.insert(key.clone(), jv);
        }
    }
    JsonValue::Object(map)
}

/// Decode a `TaskStarted` signal's `task` dict into the typed [`api::TaskView`].
fn dict_to_task_view(dict: &HashMap<String, OwnedValue>) -> Result<api::TaskView> {
    serde_json::from_value(dict_to_json_object(dict))
        .map_err(|e| anyhow::anyhow!("decoding TaskView dict: {e}"))
}

/// Decode a `TaskLogAppended` signal's `entry` dict into [`api::TaskLogEntry`].
fn dict_to_task_log_entry(dict: &HashMap<String, OwnedValue>) -> Result<api::TaskLogEntry> {
    serde_json::from_value(dict_to_json_object(dict))
        .map_err(|e| anyhow::anyhow!("decoding TaskLogEntry dict: {e}"))
}

/// Map the snake_case `TaskStatus` wire string the bridge emits on
/// `TaskCompleted` back to the typed enum, mirroring the bridge's
/// `task_status_str`. Returns `None` for an unknown value so the handler can
/// log-and-skip rather than guess a status.
fn task_status_from_str(s: &str) -> Option<api::TaskStatus> {
    match s {
        "pending" => Some(api::TaskStatus::Pending),
        "running" => Some(api::TaskStatus::Running),
        "completed" => Some(api::TaskStatus::Completed),
        "failed" => Some(api::TaskStatus::Failed),
        "cancelled" => Some(api::TaskStatus::Cancelled),
        _ => None,
    }
}

const DEFAULT_DBUS_SERVICE: &str = "org.desktopAssistant";
const DBUS_CONVERSATIONS_PATH: &str = "/org/desktopAssistant/Conversations";
const DBUS_SETTINGS_PATH: &str = "/org/desktopAssistant/Settings";
const DBUS_KNOWLEDGE_PATH: &str = "/org/desktopAssistant/Knowledge";
const DBUS_COMMANDS_PATH: &str = "/org/desktopAssistant/Commands";
const DBUS_BACKGROUND_TASKS_PATH: &str = "/org/desktopAssistant/BackgroundTasks";

/// Wire shape of a `list_conversations` row: `(id, title, message_count,
/// updated_at, archived)` — mirrors the D-Bus `a(ssusb)` reply.
type DbusConversationSummary = (String, String, u32, String, bool);

/// Wire shape of `get_conversation`: `(id, title, messages)` where each message
/// is `(role, content)` — mirrors the D-Bus `(ssa(ss))` reply.
type DbusConversationDetail = (String, String, Vec<(String, String)>);

/// The D-Bus `GetMessages` reply: `(total_raw_count, truncated, [(role, content)])`.
/// Aliased to keep the proxy signature within clippy's type-complexity bar.
type DbusMessagesPage = (u32, bool, Vec<(String, String)>);

#[zbus::proxy(interface = "org.desktopAssistant.Conversations")]
trait Conversations {
    async fn create_conversation(&self, title: &str) -> zbus::fdo::Result<String>;

    async fn list_conversations(
        &self,
        max_age_days: i32,
        include_archived: bool,
    ) -> zbus::fdo::Result<Vec<DbusConversationSummary>>;

    async fn archive_conversation(&self, id: &str) -> zbus::fdo::Result<()>;

    async fn unarchive_conversation(&self, id: &str) -> zbus::fdo::Result<()>;

    async fn get_conversation(&self, id: &str) -> zbus::fdo::Result<DbusConversationDetail>;

    async fn get_messages(
        &self,
        id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> zbus::fdo::Result<DbusMessagesPage>;

    async fn delete_conversation(&self, id: &str) -> zbus::fdo::Result<()>;

    async fn rename_conversation(&self, id: &str, title: &str) -> zbus::fdo::Result<()>;

    async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> zbus::fdo::Result<String>;

    #[zbus(signal)]
    fn response_chunk(
        &self,
        conversation_id: &str,
        request_id: &str,
        chunk: &str,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn response_complete(
        &self,
        conversation_id: &str,
        request_id: &str,
        full_response: &str,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn response_error(
        &self,
        conversation_id: &str,
        request_id: &str,
        error: &str,
    ) -> zbus::fdo::Result<()>;

    // --- #367: live cross-client sync (UserMessageAdded + the list signal) ----

    /// A user message was committed and a turn started in a conversation this
    /// client is viewing (via `SubscribeConversations`) — including turns this
    /// client did NOT initiate. The initiator dedupes on `request_id`. Mirrors
    /// the bridge's `Conversations.UserMessageAdded`.
    #[zbus(signal)]
    fn user_message_added(
        &self,
        conversation_id: &str,
        request_id: &str,
        content: &str,
    ) -> zbus::fdo::Result<()>;

    /// The user's conversation list changed (created/renamed/deleted/(un)archived)
    /// by any client or the voice daemon; carries only the affected id (the
    /// client re-fetches the list). Mirrors `Conversations.ConversationListChanged`.
    #[zbus(signal)]
    fn conversation_list_changed(&self, conversation_id: &str) -> zbus::fdo::Result<()>;

    // --- #320: client-side tool execution over D-Bus -------------------------

    /// A turn suspended on a client-side tool call, unicast to the registrant.
    /// The tool input rides as a JSON string (`arguments_json`); the handler
    /// parses it back to the `serde_json::Value` the `SignalEvent` carries
    /// (mirrors how the bridge serializes the wire event's `Value`). The client
    /// runs the tool and posts the outcome back via a `ClientToolResult` command
    /// with the same `task_id` + `tool_call_id`. Mirrors
    /// `Conversations.ClientToolCall`.
    #[zbus(signal)]
    fn client_tool_call(
        &self,
        task_id: &str,
        conversation_id: &str,
        tool_call_id: &str,
        tool_name: &str,
        arguments_json: &str,
    ) -> zbus::fdo::Result<()>;

    // --- #401: full UDS/WS signal parity for the shared reducer ---------------

    #[zbus(signal)]
    fn status(
        &self,
        conversation_id: &str,
        request_id: &str,
        message: &str,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn context_usage(
        &self,
        conversation_id: &str,
        request_id: &str,
        used_tokens: u64,
        budget_tokens: u64,
        compaction_active: bool,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn title_changed(&self, conversation_id: &str, title: &str) -> zbus::fdo::Result<()>;

    /// The structured `api::ConversationWarning` rides as a JSON string; the
    /// handler parses it back (mirrors how the bridge serializes it out).
    #[zbus(signal)]
    fn conversation_warning(
        &self,
        conversation_id: &str,
        warning_json: &str,
    ) -> zbus::fdo::Result<()>;

    #[zbus(signal)]
    fn scratchpad_changed(&self, conversation_id: &str) -> zbus::fdo::Result<()>;
}

/// Background-task lifecycle signals (#116/#367). The bridge emits these on the
/// `org.desktopAssistant.BackgroundTasks` interface (a different object path
/// from `Conversations`), so they need their own proxy. `TaskView` and
/// `TaskLogEntry` ride as `a{sv}` dictionaries keyed by their serde JSON field
/// names — the bridge builds them via `serde_json::to_value` so the same JSON
/// shape the UDS/WS clients decode is reproduced here (the handler converts the
/// dict back to a `serde_json::Value` and `from_value`s into the typed view).
#[zbus::proxy(interface = "org.desktopAssistant.BackgroundTasks")]
trait BackgroundTasks {
    /// A task transitioned to `Pending`/`Running`. `task` is the `TaskView`
    /// encoded as `a{sv}`.
    #[zbus(signal)]
    fn task_started(&self, id: &str, task: HashMap<String, OwnedValue>) -> zbus::fdo::Result<()>;

    /// Lightweight progress hint between log entries. `hint` is `""` when the
    /// upstream event carried `None`.
    #[zbus(signal)]
    fn task_progress(&self, id: &str, hint: &str) -> zbus::fdo::Result<()>;

    /// A new log entry, encoded as `a{sv}`.
    #[zbus(signal)]
    fn task_log_appended(
        &self,
        id: &str,
        entry: HashMap<String, OwnedValue>,
    ) -> zbus::fdo::Result<()>;

    /// Terminal lifecycle event. `status` is the snake_case `TaskStatus`;
    /// `last_error` is `""` when none.
    #[zbus(signal)]
    fn task_completed(&self, id: &str, status: &str, last_error: &str) -> zbus::fdo::Result<()>;
}

#[zbus::proxy(interface = "org.desktopAssistant.Settings")]
trait Settings {
    async fn generate_ws_jwt(&self, subject: &str) -> zbus::fdo::Result<String>;
}

#[zbus::proxy(interface = "org.desktopAssistant.Knowledge")]
trait Knowledge {
    async fn list_entries(
        &self,
        limit: u32,
        offset: u32,
        tag_filter_json: &str,
    ) -> zbus::fdo::Result<String>;

    async fn get_entry(&self, id: &str) -> zbus::fdo::Result<String>;

    async fn search_entries(
        &self,
        query: &str,
        tag_filter_json: &str,
        limit: u32,
    ) -> zbus::fdo::Result<String>;

    async fn create_entry(
        &self,
        content: &str,
        tags_json: &str,
        metadata_json: &str,
    ) -> zbus::fdo::Result<String>;

    async fn update_entry(
        &self,
        id: &str,
        content: &str,
        tags_json: &str,
        metadata_json: &str,
    ) -> zbus::fdo::Result<String>;

    async fn delete_entry(&self, id: &str) -> zbus::fdo::Result<()>;
}

/// Generic command channel (#213). One method maps every request/response
/// `api::Command` to its `api::CommandResult`, both passed as JSON strings —
/// the D-Bus counterpart of the socket transports' `WsRequest`/`WsFrame`
/// round-trip. This is what makes [`AssistantCommands`] reachable over D-Bus.
#[zbus::proxy(interface = "org.desktopAssistant.Commands")]
trait Commands {
    async fn send_command(&self, command_json: &str) -> zbus::fdo::Result<String>;
}

fn resolve_dbus_service_name() -> String {
    std::env::var("DESKTOP_ASSISTANT_DBUS_SERVICE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_DBUS_SERVICE.to_string())
}

pub async fn generate_ws_jwt(subject: &str) -> Result<String> {
    let connection = Connection::session().await?;
    let service_name = resolve_dbus_service_name();
    let proxy = SettingsProxy::builder(&connection)
        .destination(service_name)?
        .path(DBUS_SETTINGS_PATH)?
        .build()
        .await?;

    Ok(proxy.generate_ws_jwt(subject).await?)
}

pub struct DbusClient {
    proxy: ConversationsProxy<'static>,
    knowledge: KnowledgeProxy<'static>,
    commands: CommandsProxy<'static>,
    background_tasks: BackgroundTasksProxy<'static>,
}

impl DbusClient {
    pub async fn connect() -> Result<Self> {
        let connection = Connection::session().await?;
        let service_name = resolve_dbus_service_name();
        let proxy = ConversationsProxy::builder(&connection)
            .destination(service_name.clone())?
            .path(DBUS_CONVERSATIONS_PATH)?
            .build()
            .await?;
        let knowledge = KnowledgeProxy::builder(&connection)
            .destination(service_name.clone())?
            .path(DBUS_KNOWLEDGE_PATH)?
            .build()
            .await?;
        let commands = CommandsProxy::builder(&connection)
            .destination(service_name.clone())?
            .path(DBUS_COMMANDS_PATH)?
            .build()
            .await?;
        let background_tasks = BackgroundTasksProxy::builder(&connection)
            .destination(service_name)?
            .path(DBUS_BACKGROUND_TASKS_PATH)?
            .build()
            .await?;
        Ok(Self {
            proxy,
            knowledge,
            commands,
            background_tasks,
        })
    }

    pub async fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        let raw = self.proxy.list_conversations(0, false).await?;
        Ok(raw
            .into_iter()
            .map(
                |(id, title, message_count, _updated_at, archived)| ConversationSummary {
                    id,
                    title,
                    message_count,
                    archived,
                },
            )
            .collect())
    }

    pub async fn list_conversations_with_archived(&self) -> Result<Vec<ConversationSummary>> {
        let raw = self.proxy.list_conversations(0, true).await?;
        Ok(raw
            .into_iter()
            .map(
                |(id, title, message_count, _updated_at, archived)| ConversationSummary {
                    id,
                    title,
                    message_count,
                    archived,
                },
            )
            .collect())
    }

    pub async fn archive_conversation(&self, id: &str) -> Result<()> {
        self.proxy.archive_conversation(id).await?;
        Ok(())
    }

    pub async fn unarchive_conversation(&self, id: &str) -> Result<()> {
        self.proxy.unarchive_conversation(id).await?;
        Ok(())
    }

    pub async fn get_conversation(&self, id: &str) -> Result<ConversationDetail> {
        let (conv_id, title, messages) = self.proxy.get_conversation(id).await?;
        Ok(ConversationDetail {
            id: conv_id,
            title,
            messages: messages
                .into_iter()
                // The D-Bus conversation API predates message ids (#1) and only
                // returns (role, content); leave the id empty. The live-sync
                // cursor is a UDS/WS concern — the D-Bus client (voice) is a
                // turn producer, not a transcript renderer.
                .map(|(role, content)| ChatMessage {
                    id: String::new(),
                    role,
                    content,
                    kind: crate::MessageKind::Normal,
                    // D-Bus transcripts are daemon-sourced, not optimistic
                    // client bubbles, so they carry no idempotency stamp (#570).
                    idempotency_key: None,
                })
                .collect(),
            model_selection: None,
            conversation_personality: None,
        })
    }

    pub async fn get_messages(
        &self,
        conversation_id: &str,
        tail: i32,
        after_count: i32,
        include_roles: Vec<String>,
    ) -> Result<api::MessagesView> {
        let (total, truncated, messages) = self
            .proxy
            .get_messages(conversation_id, tail, after_count, include_roles)
            .await?;
        Ok(api::MessagesView {
            total_raw_count: total,
            truncated,
            // The D-Bus get_messages predates message ids (#1) and returns only
            // (role, content); leave the id empty — this transport is the
            // producer side, not a windowed-render consumer.
            messages: messages
                .into_iter()
                .map(|(role, content)| api::MessageView {
                    id: String::new(),
                    role,
                    content,
                })
                .collect(),
        })
    }

    pub async fn create_conversation(&self, title: &str) -> Result<String> {
        let id = self.proxy.create_conversation(title).await?;
        Ok(id)
    }

    pub async fn delete_conversation(&self, id: &str) -> Result<()> {
        self.proxy.delete_conversation(id).await?;
        Ok(())
    }

    pub async fn rename_conversation(&self, id: &str, title: &str) -> Result<()> {
        self.proxy.rename_conversation(id, title).await?;
        Ok(())
    }

    pub async fn send_prompt(&self, conversation_id: &str, prompt: &str) -> Result<String> {
        let request_id = self.proxy.send_prompt(conversation_id, prompt).await?;
        Ok(request_id)
    }

    // --- Knowledge management (issue #73) -------------------------------

    pub async fn list_knowledge_entries(
        &self,
        limit: u32,
        offset: u32,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<api::KnowledgeEntryView>> {
        let raw = self
            .knowledge
            .list_entries(limit, offset, &tag_filter_to_json(&tag_filter))
            .await?;
        decode_entries(&raw)
    }

    pub async fn get_knowledge_entry(&self, id: &str) -> Result<Option<api::KnowledgeEntryView>> {
        let raw = self.knowledge.get_entry(id).await?;
        let envelope: api::CommandResult = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("decoding get_entry response: {e}"))?;
        match envelope {
            api::CommandResult::KnowledgeEntry(entry) => Ok(entry),
            other => Err(anyhow::anyhow!(
                "unexpected dbus response for get_knowledge_entry: {other:?}"
            )),
        }
    }

    pub async fn search_knowledge_entries(
        &self,
        query: &str,
        tag_filter: Option<Vec<String>>,
        limit: u32,
    ) -> Result<Vec<api::KnowledgeEntryView>> {
        let raw = self
            .knowledge
            .search_entries(query, &tag_filter_to_json(&tag_filter), limit)
            .await?;
        decode_entries(&raw)
    }

    pub async fn create_knowledge_entry(
        &self,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView> {
        let tags_json =
            serde_json::to_string(&tags).map_err(|e| anyhow::anyhow!("encoding tags: {e}"))?;
        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| anyhow::anyhow!("encoding metadata: {e}"))?;
        let raw = self
            .knowledge
            .create_entry(content, &tags_json, &metadata_json)
            .await?;
        decode_entry_written(&raw)
    }

    pub async fn update_knowledge_entry(
        &self,
        id: &str,
        content: &str,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<api::KnowledgeEntryView> {
        let tags_json =
            serde_json::to_string(&tags).map_err(|e| anyhow::anyhow!("encoding tags: {e}"))?;
        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| anyhow::anyhow!("encoding metadata: {e}"))?;
        let raw = self
            .knowledge
            .update_entry(id, content, &tags_json, &metadata_json)
            .await?;
        decode_entry_written(&raw)
    }

    pub async fn delete_knowledge_entry(&self, id: &str) -> Result<()> {
        self.knowledge.delete_entry(id).await?;
        Ok(())
    }

    pub async fn subscribe_signals(&self) -> Result<mpsc::UnboundedReceiver<SignalEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();

        let mut chunk_stream = self.proxy.receive_response_chunk().await?;
        let tx_chunk = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = chunk_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_chunk.send(SignalEvent::Chunk {
                        conversation_id: args.conversation_id.to_string(),
                        request_id: args.request_id.to_string(),
                        chunk: args.chunk.to_string(),
                    });
                }
            }
        });

        let mut complete_stream = self.proxy.receive_response_complete().await?;
        let tx_complete = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = complete_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_complete.send(SignalEvent::Complete {
                        conversation_id: args.conversation_id.to_string(),
                        request_id: args.request_id.to_string(),
                        full_response: args.full_response.to_string(),
                    });
                }
            }
        });

        let mut error_stream = self.proxy.receive_response_error().await?;
        let tx_error = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = error_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_error.send(SignalEvent::Error {
                        conversation_id: args.conversation_id.to_string(),
                        request_id: args.request_id.to_string(),
                        error: args.error.to_string(),
                    });
                }
            }
        });

        // --- #401: per-conversation/turn parity events. Each is mapped back to
        // its `SignalEvent` so a `Connector` in `TransportMode::Dbus` reaches
        // full UDS/WS parity (a new KDE client consumes the shared reducer over
        // this transport). The #367 (UserMessageAdded / ConversationListChanged)
        // and #320 (ClientToolCall) and #116 (Task*) arms follow below. ---

        let mut status_stream = self.proxy.receive_status().await?;
        let tx_status = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = status_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_status.send(SignalEvent::Status {
                        conversation_id: args.conversation_id.to_string(),
                        request_id: args.request_id.to_string(),
                        message: args.message.to_string(),
                    });
                }
            }
        });

        let mut usage_stream = self.proxy.receive_context_usage().await?;
        let tx_usage = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = usage_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_usage.send(SignalEvent::ContextUsage {
                        conversation_id: args.conversation_id.to_string(),
                        request_id: args.request_id.to_string(),
                        used_tokens: args.used_tokens,
                        budget_tokens: args.budget_tokens,
                        compaction_active: args.compaction_active,
                    });
                }
            }
        });

        let mut title_stream = self.proxy.receive_title_changed().await?;
        let tx_title = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = title_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_title.send(SignalEvent::TitleChanged {
                        conversation_id: args.conversation_id.to_string(),
                        title: args.title.to_string(),
                    });
                }
            }
        });

        let mut warning_stream = self.proxy.receive_conversation_warning().await?;
        let tx_warning = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = warning_stream.next().await {
                if let Ok(args) = signal.args() {
                    // The warning crossed D-Bus as a JSON string; parse it back to
                    // the structured enum. Drop the event if it doesn't parse
                    // rather than fabricate a warning.
                    match serde_json::from_str(args.warning_json) {
                        Ok(warning) => {
                            let _ = tx_warning.send(SignalEvent::ConversationWarning {
                                conversation_id: args.conversation_id.to_string(),
                                warning,
                            });
                        }
                        Err(e) => {
                            tracing::warn!("dropping conversation warning: bad JSON ({e})");
                        }
                    }
                }
            }
        });

        let mut scratchpad_stream = self.proxy.receive_scratchpad_changed().await?;
        let tx_scratchpad = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = scratchpad_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_scratchpad.send(SignalEvent::ScratchpadChanged {
                        conversation_id: args.conversation_id.to_string(),
                    });
                }
            }
        });

        // --- #367: live cross-client sync ------------------------------------

        let mut user_message_stream = self.proxy.receive_user_message_added().await?;
        let tx_user_message = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = user_message_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_user_message.send(SignalEvent::UserMessageAdded {
                        conversation_id: args.conversation_id.to_string(),
                        request_id: args.request_id.to_string(),
                        content: args.content.to_string(),
                        // The D-Bus signal is `(s,s,s)` in Phase 1 and carries
                        // no idempotency key; echoing it over D-Bus is a Refs
                        // #570 follow-up.
                        idempotency_key: None,
                    });
                }
            }
        });

        let mut list_changed_stream = self.proxy.receive_conversation_list_changed().await?;
        let tx_list_changed = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = list_changed_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx_list_changed.send(SignalEvent::ConversationListChanged {
                        conversation_id: args.conversation_id.to_string(),
                    });
                }
            }
        });

        // --- #320: client-side tool execution over D-Bus ---------------------

        let mut tool_call_stream = self.proxy.receive_client_tool_call().await?;
        let tx_tool_call = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = tool_call_stream.next().await {
                if let Ok(args) = signal.args() {
                    // The tool input crossed D-Bus as a JSON string; parse it
                    // back to the `Value` the `SignalEvent` carries. Drop the
                    // event if it doesn't parse rather than run a tool with
                    // fabricated arguments.
                    match serde_json::from_str::<serde_json::Value>(args.arguments_json) {
                        Ok(arguments) => {
                            let _ = tx_tool_call.send(SignalEvent::ClientToolCall {
                                task_id: args.task_id.to_string(),
                                conversation_id: args.conversation_id.to_string(),
                                tool_call_id: args.tool_call_id.to_string(),
                                tool_name: args.tool_name.to_string(),
                                arguments,
                            });
                        }
                        Err(e) => {
                            tracing::warn!("dropping client tool call: bad arguments JSON ({e})");
                        }
                    }
                }
            }
        });

        // --- #116/#367: background-task lifecycle (BackgroundTasks interface) -
        // `TaskView`/`TaskLogEntry` ride as `a{sv}`; decode each back to the
        // typed view the reducer expects, dropping (with a warning) any dict
        // that doesn't deserialize rather than panicking on wire data.

        let mut task_started_stream = self.background_tasks.receive_task_started().await?;
        let tx_task_started = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = task_started_stream.next().await {
                if let Ok(args) = signal.args() {
                    match dict_to_task_view(&args.task) {
                        Ok(task) => {
                            let _ = tx_task_started.send(SignalEvent::TaskStarted { task });
                        }
                        Err(e) => tracing::warn!("dropping task_started: {e}"),
                    }
                }
            }
        });

        let mut task_progress_stream = self.background_tasks.receive_task_progress().await?;
        let tx_task_progress = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = task_progress_stream.next().await {
                if let Ok(args) = signal.args() {
                    // The bridge sends `""` for a `None` hint; map the empty
                    // string back to `None` so the reducer sees the same
                    // `Option` the UDS/WS clients deliver.
                    let hint = args.hint.to_string();
                    let _ = tx_task_progress.send(SignalEvent::TaskProgress {
                        id: args.id.to_string(),
                        progress_hint: if hint.is_empty() { None } else { Some(hint) },
                    });
                }
            }
        });

        let mut task_log_stream = self.background_tasks.receive_task_log_appended().await?;
        let tx_task_log = tx.clone();
        tokio::spawn(async move {
            while let Some(signal) = task_log_stream.next().await {
                if let Ok(args) = signal.args() {
                    match dict_to_task_log_entry(&args.entry) {
                        Ok(entry) => {
                            let _ = tx_task_log.send(SignalEvent::TaskLogAppended {
                                id: args.id.to_string(),
                                entry,
                            });
                        }
                        Err(e) => tracing::warn!("dropping task_log_appended: {e}"),
                    }
                }
            }
        });

        let mut task_completed_stream = self.background_tasks.receive_task_completed().await?;
        tokio::spawn(async move {
            while let Some(signal) = task_completed_stream.next().await {
                if let Ok(args) = signal.args() {
                    match task_status_from_str(args.status) {
                        Some(status) => {
                            // The bridge sends `""` for "no error"; map it back
                            // to `None`.
                            let last_error = args.last_error.to_string();
                            let _ = tx.send(SignalEvent::TaskCompleted {
                                id: args.id.to_string(),
                                status,
                                last_error: if last_error.is_empty() {
                                    None
                                } else {
                                    Some(last_error)
                                },
                            });
                        }
                        None => {
                            tracing::warn!(
                                "dropping task_completed: unknown status {:?}",
                                args.status
                            );
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}

/// Generic command channel over D-Bus (#213).
///
/// `DbusClient` now implements the shared [`AssistantCommands`] trait by
/// round-tripping each `api::Command` as a JSON string through the
/// `org.desktopAssistant.Commands.SendCommand` method (the D-Bus counterpart
/// of the WS/UDS `WsRequest`/`WsFrame` exchange). This is what lets
/// [`crate::transport::TransportClient::as_commands`] return `Some` for the
/// D-Bus transport, so the config Settings / model-override / purposes /
/// named-connection management surface is reachable over D-Bus too.
///
/// The inherent typed methods above (`list_conversations`, the streaming
/// `send_prompt`, the knowledge helpers) are retained: they win at the
/// `AssistantClient` call sites (inherent methods shadow trait methods on a
/// concrete `DbusClient`), while `&dyn AssistantCommands` callers reach this
/// trait impl. The streaming `send_prompt` in particular keeps using the typed
/// `Conversations.SendPrompt` path with its `ResponseChunk` signals — only the
/// trait's `send_command` routes through the generic channel.
#[async_trait]
impl AssistantCommands for DbusClient {
    async fn send_command(&self, command: api::Command) -> Result<api::CommandResult> {
        let command_json = serde_json::to_string(&command)
            .map_err(|e| anyhow::anyhow!("encoding command for D-Bus: {e}"))?;
        let raw = self.commands.send_command(&command_json).await?;
        serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("decoding D-Bus command result: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #401: `ConversationWarning` is the one non-trivial transform on the
    /// consume side — the structured enum crosses D-Bus as a JSON string and the
    /// `conversation_warning` handler parses it back. The bridge serializes it
    /// with the same `serde_json::to_string`, so this asserts the wire contract
    /// the two sides share: a warning round-trips through its JSON string to the
    /// identical value the reducer will receive.
    #[test]
    fn conversation_warning_round_trips_through_its_dbus_json_string() {
        let warning = api::ConversationWarning::DanglingModelSelection {
            previous_selection: api::ConversationModelSelectionView {
                connection_id: "old".into(),
                model_id: "m1".into(),
                effort: None,
            },
            fallback_to: api::ConversationModelSelectionView {
                connection_id: "new".into(),
                model_id: "m2".into(),
                effort: None,
            },
        };
        // What the bridge puts on the wire (warning_json).
        let on_wire = serde_json::to_string(&warning).expect("serialize");
        // What the consume side does with the signal's `warning_json` arg.
        let parsed: api::ConversationWarning = serde_json::from_str(&on_wire).expect("parse back");
        assert_eq!(parsed, warning);
    }

    /// A malformed `warning_json` must be dropped, never panic the signal pump:
    /// the handler's parse path returns `Err` and skips the event.
    #[test]
    fn malformed_conversation_warning_json_is_an_error_not_a_panic() {
        let parsed: Result<api::ConversationWarning, _> = serde_json::from_str("not json");
        assert!(parsed.is_err());
    }

    // --- #320: ClientToolCall arguments_json --------------------------------

    /// The tool input round-trips through the JSON string the bridge puts on the
    /// wire to the identical `serde_json::Value` the `ClientToolCall` signal
    /// carries — the wire contract the consume side relies on.
    #[test]
    fn client_tool_arguments_round_trip_through_their_dbus_json_string() {
        let arguments = serde_json::json!({ "city": "Boston", "units": "metric" });
        // What the bridge emits (arguments_json = value.to_string()).
        let on_wire = arguments.to_string();
        // What the consume side does with the signal's `arguments_json` arg.
        let parsed: serde_json::Value = serde_json::from_str(&on_wire).expect("parse back");
        assert_eq!(parsed, arguments);
    }

    /// Malformed `arguments_json` is dropped, not run with fabricated args.
    #[test]
    fn malformed_client_tool_arguments_json_is_an_error_not_a_panic() {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str("{not json");
        assert!(parsed.is_err());
    }

    // --- #116/#367: background-task `a{sv}` decode --------------------------
    //
    // These tests assert the inverse of the bridge's `task_view_to_dict` /
    // `log_entry_to_dict`. To prove the full round-trip without a dependency
    // cycle (`dbus-bridge` depends on this crate), the encoder is replicated
    // here exactly as the bridge does it: `serde_json::to_value(view)` then a
    // recursive walk into `a{sv}`. If the bridge's encoding ever changes, the
    // bridge's own tests catch it; this guards the decode half.

    /// Re-implements `dbus-bridge`'s `json_value_to_owned` so the test builds
    /// the SAME wire shape the bridge ships (u64/i64/f64 classification,
    /// `av` arrays, `a{sv}` nested objects, null-skipping).
    fn json_value_to_owned(value: JsonValue) -> Option<OwnedValue> {
        match value {
            JsonValue::Null => None,
            JsonValue::Bool(b) => OwnedValue::try_from(Value::from(b)).ok(),
            JsonValue::Number(n) => {
                if let Some(u) = n.as_u64() {
                    OwnedValue::try_from(Value::from(u)).ok()
                } else if let Some(i) = n.as_i64() {
                    OwnedValue::try_from(Value::from(i)).ok()
                } else if let Some(f) = n.as_f64() {
                    OwnedValue::try_from(Value::from(f)).ok()
                } else {
                    None
                }
            }
            JsonValue::String(s) => OwnedValue::try_from(Value::from(s)).ok(),
            JsonValue::Array(items) => {
                let mut packed: Vec<Value<'static>> = Vec::with_capacity(items.len());
                for item in items {
                    if let Some(ov) = json_value_to_owned(item) {
                        packed.push(Value::from(ov));
                    }
                }
                OwnedValue::try_from(Value::from(packed)).ok()
            }
            JsonValue::Object(map) => {
                let mut nested: HashMap<String, OwnedValue> = HashMap::with_capacity(map.len());
                for (k, v) in map {
                    if let Some(ov) = json_value_to_owned(v) {
                        nested.insert(k, ov);
                    }
                }
                OwnedValue::try_from(Value::from(nested)).ok()
            }
        }
    }

    /// Encode an already-serialized JSON object into the `a{sv}` dict the bridge
    /// sends. (Callers pass `serde_json::to_value(view).unwrap()` so this helper
    /// needs no `serde` trait bound — `serde` is not a direct dep of this crate.)
    fn to_wire_dict(value: JsonValue) -> HashMap<String, OwnedValue> {
        let JsonValue::Object(map) = value else {
            panic!("value did not serialize to a JSON object");
        };
        let mut out = HashMap::with_capacity(map.len());
        for (k, v) in map {
            if let Some(ov) = json_value_to_owned(v) {
                out.insert(k, ov);
            }
        }
        out
    }

    fn sample_task_view() -> api::TaskView {
        api::TaskView {
            id: api::TaskId("t-1".into()),
            // Nested enum → exercises the `a{sv}`-of-`a{sv}` recursion (kind
            // serializes externally-tagged: {"standalone": {...}}).
            kind: api::TaskKind::Standalone {
                name: "researcher".into(),
                conversation_id: "c-1".into(),
            },
            status: api::TaskStatus::Running,
            started_at: 1_700_000_000_000,
            ended_at: None,
            last_error: None,
            parent: None,
            // Non-empty children → exercises the `av` array path (TaskId list).
            children: vec![api::TaskId("t-2".into()), api::TaskId("t-3".into())],
            title: "Researcher: pricing data".into(),
            progress_hint: Some("step 1".into()),
        }
    }

    /// A `TaskView` survives the full encode→decode through `a{sv}`, including
    /// the nested `kind` enum and the `children` array.
    #[test]
    fn task_view_round_trips_through_its_dbus_dict() {
        let view = sample_task_view();
        let dict = to_wire_dict(serde_json::to_value(&view).unwrap());
        let decoded = dict_to_task_view(&dict).expect("decode TaskView");
        assert_eq!(decoded, view);
    }

    /// A terminal `TaskView` (Completed, with `ended_at`/`last_error` set) also
    /// round-trips — covers the `Option` fields the happy path leaves `None`.
    #[test]
    fn terminal_task_view_round_trips_through_its_dbus_dict() {
        let view = api::TaskView {
            status: api::TaskStatus::Failed,
            ended_at: Some(1_700_000_005_000),
            last_error: Some("LLM rate limit".into()),
            children: Vec::new(),
            progress_hint: None,
            ..sample_task_view()
        };
        let dict = to_wire_dict(serde_json::to_value(&view).unwrap());
        let decoded = dict_to_task_view(&dict).expect("decode terminal TaskView");
        assert_eq!(decoded, view);
    }

    /// A `TaskLogEntry` (with a structured `data` payload, exercising nested
    /// object + array decode) round-trips through `a{sv}`.
    #[test]
    fn task_log_entry_round_trips_through_its_dbus_dict() {
        let entry = api::TaskLogEntry {
            seq: 7,
            timestamp: 1_700_000_001_000,
            level: api::LogLevel::Warn,
            category: api::LogCategory::ToolResult,
            message: "fetched".into(),
            data: Some(serde_json::json!({ "items": [1, 2, 3], "ok": true })),
        };
        let dict = to_wire_dict(serde_json::to_value(&entry).unwrap());
        let decoded = dict_to_task_log_entry(&dict).expect("decode TaskLogEntry");
        assert_eq!(decoded, entry);
    }

    /// A `TaskLogEntry` with no `data` (the `skip_serializing_if` key is absent
    /// from the dict) decodes with `data: None`.
    #[test]
    fn task_log_entry_without_data_round_trips() {
        let entry = api::TaskLogEntry {
            seq: 1,
            timestamp: 1,
            level: api::LogLevel::Info,
            category: api::LogCategory::Lifecycle,
            message: "started".into(),
            data: None,
        };
        let dict = to_wire_dict(serde_json::to_value(&entry).unwrap());
        let decoded = dict_to_task_log_entry(&dict).expect("decode");
        assert_eq!(decoded, entry);
        assert!(decoded.data.is_none());
    }

    /// A malformed dict (missing required fields) is an `Err`, never a panic —
    /// the signal handler logs and skips it.
    #[test]
    fn malformed_task_dict_is_an_error_not_a_panic() {
        let mut dict: HashMap<String, OwnedValue> = HashMap::new();
        dict.insert(
            "id".to_string(),
            OwnedValue::try_from(Value::from("only-an-id")).unwrap(),
        );
        let decoded = dict_to_task_view(&dict);
        assert!(decoded.is_err(), "incomplete TaskView dict must error");
    }

    /// An empty dict decodes to an error rather than panicking.
    #[test]
    fn empty_task_dict_is_an_error_not_a_panic() {
        let dict: HashMap<String, OwnedValue> = HashMap::new();
        let decoded = dict_to_task_view(&dict);
        assert!(decoded.is_err());
    }

    /// `value_to_json` peels a boxed variant (`Value::Value`) to its inner value
    /// — a regression guard for the `Value::Value` arm, which handles the `av`
    /// (array-of-variant) signature the bridge uses for heterogeneous arrays and
    /// is what `Array::inner()` yields per element for such arrays.
    #[test]
    fn value_to_json_peels_a_boxed_variant() {
        // A heterogeneous `av` array: a string and a number, each carried as a
        // boxed variant — the shape the bridge ships when a JSON array mixes
        // types. Each element must be unwrapped to its inner value.
        let elements: Vec<Value<'static>> = vec![
            Value::Value(Box::new(Value::from("a"))),
            Value::Value(Box::new(Value::from(7u64))),
        ];
        let json = value_to_json(&Value::from(elements)).expect("array projects to JSON");
        assert_eq!(json, serde_json::json!(["a", 7]));

        // And a bare boxed variant on its own peels to the inner scalar.
        let boxed = Value::Value(Box::new(Value::from("inner")));
        assert_eq!(value_to_json(&boxed), Some(serde_json::json!("inner")));
    }

    /// A non-finite f64 has no JSON number; `value_to_json` drops it rather than
    /// producing an invalid `serde_json::Number`.
    #[test]
    fn value_to_json_drops_non_finite_floats() {
        let nan = Value::F64(f64::NAN);
        assert!(value_to_json(&nan).is_none());
        let inf = Value::F64(f64::INFINITY);
        assert!(value_to_json(&inf).is_none());
    }

    // --- TaskStatus wire string --------------------------------------------

    /// Every status string the bridge can emit (`task_status_str`) maps back to
    /// the matching enum; an unknown value is `None` (log-and-skip).
    #[test]
    fn task_status_from_str_covers_every_variant_and_rejects_unknown() {
        assert_eq!(
            task_status_from_str("pending"),
            Some(api::TaskStatus::Pending)
        );
        assert_eq!(
            task_status_from_str("running"),
            Some(api::TaskStatus::Running)
        );
        assert_eq!(
            task_status_from_str("completed"),
            Some(api::TaskStatus::Completed)
        );
        assert_eq!(
            task_status_from_str("failed"),
            Some(api::TaskStatus::Failed)
        );
        assert_eq!(
            task_status_from_str("cancelled"),
            Some(api::TaskStatus::Cancelled)
        );
        assert_eq!(task_status_from_str("bogus"), None);
        assert_eq!(task_status_from_str(""), None);
    }
}
