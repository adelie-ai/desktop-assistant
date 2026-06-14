use anyhow::Result;
use async_trait::async_trait;
use desktop_assistant_api_model as api;
use futures::StreamExt;
use tokio::sync::mpsc;
use zbus::Connection;

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

const DEFAULT_DBUS_SERVICE: &str = "org.desktopAssistant";
const DBUS_CONVERSATIONS_PATH: &str = "/org/desktopAssistant/Conversations";
const DBUS_SETTINGS_PATH: &str = "/org/desktopAssistant/Settings";
const DBUS_KNOWLEDGE_PATH: &str = "/org/desktopAssistant/Knowledge";
const DBUS_COMMANDS_PATH: &str = "/org/desktopAssistant/Commands";

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
            .destination(service_name)?
            .path(DBUS_COMMANDS_PATH)?
            .build()
            .await?;
        Ok(Self {
            proxy,
            knowledge,
            commands,
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

        // --- #401: the five events the bridge previously dropped. Each is mapped
        // back to its `SignalEvent` so a `Connector` in `TransportMode::Dbus`
        // reaches full UDS/WS parity (a new KDE client consumes the shared
        // reducer over this transport). ---

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
        tokio::spawn(async move {
            while let Some(signal) = scratchpad_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx.send(SignalEvent::ScratchpadChanged {
                        conversation_id: args.conversation_id.to_string(),
                    });
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
}
