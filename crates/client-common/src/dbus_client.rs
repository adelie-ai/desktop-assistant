use anyhow::Result;
use desktop_assistant_api_model as api;
use futures::StreamExt;
use tokio::sync::mpsc;
use zbus::Connection;

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

#[zbus::proxy(interface = "org.desktopAssistant.Conversations")]
trait Conversations {
    async fn create_conversation(&self, title: &str) -> zbus::fdo::Result<String>;

    async fn list_conversations(
        &self,
        max_age_days: i32,
        include_archived: bool,
    ) -> zbus::fdo::Result<Vec<(String, String, u32, String, bool)>>;

    async fn archive_conversation(&self, id: &str) -> zbus::fdo::Result<()>;

    async fn unarchive_conversation(&self, id: &str) -> zbus::fdo::Result<()>;

    async fn get_conversation(
        &self,
        id: &str,
    ) -> zbus::fdo::Result<(String, String, Vec<(String, String)>)>;

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
            .destination(service_name)?
            .path(DBUS_KNOWLEDGE_PATH)?
            .build()
            .await?;
        Ok(Self { proxy, knowledge })
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
                .map(|(role, content)| ChatMessage { role, content })
                .collect(),
            model_selection: None,
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
                        request_id: args.request_id.to_string(),
                        full_response: args.full_response.to_string(),
                    });
                }
            }
        });

        let mut error_stream = self.proxy.receive_response_error().await?;
        tokio::spawn(async move {
            while let Some(signal) = error_stream.next().await {
                if let Ok(args) = signal.args() {
                    let _ = tx.send(SignalEvent::Error {
                        request_id: args.request_id.to_string(),
                        error: args.error.to_string(),
                    });
                }
            }
        });

        Ok(rx)
    }
}
