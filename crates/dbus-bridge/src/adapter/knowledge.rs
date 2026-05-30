//! D-Bus adapter for `/org/desktopAssistant/Knowledge` (issue #73).
//!
//! Mirrors `crates/dbus-interface/src/knowledge.rs`. Same JSON-string
//! envelopes on the wire so KCM / TUI keep parsing identically.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use zbus::{fdo, interface};

use crate::transport::{BridgeTransport, BridgeTransportError};

fn to_fdo<E: std::fmt::Display>(error: E) -> fdo::Error {
    fdo::Error::Failed(error.to_string())
}

fn map_transport_err(error: BridgeTransportError) -> fdo::Error {
    match error {
        BridgeTransportError::Daemon(msg) => fdo::Error::Failed(msg),
        other => fdo::Error::Failed(other.to_string()),
    }
}

pub struct DbusKnowledgeAdapter<T: BridgeTransport + 'static> {
    transport: Arc<T>,
}

impl<T: BridgeTransport + 'static> DbusKnowledgeAdapter<T> {
    pub fn new(transport: Arc<T>) -> Self {
        Self { transport }
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.transport.request(cmd).await.map_err(map_transport_err)
    }
}

#[interface(name = "org.desktopAssistant.Knowledge")]
impl<T: BridgeTransport + 'static> DbusKnowledgeAdapter<T> {
    async fn list_entries(
        &self,
        limit: u32,
        offset: u32,
        tag_filter_json: &str,
    ) -> fdo::Result<String> {
        let tag_filter = parse_tag_filter(tag_filter_json)?;
        let result = self
            .dispatch(api::Command::ListKnowledgeEntries {
                limit,
                offset,
                tag_filter,
            })
            .await?;
        match &result {
            api::CommandResult::KnowledgeEntries(_) => {
                serde_json::to_string(&result).map_err(to_fdo)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListKnowledgeEntries result: {other:?}"
            ))),
        }
    }

    async fn get_entry(&self, id: &str) -> fdo::Result<String> {
        let result = self
            .dispatch(api::Command::GetKnowledgeEntry { id: id.to_string() })
            .await?;
        match &result {
            api::CommandResult::KnowledgeEntry(_) => serde_json::to_string(&result).map_err(to_fdo),
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetKnowledgeEntry result: {other:?}"
            ))),
        }
    }

    async fn search_entries(
        &self,
        query: &str,
        tag_filter_json: &str,
        limit: u32,
    ) -> fdo::Result<String> {
        let tag_filter = parse_tag_filter(tag_filter_json)?;
        let result = self
            .dispatch(api::Command::SearchKnowledgeEntries {
                query: query.to_string(),
                tag_filter,
                limit,
            })
            .await?;
        match &result {
            api::CommandResult::KnowledgeEntries(_) => {
                serde_json::to_string(&result).map_err(to_fdo)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected SearchKnowledgeEntries result: {other:?}"
            ))),
        }
    }

    async fn create_entry(
        &self,
        content: &str,
        tags_json: &str,
        metadata_json: &str,
    ) -> fdo::Result<String> {
        let tags = parse_tags(tags_json)?;
        let metadata = parse_metadata(metadata_json)?;
        let result = self
            .dispatch(api::Command::CreateKnowledgeEntry {
                content: content.to_string(),
                tags,
                metadata,
            })
            .await?;
        match &result {
            api::CommandResult::KnowledgeEntryWritten(_) => {
                serde_json::to_string(&result).map_err(to_fdo)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected CreateKnowledgeEntry result: {other:?}"
            ))),
        }
    }

    async fn update_entry(
        &self,
        id: &str,
        content: &str,
        tags_json: &str,
        metadata_json: &str,
    ) -> fdo::Result<String> {
        let tags = parse_tags(tags_json)?;
        let metadata = parse_metadata(metadata_json)?;
        let result = self
            .dispatch(api::Command::UpdateKnowledgeEntry {
                id: id.to_string(),
                content: content.to_string(),
                tags,
                metadata,
            })
            .await?;
        match &result {
            api::CommandResult::KnowledgeEntryWritten(_) => {
                serde_json::to_string(&result).map_err(to_fdo)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected UpdateKnowledgeEntry result: {other:?}"
            ))),
        }
    }

    async fn delete_entry(&self, id: &str) -> fdo::Result<()> {
        let result = self
            .dispatch(api::Command::DeleteKnowledgeEntry { id: id.to_string() })
            .await?;
        match result {
            api::CommandResult::Ack => Ok(()),
            other => Err(fdo::Error::Failed(format!(
                "unexpected DeleteKnowledgeEntry result: {other:?}"
            ))),
        }
    }
}

/// Empty or `"null"` → no filter; anything else must be a JSON array
/// of strings. Mirrors the in-process adapter's parser.
pub(crate) fn parse_tag_filter(raw: &str) -> fdo::Result<Option<Vec<String>>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "null" {
        return Ok(None);
    }
    serde_json::from_str(trimmed).map_err(to_fdo)
}

pub(crate) fn parse_tags(raw: &str) -> fdo::Result<Vec<String>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(trimmed).map_err(to_fdo)
}

pub(crate) fn parse_metadata(raw: &str) -> fdo::Result<serde_json::Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::Value::Null);
    }
    serde_json::from_str(trimmed).map_err(to_fdo)
}
