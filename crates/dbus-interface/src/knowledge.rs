//! D-Bus adapter for the knowledge management API (#73).
//!
//! Each method translates D-Bus arguments into an [`api::Command`] and
//! dispatches through the shared [`AssistantApiHandler`] (the same path
//! the WebSocket adapter takes). Complex payloads — entry views,
//! metadata blobs — are passed as JSON strings to keep the zbus
//! marshaling minimal; clients re-parse with `serde_json` (or
//! `QJsonDocument` on the Qt side).

use std::sync::Arc;

use desktop_assistant_api_model::{self as api};
use desktop_assistant_application::AssistantApiHandler;
use zbus::{fdo, interface};

fn to_fdo_error<E: std::fmt::Display>(error: E) -> fdo::Error {
    fdo::Error::Failed(error.to_string())
}

pub struct DbusKnowledgeAdapter {
    handler: Arc<dyn AssistantApiHandler>,
}

impl DbusKnowledgeAdapter {
    pub fn new(handler: Arc<dyn AssistantApiHandler>) -> Self {
        Self { handler }
    }

    async fn dispatch(&self, cmd: api::Command) -> fdo::Result<api::CommandResult> {
        self.handler
            .handle_command(cmd)
            .await
            .map_err(|e| fdo::Error::Failed(format!("{e:?}")))
    }
}

#[interface(name = "org.desktopAssistant.Knowledge")]
impl DbusKnowledgeAdapter {
    /// Paginated entry list. Returns JSON `{"knowledge_entries": [...]}`
    /// matching the WS surface. Pass `tag_filter_json="null"` (or `""`)
    /// to disable the filter.
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
                serde_json::to_string(&result).map_err(to_fdo_error)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected ListKnowledgeEntries result: {other:?}"
            ))),
        }
    }

    /// Fetch a single entry by id. Returns JSON `{"knowledge_entry": {...}}`
    /// (or `{"knowledge_entry": null}` for unknown id) so callers always
    /// see the same envelope.
    async fn get_entry(&self, id: &str) -> fdo::Result<String> {
        let result = self
            .dispatch(api::Command::GetKnowledgeEntry { id: id.to_string() })
            .await?;
        match &result {
            api::CommandResult::KnowledgeEntry(_) => {
                serde_json::to_string(&result).map_err(to_fdo_error)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected GetKnowledgeEntry result: {other:?}"
            ))),
        }
    }

    /// Full-text search. Same envelope as `list_entries` for callers
    /// that swap between browse and search modes.
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
                serde_json::to_string(&result).map_err(to_fdo_error)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected SearchKnowledgeEntries result: {other:?}"
            ))),
        }
    }

    /// Create a new entry; daemon assigns the id and embeds. Returns
    /// JSON `{"knowledge_entry_written": {...}}` carrying the persisted
    /// view (with the assigned id + timestamps).
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
                serde_json::to_string(&result).map_err(to_fdo_error)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected CreateKnowledgeEntry result: {other:?}"
            ))),
        }
    }

    /// Replace an existing entry's content/tags/metadata. Re-embeds.
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
                serde_json::to_string(&result).map_err(to_fdo_error)
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

/// Parse the wire-format tag filter. Empty string and `"null"` both map
/// to `None`. Anything else must be a JSON array of strings.
fn parse_tag_filter(raw: &str) -> fdo::Result<Option<Vec<String>>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "null" {
        return Ok(None);
    }
    serde_json::from_str(trimmed).map_err(to_fdo_error)
}

/// Parse a JSON array of tag strings; empty string maps to no tags.
fn parse_tags(raw: &str) -> fdo::Result<Vec<String>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(trimmed).map_err(to_fdo_error)
}

/// Parse a JSON metadata blob; empty string maps to `null`.
fn parse_metadata(raw: &str) -> fdo::Result<serde_json::Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::Value::Null);
    }
    serde_json::from_str(trimmed).map_err(to_fdo_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tag_filter_handles_empty_and_null() {
        assert!(parse_tag_filter("").unwrap().is_none());
        assert!(parse_tag_filter("null").unwrap().is_none());
        assert!(parse_tag_filter("  null  ").unwrap().is_none());
    }

    #[test]
    fn parse_tag_filter_parses_array() {
        let tags = parse_tag_filter("[\"a\",\"b\"]").unwrap().unwrap();
        assert_eq!(tags, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn parse_metadata_empty_is_null() {
        assert_eq!(parse_metadata("").unwrap(), serde_json::Value::Null);
    }

    #[test]
    fn parse_metadata_parses_object() {
        assert_eq!(
            parse_metadata("{\"k\":1}").unwrap(),
            serde_json::json!({"k": 1})
        );
    }
}
