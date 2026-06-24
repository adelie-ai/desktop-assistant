//! D-Bus adapter for `/org/desktopAssistant/Knowledge` (issue #73).
//!
//! Mirrors `crates/dbus-interface/src/knowledge.rs`. Same JSON-string
//! envelopes on the wire so KCM / TUI keep parsing identically.

use std::sync::Arc;

use desktop_assistant_api_model as api;
use zbus::object_server::SignalEmitter;
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

    /// Trigger an on-demand knowledge-maintenance pass (the "dream cycle"
    /// controls). `op` is one of `extraction` / `consolidation` /
    /// `recalculate_embeddings`. Returns the JSON envelope
    /// `{"maintenance_task_started":{"task_id":"…"}}`; progress/completion arrive
    /// as `BackgroundTasks.Task*` signals and the pass emits `EntriesChanged`.
    async fn start_maintenance(&self, op: &str) -> fdo::Result<String> {
        let op = parse_maintenance_op(op)?;
        let result = self
            .dispatch(api::Command::StartKnowledgeMaintenance { op })
            .await?;
        match &result {
            api::CommandResult::MaintenanceTaskStarted { .. } => {
                serde_json::to_string(&result).map_err(to_fdo)
            }
            other => Err(fdo::Error::Failed(format!(
                "unexpected StartKnowledgeMaintenance result: {other:?}"
            ))),
        }
    }

    /// Emitted when the knowledge base changes — a manual edit on another client
    /// or a maintenance pass rewrote entries. Carries no args; clients refetch.
    #[zbus(signal)]
    async fn entries_changed(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;
}

/// Map the wire op string to [`api::MaintenanceOp`]. Rejects unknown ops without
/// dispatching, mirroring the tag/metadata parsers above.
pub(crate) fn parse_maintenance_op(raw: &str) -> fdo::Result<api::MaintenanceOp> {
    match raw.trim() {
        "extraction" => Ok(api::MaintenanceOp::Extraction),
        "consolidation" => Ok(api::MaintenanceOp::Consolidation),
        "recalculate_embeddings" => Ok(api::MaintenanceOp::RecalculateEmbeddings),
        other => Err(fdo::Error::Failed(format!(
            "unknown maintenance op: {other:?} (expected extraction / consolidation / recalculate_embeddings)"
        ))),
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

#[cfg(test)]
mod tests {
    //! Contract tests for the Knowledge adapter. The spec is the canonical
    //! `api::Command` each method builds + the result it maps (the same contract
    //! every transport honors), plus the JSON-string normalization rules for
    //! tags / tag-filter / metadata. Unhappy paths (malformed JSON not
    //! dispatched, result-variant mismatch, daemon error) are named tests.
    use super::*;
    use std::sync::Mutex;

    struct FakeTransport {
        seen: Mutex<Vec<api::Command>>,
        reply: Result<api::CommandResult, String>,
    }

    impl FakeTransport {
        fn replying(reply: api::CommandResult) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply: Ok(reply),
            })
        }
        fn failing(daemon_msg: &str) -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                reply: Err(daemon_msg.to_string()),
            })
        }
        fn count(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
        fn last(&self) -> api::Command {
            self.seen
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("a command was dispatched")
        }
    }

    #[async_trait::async_trait]
    impl BridgeTransport for FakeTransport {
        async fn request(
            &self,
            command: api::Command,
        ) -> Result<api::CommandResult, BridgeTransportError> {
            self.seen.lock().unwrap().push(command);
            self.reply.clone().map_err(BridgeTransportError::Daemon)
        }
    }

    fn adapter(t: Arc<FakeTransport>) -> DbusKnowledgeAdapter<FakeTransport> {
        DbusKnowledgeAdapter::new(t)
    }

    fn sample_entry() -> api::KnowledgeEntryView {
        api::KnowledgeEntryView {
            id: "k1".to_string(),
            content: "hello".to_string(),
            tags: vec!["t".to_string()],
            metadata: serde_json::Value::Null,
            created_at: "2026-06-14T00:00:00Z".to_string(),
            updated_at: "2026-06-14T00:00:00Z".to_string(),
        }
    }

    // --- normalization: parse_tag_filter -------------------------------------

    #[test]
    fn tag_filter_blank_or_null_is_no_filter() {
        for raw in ["", "   ", "null", "  null  "] {
            assert_eq!(parse_tag_filter(raw).unwrap(), None, "input {raw:?}");
        }
    }

    #[test]
    fn tag_filter_json_array_parses_to_tags() {
        assert_eq!(
            parse_tag_filter(r#"["rust","dbus"]"#).unwrap(),
            Some(vec!["rust".to_string(), "dbus".to_string()])
        );
    }

    #[test]
    fn tag_filter_malformed_is_an_error() {
        assert!(parse_tag_filter("{ not an array").is_err());
        assert!(
            parse_tag_filter("[1, 2]").is_err(),
            "must be strings, not numbers"
        );
    }

    // --- normalization: parse_tags -------------------------------------------

    #[test]
    fn tags_blank_is_empty_vec_not_null() {
        assert_eq!(parse_tags("").unwrap(), Vec::<String>::new());
        assert_eq!(parse_tags("   ").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn tags_json_array_parses() {
        assert_eq!(
            parse_tags(r#"["a","b"]"#).unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn tags_malformed_is_an_error() {
        assert!(parse_tags("not json").is_err());
    }

    // --- normalization: parse_metadata ---------------------------------------

    #[test]
    fn metadata_blank_is_json_null() {
        assert_eq!(parse_metadata("").unwrap(), serde_json::Value::Null);
        assert_eq!(parse_metadata("  ").unwrap(), serde_json::Value::Null);
    }

    #[test]
    fn metadata_json_object_parses() {
        assert_eq!(
            parse_metadata(r#"{"source":"manual","score":3}"#).unwrap(),
            serde_json::json!({"source": "manual", "score": 3})
        );
    }

    #[test]
    fn metadata_malformed_is_an_error() {
        assert!(parse_metadata("{ broken").is_err());
    }

    // --- list_entries ---------------------------------------------------------

    #[tokio::test]
    async fn list_entries_builds_command_with_limit_offset_and_tag_filter() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntries(Vec::new()));
        let json = adapter(Arc::clone(&t))
            .list_entries(25, 50, r#"["howto"]"#)
            .await
            .unwrap();
        match t.last() {
            api::Command::ListKnowledgeEntries {
                limit,
                offset,
                tag_filter,
            } => {
                assert_eq!(limit, 25);
                assert_eq!(offset, 50);
                assert_eq!(tag_filter, Some(vec!["howto".to_string()]));
            }
            other => panic!("expected ListKnowledgeEntries, got {other:?}"),
        }
        let back: api::CommandResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, api::CommandResult::KnowledgeEntries(_)));
    }

    #[tokio::test]
    async fn list_entries_blank_tag_filter_is_none() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntries(Vec::new()));
        adapter(Arc::clone(&t))
            .list_entries(10, 0, "")
            .await
            .unwrap();
        assert!(matches!(
            t.last(),
            api::Command::ListKnowledgeEntries {
                tag_filter: None,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn list_entries_rejects_malformed_tag_filter_without_dispatching() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntries(Vec::new()));
        let err = adapter(Arc::clone(&t))
            .list_entries(10, 0, "{ bad")
            .await
            .expect_err("malformed tag filter must be rejected");
        assert!(matches!(err, fdo::Error::Failed(_)));
        assert_eq!(t.count(), 0);
    }

    #[tokio::test]
    async fn list_entries_errors_on_unexpected_result_variant() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        let err = adapter(Arc::clone(&t))
            .list_entries(10, 0, "")
            .await
            .expect_err("a non-KnowledgeEntries result must error");
        assert!(matches!(err, fdo::Error::Failed(_)));
    }

    // --- get_entry ------------------------------------------------------------

    #[tokio::test]
    async fn get_entry_builds_command_and_maps_present_entry() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntry(Some(sample_entry())));
        let json = adapter(Arc::clone(&t)).get_entry("k1").await.unwrap();
        assert!(matches!(t.last(), api::Command::GetKnowledgeEntry { id } if id == "k1"));
        let back: api::CommandResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, api::CommandResult::KnowledgeEntry(Some(_))));
    }

    #[tokio::test]
    async fn get_entry_maps_missing_entry_as_null_envelope() {
        // The contract carries "not found" as KnowledgeEntry(None); the adapter
        // must serialize it, not error.
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntry(None));
        let json = adapter(Arc::clone(&t)).get_entry("nope").await.unwrap();
        let back: api::CommandResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, api::CommandResult::KnowledgeEntry(None)));
    }

    // --- search_entries -------------------------------------------------------

    #[tokio::test]
    async fn search_entries_builds_command_with_query_filter_and_limit() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntries(Vec::new()));
        adapter(Arc::clone(&t))
            .search_entries("how to mint", r#"["jwt"]"#, 7)
            .await
            .unwrap();
        match t.last() {
            api::Command::SearchKnowledgeEntries {
                query,
                tag_filter,
                limit,
            } => {
                assert_eq!(query, "how to mint");
                assert_eq!(tag_filter, Some(vec!["jwt".to_string()]));
                assert_eq!(limit, 7);
            }
            other => panic!("expected SearchKnowledgeEntries, got {other:?}"),
        }
    }

    // --- create_entry ---------------------------------------------------------

    #[tokio::test]
    async fn create_entry_builds_command_with_parsed_tags_and_metadata() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntryWritten(sample_entry()));
        let json = adapter(Arc::clone(&t))
            .create_entry("a fact", r#"["x","y"]"#, r#"{"src":"doc"}"#)
            .await
            .unwrap();
        match t.last() {
            api::Command::CreateKnowledgeEntry {
                content,
                tags,
                metadata,
            } => {
                assert_eq!(content, "a fact");
                assert_eq!(tags, vec!["x".to_string(), "y".to_string()]);
                assert_eq!(metadata, serde_json::json!({"src": "doc"}));
            }
            other => panic!("expected CreateKnowledgeEntry, got {other:?}"),
        }
        let back: api::CommandResult = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, api::CommandResult::KnowledgeEntryWritten(_)));
    }

    #[tokio::test]
    async fn create_entry_defaults_blank_tags_and_metadata() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntryWritten(sample_entry()));
        adapter(Arc::clone(&t))
            .create_entry("c", "", "")
            .await
            .unwrap();
        match t.last() {
            api::Command::CreateKnowledgeEntry { tags, metadata, .. } => {
                assert!(tags.is_empty());
                assert_eq!(metadata, serde_json::Value::Null);
            }
            other => panic!("expected CreateKnowledgeEntry, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn create_entry_rejects_malformed_metadata_without_dispatching() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntryWritten(sample_entry()));
        let err = adapter(Arc::clone(&t))
            .create_entry("c", "[]", "{ bad")
            .await
            .expect_err("malformed metadata must be rejected");
        assert!(matches!(err, fdo::Error::Failed(_)));
        assert_eq!(t.count(), 0);
    }

    // --- update_entry ---------------------------------------------------------

    #[tokio::test]
    async fn update_entry_builds_command_with_id_content_tags_metadata() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntryWritten(sample_entry()));
        adapter(Arc::clone(&t))
            .update_entry("k9", "new body", r#"["z"]"#, "")
            .await
            .unwrap();
        match t.last() {
            api::Command::UpdateKnowledgeEntry {
                id,
                content,
                tags,
                metadata,
            } => {
                assert_eq!(id, "k9");
                assert_eq!(content, "new body");
                assert_eq!(tags, vec!["z".to_string()]);
                assert_eq!(metadata, serde_json::Value::Null);
            }
            other => panic!("expected UpdateKnowledgeEntry, got {other:?}"),
        }
    }

    // --- delete_entry ---------------------------------------------------------

    #[tokio::test]
    async fn delete_entry_builds_command_and_acks() {
        let t = FakeTransport::replying(api::CommandResult::Ack);
        adapter(Arc::clone(&t)).delete_entry("k1").await.unwrap();
        assert!(matches!(t.last(), api::Command::DeleteKnowledgeEntry { id } if id == "k1"));
    }

    #[tokio::test]
    async fn delete_entry_errors_on_unexpected_result_variant() {
        let t = FakeTransport::replying(api::CommandResult::KnowledgeEntries(Vec::new()));
        let err = adapter(Arc::clone(&t))
            .delete_entry("k1")
            .await
            .expect_err("a non-Ack result must error");
        assert!(matches!(err, fdo::Error::Failed(_)));
    }

    // --- daemon error ---------------------------------------------------------

    #[tokio::test]
    async fn daemon_error_is_propagated_verbatim() {
        let t = FakeTransport::failing("knowledge store offline");
        let err = adapter(Arc::clone(&t))
            .get_entry("k1")
            .await
            .expect_err("a daemon error must surface");
        assert!(format!("{err}").contains("knowledge store offline"));
    }
}
