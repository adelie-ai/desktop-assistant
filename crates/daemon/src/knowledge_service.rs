//! Daemon-side [`KnowledgeService`] (#73).
//!
//! Adapts the [`KnowledgeBaseStore`] outbound port + the configured
//! embedding closure into a client-facing inbound port. Mirrors the
//! `builtin_knowledge_base_*` write path so client-authored entries
//! remain discoverable via the LLM tool.
//!
//! When no Postgres pool is configured at startup, every method
//! returns `CoreError::Storage("knowledge base not configured")` —
//! same shape as the builtin tool's "no store wired" error so clients
//! get a uniform message.

use std::sync::Arc;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::inbound::KnowledgeService;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;

/// Concrete [`KnowledgeService`] backed by a Postgres-backed
/// [`KnowledgeBaseStore`] (or an in-memory test double).
///
/// Writes never embed inline; the background embedding-backfill task owns
/// vector (re)generation, so this service no longer holds an embedding
/// closure.
pub struct DaemonKnowledgeService<S>
where
    S: KnowledgeBaseStore + 'static,
{
    store: Arc<S>,
    id_generator: Box<dyn Fn() -> String + Send + Sync>,
}

impl<S> DaemonKnowledgeService<S>
where
    S: KnowledgeBaseStore + 'static,
{
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            id_generator: Box::new(|| uuid::Uuid::now_v7().to_string()),
        }
    }

    #[cfg(test)]
    pub fn with_id_generator(
        mut self,
        gen_fn: impl Fn() -> String + Send + Sync + 'static,
    ) -> Self {
        self.id_generator = Box::new(gen_fn);
        self
    }
}

impl<S> KnowledgeService for DaemonKnowledgeService<S>
where
    S: KnowledgeBaseStore + 'static,
{
    async fn list_entries(
        &self,
        limit: usize,
        offset: usize,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        self.store.list(limit, offset, tag_filter).await
    }

    async fn get_entry(&self, id: String) -> Result<Option<KnowledgeEntry>, CoreError> {
        self.store.get(&id).await
    }

    async fn search_entries(
        &self,
        query: String,
        tag_filter: Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        self.store.search_text(&query, tag_filter, limit).await
    }

    async fn create_entry(
        &self,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        let id = (self.id_generator)();
        let mut entry = KnowledgeEntry::new(id, content, tags);
        entry.metadata = metadata;
        self.store.write(entry).await
    }

    async fn update_entry(
        &self,
        id: String,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        let mut entry = KnowledgeEntry::new(id, content, tags);
        entry.metadata = metadata;
        // The store's `write` upserts on id collision, which is exactly
        // the update semantics we want; created_at is preserved by the
        // ON CONFLICT clause. The edited content's embedding is regenerated
        // later by the background backfill task (the bumped `updated_at`
        // marks the existing vector stale).
        self.store.write(entry).await
    }

    async fn delete_entry(&self, id: String) -> Result<(), CoreError> {
        self.store.delete(&id).await
    }
}

/// Runtime dispatch wrapper. The `KnowledgeService` trait uses `impl
/// Future` returns and so isn't dyn-compatible; the concrete type held
/// by the API handler must therefore be fixed at compile time. Wrap the
/// configured / unconfigured branches in this enum so the daemon can
/// pick at runtime without infecting the handler's generics.
pub enum AnyKnowledgeService<S>
where
    S: KnowledgeBaseStore + 'static,
{
    Configured(DaemonKnowledgeService<S>),
    Unconfigured(UnconfiguredKnowledgeService),
}

impl<S> KnowledgeService for AnyKnowledgeService<S>
where
    S: KnowledgeBaseStore + 'static,
{
    async fn list_entries(
        &self,
        limit: usize,
        offset: usize,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        match self {
            Self::Configured(s) => s.list_entries(limit, offset, tag_filter).await,
            Self::Unconfigured(s) => s.list_entries(limit, offset, tag_filter).await,
        }
    }

    async fn get_entry(&self, id: String) -> Result<Option<KnowledgeEntry>, CoreError> {
        match self {
            Self::Configured(s) => s.get_entry(id).await,
            Self::Unconfigured(s) => s.get_entry(id).await,
        }
    }

    async fn search_entries(
        &self,
        query: String,
        tag_filter: Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        match self {
            Self::Configured(s) => s.search_entries(query, tag_filter, limit).await,
            Self::Unconfigured(s) => s.search_entries(query, tag_filter, limit).await,
        }
    }

    async fn create_entry(
        &self,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        match self {
            Self::Configured(s) => s.create_entry(content, tags, metadata).await,
            Self::Unconfigured(s) => s.create_entry(content, tags, metadata).await,
        }
    }

    async fn update_entry(
        &self,
        id: String,
        content: String,
        tags: Vec<String>,
        metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        match self {
            Self::Configured(s) => s.update_entry(id, content, tags, metadata).await,
            Self::Unconfigured(s) => s.update_entry(id, content, tags, metadata).await,
        }
    }

    async fn delete_entry(&self, id: String) -> Result<(), CoreError> {
        match self {
            Self::Configured(s) => s.delete_entry(id).await,
            Self::Unconfigured(s) => s.delete_entry(id).await,
        }
    }
}

/// No-op [`KnowledgeService`] used when no Postgres pool is configured.
/// Every method returns a clear `not configured` error so the API
/// surface is uniform regardless of backend availability.
pub struct UnconfiguredKnowledgeService;

const UNCONFIGURED_MSG: &str = "knowledge base not configured (Postgres required)";

impl KnowledgeService for UnconfiguredKnowledgeService {
    async fn list_entries(
        &self,
        _limit: usize,
        _offset: usize,
        _tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        Err(CoreError::Storage(UNCONFIGURED_MSG.to_string()))
    }

    async fn get_entry(&self, _id: String) -> Result<Option<KnowledgeEntry>, CoreError> {
        Err(CoreError::Storage(UNCONFIGURED_MSG.to_string()))
    }

    async fn search_entries(
        &self,
        _query: String,
        _tag_filter: Option<Vec<String>>,
        _limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        Err(CoreError::Storage(UNCONFIGURED_MSG.to_string()))
    }

    async fn create_entry(
        &self,
        _content: String,
        _tags: Vec<String>,
        _metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        Err(CoreError::Storage(UNCONFIGURED_MSG.to_string()))
    }

    async fn update_entry(
        &self,
        _id: String,
        _content: String,
        _tags: Vec<String>,
        _metadata: serde_json::Value,
    ) -> Result<KnowledgeEntry, CoreError> {
        Err(CoreError::Storage(UNCONFIGURED_MSG.to_string()))
    }

    async fn delete_entry(&self, _id: String) -> Result<(), CoreError> {
        Err(CoreError::Storage(UNCONFIGURED_MSG.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct InMemoryStore {
        entries: Mutex<Vec<KnowledgeEntry>>,
    }

    impl KnowledgeBaseStore for InMemoryStore {
        async fn write(&self, entry: KnowledgeEntry) -> Result<KnowledgeEntry, CoreError> {
            let mut guard = self.entries.lock().unwrap();
            // Upsert by id — drop any prior entry with the same id.
            guard.retain(|e| e.id != entry.id);
            guard.push(entry.clone());
            Ok(entry)
        }

        async fn search(
            &self,
            _query: &str,
            _query_embedding: Vec<f32>,
            _tags: Option<Vec<String>>,
            _exclude_tags: Option<Vec<String>>,
            _limit: usize,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            Ok(Vec::new())
        }

        async fn search_text(
            &self,
            query: &str,
            _tags: Option<Vec<String>>,
            limit: usize,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            // Naive contains-match; sufficient for unit tests.
            let guard = self.entries.lock().unwrap();
            let mut hits: Vec<KnowledgeEntry> = guard
                .iter()
                .filter(|e| e.content.contains(query))
                .cloned()
                .collect();
            hits.truncate(limit);
            Ok(hits)
        }

        async fn list(
            &self,
            limit: usize,
            offset: usize,
            _tag_filter: Option<Vec<String>>,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            let guard = self.entries.lock().unwrap();
            Ok(guard.iter().cloned().skip(offset).take(limit).collect())
        }

        async fn delete(&self, id: &str) -> Result<(), CoreError> {
            let mut guard = self.entries.lock().unwrap();
            guard.retain(|e| e.id != id);
            Ok(())
        }

        async fn get(&self, id: &str) -> Result<Option<KnowledgeEntry>, CoreError> {
            let guard = self.entries.lock().unwrap();
            Ok(guard.iter().find(|e| e.id == id).cloned())
        }
    }

    #[tokio::test]
    async fn create_assigns_id_and_persists() {
        let store = Arc::new(InMemoryStore::default());
        let service = DaemonKnowledgeService::new(Arc::clone(&store))
            .with_id_generator(|| "fixed-id".into());

        let entry = service
            .create_entry(
                "user prefers dark mode".into(),
                vec!["preference".into()],
                serde_json::json!({"scope": "global"}),
            )
            .await
            .unwrap();

        assert_eq!(entry.id, "fixed-id");
        assert_eq!(entry.content, "user prefers dark mode");
        assert_eq!(entry.tags, vec!["preference"]);

        let fetched = service.get_entry("fixed-id".into()).await.unwrap();
        assert!(fetched.is_some());
    }

    #[tokio::test]
    async fn create_does_not_embed_inline() {
        // Embedding is decoupled from the write path: the service holds no
        // embedding closure and create must persist without one.
        let store = Arc::new(InMemoryStore::default());
        let service = DaemonKnowledgeService::new(Arc::clone(&store))
            .with_id_generator(|| "kb-c".into());

        service
            .create_entry("payload".into(), vec![], serde_json::json!({}))
            .await
            .unwrap();

        let stored = store.entries.lock().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, "payload");
    }

    #[tokio::test]
    async fn update_replaces_in_place() {
        let store = Arc::new(InMemoryStore::default());
        let service = DaemonKnowledgeService::new(Arc::clone(&store))
            .with_id_generator(|| "kb-x".into());

        service
            .create_entry("orig".into(), vec![], serde_json::json!({}))
            .await
            .unwrap();
        service
            .update_entry(
                "kb-x".into(),
                "updated".into(),
                vec!["t".into()],
                serde_json::json!({}),
            )
            .await
            .unwrap();

        let stored = store.entries.lock().unwrap();
        assert_eq!(stored.len(), 1, "update must not create a duplicate");
        assert_eq!(stored[0].content, "updated");
        assert_eq!(stored[0].tags, vec!["t"]);
    }

    #[tokio::test]
    async fn unconfigured_service_returns_storage_error() {
        let svc = UnconfiguredKnowledgeService;
        let err = svc.list_entries(10, 0, None).await.unwrap_err();
        assert!(matches!(err, CoreError::Storage(msg) if msg.contains("not configured")));
    }
}
