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
use desktop_assistant_core::chunking::{CHUNK_MAX_CHARS, CHUNK_OVERLAP, chunk_text};
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::embedding::EmbedFn;
use desktop_assistant_core::ports::inbound::KnowledgeService;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;

/// Concrete [`KnowledgeService`] backed by a Postgres-backed
/// [`KnowledgeBaseStore`] (or an in-memory test double).
pub struct DaemonKnowledgeService<S>
where
    S: KnowledgeBaseStore + 'static,
{
    store: Arc<S>,
    embed_fn: Option<EmbedFn>,
    embedding_model: Option<String>,
    id_generator: Box<dyn Fn() -> String + Send + Sync>,
}

impl<S> DaemonKnowledgeService<S>
where
    S: KnowledgeBaseStore + 'static,
{
    pub fn new(
        store: Arc<S>,
        embed_fn: Option<EmbedFn>,
        embedding_model: Option<String>,
    ) -> Self {
        Self {
            store,
            embed_fn,
            embedding_model,
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

    /// Chunk + embed `content` using the configured embedding closure.
    /// Returns `Ok(None)` when no embedding closure is wired (KB writes
    /// still succeed, just without semantic search coverage — same
    /// behaviour as the builtin tool's `embed_chunks` path).
    async fn embed_chunks(
        &self,
        content: &str,
    ) -> Result<Option<Vec<Vec<f32>>>, CoreError> {
        let Some(embed_fn) = self.embed_fn.as_ref() else {
            return Ok(None);
        };
        let chunks = chunk_text(content, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        if chunks.is_empty() {
            return Ok(None);
        }
        match embed_fn(chunks).await {
            Ok(vecs) if !vecs.is_empty() => Ok(Some(vecs)),
            Ok(_) => Ok(None),
            Err(e) => {
                tracing::warn!("knowledge service: embedding failed: {e}");
                Ok(None)
            }
        }
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
        let embedding = self.embed_chunks(&entry.content).await?;
        let model = embedding.as_ref().and(self.embedding_model.clone());
        self.store.write(entry, embedding, model).await
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
        let embedding = self.embed_chunks(&entry.content).await?;
        let model = embedding.as_ref().and(self.embedding_model.clone());
        // The store's `write` upserts on id collision, which is exactly
        // the update semantics we want; created_at is preserved by the
        // ON CONFLICT clause.
        self.store.write(entry, embedding, model).await
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
        entries: Mutex<Vec<(KnowledgeEntry, Option<Vec<Vec<f32>>>, Option<String>)>>,
    }

    impl KnowledgeBaseStore for InMemoryStore {
        async fn write(
            &self,
            entry: KnowledgeEntry,
            embedding: Option<Vec<Vec<f32>>>,
            embedding_model: Option<String>,
        ) -> Result<KnowledgeEntry, CoreError> {
            let mut guard = self.entries.lock().unwrap();
            // Upsert by id — drop any prior entry with the same id.
            guard.retain(|(e, _, _)| e.id != entry.id);
            guard.push((entry.clone(), embedding, embedding_model));
            Ok(entry)
        }

        async fn search(
            &self,
            _query: &str,
            _query_embedding: Vec<f32>,
            _tags: Option<Vec<String>>,
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
                .filter(|(e, _, _)| e.content.contains(query))
                .map(|(e, _, _)| e.clone())
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
            Ok(guard
                .iter()
                .map(|(e, _, _)| e.clone())
                .skip(offset)
                .take(limit)
                .collect())
        }

        async fn delete(&self, id: &str) -> Result<(), CoreError> {
            let mut guard = self.entries.lock().unwrap();
            guard.retain(|(e, _, _)| e.id != id);
            Ok(())
        }

        async fn get(&self, id: &str) -> Result<Option<KnowledgeEntry>, CoreError> {
            let guard = self.entries.lock().unwrap();
            Ok(guard
                .iter()
                .find(|(e, _, _)| e.id == id)
                .map(|(e, _, _)| e.clone()))
        }
    }

    #[tokio::test]
    async fn create_assigns_id_and_persists() {
        let store = Arc::new(InMemoryStore::default());
        let service = DaemonKnowledgeService::new(Arc::clone(&store), None, None)
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
    async fn create_embeds_when_embed_fn_configured() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_closure = Arc::clone(&calls);
        let embed_fn: EmbedFn = Arc::new(move |chunks| {
            calls_for_closure.fetch_add(1, Ordering::SeqCst);
            let n = chunks.len();
            Box::pin(async move { Ok(vec![vec![0.1, 0.2]; n]) })
        });

        let store = Arc::new(InMemoryStore::default());
        let service = DaemonKnowledgeService::new(
            Arc::clone(&store),
            Some(embed_fn),
            Some("test-model".into()),
        );

        service
            .create_entry("payload".into(), vec![], serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "embed closure must be invoked exactly once"
        );

        let stored = store.entries.lock().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].2.as_deref(), Some("test-model"));
        assert!(stored[0].1.is_some(), "embedding must be persisted");
    }

    #[tokio::test]
    async fn update_replaces_in_place_and_re_embeds() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_closure = Arc::clone(&calls);
        let embed_fn: EmbedFn = Arc::new(move |chunks| {
            calls_for_closure.fetch_add(1, Ordering::SeqCst);
            let n = chunks.len();
            Box::pin(async move { Ok(vec![vec![1.0]; n]) })
        });

        let store = Arc::new(InMemoryStore::default());
        let service = DaemonKnowledgeService::new(
            Arc::clone(&store),
            Some(embed_fn),
            Some("m".into()),
        )
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

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "create + update each embed once"
        );
        let stored = store.entries.lock().unwrap();
        assert_eq!(stored.len(), 1, "update must not create a duplicate");
        assert_eq!(stored[0].0.content, "updated");
        assert_eq!(stored[0].0.tags, vec!["t"]);
    }

    #[tokio::test]
    async fn unconfigured_service_returns_storage_error() {
        let svc = UnconfiguredKnowledgeService;
        let err = svc.list_entries(10, 0, None).await.unwrap_err();
        assert!(matches!(err, CoreError::Storage(msg) if msg.contains("not configured")));
    }
}
