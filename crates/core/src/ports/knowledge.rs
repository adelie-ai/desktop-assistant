use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::KnowledgeEntry;

/// Outbound port for the unified knowledge base (replaces preferences + memory).
pub trait KnowledgeBaseStore: Send + Sync {
    /// Write (upsert) a knowledge entry. If an entry with the same id exists,
    /// its content/tags/metadata are replaced and `updated_at` is bumped.
    ///
    /// Writes never touch the embedding columns: embedding generation is
    /// decoupled from content writes. New rows land with a NULL embedding and
    /// updates leave the existing (now stale) embedding in place; the
    /// background embedding-backfill task regenerates vectors for rows where
    /// `embedding IS NULL` or `embeddings_updated_at < updated_at`.
    fn write(
        &self,
        entry: KnowledgeEntry,
    ) -> impl Future<Output = Result<KnowledgeEntry, CoreError>> + Send;

    /// Hybrid search combining vector similarity and full-text search via RRF.
    /// The caller generates the embedding; Postgres runs both searches.
    /// `tags` requires at least one matching tag (overlap); `exclude_tags`
    /// removes any row carrying one of those tags.
    fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        tags: Option<Vec<String>>,
        exclude_tags: Option<Vec<String>>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send;

    /// Full-text search only (no vector similarity). Used by client-side
    /// browsers that need responsive search without embedding round-trips
    /// (#73). The LLM tool path keeps using [`Self::search`] for hybrid
    /// semantic+lexical match.
    fn search_text(
        &self,
        query: &str,
        tags: Option<Vec<String>>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send;

    /// Paginated listing of all entries, ordered by `updated_at DESC, id`.
    /// Used by the management API (#73).
    fn list(
        &self,
        limit: usize,
        offset: usize,
        tag_filter: Option<Vec<String>>,
    ) -> impl Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send;

    /// Delete a knowledge entry by id.
    fn delete(&self, id: &str) -> impl Future<Output = Result<(), CoreError>> + Send;

    /// Get a single knowledge entry by id.
    fn get(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<Option<KnowledgeEntry>, CoreError>> + Send;

    /// How many soft-deleted ("trashed") entries the current user has.
    ///
    /// Retired entries are hidden from every other read path, so this is the
    /// only way to see what is waiting to be reaped.
    fn trash_count(&self) -> impl Future<Output = Result<usize, CoreError>> + Send;

    /// Permanently delete every soft-deleted entry belonging to the current
    /// user, ignoring the retention window, and return how many rows were
    /// freed. An already-empty trash is a successful `0`, not an error.
    fn empty_trash(&self) -> impl Future<Output = Result<usize, CoreError>> + Send;
}

/// Boxed async closure for writing knowledge entries through non-generic
/// boundaries. Embeddings are owned by the background backfill task, not the
/// write path (see [`KnowledgeBaseStore::write`]).
pub type KnowledgeWriteFn = Arc<
    dyn Fn(
            KnowledgeEntry,
        ) -> Pin<Box<dyn Future<Output = Result<KnowledgeEntry, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for searching the knowledge base. Args:
/// `(query, query_embedding, include_tags, exclude_tags, limit)`.
pub type KnowledgeSearchFn = Arc<
    dyn Fn(
            String,
            Vec<f32>,
            Option<Vec<String>>,
            Option<Vec<String>>,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for deleting knowledge entries by id. Takes a batch of
/// ids and returns how many rows were deleted.
pub type KnowledgeDeleteFn = Arc<
    dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<usize, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for fetching a single entry by id (used by the write
/// tool to support partial updates that omit `content`).
pub type KnowledgeGetFn = Arc<
    dyn Fn(
            String,
        )
            -> Pin<Box<dyn Future<Output = Result<Option<KnowledgeEntry>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Direction for a paginated [`KnowledgeListQuery`]. Surfaced explicitly to the
/// LLM so it always knows which way it is paging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListOrder {
    OldestFirst,
    NewestFirst,
}

/// Parameters for a non-semantic, keyset-paginated listing of the knowledge
/// base. Pagination is a keyset cursor on `(created_at, id)`; `after` is the
/// opaque cursor returned by the previous page.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeListQuery {
    pub limit: usize,
    pub after: Option<String>,
    pub order: ListOrderOpt,
    /// Rows must carry at least one of these tags (overlap). `None` = no filter.
    pub tags: Option<Vec<String>>,
    /// Rows carrying any of these tags are excluded. `None` = no filter.
    pub exclude_tags: Option<Vec<String>>,
    /// Restrict to a single `source` value. `None` = no filter.
    pub source: Option<String>,
}

/// `ListOrder` with a `Default` of newest-first, for `KnowledgeListQuery`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListOrderOpt(pub ListOrder);

impl Default for ListOrderOpt {
    fn default() -> Self {
        ListOrderOpt(ListOrder::NewestFirst)
    }
}

/// One page of a [`KnowledgeListQuery`]: the entries plus an opaque cursor to
/// pass as `after` for the next page (`None` when the last page was reached).
#[derive(Debug, Clone)]
pub struct KnowledgeListPage {
    pub entries: Vec<KnowledgeEntry>,
    pub next_cursor: Option<String>,
}

/// Boxed async closure for the paginated list tool.
pub type KnowledgeListFn = Arc<
    dyn Fn(
            KnowledgeListQuery,
        ) -> Pin<Box<dyn Future<Output = Result<KnowledgeListPage, CoreError>> + Send>>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    struct MockKnowledgeStore;

    impl KnowledgeBaseStore for MockKnowledgeStore {
        async fn write(&self, entry: KnowledgeEntry) -> Result<KnowledgeEntry, CoreError> {
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
            Ok(vec![])
        }

        async fn search_text(
            &self,
            _query: &str,
            _tags: Option<Vec<String>>,
            _limit: usize,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }

        async fn list(
            &self,
            _limit: usize,
            _offset: usize,
            _tag_filter: Option<Vec<String>>,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }

        async fn delete(&self, _id: &str) -> Result<(), CoreError> {
            Ok(())
        }

        async fn get(&self, _id: &str) -> Result<Option<KnowledgeEntry>, CoreError> {
            Ok(None)
        }

        async fn trash_count(&self) -> Result<usize, CoreError> {
            Ok(0)
        }

        async fn empty_trash(&self) -> Result<usize, CoreError> {
            Ok(0)
        }
    }

    #[tokio::test]
    async fn mock_knowledge_store_write_returns_entry() {
        let store = MockKnowledgeStore;
        let entry = KnowledgeEntry::new("kb-1", "test", vec![]);
        let result = store.write(entry).await.unwrap();
        assert_eq!(result.id, "kb-1");
    }

    #[tokio::test]
    async fn mock_knowledge_store_search_returns_empty() {
        let store = MockKnowledgeStore;
        let results = store
            .search("test", vec![0.0], None, None, 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    fn _assert_knowledge_store<T: KnowledgeBaseStore>() {}
}
