use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::KnowledgeEntry;

/// Outbound port for the unified knowledge base (replaces preferences + memory).
pub trait KnowledgeBaseStore: Send + Sync {
    /// Write (upsert) a knowledge entry. If an entry with the same id exists, it is replaced.
    /// Embedding is a list of chunk vectors (one per chunk of the entry's content).
    fn write(
        &self,
        entry: KnowledgeEntry,
        embedding: Option<Vec<Vec<f32>>>,
        embedding_model: Option<String>,
    ) -> impl Future<Output = Result<KnowledgeEntry, CoreError>> + Send;

    /// Hybrid search combining vector similarity and full-text search via RRF.
    /// The caller generates the embedding; Postgres runs both searches.
    fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        tags: Option<Vec<String>>,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send;

    /// Delete a knowledge entry by id.
    fn delete(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<(), CoreError>> + Send;

    /// Get a single knowledge entry by id.
    fn get(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<Option<KnowledgeEntry>, CoreError>> + Send;
}

/// Boxed async closure for writing knowledge entries through non-generic boundaries.
/// Embedding is a list of chunk vectors (one per chunk of the entry's content).
pub type KnowledgeWriteFn = Arc<
    dyn Fn(
            KnowledgeEntry,
            Option<Vec<Vec<f32>>>,
        ) -> Pin<Box<dyn Future<Output = Result<KnowledgeEntry, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for searching the knowledge base.
pub type KnowledgeSearchFn = Arc<
    dyn Fn(
            String,
            Vec<f32>,
            Option<Vec<String>>,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send>>
        + Send
        + Sync,
>;

/// Boxed async closure for deleting knowledge entries.
pub type KnowledgeDeleteFn = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<(), CoreError>> + Send>> + Send + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;

    struct MockKnowledgeStore;

    impl KnowledgeBaseStore for MockKnowledgeStore {
        async fn write(
            &self,
            entry: KnowledgeEntry,
            _embedding: Option<Vec<Vec<f32>>>,
            _embedding_model: Option<String>,
        ) -> Result<KnowledgeEntry, CoreError> {
            Ok(entry)
        }

        async fn search(
            &self,
            _query: &str,
            _query_embedding: Vec<f32>,
            _tags: Option<Vec<String>>,
            _limit: usize,
        ) -> Result<Vec<KnowledgeEntry>, CoreError> {
            Ok(vec![])
        }

        async fn delete(&self, _id: &str) -> Result<(), CoreError> {
            Ok(())
        }

        async fn get(&self, _id: &str) -> Result<Option<KnowledgeEntry>, CoreError> {
            Ok(None)
        }
    }

    #[tokio::test]
    async fn mock_knowledge_store_write_returns_entry() {
        let store = MockKnowledgeStore;
        let entry = KnowledgeEntry::new("kb-1", "test", vec![]);
        let result = store.write(entry, None, None).await.unwrap();
        assert_eq!(result.id, "kb-1");
    }

    #[tokio::test]
    async fn mock_knowledge_store_search_returns_empty() {
        let store = MockKnowledgeStore;
        let results = store.search("test", vec![0.0], None, 10).await.unwrap();
        assert!(results.is_empty());
    }

    fn _assert_knowledge_store<T: KnowledgeBaseStore>() {}
}
