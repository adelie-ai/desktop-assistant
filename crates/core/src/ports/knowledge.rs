use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::KnowledgeEntry;

/// How many of the most recent in-scope entries the tag census reads.
///
/// Why a cap: the census is one extra aggregate on every knowledge-base search,
/// so it must never be able to become a full table scan on a large
/// multi-tenant store. It is a tail guardrail, not an optimisation of the
/// common path - a personal knowledge base never reaches it.
pub const KNOWLEDGE_TAG_CENSUS_SAMPLE: usize = 1000;

/// How many tags a search reports in [`KnowledgeSearchPage::available_tags`].
///
/// Why a cap: the list travels to a language model inside a tool result, so an
/// unbounded tag vocabulary would spend context without adding signal - the
/// frequency ordering puts the useful tags first.
pub const AVAILABLE_TAGS_LIMIT: usize = 50;

/// How many entries the searched scope holds, relative to the page returned.
///
/// "Scope" means the entries that pass the caller's `tags` and `exclude_tags`
/// filters. It is never the set that matched the query: a hybrid search whose
/// vector arm is defined for every embedded row cannot count query matches, so
/// reporting one would state a falsehood.
///
/// Why a bucket rather than a number: the count behind it comes from a capped
/// sample (see [`KNOWLEDGE_TAG_CENSUS_SAMPLE`]), and a raw figure invites the
/// caller to trust a number that is only exact below the cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeSize {
    /// The scope holds no entries. No filter can find anything here.
    None,
    /// Every entry in the scope fit in this page, so narrowing gains nothing.
    Few,
    /// The scope holds more entries than this page could show.
    Many,
}

impl ScopeSize {
    /// The value reported on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Few => "FEW",
            Self::Many => "MANY",
        }
    }

    /// Classify a scope from a capped sample of it.
    ///
    /// `sampled` is how many rows the census actually read, `cap` the cap it
    /// stopped at, and `page_limit` the caller's page size.
    ///
    /// Why a sample that reached the cap is always [`ScopeSize::Many`]: it says
    /// only "at least `cap`", so answering [`ScopeSize::Few`] there would claim
    /// the whole scope fit in a page that the caller may have sized above the
    /// cap.
    pub fn classify(_sampled: usize, _cap: usize, _page_limit: usize) -> Self {
        Self::None
    }
}

/// One page of a knowledge-base search: the entries found, plus what the caller
/// needs to judge whether it saw everything.
///
/// Why the extra fields: the caller is a language model that would otherwise
/// guess tag filters, and a guessed tag that no entry carries returns nothing.
/// Both extra fields describe the scope, never the query match set - see
/// [`ScopeSize`].
#[derive(Debug, Clone)]
pub struct KnowledgeSearchPage {
    /// The entries this page returns, best match first.
    pub entries: Vec<KnowledgeEntry>,
    /// How many entries the scope holds, relative to this page.
    pub scope_size: ScopeSize,
    /// Tags carried by entries in the scope, most frequent first and ties
    /// broken by tag name. At most [`AVAILABLE_TAGS_LIMIT`] tags, counted over
    /// at most the [`KNOWLEDGE_TAG_CENSUS_SAMPLE`] most recent entries in
    /// scope. No counts travel with them: the counts come from a sample, so
    /// they would need a caveat that the ordering does not.
    pub available_tags: Vec<String>,
}

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
    ///
    /// `embedding_model` identifies the model that produced `query_embedding`
    /// and travels with it: only rows embedded by that model take part in the
    /// vector arm, because a comparison across models is a comparison across
    /// vector dimensions, which the database answers with an error rather than
    /// a miss. The full-text arm is deliberately unscoped, so a model change
    /// degrades semantic recall to lexical search instead of hiding content.
    ///
    /// The returned [`KnowledgeSearchPage`] also reports how large the scope
    /// selected by `tags`/`exclude_tags` is and which tags that scope carries,
    /// so a caller can tell an empty page caused by a tag no entry carries from
    /// an empty page caused by a store that holds nothing.
    fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        tags: Option<Vec<String>>,
        exclude_tags: Option<Vec<String>>,
        limit: usize,
    ) -> impl Future<Output = Result<KnowledgeSearchPage, CoreError>> + Send;

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
/// `(query, query_embedding, embedding_model, include_tags, exclude_tags, limit)`,
/// where `embedding_model` identifies the model that produced `query_embedding`
/// (see [`KnowledgeBaseStore::search`]).
pub type KnowledgeSearchFn = Arc<
    dyn Fn(
            String,
            Vec<f32>,
            String,
            Option<Vec<String>>,
            Option<Vec<String>>,
            usize,
        ) -> Pin<Box<dyn Future<Output = Result<KnowledgeSearchPage, CoreError>> + Send>>
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
            _embedding_model: &str,
            _tags: Option<Vec<String>>,
            _exclude_tags: Option<Vec<String>>,
            _limit: usize,
        ) -> Result<KnowledgeSearchPage, CoreError> {
            Ok(KnowledgeSearchPage {
                entries: vec![],
                scope_size: ScopeSize::None,
                available_tags: vec![],
            })
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
        let page = store
            .search("test", vec![0.0], "test-model", None, None, 10)
            .await
            .unwrap();
        assert!(page.entries.is_empty());
        assert_eq!(page.scope_size, ScopeSize::None);
        assert!(page.available_tags.is_empty());
    }

    #[test]
    fn scope_size_is_none_when_the_sample_is_empty() {
        // An empty scope is not "small"; it is a scope in which no tag filter
        // can ever find anything, and the caller must be able to tell the two
        // apart.
        assert_eq!(ScopeSize::classify(0, 1000, 10), ScopeSize::None);
    }

    #[test]
    fn scope_size_is_few_only_when_the_whole_scope_fits_the_page() {
        assert_eq!(ScopeSize::classify(10, 1000, 10), ScopeSize::Few);
        assert_eq!(ScopeSize::classify(11, 1000, 10), ScopeSize::Many);
    }

    #[test]
    fn scope_size_is_many_when_the_sample_reached_the_cap() {
        // A sample that stopped at the cap says only "at least `cap`". Calling
        // that `Few` because the caller asked for a larger page would claim the
        // whole scope fit, which the sample cannot show.
        assert_eq!(ScopeSize::classify(1000, 1000, 5000), ScopeSize::Many);
    }

    #[test]
    fn scope_size_wire_values_are_stable() {
        assert_eq!(ScopeSize::None.as_str(), "NONE");
        assert_eq!(ScopeSize::Few.as_str(), "FEW");
        assert_eq!(ScopeSize::Many.as_str(), "MANY");
    }

    fn _assert_knowledge_store<T: KnowledgeBaseStore>() {}
}
