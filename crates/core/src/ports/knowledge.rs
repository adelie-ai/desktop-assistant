use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::CoreError;
use crate::domain::KnowledgeEntry;

/// How many of the most recent in-scope entries the tag census aggregates.
///
/// Why a cap: the census is one extra aggregate on every knowledge-base search,
/// so the work it hands the aggregate must stay bounded however large the store
/// grows. It is a tail guardrail, not an optimisation of the common path - a
/// personal knowledge base never reaches it.
///
/// This bounds rows *aggregated*, not rows *read*. The cap stops the scan once
/// this many rows pass the caller's tag filters, so a filter that excludes most
/// recent entries reads further back - up to the whole of that user's index.
/// Sampling first and filtering afterwards would bound the read and is the
/// wrong trade: it would report [`ScopeSize::None`] for a scope that is merely
/// old.
pub const KNOWLEDGE_TAG_CENSUS_SAMPLE: usize = 1000;

/// How many ids one batch read of the knowledge base may name.
///
/// Why a cap: the entries travel to a language model inside a tool result, so
/// an unbounded batch would spend context the caller cannot see. It bounds the
/// number of entries only. A batch of large entries is held by the caller's own
/// response budget, which the batch cap says nothing about.
///
/// The figure is the one
/// [`crate::ports::scratchpad::MAX_KEYS_PER_CALL`] puts on the scratchpad's
/// batch read, because the two answer the same question for the same reason.
/// It is copied rather than derived: a change to the scratchpad's cap is a
/// scratchpad decision, and would not automatically be right here.
pub const KNOWLEDGE_GET_MAX_IDS: usize = 64;

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
    /// No entry passes the filters the caller supplied. This says nothing
    /// about the store as a whole: dropping the filters may well find plenty.
    None,
    /// The scope is no larger than this page, so a plain listing would show
    /// all of it.
    ///
    /// Why this is not "the caller has seen everything": the page holds what
    /// matched the query, and the scope is what passed the filters. A query
    /// that matched nothing still reports `Few` when the scope is small.
    Few,
    /// The scope holds more entries than this page could show.
    Many,
    /// The scope was not measured, so its size is not known.
    ///
    /// The census is one extra statement after the search has already returned
    /// its entries. When that statement fails the entries still travel, and
    /// this value says the measurement is missing.
    ///
    /// Why this is not [`ScopeSize::None`]: `None` is a positive claim that no
    /// entry passes the caller's filters. Reporting it for an unmeasured scope
    /// would tell the caller the store is empty when the store may hold
    /// everything the caller asked for. Treat `Unknown` as no information.
    Unknown,
}

impl ScopeSize {
    /// The value reported on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Few => "FEW",
            Self::Many => "MANY",
            Self::Unknown => "UNKNOWN",
        }
    }

    /// Classify a scope from a capped sample of it.
    ///
    /// `sampled` is how many rows the census aggregated, `cap` the cap it
    /// stopped at, and `page_limit` the caller's page size.
    ///
    /// Why a sample that reached the cap is always [`ScopeSize::Many`]: it says
    /// only "at least `cap`", so answering [`ScopeSize::Few`] there would claim
    /// the whole scope fit in a page that the caller may have sized above the
    /// cap.
    ///
    /// This never answers [`ScopeSize::Unknown`]. A caller that has a sample
    /// has a measurement; `Unknown` belongs to the caller whose census did not
    /// run at all.
    pub fn classify(sampled: usize, cap: usize, page_limit: usize) -> Self {
        if sampled == 0 {
            Self::None
        } else if sampled < cap && sampled <= page_limit {
            Self::Few
        } else {
            Self::Many
        }
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
    ///
    /// Empty when `scope_size` is [`ScopeSize::Unknown`], because the census
    /// that would have produced them did not run.
    pub available_tags: Vec<String>,
}

/// Outbound port for the unified knowledge base (replaces preferences + memory).
///
/// Every read here returns whole [`KnowledgeEntry`] values, including the
/// one-line [`KnowledgeEntry::summary`] a caller uses to list many entries
/// without spending each whole body.
pub trait KnowledgeBaseStore: Send + Sync {
    /// Write (upsert) a knowledge entry. If an entry with the same id exists,
    /// its content/tags/metadata are replaced and `updated_at` is bumped.
    ///
    /// `source` and `summary` are the exception: `None` in either preserves the
    /// stored value rather than clearing it, so a caller that knows nothing
    /// about them cannot wipe one.
    ///
    /// `summary` carries one further meaning, so that a caller which wrote a
    /// wrong summary can take it back: an **empty** summary clears the stored
    /// one, and the entry then reads back with `None`. A cleared summary is
    /// therefore indistinguishable from one never written, which is the point -
    /// both are an entry with no short form, and the pass that fills the
    /// field for entries that have none (#1099, not yet built) will treat
    /// them alike.
    ///
    /// An id held by an entry this caller cannot write - one that was retired,
    /// or one belonging to another user - is refused with
    /// `CoreError::InvalidInput`, and nothing is stored. A retired entry is
    /// hidden from every read, so a caller that looked first sees a free id;
    /// writing to it anyway would put live content in a row nothing can read
    /// and the retention reap frees on the tombstone's own clock. No write path
    /// revives a retired entry: store the content as a new entry instead.
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

/// A tag the knowledge-base write tool proposes, with the one-line description
/// the model gave for what the tag means.
///
/// Why the description travels with the name: a formal tag vocabulary decides
/// whether two tags are the same concept by comparing embeddings, and a short
/// facet tag such as `topic:weather` carries almost no signal on its own. The
/// description is what makes `topic:forecast` recognisable as the same concept.
///
/// The description is optional. A model that omits one is not an error - the
/// vocabulary falls back to the name alone rather than refusing the write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedTag {
    /// The tag as the model wrote it, before any normalisation.
    pub name: String,
    /// One line saying what the tag means, when the model supplied one.
    pub description: Option<String>,
}

/// Boxed async closure that resolves a proposed tag to the tag name the
/// knowledge base should actually store.
///
/// The answer is the proposed name when the vocabulary accepts it as a new
/// concept, and an existing tag's name when the vocabulary considers the two
/// the same concept. Either way the caller stores what comes back, so a near
/// duplicate never becomes a second tag that no read can match.
///
/// Why fallible: resolving a genuinely new tag needs an embedding, and the
/// embedding backend is optional. An `Err` means the vocabulary could not be
/// consulted this time, and the caller falls back to storing the tag as
/// written - never to failing the write.
pub type KnowledgeTagResolveFn = Arc<
    dyn Fn(ProposedTag) -> Pin<Box<dyn Future<Output = Result<String, CoreError>> + Send>>
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

/// Boxed async closure for fetching several entries by id in one read (#1104).
///
/// Why a batch and not repeated [`KnowledgeGetFn`] calls: the `[Pinned]` block
/// resolves every attached entry on every dispatch round, so a per-pin read
/// would multiply the round's storage traffic by the pin cap. Ids that name no
/// entry the caller owns are simply absent from the result, which is what marks
/// a reference as no longer resolving.
pub type KnowledgeGetManyFn = Arc<
    dyn Fn(
            Vec<String>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<KnowledgeEntry>, CoreError>> + Send>>
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
        assert_eq!(ScopeSize::Unknown.as_str(), "UNKNOWN");
    }

    fn _assert_knowledge_store<T: KnowledgeBaseStore>() {}
}
