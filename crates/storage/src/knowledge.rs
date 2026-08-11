use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::domain::activation::LexicalMatch;
use desktop_assistant_core::domain::knowledge_use::KnowledgeUseRecord;
use desktop_assistant_core::domain::situation::{Situation, SituationRecord};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge::{
    AVAILABLE_TAGS_LIMIT, KNOWLEDGE_TAG_CENSUS_SAMPLE, KnowledgeBaseStore, KnowledgeListPage,
    KnowledgeListQuery, KnowledgeSearchPage, ListOrder, ScopeSize,
};
use desktop_assistant_core::ports::knowledge_use::current_situation_cue;
use desktop_assistant_core::ports::recall::RecallDispersion;
use pgvector::Vector;
use sqlx::PgPool;

use crate::knowledge_delete::{HardDeleteTarget, KnowledgeDeletePolicy, hard_delete_knowledge};
use crate::knowledge_search::SearchCandidate;

pub struct PgKnowledgeBaseStore {
    pool: PgPool,
    delete_policy: KnowledgeDeletePolicy,
    scan_ceiling: std::time::Duration,
}

impl PgKnowledgeBaseStore {
    /// A store that removes rows under `delete_policy`.
    ///
    /// The policy is a required argument rather than a default a caller may
    /// extend later: a construction site that forgot to attach one would
    /// silently run the permissive behaviour while the deployment believed its
    /// safety flag was on. The daemon builds the policy from `[backend_tasks]`;
    /// a caller that never deletes passes
    /// [`KnowledgeDeletePolicy::default`] and says so.
    pub fn new(pool: PgPool, delete_policy: KnowledgeDeletePolicy) -> Self {
        Self {
            pool,
            delete_policy,
            scan_ceiling: RECALL_SCAN_STATEMENT_TIMEOUT,
        }
    }

    /// The same store, whose full scans the database gives up on after
    /// `ceiling` instead of after [`RECALL_SCAN_STATEMENT_TIMEOUT`].
    ///
    /// **This exists so the bound can be proven, and proven on the path a
    /// deployment actually runs.** A test that reached past the public method
    /// to a bounded variant would go on passing if the public method stopped
    /// applying the bound - the bound would be stated and unheld, which is the
    /// defect this whole change is about. Overriding the ceiling instead leaves
    /// the delegation itself under test.
    #[must_use]
    pub fn with_scan_ceiling(mut self, ceiling: std::time::Duration) -> Self {
        self.scan_ceiling = ceiling;
        self
    }
}

impl KnowledgeBaseStore for PgKnowledgeBaseStore {
    async fn write(&self, entry: KnowledgeEntry) -> Result<KnowledgeEntry, CoreError> {
        let user_id = current_user_id();

        // Normalize tags on the way in so case/whitespace drift
        // (`Preference` / `preference ` / `preference`) can't fragment the
        // exact-match filters reads run (`tags && $2`). Facet tags keep their
        // `facet:value` colon — see `crate::tag_normalize`.
        let tags = crate::tag_normalize::normalize_tags(&entry.tags);

        // Embedding generation is decoupled from content writes: this query
        // never touches the `embedding`/`embedding_model`/`embeddings_updated_at`
        // columns. New rows insert with a NULL embedding; on update the existing
        // embedding is left in place (now stale relative to the bumped
        // `updated_at`). The background backfill task regenerates vectors for
        // rows where `embedding IS NULL` or `embeddings_updated_at < updated_at`.
        //
        // ON CONFLICT (id) inherently respects the schema's unique
        // constraint on `id`; since the KB id is a UUID we don't expect
        // collisions across users in practice. The upsert path still
        // refuses to leak rows: a writer can only land in the user's
        // own partition because the WHERE filter on the conflict update
        // matches only their row, and the insert path stamps user_id
        // from the current request.
        //
        // That WHERE also excludes a soft-deleted row. `deleted_at` is not in
        // the SET clause, so an update that reached a tombstone would write
        // real content into a row every read path hides and the retention reap
        // removes on its original clock — and report success. There is no
        // restore path through this call, deliberately: reviving a row that
        // consolidation retired would resurrect a duplicate it had merged away
        // and leave `superseded_by` pointing at the row that replaced it.
        //
        // Both exclusions land in the same place: the conflict update matches
        // no row, so nothing is written, and the caller is told rather than
        // being handed a success it did not get.
        //
        // `source` ($6) records provenance and `summary` ($7) the one-line
        // condensation. On update a NULL in either preserves the existing value
        // rather than clearing it, so a path that doesn't care about
        // provenance, or knows nothing about summaries, can't wipe one. There
        // is no way to clear the provenance through this call: absent is the
        // only meaning NULL carries for `source`.
        //
        // `summary` carries one more meaning, because a caller that wrote a
        // wrong summary must be able to take it back: an EMPTY summary clears
        // it. Cleared means NULL, not an empty string, on both halves of the
        // upsert - `NULLIF` on the insert and the `= ''` arm on the update. An
        // empty string would be a third state nothing wants: the render site
        // would show a blank line rather than falling back to the content, and
        // a pass that fills the field for entries with none (#1099) finds its
        // work with `WHERE summary IS NULL`, so it would never reach one.
        //
        // `summary_updated_at` follows the summary through all three states, so
        // the dream cycle can tell a current summary from one that describes
        // content the entry no longer holds. A supplied summary is stamped
        // `NOW()`, the same transaction time `updated_at` takes, so it reads as
        // current and the pass leaves it alone; a cleared one loses its stamp
        // with its text; and an absent one keeps the stamp it had, which is
        // what makes an update that says nothing about the summary show up as
        // drift on the next pass.
        let row: Option<KbRow> = sqlx::query_as(
            "INSERT INTO knowledge_base \
                (id, user_id, content, tags, metadata, source, summary, summary_updated_at) \
             VALUES ($1, $2, $3, $4, $5, $6, NULLIF($7, ''), \
                     CASE WHEN NULLIF($7, '') IS NULL THEN NULL ELSE NOW() END) \
             ON CONFLICT (id) DO UPDATE \
                SET content = EXCLUDED.content, \
                    tags = EXCLUDED.tags, \
                    metadata = EXCLUDED.metadata, \
                    source = COALESCE(EXCLUDED.source, knowledge_base.source), \
                    summary = CASE \
                        WHEN $7 IS NULL THEN knowledge_base.summary \
                        WHEN $7 = '' THEN NULL \
                        ELSE $7 END, \
                    summary_updated_at = CASE \
                        WHEN $7 IS NULL THEN knowledge_base.summary_updated_at \
                        WHEN $7 = '' THEN NULL \
                        ELSE NOW() END, \
                    updated_at = NOW() \
                WHERE knowledge_base.user_id = $2 \
                  AND knowledge_base.deleted_at IS NULL \
             RETURNING id, content, tags, metadata, created_at, updated_at, \
                       source, summary",
        )
        .bind(&entry.id)
        .bind(user_id.as_str())
        .bind(&entry.content)
        .bind(&tags)
        .bind(&entry.metadata)
        .bind(&entry.source)
        .bind(&entry.summary)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // No row back means the insert conflicted and the update matched
        // nothing: the id is held by an entry this caller cannot write —
        // retired, or another user's. A decline, not a failure (base rule 8.2),
        // and repeating the identical write cannot succeed.
        let row = row.ok_or_else(|| CoreError::InvalidInput {
            code: "knowledge_entry_not_writable",
            description: format!(
                "knowledge entry {} was not written: its id is held by a retired entry, or by \
                 one belonging to another user. Store this as a new entry instead, with no id.",
                entry.id
            ),
            message: "That entry cannot be updated, because the id belongs to an entry that was \
                      retired. Store this as a new entry instead, without an id."
                .to_string(),
        })?;

        Ok(row.into_entry())
    }

    async fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        tags: Option<Vec<String>>,
        exclude_tags: Option<Vec<String>>,
        limit: usize,
    ) -> Result<KnowledgeSearchPage, CoreError> {
        // Normalize the include/exclude filters the same way writes normalize
        // stored tags, so a differently-cased filter still matches (write/read
        // symmetry). Normalizing once here also keeps the census below reading
        // the same scope the search itself read.
        let tags = normalize_tag_filter(tags);
        let exclude_tags = normalize_tag_filter(exclude_tags);

        // No query embedding (e.g. the embedding backend timed out — see
        // `EMBED_TIMEOUT` in core's embedding port): the hybrid query's vector branch
        // (`chunk <=> $1`) would error on a 0-dimension vector, so fall back to
        // the full-text-only path. The fallback reports the same fields, so a
        // missing embedding backend degrades recall and nothing else.
        let entries = if query_embedding.is_empty() {
            self.search_text_scoped(query, &tags, &exclude_tags, limit)
                .await?
        } else {
            self.search_hybrid(
                query,
                query_embedding,
                embedding_model,
                &tags,
                &exclude_tags,
                limit,
            )
            .await?
        };

        // The census is a decoration on a result the caller already has, so it
        // is best-effort. A connection reset, a pool timeout, or a slow scan
        // under load must cost the measurement and not the entries: the system
        // prompt makes this search mandatory before the assistant asks the user
        // anything, so raising here would spend a whole turn on the decoration.
        //
        // The scope is then reported as `Unknown`, never `None`. `None` is the
        // positive claim that no entry passes the caller's filters, which is an
        // actively harmful falsehood when the truth is that nobody measured.
        let (scope_size, available_tags) =
            census_or_unmeasured(self.tag_census(&tags, &exclude_tags, limit).await);

        Ok(KnowledgeSearchPage {
            entries,
            scope_size,
            available_tags,
        })
    }

    async fn search_text(
        &self,
        query: &str,
        tags: Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let tags = normalize_tag_filter(tags);
        self.search_text_scoped(query, &tags, &None, limit).await
    }

    async fn list(
        &self,
        limit: usize,
        offset: usize,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let tag_filter = normalize_tag_filter(tag_filter);
        let user_id = current_user_id();
        let limit_i64 = limit as i64;
        let offset_i64 = offset as i64;
        let rows: Vec<KbRow> = sqlx::query_as(
            "SELECT id, content, tags, metadata, created_at, updated_at, source, summary
             FROM knowledge_base
             WHERE user_id = $4
               AND deleted_at IS NULL
               AND ($1::text[] IS NULL OR tags && $1)
             ORDER BY updated_at DESC, id
             LIMIT $2 OFFSET $3",
        )
        .bind(&tag_filter)
        .bind(limit_i64)
        .bind(offset_i64)
        .bind(user_id.as_str())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    async fn delete(&self, id: &str) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let ids = [id.to_string()];
        hard_delete_knowledge(
            &self.pool,
            user_id.as_str(),
            HardDeleteTarget::Ids(&ids),
            self.delete_policy,
            "knowledge::PgKnowledgeBaseStore::delete",
        )
        .await?
        .into_removed_or_refusal()?;
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<KnowledgeEntry>, CoreError> {
        let user_id = current_user_id();
        let row: Option<KbRow> = sqlx::query_as(
            "SELECT id, content, tags, metadata, created_at, updated_at, source, summary
             FROM knowledge_base \
             WHERE user_id = $1 AND id = $2 AND deleted_at IS NULL",
        )
        .bind(user_id.as_str())
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(row.map(|r| r.into_entry()))
    }

    // Both delegate to `dreaming::trash`, which owns the whole soft-delete
    // lifecycle — the retention reap, the periodic sweep, and these two
    // on-demand controls — so the user-facing action and the automatic one can
    // never drift apart on what "trash" means.
    async fn trash_count(&self) -> Result<usize, CoreError> {
        crate::dreaming::trash_count(&self.pool).await
    }

    async fn empty_trash(&self) -> Result<usize, CoreError> {
        crate::dreaming::empty_trash(&self.pool, self.delete_policy).await
    }
}

/// Inherent helpers used by the builtin KB tools (wired as closures in the
/// daemon). These sit outside the [`KnowledgeBaseStore`] port because they are
/// tool-surface concerns, not part of the application's outbound contract.
impl PgKnowledgeBaseStore {
    /// Measure the scope a search selected: how many entries it holds and which
    /// tags they carry, most frequent first with the tag name breaking ties.
    ///
    /// "Scope" is the set of entries that pass the caller's tag filters, not
    /// the set that matched the query. One aggregate answers both questions, so
    /// a search costs one extra round trip rather than two.
    ///
    /// Why the sample is ordered by `created_at DESC, id DESC` rather than
    /// merely capped, and what the cap does and does not bound:
    ///
    /// - The order rides `knowledge_base_user_id_created_at_idx` (migration
    ///   016), so the rows arrive newest first without a sort.
    /// - `created_at` alone is not a total order. Rows that share one timestamp
    ///   are then cut apart by their physical position, which moves after any
    ///   `VACUUM` or update, so two identical searches would report different
    ///   tags. `id` is unique, so adding it makes the order total. Postgres
    ///   keeps the index early-stop and sorts each timestamp group
    ///   incrementally.
    /// - The cap bounds how many rows reach the aggregate, not how many rows
    ///   the read touches. `LIMIT` stops after
    ///   [`KNOWLEDGE_TAG_CENSUS_SAMPLE`] rows *pass the filters*, so the read is
    ///   bounded by how many in-scope entries the user holds. A selective
    ///   `exclude_tags` that removes most recent entries therefore reads
    ///   further back, up to the whole of that user's index. Measured against a
    ///   200k-row table, such a filter read to the end in about 70 ms warm.
    /// - The cap is a tail guardrail for a large multi-tenant store, not an
    ///   optimisation of the common path. A personal knowledge base never
    ///   reaches it. `KnowledgeBaseStore::search` treats this whole statement
    ///   as best-effort, so a census that does turn slow and fails costs the
    ///   measurement and not the search.
    ///
    /// Both tag filters must already be normalized (`normalize_tag_filter`), or
    /// a differently-cased filter measures a different scope from the one the
    /// search itself read.
    async fn tag_census(
        &self,
        tags: &Option<Vec<String>>,
        exclude_tags: &Option<Vec<String>>,
        page_limit: usize,
    ) -> Result<(ScopeSize, Vec<String>), CoreError> {
        let user_id = current_user_id();

        // `unnest` sits in the FROM clause, not the target list: a
        // set-returning function cannot be grouped by in the target list, and
        // the lateral form makes the "one row per (entry, tag)" shape explicit.
        // Entries carrying no tags drop out of `census` but still count in
        // `scope_count`, which is what makes an untagged store report a real
        // size with an empty tag list.
        let row: CensusRow = sqlx::query_as(
            "WITH scope AS (
                 SELECT tags
                 FROM knowledge_base
                 WHERE user_id = $1
                   AND deleted_at IS NULL
                   AND ($2::text[] IS NULL OR tags && $2)
                   AND ($3::text[] IS NULL OR NOT (tags && $3))
                 ORDER BY created_at DESC, id DESC
                 LIMIT $4
             ),
             census AS (
                 SELECT t.tag, count(*) AS n
                 FROM scope, unnest(scope.tags) AS t(tag)
                 GROUP BY t.tag
                 ORDER BY n DESC, t.tag
                 LIMIT $5
             )
             SELECT (SELECT count(*) FROM scope) AS scope_count,
                    COALESCE(
                        (SELECT array_agg(tag ORDER BY n DESC, tag) FROM census),
                        ARRAY[]::text[]
                    ) AS available_tags",
        )
        .bind(user_id.as_str())
        .bind(tags)
        .bind(exclude_tags)
        .bind(KNOWLEDGE_TAG_CENSUS_SAMPLE as i64)
        .bind(AVAILABLE_TAGS_LIMIT as i64)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let sampled = row.scope_count as usize;
        let scope_size = ScopeSize::classify(sampled, KNOWLEDGE_TAG_CENSUS_SAMPLE, page_limit);
        Ok((scope_size, row.available_tags))
    }

    /// The vector + full-text arm of [`KnowledgeBaseStore::search`], ranked by
    /// the activation score (#1167).
    ///
    /// The two arms admit and the score ranks -
    /// [`crate::knowledge_search`] holds the whole argument, the shape of the
    /// scan, and the cost the change accepts. What lives here is the
    /// composition: one bounded scan, one batched read of the use log, and the
    /// ranking those two feed.
    ///
    /// **The store's spread is measured in the same pass that ranks, and is
    /// never cached.** The median and the deviation are statistics of the
    /// distances from *this* query's point, so a query in a dense region of the
    /// store has a different distribution from one in a sparse region. A store
    /// too small to state one leaves the page on
    /// [`RECALL_ASSUMED_DISPERSION`](desktop_assistant_core::recall::RECALL_ASSUMED_DISPERSION),
    /// which is the same estimate the `[Recall]` block falls back to: one
    /// estimate rather than two, read through one `activation`. That the two
    /// paths hand that function the same inputs is held mechanically (#1244):
    /// every term comes off one [`Activatable`] implementation each, so a term
    /// added to the score is a compile error in both until both answer.
    ///
    /// [`Activatable`]: desktop_assistant_core::ports::recall::Activatable
    ///
    /// The scan carries [`RECALL_SCAN_STATEMENT_TIMEOUT`]: it reads every
    /// comparable row in scope to state the spread, so it is a full scan and it
    /// is one the model can run several times inside a turn.
    ///
    /// Both tag filters must already be normalized (`normalize_tag_filter`).
    async fn search_hybrid(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        tags: &Option<Vec<String>>,
        exclude_tags: &Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let user_id = current_user_id();
        let embedding_vec = Vector::from(query_embedding);
        // The vector arm over-fetches because activation reorders it: a row it
        // lifts has to be in the admitted set to be lifted. The full-text arm
        // does not, because nothing reorders it - at most a whole page of rows
        // the vector arm cannot compare can ever show. See `rank_page`.
        let fetch_limit = (limit.saturating_mul(2)) as i64;
        let lexical_limit = limit as i64;

        let mut scan = crate::scan_bound::begin_bounded(&self.pool, self.scan_ceiling).await?;
        let rows: Vec<KbSearchRow> = sqlx::query_as(crate::knowledge_search::HYBRID_SEARCH_SQL)
            .bind(embedding_vec)
            .bind(tags)
            .bind(fetch_limit)
            .bind(query)
            .bind(lexical_limit)
            .bind(user_id.as_str())
            .bind(exclude_tags)
            .bind(embedding_model)
            .fetch_all(&mut *scan)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        scan.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Every row carries the same statistics, so the first one states them.
        // The measured dispersion is what the lexical spread is read against,
        // so both come from the same row and never from a fallback for one and
        // a measurement for the other.
        let measured = rows.first().and_then(KbSearchRow::dispersion);
        let spread = rows.first().map_or(0.0, |r| r.spread(measured));
        let dispersion =
            measured.unwrap_or(desktop_assistant_core::recall::RECALL_ASSUMED_DISPERSION);
        let mut candidates: Vec<SearchCandidate> = rows
            .into_iter()
            .map(|r| {
                let distance = r.distance;
                let lexical = LexicalMatch {
                    share: r.lexical_share,
                    spread,
                };
                SearchCandidate {
                    entry: r.into_entry(),
                    distance,
                    lexical,
                    use_record: None,
                    situation: SituationRecord::new(),
                }
            })
            .collect();
        let ids: Vec<String> = candidates.iter().map(|c| c.entry.id.clone()).collect();
        let mut records = self.use_records(ids.clone()).await;
        // The cue the running turn measured for the `[Recall]` block, handed
        // down rather than measured again - `current_situation_cue` holds why.
        // With no cue there is nothing to grade a record against, so the second
        // read is skipped rather than run and discarded: a turn with nothing
        // connected, and a deployment with recall off, pay nothing per search
        // for a term that would score every candidate zero. That is the same
        // bargain the pre-prompt recall path makes.
        let cue = current_situation_cue();
        let mut situations = match cue {
            Some(_) => self.situation_records(ids).await,
            None => std::collections::HashMap::new(),
        };
        for candidate in &mut candidates {
            candidate.use_record = records.remove(&candidate.entry.id);
            candidate.situation = situations
                .remove(&candidate.entry.id)
                .unwrap_or_else(SituationRecord::new);
        }

        Ok(crate::knowledge_search::rank_page(
            candidates,
            dispersion,
            cue.as_ref(),
            chrono::Utc::now(),
            limit,
        ))
    }

    /// What the use log knows about `ids`, keyed by id, and an empty map where
    /// it could not be read.
    ///
    /// **A read that fails costs the ranking and never the page.** The
    /// reinforcement half of the activation score is the half search worked
    /// without until now, so an entry with no record ranks on its semantic
    /// signal alone - which is exactly how every entry ranked before the log
    /// existed. This is the same bargain the recall path's `use_records` makes,
    /// and it is why the read is one batched statement after the scan rather
    /// than a join inside it: a joined read cannot degrade on its own.
    ///
    /// One round trip per search, bounded server-side by
    /// [`USE_LOG_READ_STATEMENT_TIMEOUT`](crate::USE_LOG_READ_STATEMENT_TIMEOUT),
    /// so a slow log stops the backend as well as the caller. Ids the log has
    /// never seen are simply absent, which is the same `None` a failed read
    /// gives - both mean "nothing to add".
    async fn use_records(
        &self,
        ids: Vec<String>,
    ) -> std::collections::HashMap<String, KnowledgeUseRecord> {
        use desktop_assistant_core::ports::knowledge_use::KnowledgeUseLog;

        if ids.is_empty() {
            return std::collections::HashMap::new();
        }
        match crate::knowledge_use::PgKnowledgeUseLog::new(self.pool.clone())
            .records(ids)
            .await
        {
            Ok(records) => records
                .into_iter()
                .map(|record| (record.entry_id.clone(), record))
                .collect(),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "knowledge search: the use log could not be read; ranking on the semantic \
                     signal alone"
                );
                std::collections::HashMap::new()
            }
        }
    }

    /// The situations each of `ids` has been seen in (#1125, #1244), and an
    /// empty map where the log could not be read.
    ///
    /// **A read that fails costs the order and never the page**, on exactly the
    /// terms [`Self::use_records`] states: an entry with no record scores zero
    /// on the situation term, which is how every entry on this path scored
    /// before the term reached it.
    ///
    /// The cue is not read here. It is a statistic of the whole store, so
    /// measuring one costs a full-store count per call, and the running turn
    /// has already measured one for the `[Recall]` block - see
    /// [`current_situation_cue`]. What this read fetches is the per-entry half,
    /// which is a primary-key lookup over at most one page of ids.
    ///
    /// Run after [`Self::use_records`] rather than beside it. The two are
    /// separate statements on separate connections, and the default pool holds
    /// five: a search that held two of them at once would let one turn's
    /// several searches queue the next turn behind pool acquisition, which no
    /// statement timeout bounds.
    async fn situation_records(
        &self,
        ids: Vec<String>,
    ) -> std::collections::HashMap<String, SituationRecord> {
        use desktop_assistant_core::ports::knowledge_use::KnowledgeUseLog;

        if ids.is_empty() {
            return std::collections::HashMap::new();
        }
        match crate::knowledge_use::PgKnowledgeUseLog::new(self.pool.clone())
            .situation_signal(ids, Situation::new())
            .await
        {
            Ok(signal) => signal.records.into_iter().collect(),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "knowledge search: the situation could not be read; ranking without it"
                );
                std::collections::HashMap::new()
            }
        }
    }

    /// FTS-only search with both include- and exclude-tag filters. Backs the
    /// trait `search_text` (exclude = None) and the no-embedding fallback of
    /// `search`.
    ///
    /// Both tag filters must already be normalized (`normalize_tag_filter`).
    async fn search_text_scoped(
        &self,
        query: &str,
        tags: &Option<Vec<String>>,
        exclude_tags: &Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let user_id = current_user_id();
        let result_limit = limit as i64;
        let rows: Vec<KbRow> = sqlx::query_as(
            "WITH q AS (SELECT plainto_tsquery('english', $1) AS query)
             SELECT id, content, tags, metadata, created_at, updated_at, source, summary
             FROM knowledge_base
             WHERE user_id = $4
               AND deleted_at IS NULL
               AND tsv @@ (SELECT query FROM q)
               AND ($2::text[] IS NULL OR tags && $2)
               AND ($5::text[] IS NULL OR NOT (tags && $5))
             ORDER BY ts_rank_cd(tsv, (SELECT query FROM q)) DESC,
                      updated_at DESC
             LIMIT $3",
        )
        .bind(query)
        .bind(tags)
        .bind(result_limit)
        .bind(user_id.as_str())
        .bind(exclude_tags)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    /// Fetch a batch of entries by id in a single statement (#1104).
    ///
    /// Ids that name no live entry the calling user owns are absent from the
    /// result rather than an error: a caller resolving scratchpad attachments
    /// treats an absent id as an attachment that no longer resolves, which is
    /// the same answer for a deleted entry, a trashed one, and another user's.
    /// Retired (`deleted_at`) rows are excluded, exactly as
    /// [`KnowledgeBaseStore::get`] excludes them.
    ///
    /// An id carrying a NUL byte is dropped before the statement runs. Postgres
    /// `text` cannot hold one, so no stored id can contain one and such an id
    /// names nothing - but sent as a parameter it raises rather than missing,
    /// which would break the contract above for every other id in the batch.
    pub async fn get_many(&self, ids: &[String]) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let ids: Vec<String> = ids
            .iter()
            .filter(|id| !id.contains('\0'))
            .cloned()
            .collect();
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();
        let rows: Vec<KbRow> = sqlx::query_as(
            "SELECT id, content, tags, metadata, created_at, updated_at, source, summary \
             FROM knowledge_base \
             WHERE user_id = $1 AND id = ANY($2) AND deleted_at IS NULL",
        )
        .bind(user_id.as_str())
        .bind(ids)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    /// The entries nearest a query embedding, the cosine distance that put each
    /// there, and how spread out this query's distances are over the whole
    /// scope.
    ///
    /// Backs the knowledge arm of the `[Recall]` block (#1100). It is a plain
    /// vector search rather than the hybrid `search`, because the block reads
    /// each candidate against the spread of the store's own distances, and a
    /// fused RRF score is not a quantity that has a spread of that kind: over a
    /// hybrid search every row scores non-zero against any query. A cosine
    /// distance is comparable, so a distribution over it means something.
    ///
    /// **The candidates and the spread come from one scan.** Both are functions
    /// of the same query vector: the spread says what a distance from this store
    /// is worth, and the candidates are the distances it grades, so they have to
    /// describe one query or the grading is against a geometry nothing here saw.
    /// Computing them together also costs one pass rather than two - the scan is
    /// what the query spends its time on, and it is shared.
    ///
    /// **The spread is measured over every row the scan could reach**, not over
    /// the nearest ones: the near rows are the part a cued prompt moves, so their
    /// spread is not the store's and normalizing inside it would inflate every
    /// score. Only the rows the block may show are read whole; the pass that
    /// measures the spread reads one distance per row and none of the content.
    ///
    /// Scoped to the task-local user by an explicit `WHERE user_id` predicate,
    /// on the scan and on the read of the rows it selected. Row-level security is
    /// a backstop the table owner bypasses, so the predicate is the guard, not
    /// the decoration.
    ///
    /// `embedding_model` identifies the model that produced `query_embedding`,
    /// and only rows embedded by that model take part - the same rule the
    /// hybrid search's vector arm follows, for the same reason: a comparison
    /// across models is a comparison across vector dimensions, which the
    /// database answers with an error rather than a miss.
    ///
    /// An empty `query_embedding` yields no rows and no spread. The vector
    /// operator raises on a zero-dimension vector, and the caller that has no
    /// embedding has a full-text path to fall back to (`search_text`).
    ///
    /// `metadata` is not read. The block renders an id, an entry's tags and one
    /// line of what it says, and nothing downstream of this call looks at the
    /// column, so the entries answer with [`serde_json::Value::Null`] there -
    /// the same way this file's other search row drops `source`.
    ///
    /// The scan carries [`RECALL_SCAN_STATEMENT_TIMEOUT`], so the database stops
    /// working when the caller stops waiting.
    pub async fn nearest_by_embedding(
        &self,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        limit: usize,
    ) -> Result<NearestEntries, CoreError> {
        if query_embedding.is_empty() {
            return Ok(NearestEntries::default());
        }
        let user_id = current_user_id();

        // A transaction, for one statement, because `SET LOCAL` is scoped to
        // one: it is what makes the ceiling the caller keeps a ceiling the
        // database keeps too. Abandoning the future stops the daemon waiting
        // and leaves the backend scanning, and recall runs before every turn.
        let mut scan =
            crate::scan_bound::begin_bounded(&self.pool, RECALL_SCAN_STATEMENT_TIMEOUT).await?;
        let rows: Vec<KbNearestRow> = sqlx::query_as(NEAREST_BY_EMBEDDING_SQL)
            .bind(Vector::from(query_embedding))
            .bind(user_id.as_str())
            .bind(embedding_model)
            .bind(limit as i64)
            .fetch_all(&mut *scan)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        scan.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Every row carries the same spread, so the first one states it.
        let dispersion = rows.first().and_then(KbNearestRow::dispersion);
        Ok(NearestEntries {
            entries: rows
                .into_iter()
                .map(|r| {
                    let distance = r.distance;
                    (r.into_entry(), distance)
                })
                .collect(),
            dispersion,
        })
    }

    /// Full-text search that asks for **any** of the query's terms, best match
    /// first.
    ///
    /// The degraded arm of the `[Recall]` block (#1100) when no embedding is
    /// available. It exists because `search_text` cannot serve that caller:
    /// `plainto_tsquery` joins every surviving lexeme with `AND`, which is right
    /// for a model-authored search query of two or three words, and wrong for a
    /// whole user sentence. "where does the registry live?" becomes
    /// `'registri' & 'live'`, and an entry saying "the registry is on the
    /// storage host" does not match, because it never says "live". The fallback
    /// would then answer with nothing at exactly the moment it exists to answer
    /// with something.
    ///
    /// The query is built from `to_tsvector`'s own lexemes, so stop words and
    /// stemming are handled once by the same configuration the index uses, and
    /// `quote_literal` makes every lexeme a literal - a prompt full of
    /// `tsquery` operators is text, not syntax. A prompt that reduces to no
    /// lexemes at all yields a NULL query, which matches no row.
    ///
    /// Ranking still rewards an entry that carries more of the terms, so the
    /// widened match set does not put the weakest hit first.
    ///
    /// Scoped to the task-local user by an explicit `WHERE user_id` predicate.
    ///
    /// The scan carries this store's own ceiling
    /// ([`RECALL_SCAN_STATEMENT_TIMEOUT`] unless
    /// [`Self::with_scan_ceiling`] overrode it), so the database stops working
    /// when the caller stops waiting - the same bound
    /// [`Self::nearest_by_embedding`] carries, because this is the same lookup
    /// on the turn where no embedding was available. It matters more here, not
    /// less: this read has no vector index to ride and its cost grows with the
    /// store.
    pub async fn search_text_any_term(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let user_id = current_user_id();
        let mut scan = crate::scan_bound::begin_bounded(&self.pool, self.scan_ceiling).await?;
        let rows: Vec<KbRow> = sqlx::query_as(
            "WITH q AS (
                 SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS query
                 FROM unnest(to_tsvector('english', $1))
             )
             SELECT id, content, tags, metadata, created_at, updated_at, source, summary
             FROM knowledge_base, q
             WHERE user_id = $3
               AND deleted_at IS NULL
               AND q.query IS NOT NULL
               AND tsv @@ q.query
             ORDER BY ts_rank_cd(tsv, q.query) DESC, updated_at DESC
             LIMIT $2",
        )
        .bind(query)
        .bind(limit as i64)
        .bind(user_id.as_str())
        .fetch_all(&mut *scan)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        scan.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    /// Delete a batch of entries by id in a single statement. Returns the
    /// number of rows actually removed (ids not owned by the user are no-ops).
    ///
    /// This is what `builtin_knowledge_base_delete` reaches, so the model's own
    /// judgement arrives here with no person scope installed. A policy that
    /// reserves hard deletes to a person declines it, and the caller is told
    /// why rather than being handed a count of zero.
    pub async fn delete_many(&self, ids: &[String]) -> Result<usize, CoreError> {
        if ids.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();
        let removed = hard_delete_knowledge(
            &self.pool,
            user_id.as_str(),
            HardDeleteTarget::Ids(ids),
            self.delete_policy,
            "knowledge::PgKnowledgeBaseStore::delete_many",
        )
        .await?
        .into_removed_or_refusal()?;
        Ok(removed as usize)
    }

    /// Non-semantic, keyset-paginated listing for audits. Cursor is on
    /// `(created_at, id)`; over-fetches one row to compute `next_cursor`.
    pub async fn list_page(&self, q: KnowledgeListQuery) -> Result<KnowledgeListPage, CoreError> {
        let user_id = current_user_id();
        let limit = q.limit.clamp(1, 500);
        let fetch = (limit + 1) as i64;

        let (cur_ts, cur_id) = match q.after.as_deref() {
            Some(c) => {
                let (ts, id) = decode_cursor(c)?;
                (Some(ts), Some(id))
            }
            None => (None, None),
        };

        // Two static query strings rather than splicing the comparison
        // operator, so the SQL is never assembled from runtime values.
        let sql = match q.order.0 {
            ListOrder::NewestFirst => {
                "SELECT id, content, tags, metadata, created_at, updated_at, source, summary
                 FROM knowledge_base
                 WHERE user_id = $1
                   AND deleted_at IS NULL
                   AND ($2::text[] IS NULL OR tags && $2)
                   AND ($3::text[] IS NULL OR NOT (tags && $3))
                   AND ($4::text IS NULL OR source = $4)
                   AND ($5::timestamptz IS NULL
                        OR (created_at < $5 OR (created_at = $5 AND id < $6)))
                 ORDER BY created_at DESC, id DESC
                 LIMIT $7"
            }
            ListOrder::OldestFirst => {
                "SELECT id, content, tags, metadata, created_at, updated_at, source, summary
                 FROM knowledge_base
                 WHERE user_id = $1
                   AND deleted_at IS NULL
                   AND ($2::text[] IS NULL OR tags && $2)
                   AND ($3::text[] IS NULL OR NOT (tags && $3))
                   AND ($4::text IS NULL OR source = $4)
                   AND ($5::timestamptz IS NULL
                        OR (created_at > $5 OR (created_at = $5 AND id > $6)))
                 ORDER BY created_at ASC, id ASC
                 LIMIT $7"
            }
        };

        let tags = normalize_tag_filter(q.tags);
        let exclude_tags = normalize_tag_filter(q.exclude_tags);
        let rows: Vec<KbRow> = sqlx::query_as(sql)
            .bind(user_id.as_str())
            .bind(&tags)
            .bind(&exclude_tags)
            .bind(&q.source)
            .bind(cur_ts)
            .bind(&cur_id)
            .bind(fetch)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let has_more = rows.len() as i64 > limit as i64;
        let mut rows = rows;
        rows.truncate(limit);
        let next_cursor = if has_more {
            rows.last().map(|r| encode_cursor(r.created_at, &r.id))
        } else {
            None
        };
        let entries = rows.into_iter().map(|r| r.into_entry()).collect();
        Ok(KnowledgeListPage {
            entries,
            next_cursor,
        })
    }
}

/// Normalize an optional tag filter the same way [`PgKnowledgeBaseStore::write`]
/// normalizes stored tags, so a read matches regardless of the caller's casing
/// or whitespace (`Project:MyApp` finds a row stored as `project:myapp`).
///
/// Contract: `None` (no filter) stays `None`. A present filter is normalized and
/// de-duplicated; if every entry normalizes away it collapses to an empty vec —
/// still `Some(vec![])`, never `None`. That empty-vec case is unchanged from
/// before: each read query guards with `$N::text[] IS NULL OR ...`, and
/// `tags && '{}'` is always false, so an empty include matches no rows and an
/// empty exclude drops none.
pub(crate) fn normalize_tag_filter(filter: Option<Vec<String>>) -> Option<Vec<String>> {
    filter.map(crate::tag_normalize::normalize_tags)
}

/// Encode a keyset cursor as `<created_at_micros>:<id>`.
fn encode_cursor(created_at: chrono::DateTime<chrono::Utc>, id: &str) -> String {
    format!("{}:{}", created_at.timestamp_micros(), id)
}

/// Decode a cursor produced by [`encode_cursor`]. The id may contain `:`, so
/// only the first separator is significant.
fn decode_cursor(cursor: &str) -> Result<(chrono::DateTime<chrono::Utc>, String), CoreError> {
    let (micros, id) = cursor
        .split_once(':')
        .ok_or_else(|| CoreError::Storage("invalid knowledge list cursor".to_string()))?;
    let micros: i64 = micros
        .parse()
        .map_err(|_| CoreError::Storage("invalid knowledge list cursor timestamp".to_string()))?;
    let ts = chrono::DateTime::<chrono::Utc>::from_timestamp_micros(micros)
        .ok_or_else(|| CoreError::Storage("invalid knowledge list cursor timestamp".to_string()))?;
    Ok((ts, id.to_string()))
}

/// How long the recall scan may run before the database stops it.
///
/// A ceiling the caller keeps is not a ceiling the database keeps: abandoning a
/// query future stops the daemon waiting and leaves the backend scanning, and
/// recall runs before every turn's first round - so a store slow enough to
/// exceed this would otherwise accumulate scans at the rate turns arrive.
///
/// The value leaves the embedding its own five seconds inside the ten the whole
/// recall lookup has, with a second to spare. `desktop-assistant-daemon`'s
/// recall adapter holds those three to each other.
pub const RECALL_SCAN_STATEMENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(4);

/// What [`PgKnowledgeBaseStore::nearest_by_embedding`] reads.
///
/// One scan, three uses. `d` computes one distance per row and carries nothing
/// else, so the pass that measures the store's spread reads no content: `m`
/// takes the median of those distances and `s` the median of each distance's own
/// distance from it. The rows the block may show are then read whole, by primary
/// key, and only those.
///
/// Every returned row carries the same spread, which is the price of stating it
/// in the same answer as the candidates - three numbers on at most
/// `max_recall_entries` rows.
///
/// An empty scope yields no rows at all: `d` is empty, so `s` is empty, and the
/// cross join answers with nothing rather than with a spread of nothing.
///
/// Held as its own string so the projection can be asserted on without a
/// database - see `the_recall_scan_does_not_read_metadata`.
const NEAREST_BY_EMBEDDING_SQL: &str = "\
    WITH d AS (
         SELECT id, MIN(chunk <=> $1) AS distance
         FROM knowledge_base, unnest(embedding) AS chunk
         WHERE user_id = $2
           AND deleted_at IS NULL
           AND embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND (embedding_model = $3
                OR (split_part($3, '@', 2) <> ''
                    AND split_part(embedding_model, '@', 2)
                        = split_part($3, '@', 2)))
         GROUP BY id
     ),
     m AS (
         SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY distance) AS median,
                count(*) AS rows_read
         FROM d
     ),
     s AS (
         SELECT m.median,
                m.rows_read,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(d.distance - m.median))
                    AS deviation
         FROM d CROSS JOIN m
         GROUP BY m.median, m.rows_read
     )
     SELECT kb.id, kb.content, kb.tags, kb.created_at, kb.updated_at, kb.source, kb.summary,
            d.distance, s.median, s.rows_read, s.deviation
     FROM d
     JOIN knowledge_base kb ON kb.id = d.id AND kb.user_id = $2
     CROSS JOIN s
     ORDER BY d.distance
     LIMIT $4";

/// What [`PgKnowledgeBaseStore::nearest_by_embedding`] answers with: the rows
/// the block may show, and what a distance from this store is worth.
#[derive(Debug, Default)]
pub struct NearestEntries {
    /// The nearest entries, each with the cosine distance that ranked it,
    /// nearest first.
    pub entries: Vec<(KnowledgeEntry, f64)>,
    /// The spread of this query's distances over the whole scope, or `None`
    /// where the scope holds nothing to measure. The caller then reads the
    /// source by a stated estimate.
    pub dispersion: Option<RecallDispersion>,
}

/// One entry the recall scan ranked, the cosine distance that ranked it, and the
/// spread every row of the answer repeats.
///
/// Its own row rather than a flattened [`KbRow`], because the scan does not read
/// `metadata`: recall never looks at it, and the column is the widest thing on
/// the row.
#[derive(sqlx::FromRow)]
struct KbNearestRow {
    id: String,
    content: String,
    tags: Vec<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    source: Option<String>,
    summary: Option<String>,
    distance: f64,
    median: Option<f64>,
    rows_read: i64,
    deviation: Option<f64>,
}

impl KbNearestRow {
    fn into_entry(self) -> KnowledgeEntry {
        KnowledgeEntry {
            id: self.id,
            content: self.content,
            tags: self.tags,
            metadata: serde_json::Value::Null,
            created_at: self.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            updated_at: self.updated_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            source: self.source,
            summary: self.summary,
        }
    }

    /// What this row says the store's spread is, where it says one it can be
    /// trusted for - see [`RecallDispersion::measured`].
    fn dispersion(&self) -> Option<RecallDispersion> {
        RecallDispersion::measured(
            self.median?,
            self.deviation?,
            self.rows_read.max(0) as usize,
        )
    }
}

#[derive(sqlx::FromRow)]
struct KbRow {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    source: Option<String>,
    summary: Option<String>,
}

impl KbRow {
    fn into_entry(self) -> KnowledgeEntry {
        KnowledgeEntry {
            id: self.id,
            content: self.content,
            tags: self.tags,
            metadata: self.metadata,
            created_at: self.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            updated_at: self.updated_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            source: self.source,
            summary: self.summary,
        }
    }
}

/// The single row the tag census returns: how many entries the capped sample
/// read, and the scope's tags in the order they are reported.
#[derive(sqlx::FromRow)]
struct CensusRow {
    scope_count: i64,
    available_tags: Vec<String>,
}

/// One row the hybrid search admitted: the entry, what the vector arm could
/// measure about it, and the spread every row of the answer repeats.
#[derive(sqlx::FromRow)]
struct KbSearchRow {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    /// Read by the search page, unlike the pre-#1167 projection which dropped
    /// it. Provenance is one of the signals
    /// [`SalienceReading`](desktop_assistant_core::domain::salience::SalienceReading)
    /// reads, so a page that dropped it scored every deliberately-written entry
    /// below what the `[Recall]` block scores it - which is exactly the drift
    /// this work exists to remove.
    source: Option<String>,
    summary: Option<String>,
    /// `None` for a row the full-text arm admitted and the vector arm cannot
    /// compare - no stored vector, or one from another model.
    distance: Option<f64>,
    /// Where this row stands among the rows the query's own words reached, and
    /// zero for a row those words did not reach (#1239).
    lexical_share: f64,
    median: Option<f64>,
    rows_read: i64,
    deviation: Option<f64>,
    /// The nearest and furthest distance the scan reached, which state the
    /// spread a full lexical match is worth. `None` where nothing was
    /// comparable.
    nearest: Option<f64>,
    furthest: Option<f64>,
}

impl KbSearchRow {
    /// What this row says the store's spread is, where it says one it can be
    /// trusted for - see [`RecallDispersion::measured`].
    fn dispersion(&self) -> Option<RecallDispersion> {
        RecallDispersion::measured(
            self.median?,
            self.deviation?,
            self.rows_read.max(0) as usize,
        )
    }

    /// How many of this source's own deviations separate its nearest row from
    /// its furthest, for this query - the scale a full lexical match is spent
    /// against (#1239).
    ///
    /// Zero where the source stated no dispersion to read the two extremes
    /// against, or where it reached no comparable row at all. The lexical term
    /// is then worth nothing, which is what it was worth before the term
    /// existed.
    fn spread(&self, dispersion: Option<RecallDispersion>) -> f64 {
        let (Some(dispersion), Some(nearest), Some(furthest)) =
            (dispersion, self.nearest, self.furthest)
        else {
            return 0.0;
        };
        (dispersion.deviations_below_median(nearest) - dispersion.deviations_below_median(furthest))
            .max(0.0)
    }

    fn into_entry(self) -> KnowledgeEntry {
        KnowledgeEntry {
            id: self.id,
            content: self.content,
            tags: self.tags,
            metadata: self.metadata,
            created_at: self.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            updated_at: self.updated_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            source: self.source,
            // Search does select the summary: it is what a caller reads to
            // decide whether a hit is worth pulling the body for.
            summary: self.summary,
        }
    }
}

/// Fold a tag-census result into what the search page should report.
///
/// Why this is a named function rather than an inline `match`: it is the whole
/// of the best-effort contract, and the failure it guards against is a silent
/// one. The census is an extra statement issued after the search has already
/// returned its entries, so propagating its error with `?` would discard a
/// search that succeeded. Naming it lets that be tested directly, without
/// contriving a database that fails one statement and not the other.
fn census_or_unmeasured(
    census: Result<(ScopeSize, Vec<String>), CoreError>,
) -> (ScopeSize, Vec<String>) {
    match census {
        Ok(census) => census,
        Err(e) => {
            tracing::warn!(error = %e, "knowledge base tag census failed; reporting an unmeasured scope");
            (ScopeSize::Unknown, Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_census_costs_the_measurement_not_the_search() {
        // The whole of the best-effort contract. A `?` here would discard a
        // search that had already returned its entries, and the system prompt
        // makes this search mandatory before the assistant asks the user
        // anything, so that would spend a turn on a decoration.
        let (scope_size, tags) =
            census_or_unmeasured(Err(CoreError::Storage("pool timed out".to_string())));

        assert_eq!(scope_size, ScopeSize::Unknown);
        assert!(tags.is_empty(), "an unmeasured scope reports no tags");
    }

    #[test]
    fn a_failed_census_never_reports_an_empty_scope() {
        // `None` is the positive claim that no entry passes the caller's
        // filters. Reporting it for a census that did not run tells the model
        // the store is empty when it may hold exactly what was asked for.
        let (scope_size, _) =
            census_or_unmeasured(Err(CoreError::Storage("connection reset".to_string())));

        assert_ne!(scope_size, ScopeSize::None);
    }

    #[test]
    fn a_successful_census_passes_through_unchanged() {
        let census = (ScopeSize::Few, vec!["preference".to_string()]);

        let (scope_size, tags) = census_or_unmeasured(Ok(census));

        assert_eq!(scope_size, ScopeSize::Few);
        assert_eq!(tags, vec!["preference".to_string()]);
    }

    #[test]
    fn tag_filter_none_stays_none() {
        // No filter must stay "no filter" — never coerced to an empty vec.
        assert_eq!(normalize_tag_filter(None), None);
    }

    #[test]
    fn tag_filter_empty_vec_stays_empty_some() {
        // An explicit empty filter must NOT become None or match/exclude
        // everything: `tags && '{}'` is false, so an empty include matches no
        // rows and an empty exclude drops none — identical to pre-normalization.
        assert_eq!(normalize_tag_filter(Some(vec![])), Some(vec![]));
    }

    #[test]
    fn tag_filter_normalizes_case_and_preserves_facet_colon() {
        assert_eq!(
            normalize_tag_filter(Some(vec!["Project:MyApp".to_string()])),
            Some(vec!["project:myapp".to_string()])
        );
    }

    #[test]
    fn tag_filter_dedups_after_normalization() {
        assert_eq!(
            normalize_tag_filter(Some(vec![
                "Instruction".to_string(),
                "instruction".to_string(),
            ])),
            Some(vec!["instruction".to_string()])
        );
    }

    #[test]
    fn tag_filter_all_empty_collapses_to_empty_some() {
        // A whitespace-only filter normalizes away to an empty vec — still
        // `Some`, so it behaves like an explicit empty filter, not "no filter".
        assert_eq!(
            normalize_tag_filter(Some(vec!["   ".to_string(), String::new()])),
            Some(vec![])
        );
    }

    /// Acceptance (#1121): `metadata` is not read by the recall scan. The
    /// column is the widest thing on the row and the block never looks at it.
    #[test]
    fn the_recall_scan_does_not_read_metadata() {
        assert!(
            !NEAREST_BY_EMBEDDING_SQL.contains("metadata"),
            "the recall scan selects a column recall never reads: \n{NEAREST_BY_EMBEDDING_SQL}"
        );
    }

    /// The pass that measures the store's spread reads the geometry and none of
    /// the content: one distance per row, and no column of the entry itself.
    /// Only the rows the block may show are read whole.
    #[test]
    fn the_pass_that_measures_the_spread_reads_no_entry_content() {
        let measured = NEAREST_BY_EMBEDDING_SQL
            .split("     SELECT kb.id")
            .next()
            .expect("the scan selects the rows it will show after it measures the spread");

        for column in ["metadata", "content", "summary", "tags"] {
            assert!(
                !measured.contains(column),
                "the pass that measures the spread reads {column}, which it has no use for"
            );
        }
    }

    /// The scan bounds the database's own work, not only the caller's patience.
    /// Abandoning a query future leaves the backend scanning, and recall runs
    /// before every turn.
    #[test]
    fn the_recall_scan_states_a_statement_timeout() {
        assert!(
            RECALL_SCAN_STATEMENT_TIMEOUT > std::time::Duration::ZERO,
            "a zero timeout means no timeout at all in PostgreSQL"
        );
    }
}
