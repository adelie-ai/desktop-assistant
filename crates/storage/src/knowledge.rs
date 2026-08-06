use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge::{
    AVAILABLE_TAGS_LIMIT, KNOWLEDGE_TAG_CENSUS_SAMPLE, KnowledgeBaseStore, KnowledgeListPage,
    KnowledgeListQuery, KnowledgeSearchPage, ListOrder, ScopeSize,
};
use pgvector::Vector;
use sqlx::PgPool;

pub struct PgKnowledgeBaseStore {
    pool: PgPool,
}

impl PgKnowledgeBaseStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
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
        // (COALESCE) rather than clearing it, so a path that doesn't care about
        // provenance, or knows nothing about summaries, can't wipe one. There
        // is deliberately no way to clear either through this call: absent is
        // the only meaning NULL carries here.
        let row: Option<KbRow> = sqlx::query_as(
            "INSERT INTO knowledge_base \
                (id, user_id, content, tags, metadata, source, summary) \
             VALUES ($1, $2, $3, $4, $5, $6, $7) \
             ON CONFLICT (id) DO UPDATE \
                SET content = EXCLUDED.content, \
                    tags = EXCLUDED.tags, \
                    metadata = EXCLUDED.metadata, \
                    source = COALESCE(EXCLUDED.source, knowledge_base.source), \
                    summary = COALESCE(EXCLUDED.summary, knowledge_base.summary), \
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
        sqlx::query("DELETE FROM knowledge_base WHERE user_id = $1 AND id = $2")
            .bind(user_id.as_str())
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
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
        crate::dreaming::empty_trash(&self.pool).await
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

    /// The vector + full-text (RRF) arm of [`KnowledgeBaseStore::search`].
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
        let fetch_limit = (limit * 2) as i64;
        let result_limit = limit as i64;

        // $7 = exclude_tags: drop any row carrying one of these tags.
        //
        // $8 = the model that produced $1. Only rows embedded by that model can
        // be compared against it, so the predicate belongs on this branch and
        // this branch alone: `text_ranked` below stays model-blind, which is
        // what turns a model change into degraded (lexical-only) recall instead
        // of content that cannot be found at all.
        //
        // Sameness is decided on the digest half of the `<name>@<digest>` stamp
        // wherever both sides carry one, matching
        // `embedding_backfill::invalidate_stale_embeddings`: a purely cosmetic
        // rename leaves usable vectors in place, and hiding them until the
        // sweep restamps them would blank semantic search for no reason.
        // `split_part(x, '@', 2)` yields '' when there is no '@', so the
        // non-empty test doubles as "both sides carry a digest". A NULL stamp is
        // a vector of unknown provenance, hence unknown dimension, and is
        // excluded.
        let rows: Vec<KbSearchRow> = sqlx::query_as(
            "WITH chunk_distances AS (
                SELECT id, content, tags, metadata, created_at, updated_at, summary,
                       MIN(chunk <=> $1) AS min_distance
                FROM knowledge_base, unnest(embedding) AS chunk
                WHERE user_id = $6
                  AND deleted_at IS NULL
                  AND ($2::text[] IS NULL OR tags && $2)
                  AND ($7::text[] IS NULL OR NOT (tags && $7))
                  AND embedding IS NOT NULL
                  AND embedding_model IS NOT NULL
                  AND (embedding_model = $8
                       OR (split_part($8, '@', 2) <> ''
                           AND split_part(embedding_model, '@', 2)
                               = split_part($8, '@', 2)))
                GROUP BY id, content, tags, metadata, created_at, updated_at, summary
            ),
            vector_ranked AS (
                SELECT id, content, tags, metadata, created_at, updated_at, summary,
                       ROW_NUMBER() OVER (ORDER BY min_distance) AS rank_v
                FROM chunk_distances
                LIMIT $3
            ),
            text_ranked AS (
                SELECT id, content, tags, metadata, created_at, updated_at, summary,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank_t
                FROM knowledge_base, plainto_tsquery('english', $4) query
                WHERE user_id = $6
                  AND deleted_at IS NULL
                  AND ($2::text[] IS NULL OR tags && $2)
                  AND ($7::text[] IS NULL OR NOT (tags && $7))
                  AND tsv @@ query
                ORDER BY ts_rank_cd(tsv, query) DESC
                LIMIT $3
            ),
            fused AS (
                SELECT COALESCE(v.id, t.id) AS id,
                       COALESCE(v.content, t.content) AS content,
                       COALESCE(v.tags, t.tags) AS tags,
                       COALESCE(v.metadata, t.metadata) AS metadata,
                       COALESCE(v.created_at, t.created_at) AS created_at,
                       COALESCE(v.updated_at, t.updated_at) AS updated_at,
                       COALESCE(v.summary, t.summary) AS summary,
                       (COALESCE(1.0 / (60 + v.rank_v), 0) +
                        COALESCE(1.0 / (60 + t.rank_t), 0))::FLOAT8 AS rrf_score
                FROM vector_ranked v
                FULL OUTER JOIN text_ranked t ON v.id = t.id
            )
            SELECT id, content, tags, metadata, created_at, updated_at, summary
            FROM fused ORDER BY rrf_score DESC LIMIT $5",
        )
        .bind(embedding_vec)
        .bind(tags)
        .bind(fetch_limit)
        .bind(query)
        .bind(result_limit)
        .bind(user_id.as_str())
        .bind(exclude_tags)
        .bind(embedding_model)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
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
    pub async fn get_many(&self, ids: &[String]) -> Result<Vec<KnowledgeEntry>, CoreError> {
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

    /// The entries nearest a query embedding, with the cosine distance that put
    /// them there, nearest first.
    ///
    /// Backs the knowledge arm of the `[Recall]` block (#1100). It is a plain
    /// vector search rather than the hybrid `search`, because the block applies
    /// a relevance floor and a fused RRF score is not a quantity a floor can be
    /// set against: over a hybrid search every row scores non-zero against any
    /// query. A cosine distance is comparable, so a floor over it means
    /// something.
    ///
    /// Scoped to the task-local user by an explicit `WHERE user_id` predicate.
    /// Row-level security is a backstop the table owner bypasses, so the
    /// predicate is the guard, not the decoration.
    ///
    /// `embedding_model` identifies the model that produced `query_embedding`,
    /// and only rows embedded by that model take part - the same rule the
    /// hybrid search's vector arm follows, for the same reason: a comparison
    /// across models is a comparison across vector dimensions, which the
    /// database answers with an error rather than a miss.
    ///
    /// An empty `query_embedding` yields no rows. The vector operator raises on
    /// a zero-dimension vector, and the caller that has no embedding has a
    /// full-text path to fall back to (`search_text`).
    pub async fn nearest_by_embedding(
        &self,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        limit: usize,
    ) -> Result<Vec<(KnowledgeEntry, f64)>, CoreError> {
        if query_embedding.is_empty() {
            return Ok(Vec::new());
        }
        let user_id = current_user_id();
        let rows: Vec<KbNearestRow> = sqlx::query_as(
            "SELECT id, content, tags, metadata, created_at, updated_at, source, summary,
                    MIN(chunk <=> $1) AS distance
             FROM knowledge_base, unnest(embedding) AS chunk
             WHERE user_id = $2
               AND deleted_at IS NULL
               AND embedding IS NOT NULL
               AND embedding_model IS NOT NULL
               AND (embedding_model = $3
                    OR (split_part($3, '@', 2) <> ''
                        AND split_part(embedding_model, '@', 2)
                            = split_part($3, '@', 2)))
             GROUP BY id, content, tags, metadata, created_at, updated_at, source, summary
             ORDER BY distance
             LIMIT $4",
        )
        .bind(Vector::from(query_embedding))
        .bind(user_id.as_str())
        .bind(embedding_model)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| {
                let distance = r.distance;
                (r.row.into_entry(), distance)
            })
            .collect())
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
    pub async fn search_text_any_term(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let user_id = current_user_id();
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
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    /// Delete a batch of entries by id in a single statement. Returns the
    /// number of rows actually removed (ids not owned by the user are no-ops).
    pub async fn delete_many(&self, ids: &[String]) -> Result<usize, CoreError> {
        if ids.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();
        let res = sqlx::query("DELETE FROM knowledge_base WHERE user_id = $1 AND id = ANY($2)")
            .bind(user_id.as_str())
            .bind(ids)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(res.rows_affected() as usize)
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

/// A [`KbRow`] plus the cosine distance that ranked it, for
/// [`PgKnowledgeBaseStore::nearest_by_embedding`].
#[derive(sqlx::FromRow)]
struct KbNearestRow {
    #[sqlx(flatten)]
    row: KbRow,
    distance: f64,
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

#[derive(sqlx::FromRow)]
struct KbSearchRow {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    summary: Option<String>,
}

impl KbSearchRow {
    fn into_entry(self) -> KnowledgeEntry {
        KnowledgeEntry {
            id: self.id,
            content: self.content,
            tags: self.tags,
            metadata: self.metadata,
            created_at: self.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            updated_at: self.updated_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            // Search does not select provenance; the audit/list path does.
            source: None,
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
}
