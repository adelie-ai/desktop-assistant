use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge::{
    KnowledgeBaseStore, KnowledgeListPage, KnowledgeListQuery, ListOrder,
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
        // `source` ($6) records provenance. On update a NULL `source` preserves
        // the existing value (COALESCE) rather than clearing it, so a path that
        // doesn't care about provenance can't wipe it.
        let row: KbRow = sqlx::query_as(
            "INSERT INTO knowledge_base \
                (id, user_id, content, tags, metadata, source) \
             VALUES ($1, $2, $3, $4, $5, $6) \
             ON CONFLICT (id) DO UPDATE \
                SET content = EXCLUDED.content, \
                    tags = EXCLUDED.tags, \
                    metadata = EXCLUDED.metadata, \
                    source = COALESCE(EXCLUDED.source, knowledge_base.source), \
                    updated_at = NOW() \
                WHERE knowledge_base.user_id = $2 \
             RETURNING id, content, tags, metadata, created_at, updated_at, source",
        )
        .bind(&entry.id)
        .bind(user_id.as_str())
        .bind(&entry.content)
        .bind(&tags)
        .bind(&entry.metadata)
        .bind(&entry.source)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(row.into_entry())
    }

    async fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        tags: Option<Vec<String>>,
        exclude_tags: Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        // No query embedding (e.g. the embedding backend timed out — see
        // `EMBED_TIMEOUT` in mcp-client): the hybrid query's vector branch
        // (`chunk <=> $1`) would error on a 0-dimension vector, so fall back to
        // the full-text-only path.
        if query_embedding.is_empty() {
            return self
                .search_text_filtered(query, tags, exclude_tags, limit)
                .await;
        }

        // Normalize the include/exclude filters the same way writes normalize
        // stored tags, so a differently-cased filter still matches (write/read
        // symmetry). The FTS fallback above normalizes inside
        // `search_text_filtered`, so this covers only the vector-branch path.
        let tags = normalize_tag_filter(tags);
        let exclude_tags = normalize_tag_filter(exclude_tags);
        let user_id = current_user_id();
        let embedding_vec = Vector::from(query_embedding);
        let fetch_limit = (limit * 2) as i64;
        let result_limit = limit as i64;

        // $7 = exclude_tags: drop any row carrying one of these tags.
        let rows: Vec<KbSearchRow> = sqlx::query_as(
            "WITH chunk_distances AS (
                SELECT id, content, tags, metadata, created_at, updated_at,
                       MIN(chunk <=> $1) AS min_distance
                FROM knowledge_base, unnest(embedding) AS chunk
                WHERE user_id = $6
                  AND ($2::text[] IS NULL OR tags && $2)
                  AND ($7::text[] IS NULL OR NOT (tags && $7))
                  AND embedding IS NOT NULL
                GROUP BY id, content, tags, metadata, created_at, updated_at
            ),
            vector_ranked AS (
                SELECT id, content, tags, metadata, created_at, updated_at,
                       ROW_NUMBER() OVER (ORDER BY min_distance) AS rank_v
                FROM chunk_distances
                LIMIT $3
            ),
            text_ranked AS (
                SELECT id, content, tags, metadata, created_at, updated_at,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank_t
                FROM knowledge_base, plainto_tsquery('english', $4) query
                WHERE user_id = $6
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
                       (COALESCE(1.0 / (60 + v.rank_v), 0) +
                        COALESCE(1.0 / (60 + t.rank_t), 0))::FLOAT8 AS rrf_score
                FROM vector_ranked v
                FULL OUTER JOIN text_ranked t ON v.id = t.id
            )
            SELECT id, content, tags, metadata, created_at, updated_at
            FROM fused ORDER BY rrf_score DESC LIMIT $5",
        )
        .bind(embedding_vec)
        .bind(&tags)
        .bind(fetch_limit)
        .bind(query)
        .bind(result_limit)
        .bind(user_id.as_str())
        .bind(&exclude_tags)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    async fn search_text(
        &self,
        query: &str,
        tags: Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        self.search_text_filtered(query, tags, None, limit).await
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
            "SELECT id, content, tags, metadata, created_at, updated_at, source
             FROM knowledge_base
             WHERE user_id = $4
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
            "SELECT id, content, tags, metadata, created_at, updated_at, source
             FROM knowledge_base WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(row.map(|r| r.into_entry()))
    }
}

/// Inherent helpers used by the builtin KB tools (wired as closures in the
/// daemon). These sit outside the [`KnowledgeBaseStore`] port because they are
/// tool-surface concerns, not part of the application's outbound contract.
impl PgKnowledgeBaseStore {
    /// FTS-only search with both include- and exclude-tag filters. Backs the
    /// trait `search_text` (exclude = None) and the no-embedding fallback of
    /// `search`.
    async fn search_text_filtered(
        &self,
        query: &str,
        tags: Option<Vec<String>>,
        exclude_tags: Option<Vec<String>>,
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let tags = normalize_tag_filter(tags);
        let exclude_tags = normalize_tag_filter(exclude_tags);
        let user_id = current_user_id();
        let result_limit = limit as i64;
        let rows: Vec<KbRow> = sqlx::query_as(
            "WITH q AS (SELECT plainto_tsquery('english', $1) AS query)
             SELECT id, content, tags, metadata, created_at, updated_at, source
             FROM knowledge_base
             WHERE user_id = $4
               AND tsv @@ (SELECT query FROM q)
               AND ($2::text[] IS NULL OR tags && $2)
               AND ($5::text[] IS NULL OR NOT (tags && $5))
             ORDER BY ts_rank_cd(tsv, (SELECT query FROM q)) DESC,
                      updated_at DESC
             LIMIT $3",
        )
        .bind(query)
        .bind(&tags)
        .bind(result_limit)
        .bind(user_id.as_str())
        .bind(&exclude_tags)
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
                "SELECT id, content, tags, metadata, created_at, updated_at, source
                 FROM knowledge_base
                 WHERE user_id = $1
                   AND ($2::text[] IS NULL OR tags && $2)
                   AND ($3::text[] IS NULL OR NOT (tags && $3))
                   AND ($4::text IS NULL OR source = $4)
                   AND ($5::timestamptz IS NULL
                        OR (created_at < $5 OR (created_at = $5 AND id < $6)))
                 ORDER BY created_at DESC, id DESC
                 LIMIT $7"
            }
            ListOrder::OldestFirst => {
                "SELECT id, content, tags, metadata, created_at, updated_at, source
                 FROM knowledge_base
                 WHERE user_id = $1
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

#[derive(sqlx::FromRow)]
struct KbRow {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    source: Option<String>,
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
        }
    }
}

#[derive(sqlx::FromRow)]
struct KbSearchRow {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
