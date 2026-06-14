use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::KnowledgeEntry;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge::KnowledgeBaseStore;
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
        let row: KbRow = sqlx::query_as(
            "INSERT INTO knowledge_base \
                (id, user_id, content, tags, metadata) \
             VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (id) DO UPDATE \
                SET content = EXCLUDED.content, \
                    tags = EXCLUDED.tags, \
                    metadata = EXCLUDED.metadata, \
                    updated_at = NOW() \
                WHERE knowledge_base.user_id = $2 \
             RETURNING id, content, tags, metadata, created_at, updated_at",
        )
        .bind(&entry.id)
        .bind(user_id.as_str())
        .bind(&entry.content)
        .bind(&entry.tags)
        .bind(&entry.metadata)
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
        limit: usize,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        // No query embedding (e.g. the embedding backend timed out — see
        // `EMBED_TIMEOUT` in mcp-client): the hybrid query's vector branch
        // (`chunk <=> $1`) would error on a 0-dimension vector, so fall back to
        // the full-text-only path.
        if query_embedding.is_empty() {
            return self.search_text(query, tags, limit).await;
        }

        let user_id = current_user_id();
        let embedding_vec = Vector::from(query_embedding);
        let fetch_limit = (limit * 2) as i64;
        let result_limit = limit as i64;

        let rows: Vec<KbSearchRow> = sqlx::query_as(
            "WITH chunk_distances AS (
                SELECT id, content, tags, metadata, created_at, updated_at,
                       MIN(chunk <=> $1) AS min_distance
                FROM knowledge_base, unnest(embedding) AS chunk
                WHERE user_id = $6
                  AND ($2::text[] IS NULL OR tags && $2)
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
        let user_id = current_user_id();
        let result_limit = limit as i64;
        let rows: Vec<KbRow> = sqlx::query_as(
            "WITH q AS (SELECT plainto_tsquery('english', $1) AS query)
             SELECT id, content, tags, metadata, created_at, updated_at
             FROM knowledge_base
             WHERE user_id = $4
               AND tsv @@ (SELECT query FROM q)
               AND ($2::text[] IS NULL OR tags && $2)
             ORDER BY ts_rank_cd(tsv, (SELECT query FROM q)) DESC,
                      updated_at DESC
             LIMIT $3",
        )
        .bind(query)
        .bind(&tags)
        .bind(result_limit)
        .bind(user_id.as_str())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    async fn list(
        &self,
        limit: usize,
        offset: usize,
        tag_filter: Option<Vec<String>>,
    ) -> Result<Vec<KnowledgeEntry>, CoreError> {
        let user_id = current_user_id();
        let limit_i64 = limit as i64;
        let offset_i64 = offset as i64;
        let rows: Vec<KbRow> = sqlx::query_as(
            "SELECT id, content, tags, metadata, created_at, updated_at
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
            "SELECT id, content, tags, metadata, created_at, updated_at
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

#[derive(sqlx::FromRow)]
struct KbRow {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
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
        }
    }
}
