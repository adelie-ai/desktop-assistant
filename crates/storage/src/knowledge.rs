use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::KnowledgeEntry;
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
    async fn write(
        &self,
        entry: KnowledgeEntry,
        embedding: Option<Vec<f32>>,
    ) -> Result<KnowledgeEntry, CoreError> {
        let embedding_vec = embedding.map(Vector::from);

        let row: KbRow = sqlx::query_as(
            "INSERT INTO knowledge_base (id, content, tags, metadata, embedding)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (id) DO UPDATE
                SET content = EXCLUDED.content,
                    tags = EXCLUDED.tags,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
             RETURNING id, content, tags, metadata, created_at, updated_at"
        )
        .bind(&entry.id)
        .bind(&entry.content)
        .bind(&entry.tags)
        .bind(&entry.metadata)
        .bind(embedding_vec)
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
        let embedding_vec = Vector::from(query_embedding);
        let fetch_limit = (limit * 2) as i64;
        let result_limit = limit as i64;

        let rows: Vec<KbSearchRow> = sqlx::query_as(
            "WITH vector_ranked AS (
                SELECT id, content, tags, metadata, created_at, updated_at,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank_v
                FROM knowledge_base
                WHERE ($2::text[] IS NULL OR tags && $2)
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> $1
                LIMIT $3
            ),
            text_ranked AS (
                SELECT id, content, tags, metadata, created_at, updated_at,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank_t
                FROM knowledge_base, plainto_tsquery('english', $4) query
                WHERE ($2::text[] IS NULL OR tags && $2)
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
                       COALESCE(1.0 / (60 + v.rank_v), 0) +
                       COALESCE(1.0 / (60 + t.rank_t), 0) AS rrf_score
                FROM vector_ranked v
                FULL OUTER JOIN text_ranked t ON v.id = t.id
            )
            SELECT id, content, tags, metadata, created_at, updated_at, rrf_score
            FROM fused ORDER BY rrf_score DESC LIMIT $5"
        )
        .bind(embedding_vec)
        .bind(&tags)
        .bind(fetch_limit)
        .bind(query)
        .bind(result_limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_entry()).collect())
    }

    async fn delete(&self, id: &str) -> Result<(), CoreError> {
        sqlx::query("DELETE FROM knowledge_base WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, id: &str) -> Result<Option<KnowledgeEntry>, CoreError> {
        let row: Option<KbRow> = sqlx::query_as(
            "SELECT id, content, tags, metadata, created_at, updated_at
             FROM knowledge_base WHERE id = $1"
        )
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
    #[allow(dead_code)]
    rrf_score: Option<f64>,
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
