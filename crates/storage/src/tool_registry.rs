use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
use pgvector::Vector;
use sqlx::PgPool;

pub struct PgToolRegistryStore {
    pool: PgPool,
}

impl PgToolRegistryStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

impl ToolRegistryStore for PgToolRegistryStore {
    async fn register_tools(
        &self,
        tools: Vec<ToolDefinition>,
        source: &str,
        is_core: bool,
        embeddings: Vec<Option<Vec<f32>>>,
    ) -> Result<(), CoreError> {
        let mut tx = self.pool.begin().await.map_err(|e| CoreError::Storage(e.to_string()))?;

        for (i, tool) in tools.iter().enumerate() {
            let embedding_vec = embeddings.get(i).and_then(|e| e.clone()).map(Vector::from);

            sqlx::query(
                "INSERT INTO tool_definitions (name, description, parameters, source, is_core, embedding)
                 VALUES ($1, $2, $3, $4, $5, $6)
                 ON CONFLICT (name) DO UPDATE
                    SET description = EXCLUDED.description,
                        parameters = EXCLUDED.parameters,
                        source = EXCLUDED.source,
                        is_core = EXCLUDED.is_core,
                        embedding = EXCLUDED.embedding,
                        registered_at = NOW()"
            )
            .bind(&tool.name)
            .bind(&tool.description)
            .bind(&tool.parameters)
            .bind(source)
            .bind(is_core)
            .bind(embedding_vec)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        tx.commit().await.map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn unregister_source(&self, source: &str) -> Result<(), CoreError> {
        sqlx::query("DELETE FROM tool_definitions WHERE source = $1")
            .bind(source)
            .execute(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn core_tools(&self) -> Result<Vec<ToolDefinition>, CoreError> {
        let rows: Vec<ToolRow> = sqlx::query_as(
            "SELECT name, description, parameters FROM tool_definitions WHERE is_core = TRUE"
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_definition()).collect())
    }

    async fn search_tools(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<ToolDefinition>, CoreError> {
        let embedding_vec = Vector::from(query_embedding);
        let fetch_limit = (limit * 2) as i64;
        let result_limit = limit as i64;

        let rows: Vec<ToolSearchRow> = sqlx::query_as(
            "WITH vector_ranked AS (
                SELECT name, description, parameters,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank_v
                FROM tool_definitions
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1
                LIMIT $2
            ),
            text_ranked AS (
                SELECT name, description, parameters,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank_t
                FROM tool_definitions, plainto_tsquery('english', $3) query
                WHERE tsv @@ query
                ORDER BY ts_rank_cd(tsv, query) DESC
                LIMIT $2
            ),
            fused AS (
                SELECT COALESCE(v.name, t.name) AS name,
                       COALESCE(v.description, t.description) AS description,
                       COALESCE(v.parameters, t.parameters) AS parameters,
                       COALESCE(1.0 / (60 + v.rank_v), 0) +
                       COALESCE(1.0 / (60 + t.rank_t), 0) AS rrf_score
                FROM vector_ranked v
                FULL OUTER JOIN text_ranked t ON v.name = t.name
            )
            SELECT name, description, parameters, rrf_score
            FROM fused ORDER BY rrf_score DESC LIMIT $4"
        )
        .bind(embedding_vec)
        .bind(fetch_limit)
        .bind(query)
        .bind(result_limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_definition()).collect())
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        let row: Option<ToolRow> = sqlx::query_as(
            "SELECT name, description, parameters FROM tool_definitions WHERE name = $1"
        )
        .bind(name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(row.map(|r| r.into_definition()))
    }
}

#[derive(sqlx::FromRow)]
struct ToolRow {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl ToolRow {
    fn into_definition(self) -> ToolDefinition {
        ToolDefinition::new(self.name, self.description, self.parameters)
    }
}

#[derive(sqlx::FromRow)]
struct ToolSearchRow {
    name: String,
    description: String,
    parameters: serde_json::Value,
    #[allow(dead_code)]
    rrf_score: Option<f64>,
}

impl ToolSearchRow {
    fn into_definition(self) -> ToolDefinition {
        ToolDefinition::new(self.name, self.description, self.parameters)
    }
}
