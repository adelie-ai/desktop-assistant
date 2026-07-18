use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ToolDefinition;
use desktop_assistant_core::ports::tool_registry::ToolRegistryStore;
use pgvector::Vector;
use sqlx::PgPool;

pub struct PgToolRegistryStore {
    pool: PgPool,
}

/// Weight applied to a matched provider row's fused score when boosting its
/// member tools. A provider match contributes at most `0.5 x` a *maximal*
/// (rank-1 in both branches) RRF hit to each member, so a provider match can
/// lift a member several ranks - enough to pull one just below the top-N cutoff
/// into it - but cannot, on its own, carry a weakly-matching member past a tool
/// that matched the query maximally on its own. (At `1.0` a single provider
/// match added a whole standalone hit and could leapfrog strong direct matches;
/// see the `weak_member_..._does_not_outrank_strong_standalone` test.) Named so
/// it stays tunable.
pub const PROVIDER_BOOST_WEIGHT: f64 = 0.5;

impl PgToolRegistryStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Like [`ToolRegistryStore::search_tools`], but also returns each result's
    /// final (boosted) fused score. `search_tools` is this with the scores
    /// dropped; the scores are exposed for ranking diagnostics and to let tests
    /// assert the exact boost arithmetic (e.g. that a matched provider's score is
    /// added exactly once, not per branch).
    ///
    /// Both paths apply the provider score-boost and BOTH exclude the synthetic
    /// `provider:%` rows from the output (guard #2): a provider row participates
    /// in scoring and drives the boost, but is never returned as a callable tool.
    pub async fn search_tools_scored(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<(ToolDefinition, f64)>, CoreError> {
        // No query embedding (e.g. the embedding backend timed out — see
        // `EMBED_TIMEOUT` in mcp-client): the hybrid query's vector branch
        // (`chunk <=> $1`) would error on a 0-dimension vector, so fall back to
        // full-text search only. The FTS fallback carries the SAME provider
        // boost + `provider:%` exclusion as the hybrid path.
        if query_embedding.is_empty() {
            // Real tools and provider rows are scored in SEPARATE candidate sets
            // (FIX 1): `real_ranked` excludes `provider:%` so synthetic rows never
            // consume a real-tool result slot, while `matched_providers` still
            // scores the provider rows so the boost works. The final
            // `WHERE name NOT LIKE 'provider:%'` remains as guard #2.
            let rows: Vec<ToolSearchRow> = sqlx::query_as(
                "WITH real_ranked AS (
                    SELECT name, description, parameters, provider,
                           ts_rank_cd(tsv, query)::FLOAT8 AS score
                    FROM tool_definitions, plainto_tsquery('english', $1) query
                    WHERE tsv @@ query AND name NOT LIKE 'provider:%'
                ),
                matched_providers AS (
                    SELECT provider, ts_rank_cd(tsv, query)::FLOAT8 AS provider_score
                    FROM tool_definitions, plainto_tsquery('english', $1) query
                    WHERE tsv @@ query AND name LIKE 'provider:%'
                ),
                boosted AS (
                    SELECT r.name, r.description, r.parameters,
                           (r.score + COALESCE(m.provider_score, 0) * $2)::FLOAT8 AS boosted_score
                    FROM real_ranked r
                    LEFT JOIN matched_providers m ON r.provider = m.provider
                )
                SELECT name, description, parameters, boosted_score AS rrf_score
                FROM boosted
                WHERE name NOT LIKE 'provider:%'
                ORDER BY boosted_score DESC LIMIT $3",
            )
            .bind(query)
            .bind(PROVIDER_BOOST_WEIGHT)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
            return Ok(rows.into_iter().map(|r| r.into_scored()).collect());
        }

        let embedding_vec = Vector::from(query_embedding);
        let fetch_limit = (limit * 2) as i64;
        let result_limit = limit as i64;

        // Real tools and provider rows are ranked in SEPARATE candidate sets
        // (FIX 1): the real-tool `vector_ranked` / `text_ranked` windows exclude
        // `provider:%`, so synthetic rows never consume a real-tool slot and a
        // plain query returns the full requested count of real tools. The
        // provider rows are ranked among THEMSELVES (a small set) with the same
        // RRF formula to compute a comparable `provider_score` for the boost.
        let rows: Vec<ToolSearchRow> = sqlx::query_as(
            "WITH real_chunk_distances AS (
                SELECT name, description, parameters, provider,
                       MIN(chunk <=> $1) AS min_distance
                FROM tool_definitions, unnest(embedding) AS chunk
                WHERE embedding IS NOT NULL AND name NOT LIKE 'provider:%'
                GROUP BY name, description, parameters, provider
            ),
            vector_ranked AS (
                SELECT name, description, parameters, provider,
                       ROW_NUMBER() OVER (ORDER BY min_distance) AS rank_v
                FROM real_chunk_distances
                LIMIT $2
            ),
            text_ranked AS (
                SELECT name, description, parameters, provider,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank_t
                FROM tool_definitions, plainto_tsquery('english', $3) query
                WHERE tsv @@ query AND name NOT LIKE 'provider:%'
                ORDER BY ts_rank_cd(tsv, query) DESC
                LIMIT $2
            ),
            fused AS (
                SELECT COALESCE(v.name, t.name) AS name,
                       COALESCE(v.description, t.description) AS description,
                       COALESCE(v.parameters, t.parameters) AS parameters,
                       COALESCE(v.provider, t.provider) AS provider,
                       (COALESCE(1.0 / (60 + v.rank_v), 0) +
                        COALESCE(1.0 / (60 + t.rank_t), 0))::FLOAT8 AS rrf_score
                FROM vector_ranked v
                FULL OUTER JOIN text_ranked t ON v.name = t.name
            ),
            provider_vector AS (
                SELECT name, provider, MIN(chunk <=> $1) AS min_distance
                FROM tool_definitions, unnest(embedding) AS chunk
                WHERE embedding IS NOT NULL AND name LIKE 'provider:%'
                GROUP BY name, provider
            ),
            provider_vector_ranked AS (
                SELECT name, provider,
                       ROW_NUMBER() OVER (ORDER BY min_distance) AS rank_v
                FROM provider_vector
            ),
            provider_text_ranked AS (
                SELECT name, provider,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank_t
                FROM tool_definitions, plainto_tsquery('english', $3) query
                WHERE tsv @@ query AND name LIKE 'provider:%'
            ),
            matched_providers AS (
                SELECT COALESCE(v.provider, t.provider) AS provider,
                       (COALESCE(1.0 / (60 + v.rank_v), 0) +
                        COALESCE(1.0 / (60 + t.rank_t), 0))::FLOAT8 AS provider_score
                FROM provider_vector_ranked v
                FULL OUTER JOIN provider_text_ranked t ON v.name = t.name
            ),
            boosted AS (
                SELECT f.name, f.description, f.parameters,
                       (f.rrf_score + COALESCE(m.provider_score, 0) * $5)::FLOAT8 AS boosted_score
                FROM fused f
                LEFT JOIN matched_providers m ON f.provider = m.provider
            )
            SELECT name, description, parameters, boosted_score AS rrf_score
            FROM boosted
            WHERE name NOT LIKE 'provider:%'
            ORDER BY boosted_score DESC LIMIT $4",
        )
        .bind(embedding_vec)
        .bind(fetch_limit)
        .bind(query)
        .bind(result_limit)
        .bind(PROVIDER_BOOST_WEIGHT)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(|r| r.into_scored()).collect())
    }
}

impl ToolRegistryStore for PgToolRegistryStore {
    async fn register_tools(
        &self,
        tools: Vec<ToolDefinition>,
        source: &str,
        is_core: bool,
        provider: Option<&str>,
        embeddings: Vec<Option<Vec<Vec<f32>>>>,
        embedding_model: Option<String>,
    ) -> Result<(), CoreError> {
        // Guard #4 (defense in depth): the `provider:*` name space is reserved
        // for the synthetic, non-routable provider rows. A batch may carry its
        // own provider's synthetic row (`provider:<provider>`); any other tool
        // literally named `provider:*` is refused so a real, dispatchable tool
        // can never masquerade as a provider row. Checked before opening the tx
        // so a rejected batch writes nothing.
        let own_synthetic = provider.map(|p| format!("provider:{p}"));
        for tool in &tools {
            if tool.name.starts_with("provider:") && Some(&tool.name) != own_synthetic.as_ref() {
                return Err(CoreError::Storage(format!(
                    "refusing to register reserved tool name '{}': the 'provider:' \
                     prefix is reserved for synthetic provider rows",
                    tool.name
                )));
            }
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        for (i, tool) in tools.iter().enumerate() {
            let embedding_vecs: Option<Vec<Vector>> = embeddings
                .get(i)
                .and_then(|e| e.clone())
                .map(|chunks| chunks.into_iter().map(Vector::from).collect());

            sqlx::query(
                "INSERT INTO tool_definitions (name, description, parameters, source, is_core, provider, embedding, embedding_model)
                 VALUES ($1, $2, $3, $4, $5, $6, $7::vector[], $8)
                 ON CONFLICT (name) DO UPDATE
                    SET description = EXCLUDED.description,
                        parameters = EXCLUDED.parameters,
                        source = EXCLUDED.source,
                        is_core = EXCLUDED.is_core,
                        provider = EXCLUDED.provider,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        registered_at = NOW()"
            )
            .bind(&tool.name)
            .bind(&tool.description)
            .bind(&tool.parameters)
            .bind(source)
            .bind(is_core)
            .bind(provider)
            .bind(&embedding_vecs)
            .bind(&embedding_model)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
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
            "SELECT name, description, parameters FROM tool_definitions WHERE is_core = TRUE",
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
        Ok(self
            .search_tools_scored(query, query_embedding, limit)
            .await?
            .into_iter()
            .map(|(def, _score)| def)
            .collect())
    }

    async fn tool_definition(&self, name: &str) -> Result<Option<ToolDefinition>, CoreError> {
        let row: Option<ToolRow> = sqlx::query_as(
            "SELECT name, description, parameters FROM tool_definitions WHERE name = $1",
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
    rrf_score: Option<f64>,
}

impl ToolSearchRow {
    /// The definition paired with its final (boosted) score. The score is never
    /// NULL in practice (it is a computed FLOAT8), so a NULL degrades to `0.0`.
    fn into_scored(self) -> (ToolDefinition, f64) {
        let score = self.rrf_score.unwrap_or(0.0);
        (
            ToolDefinition::new(self.name, self.description, self.parameters),
            score,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td(name: &str) -> ToolDefinition {
        ToolDefinition::new(name, "does things", serde_json::json!({}))
    }

    #[test]
    fn validate_reserved_names_rejects_foreign_provider_row() {
        // A batch registered under `mcp:weather` may carry its OWN synthetic row,
        // but a *different* `provider:*` row must be refused (guard #4). This is
        // the pure check `reindex_source` runs over every batch before opening a
        // transaction, so a rejected batch never runs the sweep.
        let tools = vec![td("weather__forecast"), td("provider:mcp:other")];
        let err = PgToolRegistryStore::validate_reserved_names(&tools, Some("mcp:weather"))
            .expect_err("a foreign provider row must be rejected");
        assert!(
            matches!(err, CoreError::Storage(_)),
            "reserved-name violations surface as CoreError::Storage"
        );
    }

    #[test]
    fn validate_reserved_names_accepts_own_synthetic_row() {
        let tools = vec![td("weather__forecast"), td("provider:mcp:weather")];
        PgToolRegistryStore::validate_reserved_names(&tools, Some("mcp:weather"))
            .expect("a batch may carry its own synthetic provider row");
    }

    #[test]
    fn validate_reserved_names_rejects_provider_row_when_unclassified() {
        // With no owning provider identity, ANY `provider:*` row is foreign.
        let tools = vec![td("provider:mcp:weather")];
        PgToolRegistryStore::validate_reserved_names(&tools, None)
            .expect_err("a provider row with no owning provider must be rejected");
    }
}
