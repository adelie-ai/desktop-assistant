//! Postgres adapter for the host-global skill index (#573).
//!
//! Mirrors [`crate::tool_registry`]: a hybrid vector + full-text (RRF) search
//! over a host-global table with no `user_id`/RLS. Two deliberate differences
//! from the tool registry's `reindex_source`:
//!
//! - **Reindex upserts and prunes** rather than delete-all-then-insert, so a
//!   skill unchanged since the last scan keeps its embedding instead of being
//!   re-embedded on every daemon boot (the scanner runs at startup).
//! - **Embeddings are preserved across a rescan iff the content hash is
//!   unchanged**; a content change (including any attachment) nulls the vector
//!   so [`crate::embedding_backfill::backfill_skill_embeddings`] re-embeds it.
//!
//! All SQL is static with bound parameters (no dynamic string building); the
//! only "search input" is the bound `$query` text and `$embedding` vector.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, TrustTier};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::skill_index::SkillIndexStore;
use pgvector::Vector;
use sqlx::PgPool;

/// Postgres-backed [`SkillIndexStore`].
pub struct PgSkillIndexStore {
    pool: PgPool,
}

impl PgSkillIndexStore {
    /// Construct a store over the given pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Upsert one skill, preserving the row's embedding when its content hash is
    /// unchanged and nulling it (for re-embedding) when the content changed.
    async fn upsert(conn: &mut sqlx::PgConnection, skill: &IndexedSkill) -> Result<(), CoreError> {
        sqlx::query(
            "INSERT INTO skill_index \
                (name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                 trust_tier, source, tags, attachments, body, metadata, embedding, embedding_model) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13, NULL, NULL) \
             ON CONFLICT (name, owner_key) DO UPDATE SET \
                description = EXCLUDED.description, \
                kind = EXCLUDED.kind, \
                disk_path = EXCLUDED.disk_path, \
                locality = EXCLUDED.locality, \
                content_hash = EXCLUDED.content_hash, \
                trust_tier = EXCLUDED.trust_tier, \
                source = EXCLUDED.source, \
                tags = EXCLUDED.tags, \
                attachments = EXCLUDED.attachments, \
                body = EXCLUDED.body, \
                metadata = EXCLUDED.metadata, \
                embedding = CASE \
                    WHEN skill_index.content_hash IS DISTINCT FROM EXCLUDED.content_hash \
                    THEN NULL ELSE skill_index.embedding END, \
                embedding_model = CASE \
                    WHEN skill_index.content_hash IS DISTINCT FROM EXCLUDED.content_hash \
                    THEN NULL ELSE skill_index.embedding_model END, \
                indexed_at = NOW()",
        )
        .bind(&skill.name)
        .bind(&skill.owner_user_id)
        .bind(&skill.description)
        .bind(skill.kind.as_str())
        .bind(&skill.disk_path)
        .bind(skill.locality.as_str())
        .bind(&skill.content_hash)
        .bind(skill.trust_tier.as_str())
        .bind(&skill.source)
        .bind(serde_json::json!(skill.tags))
        .bind(serde_json::json!(skill.attachments))
        .bind(&skill.body)
        .bind(&skill.metadata)
        .execute(&mut *conn)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn search_fts_only(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<IndexedSkill>, CoreError> {
        let user = current_user_id();
        let rows: Vec<SkillRow> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata \
             FROM skill_index \
             WHERE (owner_user_id IS NULL OR owner_user_id = $1) \
               AND tsv @@ plainto_tsquery('english', $2) \
             ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $2)) DESC \
             LIMIT $3",
        )
        .bind(user.as_str())
        .bind(query)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SkillRow::into_domain).collect())
    }

    async fn search_hybrid(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<IndexedSkill>, CoreError> {
        let user = current_user_id();
        let rows: Vec<SkillRow> = sqlx::query_as(
            "WITH scope AS ( \
                 SELECT * FROM skill_index \
                 WHERE (owner_user_id IS NULL OR owner_user_id = $1) \
             ), \
             vector_ranked AS ( \
                 SELECT name, owner_key, MIN(chunk <=> $2) AS dist \
                 FROM scope, unnest(embedding) AS chunk \
                 WHERE embedding IS NOT NULL \
                 GROUP BY name, owner_key \
             ), \
             vr AS ( \
                 SELECT name, owner_key, ROW_NUMBER() OVER (ORDER BY dist) AS rank_v \
                 FROM vector_ranked LIMIT $4 \
             ), \
             tr AS ( \
                 SELECT name, owner_key, \
                        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank_t \
                 FROM scope, plainto_tsquery('english', $3) query \
                 WHERE tsv @@ query \
                 ORDER BY ts_rank_cd(tsv, query) DESC LIMIT $4 \
             ), \
             fused AS ( \
                 SELECT COALESCE(vr.name, tr.name) AS name, \
                        COALESCE(vr.owner_key, tr.owner_key) AS owner_key, \
                        (COALESCE(1.0 / (60 + vr.rank_v), 0) \
                         + COALESCE(1.0 / (60 + tr.rank_t), 0))::float8 AS score \
                 FROM vr FULL OUTER JOIN tr \
                   ON vr.name = tr.name AND vr.owner_key = tr.owner_key \
             ) \
             SELECT s.name, s.owner_user_id, s.description, s.kind, s.disk_path, s.locality, \
                    s.content_hash, s.trust_tier, s.source, s.tags, s.attachments, s.body, \
                    s.metadata \
             FROM fused f JOIN scope s ON s.name = f.name AND s.owner_key = f.owner_key \
             ORDER BY f.score DESC LIMIT $5",
        )
        .bind(user.as_str())
        .bind(Vector::from(query_embedding))
        .bind(query)
        .bind((limit * 2) as i64)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SkillRow::into_domain).collect())
    }
}

#[async_trait::async_trait]
impl SkillIndexStore for PgSkillIndexStore {
    async fn reindex_global(&self, skills: Vec<IndexedSkill>) -> Result<(), CoreError> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        for skill in &skills {
            Self::upsert(&mut tx, skill).await?;
        }

        // Prune global rows no longer present on disk. An empty scan clears the
        // whole global catalog (every name fails `<> ALL('{}')` -> nothing kept).
        let names: Vec<String> = skills.iter().map(|s| s.name.clone()).collect();
        sqlx::query("DELETE FROM skill_index WHERE owner_user_id IS NULL AND name <> ALL($1)")
            .bind(&names)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn reindex_for_owner(
        &self,
        owner: &str,
        skills: Vec<IndexedSkill>,
    ) -> Result<(), CoreError> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        for skill in &skills {
            Self::upsert(&mut tx, skill).await?;
        }

        // Prune this owner's rows no longer present on disk (global and other
        // users' rows are untouched). An empty scan clears the owner's catalog.
        let names: Vec<String> = skills.iter().map(|s| s.name.clone()).collect();
        sqlx::query("DELETE FROM skill_index WHERE owner_user_id = $1 AND name <> ALL($2)")
            .bind(owner)
            .bind(&names)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<IndexedSkill>, CoreError> {
        // Empty embedding (backend down/unavailable) -> full-text only, exactly
        // like the knowledge-base search. A zero-dim vector would also make the
        // `<=>` operator error, so this branch is required, not just an
        // optimization.
        if query_embedding.is_empty() {
            self.search_fts_only(query, limit).await
        } else {
            self.search_hybrid(query, query_embedding, limit).await
        }
    }

    async fn get(
        &self,
        name: &str,
        owner: Option<&str>,
    ) -> Result<Option<IndexedSkill>, CoreError> {
        let row: Option<SkillRow> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata \
             FROM skill_index \
             WHERE name = $1 AND owner_key = COALESCE($2, '')",
        )
        .bind(name)
        .bind(owner)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(row.map(SkillRow::into_domain))
    }

    async fn list(&self, limit: Option<u32>) -> Result<Vec<IndexedSkill>, CoreError> {
        let user = current_user_id();
        let rows: Vec<SkillRow> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata \
             FROM skill_index \
             WHERE (owner_user_id IS NULL OR owner_user_id = $1) \
             ORDER BY indexed_at DESC LIMIT $2",
        )
        .bind(user.as_str())
        .bind(limit.map(i64::from).unwrap_or(i64::MAX))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SkillRow::into_domain).collect())
    }
}

/// A row read from `skill_index`, decoded straight from the projected columns.
#[derive(sqlx::FromRow)]
struct SkillRow {
    name: String,
    owner_user_id: Option<String>,
    description: String,
    kind: String,
    disk_path: String,
    locality: String,
    content_hash: String,
    trust_tier: String,
    source: Option<String>,
    tags: serde_json::Value,
    attachments: serde_json::Value,
    body: String,
    metadata: serde_json::Value,
}

impl SkillRow {
    fn into_domain(self) -> IndexedSkill {
        IndexedSkill {
            name: self.name,
            description: self.description,
            kind: SkillKind::from_db(&self.kind),
            disk_path: self.disk_path,
            owner_user_id: self.owner_user_id,
            locality: Locality::from_db(&self.locality),
            content_hash: self.content_hash,
            trust_tier: TrustTier::from_db(&self.trust_tier),
            source: self.source,
            tags: json_to_string_vec(self.tags),
            attachments: json_to_string_vec(self.attachments),
            body: self.body,
            metadata: self.metadata,
        }
    }
}

/// Decode a JSONB array column into `Vec<String>`, defaulting to empty on any
/// shape mismatch (a malformed stored value must not fail a whole search).
fn json_to_string_vec(v: serde_json::Value) -> Vec<String> {
    serde_json::from_value(v).unwrap_or_default()
}
