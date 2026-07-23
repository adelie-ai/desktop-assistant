//! SQLite adapter for the host-global skill index (#594).
//!
//! The SQLite mirror of [`desktop_assistant_storage::PgSkillIndexStore`], behind
//! the same [`SkillIndexStore`] port. Search is **full-text only** (FTS5); there
//! is no vector column until sqlite-vec lands (#544 inc2), so the pre-computed
//! query embedding is ignored here. `reindex_global` uses delete-then-insert
//! (there is no embedding to preserve), and the FTS index stays in sync via the
//! triggers in migration `002_skill_index.sql`.
//!
//! Host-global like the Postgres table: no `user_id`/RLS; `owner_user_id` is
//! NULL for a global skill. All SQL is static with bound parameters — the FTS
//! `MATCH` string is a bound parameter built from sanitized query tokens.

use async_trait::async_trait;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, TrustTier};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::skill_index::SkillIndexStore;
use sqlx::SqlitePool;

/// SQLite adapter for the `skill_index` table.
pub struct SqliteSkillIndexStore {
    pool: SqlitePool,
}

impl SqliteSkillIndexStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

/// A `skill_index` row decoded as a positional tuple (SQLite adapters map into
/// tuples then a free function, rather than `FromRow`).
type SkillTuple = (
    String,         // name
    Option<String>, // owner_user_id
    String,         // description
    String,         // kind
    String,         // disk_path
    String,         // locality
    String,         // content_hash
    String,         // trust_tier
    Option<String>, // source
    String,         // tags (JSON text)
    String,         // attachments (JSON text)
    String,         // body
    String,         // metadata (JSON text)
);

fn row_from_tuple(t: SkillTuple) -> IndexedSkill {
    IndexedSkill {
        name: t.0,
        owner_user_id: t.1,
        description: t.2,
        kind: SkillKind::from_db(&t.3),
        disk_path: t.4,
        locality: Locality::from_db(&t.5),
        content_hash: t.6,
        trust_tier: TrustTier::from_db(&t.7),
        source: t.8,
        tags: json_to_string_vec(&t.9),
        attachments: json_to_string_vec(&t.10),
        body: t.11,
        metadata: serde_json::from_str(&t.12).unwrap_or(serde_json::Value::Null),
    }
}

fn json_to_string_vec(s: &str) -> Vec<String> {
    serde_json::from_str(s).unwrap_or_default()
}

/// Build an FTS5 `MATCH` string from a free-text query: sanitized tokens quoted
/// as string literals and OR'd for recall. Returns `None` when the query has no
/// usable token, so the caller returns no results rather than issuing an
/// invalid `MATCH`.
fn fts_match(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|tok| {
            tok.chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
        })
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{t}\""))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

#[async_trait]
impl SkillIndexStore for SqliteSkillIndexStore {
    async fn reindex_global(&self, skills: Vec<IndexedSkill>) -> Result<(), CoreError> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // No embedding to preserve on SQLite, so a plain delete-then-insert is
        // simplest; the FTS triggers keep `skill_index_fts` in sync.
        sqlx::query("DELETE FROM skill_index WHERE owner_user_id IS NULL")
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        for skill in &skills {
            sqlx::query(
                "INSERT INTO skill_index \
                    (name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                     trust_tier, source, tags, attachments, body, metadata) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            .bind(serde_json::to_string(&skill.tags).unwrap_or_else(|_| "[]".into()))
            .bind(serde_json::to_string(&skill.attachments).unwrap_or_else(|_| "[]".into()))
            .bind(&skill.body)
            .bind(serde_json::to_string(&skill.metadata).unwrap_or_else(|_| "{}".into()))
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        query: &str,
        _query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<IndexedSkill>, CoreError> {
        // Full-text only: the SQLite adapter has no vector column yet, so the
        // embedding is ignored.
        let Some(match_query) = fts_match(query) else {
            return Ok(Vec::new());
        };
        let user = current_user_id();
        let rows: Vec<SkillTuple> = sqlx::query_as(
            "SELECT s.name, s.owner_user_id, s.description, s.kind, s.disk_path, s.locality, \
                    s.content_hash, s.trust_tier, s.source, s.tags, s.attachments, s.body, \
                    s.metadata \
             FROM skill_index s JOIN skill_index_fts f ON f.rowid = s.id \
             WHERE skill_index_fts MATCH ? \
               AND (s.owner_user_id IS NULL OR s.owner_user_id = ?) \
             ORDER BY bm25(skill_index_fts) \
             LIMIT ?",
        )
        .bind(match_query)
        .bind(user.as_str())
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(row_from_tuple).collect())
    }

    async fn get(
        &self,
        name: &str,
        owner: Option<&str>,
    ) -> Result<Option<IndexedSkill>, CoreError> {
        let row: Option<SkillTuple> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata \
             FROM skill_index WHERE name = ? AND owner_key = ifnull(?, '')",
        )
        .bind(name)
        .bind(owner)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(row.map(row_from_tuple))
    }

    async fn list(&self, limit: Option<u32>) -> Result<Vec<IndexedSkill>, CoreError> {
        let user = current_user_id();
        // SQLite treats LIMIT -1 as "no limit".
        let lim = limit.map(i64::from).unwrap_or(-1);
        let rows: Vec<SkillTuple> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata \
             FROM skill_index \
             WHERE owner_user_id IS NULL OR owner_user_id = ? \
             ORDER BY indexed_at DESC, id DESC LIMIT ?",
        )
        .bind(user.as_str())
        .bind(lim)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(row_from_tuple).collect())
    }
}
