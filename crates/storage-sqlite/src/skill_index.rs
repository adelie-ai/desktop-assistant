//! SQLite adapter for the host-global skill index (#594).
//!
//! The SQLite mirror of [`desktop_assistant_storage::PgSkillIndexStore`], behind
//! the same [`SkillIndexStore`] port. Search is **full-text only** (FTS5); there
//! is no vector column until sqlite-vec lands (#544 inc2), so the pre-computed
//! query embedding is ignored here. The FTS index stays in sync via the triggers
//! in migration `002_skill_index.sql` -- including on update, which the upsert
//! path relies on.
//!
//! Nothing here deletes: the catalog is cumulative (#639) and this adapter
//! implements storage primitives only. What accretes or is marked absent is
//! decided once in `core`'s reconcile pass, which is also what keeps this
//! adapter and the Postgres one from drifting apart.
//!
//! `last_seen_at` is stored as RFC 3339 text: this crate's `sqlx` build has no
//! `chrono` feature, so the conversion is explicit rather than implicit.
//!
//! Host-global like the Postgres table: no `user_id`/RLS; `owner_user_id` is
//! NULL for a global skill. All SQL is static with bound parameters — the FTS
//! `MATCH` string is a bound parameter built from sanitized query tokens.

use async_trait::async_trait;

use chrono::{DateTime, SecondsFormat, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{IndexedSkill, Locality, SkillKind, SkillScope, TrustTier};
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

    /// Insert or update one skill row, keyed on the unique `(name, owner_key)`
    /// index (the FTS triggers keep `skill_index_fts` in sync on both paths).
    /// JSON columns store `tags`/`attachments`/`metadata` as text.
    ///
    /// `seen_at` stamps `last_seen_at` and marks the row present: presence is
    /// index state derived from the scan that produced `skill`, never read off
    /// the argument.
    async fn upsert_row(
        conn: &mut sqlx::SqliteConnection,
        skill: &IndexedSkill,
        seen_at: DateTime<Utc>,
    ) -> Result<(), CoreError> {
        sqlx::query(
            "INSERT INTO skill_index \
                (name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                 trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                 last_seen_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?) \
             ON CONFLICT (name, owner_key) DO UPDATE SET \
                description = excluded.description, \
                kind = excluded.kind, \
                disk_path = excluded.disk_path, \
                locality = excluded.locality, \
                content_hash = excluded.content_hash, \
                trust_tier = excluded.trust_tier, \
                source = excluded.source, \
                tags = excluded.tags, \
                attachments = excluded.attachments, \
                body = excluded.body, \
                metadata = excluded.metadata, \
                present_on_disk = 1, \
                last_seen_at = excluded.last_seen_at, \
                indexed_at = datetime('now')",
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
        .bind(seen_at.to_rfc3339_opts(SecondsFormat::Secs, true))
        .execute(&mut *conn)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
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
    i64,            // present_on_disk (0/1)
    Option<String>, // last_seen_at (RFC 3339 text)
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
        present_on_disk: t.13 != 0,
        last_seen_at: t.14.as_deref().and_then(parse_ts),
    }
}

fn json_to_string_vec(s: &str) -> Vec<String> {
    serde_json::from_str(s).unwrap_or_default()
}

/// Decode a stored RFC 3339 timestamp, treating an unparseable value as absent
/// rather than failing the read -- a malformed stored value must not take down a
/// whole search, exactly like the JSON columns above.
fn parse_ts(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
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
    async fn upsert(&self, skill: &IndexedSkill, seen_at: DateTime<Utc>) -> Result<(), CoreError> {
        let mut conn = self
            .pool
            .acquire()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Self::upsert_row(&mut conn, skill, seen_at).await
    }

    async fn list_scope(&self, scope: &SkillScope) -> Result<Vec<IndexedSkill>, CoreError> {
        // Unfiltered by the calling user by design: the reconcile pass runs at
        // startup with no request scope and must see the whole partition it is
        // about to update. `owner_key` is the generated NULL -> '' mirror, so one
        // bound parameter addresses either scope.
        let rows: Vec<SkillTuple> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                    last_seen_at \
             FROM skill_index WHERE owner_key = ? ORDER BY name",
        )
        .bind(scope.owner().unwrap_or(""))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(row_from_tuple).collect())
    }

    async fn set_presence(
        &self,
        scope: &SkillScope,
        names: &[String],
        present: bool,
    ) -> Result<(), CoreError> {
        if names.is_empty() {
            return Ok(());
        }
        // SQLite has no array binding, so the names go through a JSON array and
        // `json_each` -- still one bound parameter, no SQL built from input.
        // Names absent from the scope match nothing, so a concurrent removal
        // cannot fail a reconcile. Nothing else on the row is touched,
        // `last_seen_at` included: it records when the skill was last on disk.
        let names_json =
            serde_json::to_string(names).map_err(|e| CoreError::Storage(e.to_string()))?;
        sqlx::query(
            "UPDATE skill_index SET present_on_disk = ? \
             WHERE owner_key = ? AND name IN (SELECT value FROM json_each(?))",
        )
        .bind(i64::from(present))
        .bind(scope.owner().unwrap_or(""))
        .bind(names_json)
        .execute(&self.pool)
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
                    s.metadata, s.present_on_disk, s.last_seen_at \
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
                    trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                    last_seen_at \
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
                    trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                    last_seen_at \
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
