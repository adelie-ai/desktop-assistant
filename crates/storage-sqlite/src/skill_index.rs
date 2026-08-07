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
//! `approved_at` follows the same convention.
//!
//! Approval (#1155) is a third column pair (`approved_at`/`approved_by`),
//! orthogonal to `trust_tier`'s provenance: [`upsert_row`](Self::upsert_row)
//! honours it on insert and preserves it on update, while
//! [`write_authored_row`](Self::write_authored_row) forces it cleared on both
//! branches. See [`SkillIndexStore::upsert`] and
//! [`SkillIndexStore::write_authored`] for why.
//!
//! Host-global like the Postgres table: no `user_id`/RLS; `owner_user_id` is
//! NULL for a global skill. All SQL is static with bound parameters — the FTS
//! `MATCH` string is a bound parameter built from sanitized query tokens.

use async_trait::async_trait;

use chrono::{DateTime, SecondsFormat, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    IndexedSkill, Locality, SkillApproval, SkillKind, SkillScope, TrustTier,
};
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
    ///
    /// Approval (`approved_at`/`approved_by`) is honoured on insert -- a
    /// first-seen scan is how a skill records "a person put this file in a
    /// skill root" (#1155) -- but the `ON CONFLICT` `SET` list below
    /// deliberately omits both columns, so an update leaves the stored
    /// approval exactly where it was. A rescan re-reads a file; it does not
    /// re-decide whether a person consented to it.
    async fn upsert_row(
        conn: &mut sqlx::SqliteConnection,
        skill: &IndexedSkill,
        seen_at: DateTime<Utc>,
    ) -> Result<(), CoreError> {
        sqlx::query(
            "INSERT INTO skill_index \
                (name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                 trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                 last_seen_at, approved_at, approved_by) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?) \
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
        .bind(to_text(seen_at))
        .bind(skill.approved_at.map(to_text))
        .bind(&skill.approved_by)
        .execute(&mut *conn)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Insert or update a skill the assistant authored from a completed plan
    /// (#1155), keyed on `(name, owner_key)` like [`Self::upsert_row`].
    ///
    /// Both branches force `present_on_disk = 0`, `approved_at = NULL` and
    /// `approved_by = NULL`: nothing was read off disk, and unattended
    /// authoring records no consent, so the row cannot wear either claim
    /// whatever the caller's argument says (see
    /// [`SkillIndexStore::write_authored`]'s doc for why that has to be
    /// forced here rather than trusted from the caller). `last_seen_at` is
    /// deliberately absent from the `ON CONFLICT` `SET` list, so an amend
    /// leaves it exactly as the last scan (if any) left it -- nothing here was
    /// scanned, so there is nothing to mark freshly seen.
    ///
    /// `authored_at` stamps `indexed_at` on both branches, in place of the
    /// wall clock, mirroring how [`Self::upsert_row`]'s `seen_at` stamps
    /// `last_seen_at`: the instant is injected so a caller's write is
    /// deterministic under test, per `desktop_assistant_core::clock`'s
    /// convention.
    async fn write_authored_row(
        conn: &mut sqlx::SqliteConnection,
        skill: &IndexedSkill,
        authored_at: DateTime<Utc>,
    ) -> Result<(), CoreError> {
        sqlx::query(
            "INSERT INTO skill_index \
                (name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                 trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                 last_seen_at, approved_at, approved_by, indexed_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, NULL, ?) \
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
                present_on_disk = 0, \
                approved_at = NULL, \
                approved_by = NULL, \
                indexed_at = excluded.indexed_at",
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
        .bind(skill.last_seen_at.map(to_text))
        .bind(to_text(authored_at))
        .execute(&mut *conn)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }
}

/// Encode a UTC instant the same way `last_seen_at`/`approved_at` are stored:
/// RFC 3339 text, seconds precision.
fn to_text(dt: DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(SecondsFormat::Secs, true)
}

/// A `skill_index` row, decoded via `#[derive(sqlx::FromRow)]` (matching
/// `conversation.rs` elsewhere in this crate) rather than the positional
/// tuple this file used before #1155: sqlx's tuple `FromRow` impl tops out at
/// 16 elements, and the two approval columns push this row to 17.
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
    tags: String,        // JSON text
    attachments: String, // JSON text
    body: String,
    metadata: String,             // JSON text
    present_on_disk: i64,         // 0/1
    last_seen_at: Option<String>, // RFC 3339 text
    approved_at: Option<String>,  // RFC 3339 text
    approved_by: Option<String>,
}

impl SkillRow {
    fn into_domain(self) -> IndexedSkill {
        IndexedSkill {
            name: self.name,
            owner_user_id: self.owner_user_id,
            description: self.description,
            kind: SkillKind::from_db(&self.kind),
            disk_path: self.disk_path,
            locality: Locality::from_db(&self.locality),
            content_hash: self.content_hash,
            trust_tier: TrustTier::from_db(&self.trust_tier),
            source: self.source,
            tags: json_to_string_vec(&self.tags),
            attachments: json_to_string_vec(&self.attachments),
            body: self.body,
            metadata: serde_json::from_str(&self.metadata).unwrap_or(serde_json::Value::Null),
            present_on_disk: self.present_on_disk != 0,
            last_seen_at: self.last_seen_at.as_deref().and_then(parse_ts),
            approved_at: self.approved_at.as_deref().and_then(parse_ts),
            approved_by: self.approved_by,
        }
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

    async fn write_authored(
        &self,
        skill: &IndexedSkill,
        authored_at: DateTime<Utc>,
    ) -> Result<(), CoreError> {
        let mut conn = self
            .pool
            .acquire()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Self::write_authored_row(&mut conn, skill, authored_at).await
    }

    async fn list_scope(&self, scope: &SkillScope) -> Result<Vec<IndexedSkill>, CoreError> {
        // Unfiltered by the calling user by design: the reconcile pass runs at
        // startup with no request scope and must see the whole partition it is
        // about to update. `owner_key` is the generated NULL -> '' mirror, so one
        // bound parameter addresses either scope.
        let rows: Vec<SkillRow> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                    last_seen_at, approved_at, approved_by \
             FROM skill_index WHERE owner_key = ? ORDER BY name",
        )
        .bind(scope.owner().unwrap_or(""))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SkillRow::into_domain).collect())
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

    async fn set_approval(
        &self,
        scope: &SkillScope,
        names: &[String],
        approval: Option<SkillApproval>,
    ) -> Result<(), CoreError> {
        if names.is_empty() {
            return Ok(());
        }
        // Mirrors `set_presence` exactly in shape: names go through a JSON
        // array and `json_each` -- still one bound parameter, no SQL built
        // from input. Names absent from the scope match nothing, and nothing
        // else on the row is touched.
        let names_json =
            serde_json::to_string(names).map_err(|e| CoreError::Storage(e.to_string()))?;
        let approved_at = approval.as_ref().map(|a| to_text(a.at));
        let approved_by = approval.as_ref().and_then(|a| a.by.clone());
        sqlx::query(
            "UPDATE skill_index SET approved_at = ?, approved_by = ? \
             WHERE owner_key = ? AND name IN (SELECT value FROM json_each(?))",
        )
        .bind(approved_at)
        .bind(approved_by)
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
        _embedding_model: &str,
        limit: usize,
    ) -> Result<Vec<IndexedSkill>, CoreError> {
        // Full-text only: the SQLite adapter has no vector column yet, so the
        // embedding and the model that produced it are both ignored.
        let Some(match_query) = fts_match(query) else {
            return Ok(Vec::new());
        };
        let user = current_user_id();
        let rows: Vec<SkillRow> = sqlx::query_as(
            "SELECT s.name, s.owner_user_id, s.description, s.kind, s.disk_path, s.locality, \
                    s.content_hash, s.trust_tier, s.source, s.tags, s.attachments, s.body, \
                    s.metadata, s.present_on_disk, s.last_seen_at, s.approved_at, s.approved_by \
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
        Ok(rows.into_iter().map(SkillRow::into_domain).collect())
    }

    async fn get(
        &self,
        name: &str,
        owner: Option<&str>,
    ) -> Result<Option<IndexedSkill>, CoreError> {
        // `owner` only distinguishes "give me the global one" (None) from
        // "give me mine" (Some) -- the string it carries is untrusted (an
        // LLM-forwarded MCP tool argument, #911) and is never used to
        // address another user's row. `Some(_)` always resolves to
        // `current_user_id()`, the same source `search`/`list` scope to, so
        // a caller naming a different user's id gets a silent miss rather
        // than that user's skill.
        let user = current_user_id();
        let owner_key = owner.map(|_| user.as_str());
        let row: Option<SkillRow> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                    last_seen_at, approved_at, approved_by \
             FROM skill_index WHERE name = ? AND owner_key = ifnull(?, '')",
        )
        .bind(name)
        .bind(owner_key)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(row.map(SkillRow::into_domain))
    }

    async fn list(&self, limit: Option<u32>) -> Result<Vec<IndexedSkill>, CoreError> {
        let user = current_user_id();
        // SQLite treats LIMIT -1 as "no limit".
        let lim = limit.map(i64::from).unwrap_or(-1);
        let rows: Vec<SkillRow> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                    last_seen_at, approved_at, approved_by \
             FROM skill_index \
             WHERE owner_user_id IS NULL OR owner_user_id = ? \
             ORDER BY indexed_at DESC, id DESC LIMIT ?",
        )
        .bind(user.as_str())
        .bind(lim)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SkillRow::into_domain).collect())
    }
}
