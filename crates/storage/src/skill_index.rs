//! Postgres adapter for the host-global skill index (#573).
//!
//! Mirrors [`crate::tool_registry`]: a hybrid vector + full-text (RRF) search
//! over a host-global table with no `user_id`/RLS. Two deliberate differences
//! from the tool registry's `reindex_source`:
//!
//! - **Nothing here deletes.** The catalog is cumulative (#639): this adapter
//!   implements storage primitives, and what accretes or is marked absent is
//!   decided once in `core`'s reconcile pass, not in SQL here.
//! - **Embeddings are preserved across a rescan iff the content hash is
//!   unchanged**; a content change (including any attachment) nulls the vector
//!   so [`crate::embedding_backfill::backfill_skill_embeddings`] re-embeds it.
//!   This is the one behavior genuinely local to this adapter -- SQLite has no
//!   vector column -- so it is tested here rather than in the shared contract.
//!
//! Approval (#1155) is a third column pair (`approved_at`/`approved_by`),
//! orthogonal to `trust_tier`'s provenance: `upsert_row` honours it on insert
//! and preserves it on update, while `write_authored_row` forces it cleared on
//! both branches. See [`SkillIndexStore::upsert`] and
//! [`SkillIndexStore::write_authored`] for why.
//!
//! All SQL is static with bound parameters (no dynamic string building); the
//! only "search input" is the bound `$query` text and `$embedding` vector.

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::{
    IndexedSkill, Locality, SkillApproval, SkillKind, SkillScope, TrustTier,
};
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
        conn: &mut sqlx::PgConnection,
        skill: &IndexedSkill,
        seen_at: DateTime<Utc>,
    ) -> Result<(), CoreError> {
        sqlx::query(
            "INSERT INTO skill_index \
                (name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                 trust_tier, source, tags, attachments, body, metadata, embedding, embedding_model, \
                 present_on_disk, last_seen_at, approved_at, approved_by) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13, NULL, NULL, TRUE, $14, $15, $16) \
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
                present_on_disk = TRUE, \
                last_seen_at = EXCLUDED.last_seen_at, \
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
        .bind(seen_at)
        .bind(skill.approved_at)
        .bind(&skill.approved_by)
        .execute(&mut *conn)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Insert or update a skill the assistant authored from a completed plan
    /// (#1155), keyed on `(name, owner_key)` like [`Self::upsert_row`].
    ///
    /// Both branches force `present_on_disk = FALSE`, `approved_at = NULL`
    /// and `approved_by = NULL`: nothing was read off disk, and unattended
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
    /// deterministic under test, per `crate::clock`'s convention.
    ///
    /// Embedding retention mirrors [`Self::upsert_row`]: preserved when
    /// `content_hash` is unchanged, nulled for re-embedding when it changes.
    async fn write_authored_row(
        conn: &mut sqlx::PgConnection,
        skill: &IndexedSkill,
        authored_at: DateTime<Utc>,
    ) -> Result<(), CoreError> {
        sqlx::query(
            "INSERT INTO skill_index \
                (name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                 trust_tier, source, tags, attachments, body, metadata, embedding, embedding_model, \
                 present_on_disk, last_seen_at, approved_at, approved_by, indexed_at) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13, NULL, NULL, FALSE, $14, NULL, NULL, $15) \
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
                present_on_disk = FALSE, \
                approved_at = NULL, \
                approved_by = NULL, \
                indexed_at = EXCLUDED.indexed_at",
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
        .bind(skill.last_seen_at)
        .bind(authored_at)
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
                    trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                    last_seen_at, approved_at, approved_by \
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
        embedding_model: &str,
        limit: usize,
    ) -> Result<Vec<IndexedSkill>, CoreError> {
        let user = current_user_id();
        // $6 = the model that produced $2. Only rows embedded by that model can
        // be compared against it (a cross-model comparison is a cross-dimension
        // comparison, which raises rather than missing), so the predicate sits
        // on `vector_ranked` alone -- `tr` below stays model-blind so a model
        // change degrades to full-text recall instead of hiding skills.
        //
        // Sameness is decided on the digest half of the `<name>@<digest>` stamp
        // wherever both sides carry one, matching
        // `embedding_backfill::invalidate_stale_embeddings`, so a cosmetic
        // rename does not blank search until the sweep restamps the rows.
        //
        // `vr` and `tr` both carry an explicit `ORDER BY` before their `LIMIT`,
        // and both are load-bearing rather than decorative (#1107). `ORDER BY`
        // inside `OVER (…)` orders the window computation, not the statement's
        // output, so a `LIMIT` with no statement-level order truncates an
        // undefined set: the arm still returns rows and the fusion still ranks
        // them, so the caller gets a plausible page that quietly omits the
        // best matches.
        //
        // The final `ORDER BY f.score DESC, s.name ASC, s.owner_key ASC` is
        // the same defect one level out: RRF ties exactly by construction (a
        // row found by only one arm at rank 1 scores exactly `1/(60+1)`,
        // whichever arm found it), so `f.score` alone would leave the final
        // `LIMIT` truncation undefined between tied skills. This table has no
        // single surrogate id column -- its uniqueness is the composite
        // `(name, owner_key)` (`idx_skill_index_name_owner`) -- so that
        // composite, not a recency column, is the natural total-order
        // tiebreak; a skill's catalog `indexed_at` reflects when the last
        // reconcile scan touched it, not the search-relevant recency
        // `updated_at`/`id` give the other three tables.
        let rows: Vec<SkillRow> = sqlx::query_as(
            "WITH scope AS ( \
                 SELECT * FROM skill_index \
                 WHERE (owner_user_id IS NULL OR owner_user_id = $1) \
             ), \
             vector_ranked AS ( \
                 SELECT name, owner_key, MIN(chunk <=> $2) AS dist \
                 FROM scope, unnest(embedding) AS chunk \
                 WHERE embedding IS NOT NULL \
                   AND embedding_model IS NOT NULL \
                   AND (embedding_model = $6 \
                        OR (split_part($6, '@', 2) <> '' \
                            AND split_part(embedding_model, '@', 2) \
                                = split_part($6, '@', 2))) \
                 GROUP BY name, owner_key \
             ), \
             vr AS ( \
                 SELECT name, owner_key, ROW_NUMBER() OVER (ORDER BY dist) AS rank_v \
                 FROM vector_ranked ORDER BY dist LIMIT $4 \
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
                    s.metadata, s.present_on_disk, s.last_seen_at, s.approved_at, s.approved_by \
             FROM fused f JOIN scope s ON s.name = f.name AND s.owner_key = f.owner_key \
             ORDER BY f.score DESC, s.name ASC, s.owner_key ASC LIMIT $5",
        )
        .bind(user.as_str())
        .bind(Vector::from(query_embedding))
        .bind(query)
        .bind((limit * 2) as i64)
        .bind(limit as i64)
        .bind(embedding_model)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SkillRow::into_domain).collect())
    }
}

#[async_trait::async_trait]
impl SkillIndexStore for PgSkillIndexStore {
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

    async fn set_approval(
        &self,
        scope: &SkillScope,
        names: &[String],
        approval: Option<SkillApproval>,
    ) -> Result<(), CoreError> {
        if names.is_empty() {
            return Ok(());
        }
        // `Some` approves, `None` withdraws -- unpacked here rather than
        // bound as a whole, since `SkillApproval` carries no SQL encoding of
        // its own and the two columns it wraps are independently nullable.
        let (approved_at, approved_by) = match approval {
            Some(a) => (Some(a.at), a.by),
            None => (None, None),
        };
        // Names absent from the scope simply match nothing -- the same
        // tolerance `set_presence` has, and for the same reason.
        sqlx::query(
            "UPDATE skill_index SET approved_at = $3, approved_by = $4 \
             WHERE owner_key = $1 AND name = ANY($2)",
        )
        .bind(scope.owner().unwrap_or(""))
        .bind(names)
        .bind(approved_at)
        .bind(approved_by)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
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
             FROM skill_index WHERE owner_key = $1 ORDER BY name",
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
        // Names absent from the scope simply match nothing -- a concurrent
        // removal must not fail a reconcile. Nothing else on the row is touched,
        // `last_seen_at` included: it records when the skill was last on disk.
        sqlx::query(
            "UPDATE skill_index SET present_on_disk = $3 \
             WHERE owner_key = $1 AND name = ANY($2)",
        )
        .bind(scope.owner().unwrap_or(""))
        .bind(names)
        .bind(present)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        query: &str,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        limit: usize,
    ) -> Result<Vec<IndexedSkill>, CoreError> {
        // Empty embedding (backend down/unavailable) -> full-text only, exactly
        // like the knowledge-base search. A zero-dim vector would also make the
        // `<=>` operator error, so this branch is required, not just an
        // optimization.
        if query_embedding.is_empty() {
            self.search_fts_only(query, limit).await
        } else {
            self.search_hybrid(query, query_embedding, embedding_model, limit)
                .await
        }
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
             FROM skill_index \
             WHERE name = $1 AND owner_key = COALESCE($2, '')",
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
        let rows: Vec<SkillRow> = sqlx::query_as(
            "SELECT name, owner_user_id, description, kind, disk_path, locality, content_hash, \
                    trust_tier, source, tags, attachments, body, metadata, present_on_disk, \
                    last_seen_at, approved_at, approved_by \
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
    present_on_disk: bool,
    last_seen_at: Option<DateTime<Utc>>,
    approved_at: Option<DateTime<Utc>>,
    approved_by: Option<String>,
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
            present_on_disk: self.present_on_disk,
            last_seen_at: self.last_seen_at,
            approved_at: self.approved_at,
            approved_by: self.approved_by,
        }
    }
}

/// Decode a JSONB array column into `Vec<String>`, defaulting to empty on any
/// shape mismatch (a malformed stored value must not fail a whole search).
fn json_to_string_vec(v: serde_json::Value) -> Vec<String> {
    serde_json::from_value(v).unwrap_or_default()
}
