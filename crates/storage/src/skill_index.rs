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
use desktop_assistant_core::ports::recall::RecallDispersion;
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

    /// The skills nearest a prompt embedding, nearest first, with what a
    /// distance from this catalog is worth (#1154).
    ///
    /// The read behind the `[Recall]` block's skill arm, and deliberately not a
    /// [`SkillIndexStore`] method: `vector[]` is a Postgres column with no
    /// SQLite counterpart, and the shared contract would then hold a method one
    /// adapter cannot answer.
    ///
    /// **Only approved skills take part, in the candidates and in the spread.**
    /// A skill nobody has consented to cannot be followed - `builtin_skill_get`
    /// refuses its body - so offering one would put a line in front of the
    /// model that it can only fail on, and would accrue an offer every turn and
    /// never an open. Filtering inside the scan also keeps the spread a
    /// statement about the catalog the arm actually draws from.
    ///
    /// **Only a locally authored skill is offered.** A skill from a GitHub or
    /// `.well-known` source carries a description its author wrote, and the
    /// platform already rules that such text is third-party content:
    /// `builtin_skill_search` returns the same field and is classified
    /// `Declared(SkillTrustTier)`, so a non-local hit taints the turn and
    /// closes the tool gate. This block has no tool call in it, so nothing
    /// would taint - the text would land in a system message, ahead of the
    /// user prompt, with every tier still open. Dropping is the answer rather
    /// than tainting, for the reason the scratchpad arm drops a note stamped
    /// as external: a catalog row lives indefinitely, and closing the gate
    /// whenever one happened to rank near the prompt would degrade the
    /// conversation permanently. An installed skill stays reachable through
    /// `builtin_skill_search`, which taints correctly.
    ///
    /// The predicate sits on the final join rather than inside `d`, so it
    /// applies after a name has resolved to one row. Filtering earlier would
    /// let a local global skill be offered while the fetch returned the
    /// non-local personal one that shadows it.
    ///
    /// **One row per name, and it is the row the fetch returns.** The catalog
    /// can hold a global skill and this user's own under one name. Two lines
    /// for one openable procedure would be two lines the model cannot tell
    /// apart, and a line describing a procedure other than the one
    /// `builtin_skill_get` hands back would be worse - the model would be
    /// briefed on one method and given another's steps. So `pick` applies that
    /// tool's own rule: the user's own row when its files are on disk, else
    /// the global one, else the user's own tombstone.
    ///
    /// **The body is not read.** The arm renders a name and one line of what
    /// the skill is for; the body is the widest column on the row and nothing
    /// downstream of this call looks at it.
    ///
    /// An empty `query_embedding` yields no rows and no spread: the vector
    /// operator raises on a zero-dimension vector, and the caller with no
    /// embedding has [`Self::search_text_any_term`] to fall back to.
    ///
    /// `embedding_model` scopes the comparison the same way the store's own
    /// `search_hybrid` does, and for the same reason: a comparison across
    /// models is a comparison across vector dimensions.
    ///
    /// The scan carries [`SKILL_RECALL_SCAN_STATEMENT_TIMEOUT`], so the
    /// database stops working when the caller stops waiting.
    pub async fn nearest_by_embedding(
        &self,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        limit: usize,
    ) -> Result<NearestSkills, CoreError> {
        if query_embedding.is_empty() {
            return Ok(NearestSkills::default());
        }
        let user = current_user_id();

        // A transaction, for one statement, because `SET LOCAL` is scoped to
        // one: it is what makes the ceiling the caller keeps a ceiling the
        // database keeps too.
        let mut scan = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        sqlx::query("SELECT set_config('statement_timeout', $1, true)")
            .bind(SKILL_RECALL_SCAN_STATEMENT_TIMEOUT.as_millis().to_string())
            .execute(&mut *scan)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows: Vec<SkillNearestRow> = sqlx::query_as(NEAREST_SKILLS_BY_EMBEDDING_SQL)
            .bind(Vector::from(query_embedding))
            .bind(user.as_str())
            .bind(embedding_model)
            .bind(limit as i64)
            .fetch_all(&mut *scan)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        scan.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Every row carries the same spread, so the first one states it.
        let dispersion = rows.first().and_then(SkillNearestRow::dispersion);
        Ok(NearestSkills {
            skills: rows
                .into_iter()
                .map(SkillNearestRow::into_candidate)
                .collect(),
            dispersion,
        })
    }

    /// Approved skills carrying **any** of a prompt's terms, best match first
    /// (#1154).
    ///
    /// The degraded arm of the block's skill lookup, on the same terms as
    /// [`crate::PgKnowledgeBaseStore::search_text_any_term`]: the store's own
    /// `search` joins a query's lexemes with `AND`, which is right for a
    /// model-authored query of two or three words and wrong for a whole user
    /// sentence. A fallback that answers nothing is not a fallback.
    ///
    /// The same approval filter and the same one-row-per-name rule as
    /// [`Self::nearest_by_embedding`], for the same reasons.
    pub async fn search_text_any_term(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<NearestSkill>, CoreError> {
        let user = current_user_id();
        let rows: Vec<SkillTextRow> = sqlx::query_as(
            "WITH q AS (
                 SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS query
                 FROM unnest(to_tsvector('english', $1))
             ),
             matched AS (
                 SELECT DISTINCT ON (s.name)
                        s.name, s.description, s.present_on_disk, s.trust_tier,
                        ts_rank_cd(s.tsv, q.query) AS rank
                 FROM skill_index s, q
                 WHERE (s.owner_user_id IS NULL OR s.owner_user_id = $3)
                   AND s.approved_at IS NOT NULL
                   AND q.query IS NOT NULL
                   AND s.tsv @@ q.query
                 ORDER BY s.name,
                          CASE WHEN s.owner_key <> '' AND s.present_on_disk THEN 0
                               WHEN s.owner_key = '' THEN 1
                               ELSE 2 END
             )
             SELECT name, description, present_on_disk
             FROM matched
             WHERE trust_tier = 'local'
             ORDER BY rank DESC, name
             LIMIT $2",
        )
        .bind(query)
        .bind(limit as i64)
        .bind(user.as_str())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(SkillTextRow::into_candidate).collect())
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

/// How long the skill recall scan may run before the database stops it.
///
/// The same figure and the same reasoning as the knowledge base's
/// [`RECALL_SCAN_STATEMENT_TIMEOUT`](crate::RECALL_SCAN_STATEMENT_TIMEOUT): a
/// ceiling the caller keeps is not a ceiling the database keeps, and recall
/// runs before every turn's first round. The two arms run together, so they
/// share one ceiling rather than adding to each other.
pub const SKILL_RECALL_SCAN_STATEMENT_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(4);

/// What [`PgSkillIndexStore::nearest_by_embedding`] reads.
///
/// One scan, four uses. `d` computes one distance per catalog row and carries
/// only what resolving a name needs. `pick` cuts that to one row per name,
/// keeping the row `builtin_skill_get` would return - its own row when the
/// files are on disk, else the global one, else its own tombstone. `m` and `s`
/// then take the median of those distances and the median of each distance's
/// own distance from it, so the pass that measures the catalog's spread reads
/// no description and no body. The rows the block may show are read last, by
/// name and owner, and only those.
///
/// The approval predicate sits inside `d`, so an unapproved skill is absent
/// from the candidates *and* from the spread they are graded against. The
/// trust predicate sits on the final join instead, because it must apply to
/// the row a name resolved to rather than to the set a name resolves from.
///
/// Every returned row carries the same spread, which is the price of stating it
/// in the same answer as the candidates - three numbers on at most
/// `MAX_RECALL_SKILLS` rows.
///
/// The scope predicate is repeated on the final join rather than trusted from
/// `pick`. The composite is unique, so the join can only resolve to the row
/// `pick` came from - but a scope predicate that appears once is a scope
/// predicate one refactor can lose, and this is a host-global table.
///
/// An empty catalog yields no rows at all: `pick` is empty, so `s` is empty,
/// and the cross join answers with nothing rather than with a spread of
/// nothing.
///
/// Held as its own string so the projection can be asserted on without a
/// database - see `the_skill_recall_scan_does_not_read_the_body`.
const NEAREST_SKILLS_BY_EMBEDDING_SQL: &str = "\
    WITH d AS (
         SELECT name, owner_key, present_on_disk, MIN(chunk <=> $1) AS distance
         FROM skill_index, unnest(embedding) AS chunk
         WHERE (owner_user_id IS NULL OR owner_user_id = $2)
           AND approved_at IS NOT NULL
           AND embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND (embedding_model = $3
                OR (split_part($3, '@', 2) <> ''
                    AND split_part(embedding_model, '@', 2)
                        = split_part($3, '@', 2)))
         GROUP BY name, owner_key, present_on_disk
     ),
     pick AS (
         SELECT DISTINCT ON (name) name, owner_key, distance
         FROM d
         ORDER BY name,
                  CASE WHEN owner_key <> '' AND present_on_disk THEN 0
                       WHEN owner_key = '' THEN 1
                       ELSE 2 END
     ),
     m AS (
         SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY distance) AS median,
                count(*) AS rows_read
         FROM pick
     ),
     s AS (
         SELECT m.median,
                m.rows_read,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(pick.distance - m.median))
                    AS deviation
         FROM pick CROSS JOIN m
         GROUP BY m.median, m.rows_read
     )
     SELECT si.name, si.description, si.present_on_disk,
            pick.distance, s.median, s.rows_read, s.deviation
     FROM pick
     JOIN skill_index si
       ON si.name = pick.name
      AND si.owner_key = pick.owner_key
      AND (si.owner_user_id IS NULL OR si.owner_user_id = $2)
      AND si.trust_tier = 'local'
     CROSS JOIN s
     ORDER BY pick.distance, si.name
     LIMIT $4";

/// One skill the recall scan ranked: what a line needs, and nothing else.
#[derive(Debug, Clone, PartialEq)]
pub struct NearestSkill {
    /// The catalog name, which is the handle the skill is fetched by.
    pub name: String,
    /// The skill's own "when to use" line.
    pub description: String,
    /// Whether the skill's files were on disk at the last scan of its scope.
    pub present_on_disk: bool,
    /// The cosine distance that ranked it. `None` from the degraded full-text
    /// read, which carries no distance to state.
    pub distance: Option<f64>,
}

/// What [`PgSkillIndexStore::nearest_by_embedding`] answers with: the skills the
/// block may show, and what a distance from this catalog is worth.
#[derive(Debug, Default)]
pub struct NearestSkills {
    /// The nearest skills, nearest first.
    pub skills: Vec<NearestSkill>,
    /// The spread of this query's distances over the approved catalog, or
    /// `None` where there was nothing to measure. The caller then reads the
    /// source by a stated estimate.
    pub dispersion: Option<RecallDispersion>,
}

/// One row of [`NEAREST_SKILLS_BY_EMBEDDING_SQL`]: a candidate, its distance,
/// and the spread every row of the answer repeats.
#[derive(sqlx::FromRow)]
struct SkillNearestRow {
    name: String,
    description: String,
    present_on_disk: bool,
    distance: f64,
    median: Option<f64>,
    rows_read: i64,
    deviation: Option<f64>,
}

impl SkillNearestRow {
    fn into_candidate(self) -> NearestSkill {
        NearestSkill {
            name: self.name,
            description: self.description,
            present_on_disk: self.present_on_disk,
            distance: Some(self.distance),
        }
    }

    /// What this row says the catalog's spread is, where it says one it can be
    /// trusted for - see [`RecallDispersion::measured`].
    fn dispersion(&self) -> Option<RecallDispersion> {
        RecallDispersion::measured(
            self.median?,
            self.deviation?,
            self.rows_read.max(0) as usize,
        )
    }
}

/// One row of the degraded full-text read, which carries no distance.
#[derive(sqlx::FromRow)]
struct SkillTextRow {
    name: String,
    description: String,
    present_on_disk: bool,
}

impl SkillTextRow {
    fn into_candidate(self) -> NearestSkill {
        NearestSkill {
            name: self.name,
            description: self.description,
            present_on_disk: self.present_on_disk,
            distance: None,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Acceptance (#1154): the block never carries a skill body, and the scan
    /// behind it never reads one. The body is the widest column on the row, and
    /// the arm's whole economy is that recognition costs less than recall.
    #[test]
    fn the_skill_recall_scan_does_not_read_the_body() {
        let projection = NEAREST_SKILLS_BY_EMBEDDING_SQL
            .rsplit("SELECT si.name")
            .next()
            .expect("the scan selects the rows it will show after it measures the spread");
        assert!(
            !projection.contains("body"),
            "the recall scan selects a column the block never renders: \
             \n{NEAREST_SKILLS_BY_EMBEDDING_SQL}"
        );
        assert!(
            !NEAREST_SKILLS_BY_EMBEDDING_SQL.contains("si.body"),
            "the recall scan selects a column the block never renders: \
             \n{NEAREST_SKILLS_BY_EMBEDDING_SQL}"
        );
    }

    /// The pass that measures the catalog's spread reads the geometry and none
    /// of the content: one distance per row, and no column of the skill itself.
    #[test]
    fn the_pass_that_measures_the_skill_spread_reads_no_skill_content() {
        let measured = NEAREST_SKILLS_BY_EMBEDDING_SQL
            .split("     SELECT si.name")
            .next()
            .expect("the scan selects the rows it will show after it measures the spread");

        for column in ["description", "body", "tags", "metadata"] {
            assert!(
                !measured.contains(column),
                "the pass that measures the spread reads {column}, which it has no use for"
            );
        }
    }

    /// Acceptance (#1154): a skill nobody approved is excluded inside the scan,
    /// so it reaches neither the candidates nor the spread they are graded
    /// against. Proven against a real database in `tests/skill_recall.rs`; this
    /// pins where the predicate sits, which no behavioural test can see.
    #[test]
    fn the_approval_predicate_sits_inside_the_pass_that_measures_the_spread() {
        let measured = NEAREST_SKILLS_BY_EMBEDDING_SQL
            .split("     SELECT si.name")
            .next()
            .expect("the scan measures the spread before it reads the rows");
        assert!(
            measured.contains("approved_at IS NOT NULL"),
            "an unapproved skill must be absent from the spread as well as from the \
             candidates: \n{NEAREST_SKILLS_BY_EMBEDDING_SQL}"
        );
    }

    /// The scan's stated timeout is a real one. In PostgreSQL a
    /// `statement_timeout` of zero means no timeout at all, so the constant
    /// being positive is what makes the ceiling exist - and abandoning a query
    /// future leaves the backend scanning, on a path that runs before every
    /// turn.
    #[test]
    fn the_skill_recall_scans_statement_timeout_is_not_zero() {
        assert!(
            SKILL_RECALL_SCAN_STATEMENT_TIMEOUT > std::time::Duration::ZERO,
            "a zero timeout means no timeout at all in PostgreSQL"
        );
    }
}
