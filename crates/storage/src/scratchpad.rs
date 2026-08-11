//! Postgres-backed adapter for `ScratchpadStore` (issue #184).
//!
//! Mirrors the patterns established by `PgKnowledgeBaseStore` and
//! `PgConversationSearchStore`:
//! - `(user_id, conversation_id)`-scoped queries throughout, with
//!   `current_user_id()` read from the task-local — nothing here takes a
//!   `UserId` parameter (see `desktop-assistant-core::ports::auth`).
//! - Cross-user reads/deletes/upserts all fail closed: reads and deletes via
//!   a plain `WHERE user_id = $1`, and `write`'s upsert via a
//!   `WHERE scratchpads.user_id = EXCLUDED.user_id` guard on its
//!   `ON CONFLICT ... DO UPDATE` (the conflict target itself is not
//!   user-scoped, so the guard is the only thing standing between a
//!   colliding key and another tenant's row — see `write`, #809).
//! - Cross-user reads return empty, not an error.
//! - `search` is hybrid: a vector arm and a `plainto_tsquery` / `ts_rank_cd`
//!   arm fused by reciprocal rank, exactly as `PgKnowledgeBaseStore` does it
//!   (#717).
//! - [`PgScratchpadStore::nearest_by_embedding`] and
//!   [`PgScratchpadStore::search_text_any_term`] are the two reads behind the
//!   scratchpad arm of the `[Recall]` block (#1101). They answer a different
//!   question from `search` - "what is near this whole user sentence", not
//!   "find what I asked for" - so they rank differently, and neither replaces
//!   the other. Both leave out the reserved `goal` note, which every turn
//!   already renders as `[Current task]`.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ScratchpadNote;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::recall::RecallDispersion;
use desktop_assistant_core::ports::scratchpad::{
    NewScratchpadNote, SCRATCHPAD_GOAL_KEY, ScratchpadStore,
};
use desktop_assistant_core::ports::scratchpad_scope::{
    current_ancestors, current_owner_todo, current_visible_before,
};
use pgvector::Vector;
use sqlx::PgPool;

/// The `owner_todo` namespace the current turn writes under and is confined to
/// on delete. Root sentinel `""` when no subagent scope is installed (the
/// top-level session's own notes) — byte-for-byte the pre-#287 behavior.
fn current_namespace() -> String {
    current_owner_todo().unwrap_or_default()
}

/// The read snapshot bound: `(visible_before, own_namespace, ancestor_chain)`.
///
/// When `visible_before` is `Some` (a subagent turn) the read is a spawn-time
/// SNAPSHOT: the subagent's own subtree at any id, PLUS pre-marker rows from
/// its ancestor namespaces only — never a concurrent sibling's/cousin's
/// in-flight notes (#287 finding: the `id < marker` bound must be
/// ancestor-restricted, not namespace-blind). When `None` (top-level turn) the
/// read is unbounded across all namespaces — byte-for-byte the pre-#287 pad.
/// Consumers bind these three after their existing params and gate the extra
/// predicate on `$vb IS NULL`.
fn read_snapshot() -> (Option<String>, String, Vec<String>) {
    (
        current_visible_before(),
        current_namespace(),
        current_ancestors().unwrap_or_default(),
    )
}

/// Postgres adapter for the per-conversation scratchpad table.
pub struct PgScratchpadStore {
    pool: PgPool,
    scan_ceiling: std::time::Duration,
}

impl PgScratchpadStore {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            scan_ceiling: crate::RECALL_SCAN_STATEMENT_TIMEOUT,
        }
    }

    /// The same store, whose full scans the database gives up on after
    /// `ceiling` instead of after
    /// [`RECALL_SCAN_STATEMENT_TIMEOUT`](crate::RECALL_SCAN_STATEMENT_TIMEOUT).
    ///
    /// Exists so the bound can be proven on the path a deployment actually
    /// runs - see
    /// [`PgKnowledgeBaseStore::with_scan_ceiling`](crate::PgKnowledgeBaseStore::with_scan_ceiling)
    /// for why a bounded variant reached past the public method would prove
    /// nothing.
    #[must_use]
    pub fn with_scan_ceiling(mut self, ceiling: std::time::Duration) -> Self {
        self.scan_ceiling = ceiling;
        self
    }
}

/// An [`SpRow`] plus the cosine distance that ranked it and the spread every
/// row of the answer repeats, for
/// [`PgScratchpadStore::nearest_by_embedding`].
#[derive(sqlx::FromRow)]
struct SpNearestRow {
    #[sqlx(flatten)]
    row: SpRow,
    distance: f64,
    median: Option<f64>,
    rows_read: i64,
    deviation: Option<f64>,
}

impl SpNearestRow {
    /// What this row says the pad's spread is, where it says one it can be
    /// trusted for - see [`RecallDispersion::measured`].
    fn dispersion(&self) -> Option<RecallDispersion> {
        RecallDispersion::measured(
            self.median?,
            self.deviation?,
            self.rows_read.max(0) as usize,
        )
    }
}

/// What [`PgScratchpadStore::nearest_by_embedding`] answers with: the notes the
/// block may show, and what a distance from this pad is worth.
#[derive(Debug, Default)]
pub struct NearestNotes {
    /// The nearest notes, each with the cosine distance that ranked it,
    /// nearest first.
    pub notes: Vec<(ScratchpadNote, f64)>,
    /// The spread of this query's distances over the whole pad, or `None`
    /// where the pad holds too little to measure one. The caller then reads the
    /// source by a stated estimate.
    pub dispersion: Option<RecallDispersion>,
}

/// What [`PgScratchpadStore::nearest_by_embedding`] reads.
///
/// One scan, three uses, exactly as the knowledge base's own scan does it. `d`
/// computes one distance per note and carries nothing else, so the pass that
/// measures the pad's spread reads no note content: `m` takes the median of
/// those distances and `s` the median of each distance's own distance from it.
/// The notes the block may show are then read whole, by primary key, and only
/// those.
///
/// The three scope bounds - the user, the conversation, and the caller's
/// `owner_todo` read snapshot - sit on `d`, so they bound the measurement as
/// well as the candidates. The join that reads the rows repeats the two that a
/// row can be addressed by, the user and the conversation: a scope predicate
/// that appears once is a scope predicate one refactor can lose, and this table
/// holds every tenant's working notes. The `owner_todo` snapshot and the goal
/// exclusion are not repeated, because the join is on `d.id` and a row `d` never
/// selected cannot arrive through it.
///
/// An empty pad yields no rows at all: `d` is empty, so `s` is empty, and the
/// cross join answers with nothing rather than with a spread of nothing.
///
/// Held as its own string so the projection can be asserted on without a
/// database - see `the_pad_scan_measures_before_it_reads_any_note`.
const NEAREST_NOTES_BY_EMBEDDING_SQL: &str = "\
    WITH d AS (
         SELECT id, MIN(chunk <=> $1) AS distance
         FROM scratchpads, unnest(embedding) AS chunk
         WHERE user_id = $2 AND conversation_id = $3
           AND ($4::text IS NULL OR (owner_todo = $5 OR owner_todo LIKE $5 || '.%'
                OR (id COLLATE \"C\" < $4 AND owner_todo = ANY($6::text[]))))
           AND note_key <> $9
           AND embedding IS NOT NULL
           AND embedding_model IS NOT NULL
           AND (embedding_model = $7
                OR (split_part($7, '@', 2) <> ''
                    AND split_part(embedding_model, '@', 2)
                        = split_part($7, '@', 2)))
         GROUP BY id
     ),
     m AS (
         SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY distance) AS median,
                count(*) AS rows_read
         FROM d
     ),
     s AS (
         SELECT m.median,
                m.rows_read,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(d.distance - m.median))
                    AS deviation
         FROM d CROSS JOIN m
         GROUP BY m.median, m.rows_read
     )
     SELECT sp.id, sp.conversation_id, sp.owner_todo, sp.note_key, sp.content, sp.note_type,
            sp.seq, sp.done, sp.pinned, sp.knowledge_entry_id, sp.created_at, sp.updated_at,
            d.distance, s.median, s.rows_read, s.deviation
     FROM d
     JOIN scratchpads sp
       ON sp.id = d.id AND sp.user_id = $2 AND sp.conversation_id = $3
     CROSS JOIN s
     ORDER BY d.distance, sp.id DESC
     LIMIT $8";

#[derive(sqlx::FromRow)]
struct SpRow {
    id: String,
    conversation_id: String,
    owner_todo: String,
    note_key: String,
    content: String,
    note_type: String,
    seq: Option<i32>,
    done: bool,
    pinned: bool,
    knowledge_entry_id: Option<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

impl SpRow {
    fn into_note(self) -> ScratchpadNote {
        ScratchpadNote {
            id: self.id,
            conversation_id: self.conversation_id,
            owner_todo: self.owner_todo,
            key: self.note_key,
            content: self.content,
            note_type: self.note_type,
            sequence: self.seq,
            done: self.done,
            pinned: self.pinned,
            knowledge_entry_id: self.knowledge_entry_id,
            created_at: self.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            updated_at: self.updated_at.format("%Y-%m-%d %H:%M:%S").to_string(),
        }
    }
}

impl ScratchpadStore for PgScratchpadStore {
    async fn write(
        &self,
        conversation_id: &str,
        notes: &[NewScratchpadNote],
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        if notes.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();

        // Batch upsert via UNNEST so a variable-length batch is a single
        // prepared statement. Zipping the parallel arrays yields one row per
        // note; the conflict target is `(conversation_id, note_key)` so a
        // repeated key replaces content/type/sequence/done and bumps
        // `updated_at`. `id` and `user_id` are only used on insert — an
        // existing note keeps its original id/owner on update.
        let ids: Vec<String> = (0..notes.len())
            .map(|_| uuid::Uuid::now_v7().to_string())
            .collect();
        let user_ids: Vec<String> = vec![user_id.as_str().to_string(); notes.len()];
        let conv_ids: Vec<String> = vec![conversation_id.to_string(); notes.len()];
        // Stamp every note in the batch with the writer's current namespace,
        // read from the task-local scope (root "" outside any subagent scope).
        // A tool call cannot set a task-local, so confinement is spoof-proof.
        let owner_todos: Vec<String> = vec![current_namespace(); notes.len()];
        let keys: Vec<String> = notes.iter().map(|n| n.key.clone()).collect();
        let contents: Vec<String> = notes.iter().map(|n| n.content.clone()).collect();
        let types: Vec<String> = notes.iter().map(|n| n.note_type.clone()).collect();
        // `seq` is nullable; UNNEST of a `Vec<Option<i32>>` preserves NULLs.
        let seqs: Vec<Option<i32>> = notes.iter().map(|n| n.sequence).collect();
        let dones: Vec<bool> = notes.iter().map(|n| n.done).collect();
        // `None` means "leave whatever is attached alone", not "detach", so a
        // caller that rewrites a note's text without knowing about the
        // attachment cannot drop it — the same rule `source` and `summary`
        // follow on a knowledge write. The COALESCE below is what applies it.
        let entry_ids: Vec<Option<String>> =
            notes.iter().map(|n| n.knowledge_entry_id.clone()).collect();

        // The conflict target `(conversation_id, owner_todo, note_key)`
        // (migration 031) has no `user_id` component, so without a `WHERE`
        // guard on the update, a colliding key from a DIFFERENT tenant would
        // silently overwrite another user's row in place — the FK to
        // `conversations(id)` isn't user-scoped, so any conversation id is
        // syntactically writable. `scratchpads.user_id = EXCLUDED.user_id`
        // fails closed exactly like `PgKnowledgeBaseStore::write`
        // (knowledge.rs): when the guard is false, Postgres treats the
        // conflict as a no-op for that row (no update, no error) and
        // `RETURNING` omits it, so a cross-tenant write neither changes the
        // victim's row nor leaks its content back to the writer (#809).
        //
        // The upsert always CLEARS the vector, and the statement after it
        // writes back whatever the caller embedded inline (#717). Clearing is
        // what keeps a vector honest: an upsert replaces the content, so a
        // vector left in place would describe text that is no longer there,
        // while its stamp still named the current model — putting it beyond
        // both the stale sweep and the backfill, which act only on a missing or
        // superseded stamp. A cleared row is simply re-embedded.
        //
        // Both statements run in one transaction, so no reader ever sees a note
        // carrying the previous content's vector.
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let rows: Vec<SpRow> = sqlx::query_as(
            "INSERT INTO scratchpads \
                 (id, user_id, conversation_id, owner_todo, note_key, content, note_type, seq, done, \
                  knowledge_entry_id) \
             SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[], \
                                  $5::text[], $6::text[], $7::text[], $8::int4[], $9::bool[], \
                                  $10::text[]) \
             ON CONFLICT (conversation_id, owner_todo, note_key) \
             DO UPDATE SET content = EXCLUDED.content, note_type = EXCLUDED.note_type, \
                           seq = EXCLUDED.seq, done = EXCLUDED.done, updated_at = NOW(), \
                           embedding = NULL, embedding_model = NULL, \
                           knowledge_entry_id = COALESCE(EXCLUDED.knowledge_entry_id, \
                                                         scratchpads.knowledge_entry_id) \
             WHERE scratchpads.user_id = EXCLUDED.user_id \
             RETURNING id, conversation_id, owner_todo, note_key, content, note_type, seq, done, pinned, \
                       knowledge_entry_id, created_at, updated_at",
        )
        .bind(&ids)
        .bind(&user_ids)
        .bind(&conv_ids)
        .bind(&owner_todos)
        .bind(&keys)
        .bind(&contents)
        .bind(&types)
        .bind(&seqs)
        .bind(&dones)
        .bind(&entry_ids)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // One statement per embedded note. A single statement cannot carry them
        // all: every note's `vector[]` has its own chunk count, and a Postgres
        // array of arrays must be rectangular. The batch is capped at
        // `MAX_NOTES_PER_WRITE`, and a real write carries one or two notes.
        //
        // A note finds its stored row by `note_key`, which is unique within the
        // batch (the tool layer de-duplicates last-wins before it calls) and
        // unique in the table within one `(conversation_id, owner_todo)`. A row
        // the cross-tenant guard refused is absent from `rows`, so it is skipped
        // rather than written to.
        for note in notes {
            let Some(embedding) = note.embedding.as_ref() else {
                continue;
            };
            if embedding.chunks.is_empty() {
                continue;
            }
            let Some(row) = rows.iter().find(|r| r.note_key == note.key) else {
                continue;
            };
            let vectors: Vec<Vector> = embedding.chunks.iter().cloned().map(Vector::from).collect();
            sqlx::query(
                "UPDATE scratchpads SET embedding = $1::vector[], embedding_model = $2 \
                 WHERE id = $3 AND user_id = $4",
            )
            .bind(&vectors)
            .bind(&embedding.model)
            .bind(&row.id)
            .bind(user_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.into_iter().map(SpRow::into_note).collect())
    }

    async fn get_many(
        &self,
        conversation_id: &str,
        keys: &[String],
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        if keys.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();
        let (vb, me, ancestors) = read_snapshot();
        let rows: Vec<SpRow> = sqlx::query_as(
            "SELECT id, conversation_id, owner_todo, note_key, content, note_type, seq, done, pinned, \
                    knowledge_entry_id, created_at, updated_at \
             FROM scratchpads \
             WHERE user_id = $1 AND conversation_id = $2 AND note_key = ANY($3) \
               AND ($5::text IS NULL OR (owner_todo = $6 OR owner_todo LIKE $6 || '.%' \
                    OR (id COLLATE \"C\" < $5 AND owner_todo = ANY($7::text[])))) \
             ORDER BY updated_at DESC LIMIT $4",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(keys)
        .bind(limit as i64)
        .bind(vb)
        .bind(me)
        .bind(ancestors)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SpRow::into_note).collect())
    }

    async fn list(
        &self,
        conversation_id: &str,
        note_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        let user_id = current_user_id();
        // Order by type, then sequence ascending (nulls last), then recency —
        // so a sequenced plan of `todo`s reads in order. The optional
        // `note_type` filter rides a single static query via `IS NULL OR`.
        let (vb, me, ancestors) = read_snapshot();
        let rows: Vec<SpRow> = sqlx::query_as(
            "SELECT id, conversation_id, owner_todo, note_key, content, note_type, seq, done, pinned, \
                    knowledge_entry_id, created_at, updated_at \
             FROM scratchpads \
             WHERE user_id = $1 AND conversation_id = $2 \
               AND ($3::text IS NULL OR note_type = $3) \
               AND ($5::text IS NULL OR (owner_todo = $6 OR owner_todo LIKE $6 || '.%' \
                    OR (id COLLATE \"C\" < $5 AND owner_todo = ANY($7::text[])))) \
             ORDER BY pinned DESC, note_type ASC, seq ASC NULLS LAST, updated_at DESC LIMIT $4",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(note_type)
        .bind(limit as i64)
        .bind(vb)
        .bind(me)
        .bind(ancestors)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SpRow::into_note).collect())
    }

    async fn search(
        &self,
        conversation_id: &str,
        query: &str,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        note_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        // No query vector (no embedding backend, or one that stalled — see
        // `EMBED_TIMEOUT` in core's embedding port): the hybrid query's vector
        // branch (`chunk <=> $1`) errors on a zero-dimension vector, so take the
        // full-text-only path. Both paths report the same fields, so a missing
        // embedding backend costs recall and nothing else.
        if query_embedding.is_empty() {
            self.search_text(conversation_id, query, note_type, limit)
                .await
        } else {
            self.search_hybrid(
                conversation_id,
                query,
                query_embedding,
                embedding_model,
                note_type,
                limit,
            )
            .await
        }
    }

    async fn delete_many(&self, conversation_id: &str, keys: &[String]) -> Result<u64, CoreError> {
        if keys.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();
        // Confine to the caller's own namespace: a subagent can only delete its
        // own entries, never the parent's or a sibling's (#287). Top-level
        // (namespace "") deletes only root notes, which is byte-for-byte the
        // pre-#287 behavior since all top-level notes are owner_todo="".
        let me = current_namespace();
        let result = sqlx::query(
            "DELETE FROM scratchpads WHERE user_id = $1 AND conversation_id = $2 \
             AND owner_todo = $3 AND note_key = ANY($4)",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(&me)
        .bind(keys)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result.rows_affected())
    }

    async fn set_pinned(
        &self,
        conversation_id: &str,
        keys: &[String],
        pinned: bool,
    ) -> Result<u64, CoreError> {
        if keys.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();
        // Confined to the caller's own namespace, exactly as `delete_many` is: a
        // subagent may pin its own notes but never reach into the parent's or a
        // sibling's context budget (#287).
        let me = current_namespace();
        // `pinned IS DISTINCT FROM $5` makes `rows_affected` the count of notes
        // actually CHANGED, so re-pinning an already-pinned note reports 0 and
        // the caller can tell a real change from a no-op.
        let result = sqlx::query(
            "UPDATE scratchpads SET pinned = $5, updated_at = NOW() \
             WHERE user_id = $1 AND conversation_id = $2 \
               AND owner_todo = $3 AND note_key = ANY($4) \
               AND pinned IS DISTINCT FROM $5",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(&me)
        .bind(keys)
        .bind(pinned)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result.rows_affected())
    }

    async fn release_knowledge_references(
        &self,
        conversation_id: &str,
        note_ids: &[String],
    ) -> Result<u64, CoreError> {
        if note_ids.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();
        // Confined to the caller's own SUBTREE, which is not what `set_pinned`
        // and `delete_many` do - they confine to the caller's own namespace
        // exactly. Two failures bound this, one on each side.
        //
        // Reaching too far: a subagent round must never clear a pin in an
        // ancestor's namespace. Its read spans its ancestors, so an unconfined
        // repair let it release the parent's pin, and the line saying so
        // rendered into the subagent's block where the parent never sees it.
        //
        // Reaching too little: the top-level read is namespace-blind, so a
        // top-level round sees a subagent's note. Confining the repair to the
        // caller's own namespace alone left that note stuck for the life of the
        // conversation - a dead attachment, a slot of the pin cap consumed, a
        // knowledge read spent every round, and no verb the model could use to
        // clear it, because `set_pinned` and `delete_many` are confined too.
        //
        // Own-subtree is the rule that fits both: the root namespace repairs
        // anything, matching its namespace-blind read, and a subagent repairs
        // itself and its descendants but never an ancestor or a sibling. The
        // repair only ever touches rows this round has just read and just named
        // to the model, so no round clears something it did not report.
        let me = current_namespace();
        //
        // The pin goes with the attachment. A note whose entry has gone renders
        // nothing under `[Pinned]`, and a pin that renders nothing is a fact the
        // model believes it has and does not, so the pin is released and the
        // note falls back to the `[Scratchpad]` index like any other note.
        //
        // `updated_at` is deliberately left alone: this is a repair, not an edit
        // the model made, and bumping it would move the note to the top of the
        // pad for a reason nothing can see.
        //
        // `knowledge_entry_id IS NOT NULL` makes `rows_affected` the count
        // actually repaired and makes a second call a true no-op.
        let result = sqlx::query(
            "UPDATE scratchpads SET knowledge_entry_id = NULL, pinned = FALSE \
             WHERE user_id = $1 AND conversation_id = $2 AND id = ANY($4) \
               AND ($3 = '' OR owner_todo = $3 OR owner_todo LIKE $3 || '.%') \
               AND knowledge_entry_id IS NOT NULL",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(&me)
        .bind(note_ids)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result.rows_affected())
    }

    async fn clear(&self, conversation_id: &str) -> Result<u64, CoreError> {
        let user_id = current_user_id();
        // Namespace-confined (see delete_many): `clear`/`delete all:true` from a
        // subagent wipes only its own namespace, never the parent pad.
        let me = current_namespace();
        let result = sqlx::query(
            "DELETE FROM scratchpads WHERE user_id = $1 AND conversation_id = $2 \
             AND owner_todo = $3",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(&me)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result.rows_affected())
    }

    async fn delete_owner_subtree(
        &self,
        conversation_id: &str,
        owner_todo: &str,
    ) -> Result<u64, CoreError> {
        let user_id = current_user_id();
        // Delete the namespace itself AND every descendant. The dot-delimited
        // LIKE is a real prefix match ('1' does not match '10'/'11'); owner_todo
        // is a bound parameter and the migration-031 [0-9.] CHECK guarantees no
        // LIKE metacharacters. user_id = $1 first keeps the tenant guard.
        let result = sqlx::query(
            "DELETE FROM scratchpads WHERE user_id = $1 AND conversation_id = $2 \
             AND (owner_todo = $3 OR owner_todo LIKE $3 || '.%')",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(owner_todo)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result.rows_affected())
    }
}

/// The two reads behind the scratchpad arm of the `[Recall]` block (#1101).
///
/// The block searches this conversation's pad with the same prompt embedding it
/// asks the knowledge base, so a note the model stashed earlier comes back when
/// the prompt is about it - rather than only once the `[Scratchpad]` index
/// opens, which is gated on context starting to drop.
///
/// Neither read is on [`ScratchpadStore`], for the same reason
/// `PgKnowledgeBaseStore::nearest_by_embedding` is not on its trait: they serve
/// one caller with one ranking need, and the port is what the tools use.
///
/// ## Both leave out the `goal` note
///
/// One exclusion, and only one, belongs in the query. The reserved `goal` note
/// is what every turn renders as `[Current task]`, and it is by construction the
/// pad row nearest a prompt about the current task - so without this the arm
/// would spend its first line restating the task the prompt already carries, on
/// every turn. Excluding it here rather than after the read means it never
/// occupies a slot in the scan the block's "and N more" count is measured
/// against. The key is bound from the core constant that defines it.
///
/// Nothing else is excluded here, and the omission is deliberate. A `todo` step
/// and an `outcome:<step>` finding are rendered by `[Plan]` only while the tree
/// still shows them: a finding is dropped once its parent step is done, and the
/// tree elides past its cap. Such a note is then durable and invisible, which is
/// exactly the condition this arm exists for. What the turn actually showed is
/// the core's business, and it drops those keys at render time.
impl PgScratchpadStore {
    /// The conversation's notes nearest a query embedding, with the cosine
    /// distance that put them there, nearest first, and how spread out this
    /// query's distances are over the whole pad.
    ///
    /// A plain vector search rather than the hybrid [`ScratchpadStore::search`],
    /// because the block reads each candidate against the spread of its
    /// source's own distances, and a fused RRF score is not a quantity that has
    /// a spread of that kind: over a hybrid search every row scores non-zero
    /// against any query. A cosine distance is comparable, so a distribution
    /// over it means something.
    ///
    /// **The pad is its own source, so it states its own spread** (#1167). A
    /// note embeds `"<key> <content>"`, which is terser and more telegraphic
    /// than a knowledge entry's body, so the pad puts its distances somewhere
    /// else than the store does - and a bar read against one source says
    /// nothing about the other. Reading the pad by the estimate a source falls
    /// back to when it cannot state its own geometry was the weakest part of
    /// the arm, because that estimate is fitted to neither.
    ///
    /// **The candidates and the spread come from one scan**, on the same terms
    /// as [`crate::PgKnowledgeBaseStore::nearest_by_embedding`]: both are
    /// functions of the same query vector, and the pass that measures reads one
    /// distance per note and none of its content. Only the notes the block may
    /// show are read whole.
    ///
    /// **A small pad states nothing**, and the caller falls back to its stated
    /// estimate - see
    /// [`RecallDispersion::measured`](desktop_assistant_core::ports::recall::RecallDispersion::measured).
    /// One conversation's pad is usually under the sample floor, so that is the
    /// ordinary answer rather than the exceptional one; what the measurement
    /// buys is the long conversation, whose pad is both large enough to measure
    /// and least like the store.
    ///
    /// **A pad whose distances are widely spread renders nothing, and that is
    /// the bar working rather than the arm failing.** The bar is stated in
    /// deviations, so a source whose deviation is more than about a seventh of
    /// its median puts the bar past any distance a cosine can take, and no note
    /// of it is exceptional enough to show. That is the same rule
    /// `no_raw_cosine_constant_decides_whether_the_block_renders` (#1121) pins
    /// for the knowledge store, and refusing such a measurement would put a
    /// fixed distance back in charge of who renders - which is what the whole
    /// dimensionless bar exists to prevent.
    ///
    /// It does mean a real behaviour change the moment a pad is large enough to
    /// measure: before this the pad was always read by the stated estimate, so
    /// it rendered whatever sat inside 0.31 of cosine distance. A pad that
    /// measures wide now renders nothing where it used to render lines. #1243
    /// is where a real pad's geometry is measured against a real embedding
    /// model, because nothing here knows whether real pads are wide.
    ///
    /// Scoped by an explicit `WHERE user_id` **and** `conversation_id`
    /// predicate, plus the caller's `owner_todo` read snapshot - the same three
    /// bounds every other read here carries, on the pass that measures as much
    /// as on the one that reads. Row-level security is a backstop the table
    /// owner bypasses, so the predicates are the guard.
    ///
    /// The `goal` note is left out, as the impl block above states - of the
    /// spread as well as of the candidates, so the geometry describes the notes
    /// the arm can actually offer.
    ///
    /// `embedding_model` identifies the model that produced `query_embedding`,
    /// and only rows embedded by that model take part, matched on the digest
    /// half of the `<name>@<digest>` stamp wherever both sides carry one. A
    /// comparison across models is a comparison across vector dimensions, which
    /// the database answers with an error rather than a miss.
    ///
    /// Ties break on `id`, which is unique, so two identical reads return the
    /// same page. Without a total order the rest is decided by physical row
    /// position, which moves after any `VACUUM` or update - and notes written in
    /// one batch carry one vector each, so exact ties are ordinary here.
    ///
    /// An empty `query_embedding` yields no rows and no spread: the vector
    /// operator raises on a zero-dimension vector, and a caller with no
    /// embedding has [`Self::search_text_any_term`] to fall back to.
    ///
    /// The scan carries
    /// [`RECALL_SCAN_STATEMENT_TIMEOUT`](crate::RECALL_SCAN_STATEMENT_TIMEOUT),
    /// so the database stops working when the caller stops waiting. It is the
    /// block's most expensive read - the pad's vectors have no index, because
    /// the query unnests every note's chunks and groups on the row - so it is
    /// the one that most needs the bound.
    pub async fn nearest_by_embedding(
        &self,
        conversation_id: &str,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        limit: usize,
    ) -> Result<NearestNotes, CoreError> {
        if query_embedding.is_empty() {
            return Ok(NearestNotes::default());
        }
        let user_id = current_user_id();
        let (vb, me, ancestors) = read_snapshot();
        let mut scan = crate::scan_bound::begin_bounded(&self.pool, self.scan_ceiling).await?;
        let rows: Vec<SpNearestRow> = sqlx::query_as(NEAREST_NOTES_BY_EMBEDDING_SQL)
            .bind(Vector::from(query_embedding))
            .bind(user_id.as_str())
            .bind(conversation_id)
            .bind(vb)
            .bind(me)
            .bind(ancestors)
            .bind(embedding_model)
            .bind(limit as i64)
            .bind(SCRATCHPAD_GOAL_KEY)
            .fetch_all(&mut *scan)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        scan.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Every row carries the same spread, so the first one states it.
        let dispersion = rows.first().and_then(SpNearestRow::dispersion);
        Ok(NearestNotes {
            notes: rows
                .into_iter()
                .map(|r| {
                    let distance = r.distance;
                    (r.row.into_note(), distance)
                })
                .collect(),
            dispersion,
        })
    }

    /// Full-text search over the conversation's notes that asks for **any** of
    /// the query's terms, best match first.
    ///
    /// The degraded arm, for the turn where no embedding is available. It
    /// cannot be the adapter's own `search_text`: `plainto_tsquery` joins every
    /// surviving lexeme with `AND`, which is right for a model-authored search of two or
    /// three words and wrong for a whole user sentence. "when is the next
    /// deploy?" becomes `'next' & 'deploy'`, and a note saying "the deploy runs
    /// on Fridays" does not match, because it never says "next". The fallback
    /// would then answer with nothing at exactly the moment it exists to answer
    /// with something (#1100).
    ///
    /// The query is built from `to_tsvector`'s own lexemes, so stop words and
    /// stemming are handled once by the same configuration the index uses, and
    /// `quote_literal` makes every lexeme a literal - a prompt full of
    /// `tsquery` operators is text, not syntax. A prompt that reduces to no
    /// lexemes at all yields a NULL query, which matches no row.
    ///
    /// Carries the same `user_id` / `conversation_id` / `owner_todo` scope as
    /// every other read here, and leaves out the same `goal` note as
    /// [`Self::nearest_by_embedding`].
    ///
    /// The scan carries this store's own ceiling, like its measured
    /// counterpart. This pad's rows are few, so the bound is cheaper insurance
    /// here than on the knowledge base - but a read that states no ceiling at
    /// all leaves the backend working for a caller that has given up, and one
    /// of two sibling reads being bounded is the shape a later reader misreads.
    pub async fn search_text_any_term(
        &self,
        conversation_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        let user_id = current_user_id();
        let (vb, me, ancestors) = read_snapshot();
        let mut scan = crate::scan_bound::begin_bounded(&self.pool, self.scan_ceiling).await?;
        let rows: Vec<SpRow> = sqlx::query_as(
            "WITH q AS (
                 SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS query
                 FROM unnest(to_tsvector('english', $1))
             )
             SELECT id, conversation_id, owner_todo, note_key, content, note_type,
                    seq, done, pinned, knowledge_entry_id, created_at, updated_at
             FROM scratchpads, q
             WHERE user_id = $2 AND conversation_id = $3
               AND q.query IS NOT NULL
               AND tsv @@ q.query
               AND ($4::text IS NULL OR (owner_todo = $5 OR owner_todo LIKE $5 || '.%'
                    OR (id COLLATE \"C\" < $4 AND owner_todo = ANY($6::text[]))))
               AND note_key <> $8
             ORDER BY ts_rank_cd(tsv, q.query) DESC, updated_at DESC, id DESC
             LIMIT $7",
        )
        .bind(query)
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(vb)
        .bind(me)
        .bind(ancestors)
        .bind(limit as i64)
        .bind(SCRATCHPAD_GOAL_KEY)
        .fetch_all(&mut *scan)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        scan.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SpRow::into_note).collect())
    }
}

/// The two arms behind [`ScratchpadStore::search`]. Private to the adapter:
/// which arm runs is decided by whether the caller has a query vector, never by
/// the caller itself.
impl PgScratchpadStore {
    /// Full-text-only search: `plainto_tsquery` + `ts_rank_cd`, scoped. Backs
    /// the no-embedding fallback, and is byte-for-byte the pre-#717 search.
    async fn search_text(
        &self,
        conversation_id: &str,
        query: &str,
        note_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        let user_id = current_user_id();
        // Search stays relevance-ranked; the optional `note_type` filter rides
        // a single static query via `IS NULL OR`.
        let (vb, me, ancestors) = read_snapshot();
        let rows: Vec<SpRow> = sqlx::query_as(
            "WITH q AS (SELECT plainto_tsquery('english', $3) AS query) \
             SELECT id, conversation_id, owner_todo, note_key, content, note_type, seq, done, pinned, \
                    knowledge_entry_id, created_at, updated_at \
             FROM scratchpads, q \
             WHERE user_id = $1 AND conversation_id = $2 AND tsv @@ q.query \
               AND ($4::text IS NULL OR note_type = $4) \
               AND ($6::text IS NULL OR (owner_todo = $7 OR owner_todo LIKE $7 || '.%' \
                    OR (id COLLATE \"C\" < $6 AND owner_todo = ANY($8::text[])))) \
             ORDER BY ts_rank_cd(tsv, q.query) DESC, updated_at DESC LIMIT $5",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(query)
        .bind(note_type)
        .bind(limit as i64)
        .bind(vb)
        .bind(me)
        .bind(ancestors)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SpRow::into_note).collect())
    }

    /// Vector + full-text search, fused by reciprocal rank — the same shape
    /// `PgKnowledgeBaseStore::search_hybrid` uses (#717).
    ///
    /// Two properties carry the design:
    ///
    /// * **Only the vector arm is model-scoped** ($9). A vector of another
    ///   dimension makes pgvector raise rather than miss, and a table
    ///   legitimately holds two models' vectors during any reindex and for the
    ///   whole of a live backend swap. Sameness is decided on the digest half of
    ///   the `<name>@<digest>` stamp wherever both sides carry one, matching
    ///   `embedding_backfill::invalidate_stale_embeddings`, so a purely cosmetic
    ///   rename does not blind the search until the sweep restamps the rows.
    ///   `split_part(x, '@', 2)` yields '' where there is no '@', so the
    ///   non-empty test doubles as "both sides carry a digest". A NULL stamp is a
    ///   vector of unknown provenance, hence unknown dimension, and is excluded.
    /// * **The full-text arm is never model-scoped**, so changing the embedding
    ///   model costs recall quality and not all recall.
    ///
    /// Both arms carry the full `user_id` / `conversation_id` / `owner_todo`
    /// scope. A predicate present on one arm and missing from the other would
    /// make the weaker arm a way around the confinement the other enforces.
    ///
    /// The final order breaks ties on `id`, which is unique, so two identical
    /// searches return the same page. Fused scores collide readily -- a note
    /// found by only one arm scores exactly `1/(60 + rank)` -- and without a
    /// total order the rest is decided by physical row position, which moves
    /// after any `VACUUM` or update.
    async fn search_hybrid(
        &self,
        conversation_id: &str,
        query: &str,
        query_embedding: Vec<f32>,
        embedding_model: &str,
        note_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        let user_id = current_user_id();
        let (vb, me, ancestors) = read_snapshot();
        let embedding_vec = Vector::from(query_embedding);
        // Over-fetch each arm so the fusion has something to fuse: a row that
        // ranks well on one arm and modestly on the other must still be
        // reachable from both lists.
        //
        // Both arms carry an explicit `ORDER BY` before that `LIMIT`, and both
        // are load-bearing rather than decorative. `ORDER BY` inside `OVER (…)`
        // orders the window computation, not the statement's output, so a
        // `LIMIT` with no statement-level order truncates an undefined set: the
        // arm still returns rows and the fusion still ranks them, so the caller
        // gets a plausible page that quietly omits the best matches. Removing
        // either one costs nothing at the time and breaks recall later.
        let fetch_limit = (limit.saturating_mul(2)) as i64;

        let rows: Vec<SpRow> = sqlx::query_as(
            "WITH chunk_distances AS (
                SELECT id, conversation_id, owner_todo, note_key, content, note_type,
                       seq, done, pinned, knowledge_entry_id, created_at, updated_at,
                       MIN(chunk <=> $1) AS min_distance
                FROM scratchpads, unnest(embedding) AS chunk
                WHERE user_id = $2 AND conversation_id = $3
                  AND ($4::text IS NULL OR note_type = $4)
                  AND ($6::text IS NULL OR (owner_todo = $7 OR owner_todo LIKE $7 || '.%'
                       OR (id COLLATE \"C\" < $6 AND owner_todo = ANY($8::text[]))))
                  AND embedding IS NOT NULL
                  AND embedding_model IS NOT NULL
                  AND (embedding_model = $9
                       OR (split_part($9, '@', 2) <> ''
                           AND split_part(embedding_model, '@', 2)
                               = split_part($9, '@', 2)))
                GROUP BY id, conversation_id, owner_todo, note_key, content, note_type,
                         seq, done, pinned, knowledge_entry_id, created_at, updated_at
            ),
            vector_ranked AS (
                SELECT id, conversation_id, owner_todo, note_key, content, note_type,
                       seq, done, pinned, knowledge_entry_id, created_at, updated_at,
                       ROW_NUMBER() OVER (ORDER BY min_distance) AS rank_v
                FROM chunk_distances
                ORDER BY min_distance
                LIMIT $10
            ),
            text_ranked AS (
                SELECT id, conversation_id, owner_todo, note_key, content, note_type,
                       seq, done, pinned, knowledge_entry_id, created_at, updated_at,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q.query) DESC) AS rank_t
                FROM scratchpads, plainto_tsquery('english', $5) AS q(query)
                WHERE user_id = $2 AND conversation_id = $3
                  AND tsv @@ q.query
                  AND ($4::text IS NULL OR note_type = $4)
                  AND ($6::text IS NULL OR (owner_todo = $7 OR owner_todo LIKE $7 || '.%'
                       OR (id COLLATE \"C\" < $6 AND owner_todo = ANY($8::text[]))))
                ORDER BY ts_rank_cd(tsv, q.query) DESC
                LIMIT $10
            ),
            fused AS (
                SELECT COALESCE(v.id, t.id) AS id,
                       COALESCE(v.conversation_id, t.conversation_id) AS conversation_id,
                       COALESCE(v.owner_todo, t.owner_todo) AS owner_todo,
                       COALESCE(v.note_key, t.note_key) AS note_key,
                       COALESCE(v.content, t.content) AS content,
                       COALESCE(v.note_type, t.note_type) AS note_type,
                       COALESCE(v.seq, t.seq) AS seq,
                       COALESCE(v.done, t.done) AS done,
                       COALESCE(v.pinned, t.pinned) AS pinned,
                       COALESCE(v.knowledge_entry_id, t.knowledge_entry_id) AS knowledge_entry_id,
                       COALESCE(v.created_at, t.created_at) AS created_at,
                       COALESCE(v.updated_at, t.updated_at) AS updated_at,
                       (COALESCE(1.0 / (60 + v.rank_v), 0) +
                        COALESCE(1.0 / (60 + t.rank_t), 0))::FLOAT8 AS rrf_score
                FROM vector_ranked v
                FULL OUTER JOIN text_ranked t ON v.id = t.id
            )
            SELECT id, conversation_id, owner_todo, note_key, content, note_type,
                   seq, done, pinned, knowledge_entry_id, created_at, updated_at
            FROM fused ORDER BY rrf_score DESC, updated_at DESC, id DESC LIMIT $11",
        )
        .bind(embedding_vec)
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(note_type)
        .bind(query)
        .bind(vb)
        .bind(me)
        .bind(ancestors)
        .bind(embedding_model)
        .bind(fetch_limit)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SpRow::into_note).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acceptance (#1167): the pass that measures the pad's spread reads the
    /// geometry and none of the content - one distance per note, and no column
    /// of the note itself. Only the notes the block may show are read whole.
    #[test]
    fn the_pad_scan_measures_before_it_reads_any_note() {
        let measured = NEAREST_NOTES_BY_EMBEDDING_SQL
            .split("     SELECT sp.id")
            .next()
            .expect("the scan selects the notes it will show after it measures the spread");

        for column in ["content", "note_type", "knowledge_entry_id"] {
            assert!(
                !measured.contains(column),
                "the pass that measures the spread reads {column}, which it has no use for"
            );
        }
    }

    /// The three scope bounds sit on the pass that measures, not only on the one
    /// that reads. A spread measured over another tenant's pad, or another
    /// conversation's, would grade this pad's notes against a geometry nothing
    /// here saw.
    #[test]
    fn the_pad_scan_scopes_the_pass_that_measures_the_spread() {
        let measured = NEAREST_NOTES_BY_EMBEDDING_SQL
            .split("     SELECT sp.id")
            .next()
            .expect("the scan measures the spread before it reads the notes");

        for bound in ["user_id = $2", "conversation_id = $3", "owner_todo"] {
            assert!(
                measured.contains(bound),
                "the pass that measures the spread is not bounded by {bound}: \
                 \n{NEAREST_NOTES_BY_EMBEDDING_SQL}"
            );
        }
    }

    /// The reserved `goal` note is out of the spread as well as out of the
    /// candidates, so the geometry describes the notes the arm can offer.
    #[test]
    fn the_pad_scan_leaves_the_goal_note_out_of_the_spread() {
        let measured = NEAREST_NOTES_BY_EMBEDDING_SQL
            .split("     SELECT sp.id")
            .next()
            .expect("the scan measures the spread before it reads the notes");

        assert!(
            measured.contains("note_key <> $9"),
            "the goal note must be absent from the spread as well as from the candidates: \
             \n{NEAREST_NOTES_BY_EMBEDDING_SQL}"
        );
    }
}
