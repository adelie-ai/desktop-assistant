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

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ScratchpadNote;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::scratchpad::{NewScratchpadNote, ScratchpadStore};
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
}

impl PgScratchpadStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

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
        // Not confined by `owner_todo`: this repairs rows the render path has
        // just read, so there is no model-driven caller to confine (see the
        // port doc). `user_id = $1` stays, because that is the tenant guard.
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
             WHERE user_id = $1 AND conversation_id = $2 AND id = ANY($3) \
               AND knowledge_entry_id IS NOT NULL",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
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
