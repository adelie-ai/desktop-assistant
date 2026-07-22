//! Postgres-backed adapter for `ScratchpadStore` (issue #184).
//!
//! Mirrors the patterns established by `PgKnowledgeBaseStore` and
//! `PgConversationSearchStore`:
//! - `(user_id, conversation_id)`-scoped queries throughout, with
//!   `current_user_id()` read from the task-local — nothing here takes a
//!   `UserId` parameter (see `desktop-assistant-core::ports::auth`).
//! - Cross-user reads return empty, not an error.
//! - The full-text `search` reuses the `plainto_tsquery` / `ts_rank_cd`
//!   shape from the conversation search adapter.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::ScratchpadNote;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::scratchpad::{NewScratchpadNote, ScratchpadStore};
use desktop_assistant_core::ports::scratchpad_scope::{
    current_ancestors, current_owner_todo, current_visible_before,
};
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

        let rows: Vec<SpRow> = sqlx::query_as(
            "INSERT INTO scratchpads \
                 (id, user_id, conversation_id, owner_todo, note_key, content, note_type, seq, done) \
             SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[], \
                                  $5::text[], $6::text[], $7::text[], $8::int4[], $9::bool[]) \
             ON CONFLICT (conversation_id, owner_todo, note_key) \
             DO UPDATE SET content = EXCLUDED.content, note_type = EXCLUDED.note_type, \
                           seq = EXCLUDED.seq, done = EXCLUDED.done, updated_at = NOW() \
             RETURNING id, conversation_id, owner_todo, note_key, content, note_type, seq, done, \
                       created_at, updated_at",
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
        .fetch_all(&self.pool)
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
            "SELECT id, conversation_id, owner_todo, note_key, content, note_type, seq, done, \
                    created_at, updated_at \
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
            "SELECT id, conversation_id, owner_todo, note_key, content, note_type, seq, done, \
                    created_at, updated_at \
             FROM scratchpads \
             WHERE user_id = $1 AND conversation_id = $2 \
               AND ($3::text IS NULL OR note_type = $3) \
               AND ($5::text IS NULL OR (owner_todo = $6 OR owner_todo LIKE $6 || '.%' \
                    OR (id COLLATE \"C\" < $5 AND owner_todo = ANY($7::text[])))) \
             ORDER BY note_type ASC, seq ASC NULLS LAST, updated_at DESC LIMIT $4",
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
        note_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        let user_id = current_user_id();
        // plainto_tsquery + ts_rank_cd, scoped — mirrors PgConversationSearchStore.
        // Search stays relevance-ranked; the optional `note_type` filter rides
        // a single static query via `IS NULL OR`.
        let (vb, me, ancestors) = read_snapshot();
        let rows: Vec<SpRow> = sqlx::query_as(
            "WITH q AS (SELECT plainto_tsquery('english', $3) AS query) \
             SELECT id, conversation_id, owner_todo, note_key, content, note_type, seq, done, \
                    created_at, updated_at \
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
}
