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
use desktop_assistant_core::ports::scratchpad::ScratchpadStore;
use sqlx::PgPool;

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
    note_key: String,
    content: String,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

impl SpRow {
    fn into_note(self) -> ScratchpadNote {
        ScratchpadNote {
            id: self.id,
            conversation_id: self.conversation_id,
            key: self.note_key,
            content: self.content,
            created_at: self.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
            updated_at: self.updated_at.format("%Y-%m-%d %H:%M:%S").to_string(),
        }
    }
}

impl ScratchpadStore for PgScratchpadStore {
    async fn write(
        &self,
        conversation_id: &str,
        notes: &[(String, String)],
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        if notes.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();

        // Batch upsert via UNNEST so a variable-length batch is a single
        // prepared statement. Zipping the parallel arrays yields one row per
        // note; the conflict target is `(conversation_id, note_key)` so a
        // repeated key replaces content and bumps `updated_at`. `id` and
        // `user_id` are only used on insert — an existing note keeps its
        // original id/owner on update.
        let ids: Vec<String> = (0..notes.len())
            .map(|_| uuid::Uuid::now_v7().to_string())
            .collect();
        let user_ids: Vec<String> = vec![user_id.as_str().to_string(); notes.len()];
        let conv_ids: Vec<String> = vec![conversation_id.to_string(); notes.len()];
        let keys: Vec<String> = notes.iter().map(|(k, _)| k.clone()).collect();
        let contents: Vec<String> = notes.iter().map(|(_, c)| c.clone()).collect();

        let rows: Vec<SpRow> = sqlx::query_as(
            "INSERT INTO scratchpads (id, user_id, conversation_id, note_key, content) \
             SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[], $5::text[]) \
             ON CONFLICT (conversation_id, note_key) \
             DO UPDATE SET content = EXCLUDED.content, updated_at = NOW() \
             RETURNING id, conversation_id, note_key, content, created_at, updated_at",
        )
        .bind(&ids)
        .bind(&user_ids)
        .bind(&conv_ids)
        .bind(&keys)
        .bind(&contents)
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
        let rows: Vec<SpRow> = sqlx::query_as(
            "SELECT id, conversation_id, note_key, content, created_at, updated_at \
             FROM scratchpads \
             WHERE user_id = $1 AND conversation_id = $2 AND note_key = ANY($3) \
             ORDER BY updated_at DESC LIMIT $4",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(keys)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SpRow::into_note).collect())
    }

    async fn list(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        let user_id = current_user_id();
        let rows: Vec<SpRow> = sqlx::query_as(
            "SELECT id, conversation_id, note_key, content, created_at, updated_at \
             FROM scratchpads \
             WHERE user_id = $1 AND conversation_id = $2 \
             ORDER BY updated_at DESC LIMIT $3",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.into_iter().map(SpRow::into_note).collect())
    }

    async fn search(
        &self,
        conversation_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScratchpadNote>, CoreError> {
        let user_id = current_user_id();
        // plainto_tsquery + ts_rank_cd, scoped — mirrors PgConversationSearchStore.
        let rows: Vec<SpRow> = sqlx::query_as(
            "WITH q AS (SELECT plainto_tsquery('english', $3) AS query) \
             SELECT id, conversation_id, note_key, content, created_at, updated_at \
             FROM scratchpads, q \
             WHERE user_id = $1 AND conversation_id = $2 AND tsv @@ q.query \
             ORDER BY ts_rank_cd(tsv, q.query) DESC, updated_at DESC LIMIT $4",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(query)
        .bind(limit as i64)
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
        let result =
            sqlx::query("DELETE FROM scratchpads WHERE user_id = $1 AND conversation_id = $2 AND note_key = ANY($3)")
                .bind(user_id.as_str())
                .bind(conversation_id)
                .bind(keys)
                .execute(&self.pool)
                .await
                .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result.rows_affected())
    }

    async fn clear(&self, conversation_id: &str) -> Result<u64, CoreError> {
        let user_id = current_user_id();
        let result =
            sqlx::query("DELETE FROM scratchpads WHERE user_id = $1 AND conversation_id = $2")
                .bind(user_id.as_str())
                .bind(conversation_id)
                .execute(&self.pool)
                .await
                .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result.rows_affected())
    }
}
