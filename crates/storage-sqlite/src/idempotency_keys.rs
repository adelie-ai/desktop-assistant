//! SQLite adapter for [`IdempotencyKeyStore`] (issue #204).
//!
//! Records the committed reply of a completed `SendMessage` turn keyed by
//! `(user_id, conversation_id, idempotency_key)` so a dropped-then-retried turn
//! replays instead of re-running. Every query scopes to `current_user_id()`;
//! a cross-user lookup behaves like the row doesn't exist. `record_response`
//! is an idempotent upsert (`ON CONFLICT … DO UPDATE`).

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::store::IdempotencyKeyStore;
use sqlx::SqlitePool;

/// SQLite adapter for the `idempotency_keys` table.
pub struct SqliteIdempotencyKeyStore {
    pool: SqlitePool,
}

impl SqliteIdempotencyKeyStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl IdempotencyKeyStore for SqliteIdempotencyKeyStore {
    async fn lookup_completed(
        &self,
        conversation_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<String>, CoreError> {
        let user_id = current_user_id();
        let row: Option<(String,)> = sqlx::query_as(
            "SELECT response FROM idempotency_keys \
             WHERE user_id = ? AND conversation_id = ? AND idempotency_key = ?",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(idempotency_key)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(row.map(|(response,)| response))
    }

    async fn record_response(
        &self,
        conversation_id: &str,
        idempotency_key: &str,
        request_id: &str,
        response: &str,
    ) -> Result<(), CoreError> {
        let user_id = current_user_id();
        // Upsert so a turn that raced past the dedup check and ran twice still
        // converges to a single row (idempotent record).
        sqlx::query(
            "INSERT INTO idempotency_keys \
                (user_id, conversation_id, idempotency_key, request_id, response) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT (user_id, conversation_id, idempotency_key) \
             DO UPDATE SET response = excluded.response, request_id = excluded.request_id",
        )
        .bind(user_id.as_str())
        .bind(conversation_id)
        .bind(idempotency_key)
        .bind(request_id)
        .bind(response)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }
}
