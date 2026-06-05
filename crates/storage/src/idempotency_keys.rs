//! Postgres adapter for `IdempotencyKeyStore` (#204).
//!
//! Records the committed reply of a completed `SendMessage` turn keyed by
//! `(user_id, conversation_id, idempotency_key)` so a client can safely retry
//! a dropped turn: a retry whose original already finished replays the stored
//! reply instead of re-running the LLM/tools. Mirrors the other Pg stores —
//! every query scopes to `current_user_id()` (read from the task-local),
//! nothing takes a `UserId` parameter, and a cross-user lookup behaves like
//! the row doesn't exist.

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::store::IdempotencyKeyStore;
use sqlx::PgPool;

/// Postgres adapter for the `idempotency_keys` table.
pub struct PgIdempotencyKeyStore {
    pool: PgPool,
}

impl PgIdempotencyKeyStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl IdempotencyKeyStore for PgIdempotencyKeyStore {
    async fn lookup_completed(
        &self,
        conversation_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<String>, CoreError> {
        let user_id = current_user_id();
        let row: Option<(String,)> = sqlx::query_as(
            "SELECT response FROM idempotency_keys \
             WHERE user_id = $1 AND conversation_id = $2 AND idempotency_key = $3",
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
             VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (user_id, conversation_id, idempotency_key) \
             DO UPDATE SET response = EXCLUDED.response, request_id = EXCLUDED.request_id",
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
