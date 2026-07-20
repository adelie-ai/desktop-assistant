//! SQLite adapter for [`IdempotencyKeyStore`] (issue #204).

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
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
        let _ = (&self.pool, conversation_id, idempotency_key);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn record_response(
        &self,
        conversation_id: &str,
        idempotency_key: &str,
        request_id: &str,
        response: &str,
    ) -> Result<(), CoreError> {
        let _ = (
            &self.pool,
            conversation_id,
            idempotency_key,
            request_id,
            response,
        );
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }
}
