//! SQLite adapter for [`ErrorClassificationStore`] (epic #178). Global store,
//! no `user_id` — connector knowledge, not personal data.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{ErrorClassificationStore, LearnedClassification};
use sqlx::SqlitePool;

/// SQLite adapter for the `error_classifications` table.
pub struct SqliteErrorClassificationStore {
    pool: SqlitePool,
}

impl SqliteErrorClassificationStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait::async_trait]
impl ErrorClassificationStore for SqliteErrorClassificationStore {
    async fn lookup(
        &self,
        connector: &str,
        message: &str,
    ) -> Result<Option<LearnedClassification>, CoreError> {
        let _ = (&self.pool, connector, message);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn record(&self, connector: &str, signature: &str, cause: &str) -> Result<(), CoreError> {
        let _ = (&self.pool, connector, signature, cause);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }
}
