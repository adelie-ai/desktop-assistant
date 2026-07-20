//! SQLite adapter for [`LearnedWindowStore`] (issues #343 / #425). Global
//! store, no `user_id` — connector/model knowledge, not personal data.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{LearnedWindow, LearnedWindowStore};
use sqlx::SqlitePool;

/// SQLite adapter for the `context_window_observations` table.
pub struct SqliteLearnedWindowStore {
    pool: SqlitePool,
}

impl SqliteLearnedWindowStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait::async_trait]
impl LearnedWindowStore for SqliteLearnedWindowStore {
    async fn lookup(
        &self,
        connector: &str,
        model: &str,
    ) -> Result<Option<LearnedWindow>, CoreError> {
        let _ = (&self.pool, connector, model);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn record_overflow(
        &self,
        connector: &str,
        model: &str,
        observed_limit: u64,
        configured_window: u64,
    ) -> Result<(), CoreError> {
        let _ = (
            &self.pool,
            connector,
            model,
            observed_limit,
            configured_window,
        );
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn record_success(
        &self,
        connector: &str,
        model: &str,
        input_tokens: u64,
    ) -> Result<(), CoreError> {
        let _ = (&self.pool, connector, model, input_tokens);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }
}
