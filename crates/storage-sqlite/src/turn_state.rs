//! SQLite adapter for [`TurnStateStore`] (issue #107).

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{TurnRow, TurnStateJson, TurnStateStore, TurnStatus};
use sqlx::SqlitePool;

/// SQLite adapter for the `turns` table.
pub struct SqliteTurnStateStore {
    pool: SqlitePool,
}

impl SqliteTurnStateStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl TurnStateStore for SqliteTurnStateStore {
    async fn create_turn(&self, row: TurnRow) -> Result<(), CoreError> {
        let _ = (&self.pool, row);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn get_turn(&self, id: &str) -> Result<Option<TurnRow>, CoreError> {
        let _ = (&self.pool, id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn update_turn(
        &self,
        id: &str,
        status: TurnStatus,
        state: &TurnStateJson,
        last_error: Option<&str>,
    ) -> Result<(), CoreError> {
        let _ = (&self.pool, id, status, state, last_error);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn scan_non_terminal(&self) -> Result<Vec<TurnRow>, CoreError> {
        let _ = &self.pool;
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }
}
