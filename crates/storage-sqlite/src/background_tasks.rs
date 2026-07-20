//! SQLite adapter for [`BackgroundTaskStore`] (issue #115).

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{
    BackgroundTaskRow, BackgroundTaskStatus, BackgroundTaskStore,
};
use sqlx::SqlitePool;

/// SQLite adapter for the `background_tasks` table.
pub struct SqliteBackgroundTaskStore {
    pool: SqlitePool,
}

impl SqliteBackgroundTaskStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl BackgroundTaskStore for SqliteBackgroundTaskStore {
    async fn create_task(&self, row: BackgroundTaskRow) -> Result<(), CoreError> {
        let _ = (&self.pool, row);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn get_task(&self, id: &str) -> Result<Option<BackgroundTaskRow>, CoreError> {
        let _ = (&self.pool, id);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn update_task(
        &self,
        id: &str,
        status: BackgroundTaskStatus,
        last_error: Option<&str>,
        progress_hint: Option<&str>,
        ended_at: Option<i64>,
    ) -> Result<(), CoreError> {
        let _ = (&self.pool, id, status, last_error, progress_hint, ended_at);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn list_tasks_for_user(
        &self,
        user_id: &str,
        include_finished: bool,
        limit: Option<u32>,
    ) -> Result<Vec<BackgroundTaskRow>, CoreError> {
        let _ = (&self.pool, user_id, include_finished, limit);
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }

    async fn scan_non_terminal(&self) -> Result<Vec<BackgroundTaskRow>, CoreError> {
        let _ = &self.pool;
        Err(CoreError::Storage("inc1 stub: unimplemented".into()))
    }
}
