//! SQLite adapter for [`BackgroundTaskStore`] (issue #115).
//!
//! Mirrors `PgBackgroundTaskStore`: `(user_id, …)` scoped everywhere except the
//! `scan_non_terminal` system hook, cross-user reads return `Ok(None)`, and the
//! owning user is supplied by the caller on `create_task`/`list_tasks_for_user`.
//! `kind_json` (Postgres JSONB) is a TEXT column holding JSON; `started_at` /
//! `ended_at` (Postgres BIGINT) are INTEGER epoch-millis.

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
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

/// Column tuple selected for a background-task row.
type TaskTuple = (
    String,
    String,
    serde_json::Value,
    String,
    Option<String>,
    String,
    Option<String>,
    Option<String>,
    i64,
    Option<i64>,
);

// Argument count matches the column list; folding into a struct would just move
// the verbosity to the caller.
#[allow(clippy::too_many_arguments)]
fn row_from_db(
    id: String,
    user_id: String,
    kind_json: serde_json::Value,
    status_key: String,
    parent_task_id: Option<String>,
    title: String,
    last_error: Option<String>,
    progress_hint: Option<String>,
    started_at: i64,
    ended_at: Option<i64>,
) -> Result<BackgroundTaskRow, CoreError> {
    let status = BackgroundTaskStatus::from_key(&status_key).ok_or_else(|| {
        CoreError::Storage(format!(
            "unknown background task status in DB: {status_key} for task {id}"
        ))
    })?;
    Ok(BackgroundTaskRow {
        id,
        user_id,
        kind_json,
        status,
        parent_task_id,
        title,
        last_error,
        progress_hint,
        started_at,
        ended_at,
    })
}

fn row_from_tuple(t: TaskTuple) -> Result<BackgroundTaskRow, CoreError> {
    row_from_db(t.0, t.1, t.2, t.3, t.4, t.5, t.6, t.7, t.8, t.9)
}

#[async_trait]
impl BackgroundTaskStore for SqliteBackgroundTaskStore {
    async fn create_task(&self, row: BackgroundTaskRow) -> Result<(), CoreError> {
        let result = sqlx::query(
            "INSERT INTO background_tasks (\
                id, user_id, kind_json, task_status, parent_task_id, title, \
                last_error, progress_hint, started_at, ended_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&row.id)
        .bind(&row.user_id)
        .bind(&row.kind_json)
        .bind(row.status.as_key())
        .bind(row.parent_task_id.as_deref())
        .bind(&row.title)
        .bind(row.last_error.as_deref())
        .bind(row.progress_hint.as_deref())
        .bind(row.started_at)
        .bind(row.ended_at)
        .execute(&self.pool)
        .await;
        match result {
            Ok(_) => Ok(()),
            Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => Err(
                CoreError::Storage(format!("background task id already exists: {}", row.id)),
            ),
            Err(e) => Err(CoreError::Storage(e.to_string())),
        }
    }

    async fn get_task(&self, id: &str) -> Result<Option<BackgroundTaskRow>, CoreError> {
        let user_id = current_user_id();
        let row: Option<TaskTuple> = sqlx::query_as(
            "SELECT id, user_id, kind_json, task_status, parent_task_id, \
                    title, last_error, progress_hint, started_at, ended_at \
             FROM background_tasks WHERE user_id = ? AND id = ?",
        )
        .bind(user_id.as_str())
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        match row {
            None => Ok(None),
            Some(t) => Ok(Some(row_from_tuple(t)?)),
        }
    }

    async fn update_task(
        &self,
        id: &str,
        status: BackgroundTaskStatus,
        last_error: Option<&str>,
        progress_hint: Option<&str>,
        ended_at: Option<i64>,
    ) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let result = sqlx::query(
            "UPDATE background_tasks \
             SET task_status = ?, last_error = ?, progress_hint = ?, \
                 ended_at = ?, updated_at = CURRENT_TIMESTAMP \
             WHERE user_id = ? AND id = ?",
        )
        .bind(status.as_key())
        .bind(last_error)
        .bind(progress_hint)
        .bind(ended_at)
        .bind(user_id.as_str())
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        if result.rows_affected() == 0 {
            return Err(CoreError::Storage(format!(
                "background task not found for current user: {id}"
            )));
        }
        Ok(())
    }

    async fn list_tasks_for_user(
        &self,
        user_id: &str,
        include_finished: bool,
        limit: Option<u32>,
    ) -> Result<Vec<BackgroundTaskRow>, CoreError> {
        // SQLite `LIMIT -1` means "no limit"; use it when the caller passes None.
        // Two literal statements (rather than an interpolated status filter) keep
        // the SQL statically typed and injection-free.
        let limit_value: i64 = limit.map(|l| l as i64).unwrap_or(-1);
        let rows: Vec<TaskTuple> = if include_finished {
            sqlx::query_as(
                "SELECT id, user_id, kind_json, task_status, parent_task_id, \
                        title, last_error, progress_hint, started_at, ended_at \
                 FROM background_tasks WHERE user_id = ? \
                 ORDER BY started_at DESC LIMIT ?",
            )
            .bind(user_id)
            .bind(limit_value)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
        } else {
            sqlx::query_as(
                "SELECT id, user_id, kind_json, task_status, parent_task_id, \
                        title, last_error, progress_hint, started_at, ended_at \
                 FROM background_tasks \
                 WHERE user_id = ? AND task_status IN ('pending', 'running') \
                 ORDER BY started_at DESC LIMIT ?",
            )
            .bind(user_id)
            .bind(limit_value)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
        };
        rows.into_iter().map(row_from_tuple).collect()
    }

    async fn scan_non_terminal(&self) -> Result<Vec<BackgroundTaskRow>, CoreError> {
        // No user_id filter — see method doc on the trait.
        let rows: Vec<TaskTuple> = sqlx::query_as(
            "SELECT id, user_id, kind_json, task_status, parent_task_id, \
                    title, last_error, progress_hint, started_at, ended_at \
             FROM background_tasks \
             WHERE task_status NOT IN ('completed', 'failed', 'cancelled')",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        rows.into_iter().map(row_from_tuple).collect()
    }
}
