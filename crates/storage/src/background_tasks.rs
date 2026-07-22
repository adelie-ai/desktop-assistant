//! Postgres-backed adapter for `BackgroundTaskStore` (issue #115).
//!
//! Mirrors the patterns established by `PgTurnStateStore`:
//! - `(user_id, …)` scoped queries for everything except the
//!   `scan_non_terminal` system hook
//! - cross-user reads return `Ok(None)`, not an error (#105)
//! - `current_user_id()` from the task-local; user-id is only passed
//!   in explicitly on `create_task` (the row's owner is supplied by
//!   the caller) and on `list_tasks_for_user` (a small ergonomic
//!   exception — the registry already has the owning user in hand).

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::store::{
    BackgroundTaskRow, BackgroundTaskStatus, BackgroundTaskStore,
};
use sqlx::PgPool;

/// Postgres adapter for the `background_tasks` table.
pub struct PgBackgroundTaskStore {
    pool: PgPool,
}

impl PgBackgroundTaskStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

// Tuple-to-row helper. Argument count matches the column list; folding
// these into a struct would just shift the verbosity into the caller.
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
    owner_todo: String,
    spawn_marker: Option<String>,
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
        owner_todo,
        spawn_marker,
    })
}

#[async_trait]
impl BackgroundTaskStore for PgBackgroundTaskStore {
    async fn create_task(&self, row: BackgroundTaskRow) -> Result<(), CoreError> {
        // The caller owns the identity decision — typically the
        // application layer reads `current_user_id()` and bakes it
        // into the row before calling. Mirrors `PgTurnStateStore::create_turn`.
        let result = sqlx::query(
            "INSERT INTO background_tasks (\
                id, user_id, kind_json, task_status, parent_task_id, title, \
                last_error, progress_hint, started_at, ended_at, owner_todo, spawn_marker) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
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
        .bind(&row.owner_todo)
        .bind(row.spawn_marker.as_deref())
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
        // (user_id, id) — the composite filter keeps cross-user probes
        // from leaking existence.
        let row: Option<(
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
            String,
            Option<String>,
        )> = sqlx::query_as(
            "SELECT id, user_id, kind_json, task_status, parent_task_id, \
                    title, last_error, progress_hint, started_at, ended_at, \
                    owner_todo, spawn_marker \
             FROM background_tasks \
             WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        match row {
            None => Ok(None),
            Some((
                id,
                user_id,
                kind_json,
                status_key,
                parent_task_id,
                title,
                last_error,
                progress_hint,
                started_at,
                ended_at,
                owner_todo,
                spawn_marker,
            )) => Ok(Some(row_from_db(
                id,
                user_id,
                kind_json,
                status_key,
                parent_task_id,
                title,
                last_error,
                progress_hint,
                started_at,
                ended_at,
                owner_todo,
                spawn_marker,
            )?)),
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
             SET task_status = $3, last_error = $4, progress_hint = $5, \
                 ended_at = $6, updated_at = now() \
             WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(id)
        .bind(status.as_key())
        .bind(last_error)
        .bind(progress_hint)
        .bind(ended_at)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        if result.rows_affected() == 0 {
            // Same opacity rule as `get_task`: don't distinguish
            // "doesn't exist" from "not yours".
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
        // Build the query in two parts so the optional WHERE clause
        // stays statically typed in sqlx.
        let limit_value: i64 = limit.map(|l| l as i64).unwrap_or(i64::MAX);
        let rows: Vec<(
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
            String,
            Option<String>,
        )> = if include_finished {
            sqlx::query_as(
                "SELECT id, user_id, kind_json, task_status, parent_task_id, \
                        title, last_error, progress_hint, started_at, ended_at, \
                        owner_todo, spawn_marker \
                 FROM background_tasks \
                 WHERE user_id = $1 \
                 ORDER BY started_at DESC \
                 LIMIT $2",
            )
            .bind(user_id)
            .bind(limit_value)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
        } else {
            sqlx::query_as(
                "SELECT id, user_id, kind_json, task_status, parent_task_id, \
                        title, last_error, progress_hint, started_at, ended_at, \
                        owner_todo, spawn_marker \
                 FROM background_tasks \
                 WHERE user_id = $1 AND task_status IN ('pending', 'running') \
                 ORDER BY started_at DESC \
                 LIMIT $2",
            )
            .bind(user_id)
            .bind(limit_value)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
        };
        let mut out = Vec::with_capacity(rows.len());
        for (
            id,
            user_id,
            kind_json,
            status_key,
            parent_task_id,
            title,
            last_error,
            progress_hint,
            started_at,
            ended_at,
            owner_todo,
            spawn_marker,
        ) in rows
        {
            out.push(row_from_db(
                id,
                user_id,
                kind_json,
                status_key,
                parent_task_id,
                title,
                last_error,
                progress_hint,
                started_at,
                ended_at,
                owner_todo,
                spawn_marker,
            )?);
        }
        Ok(out)
    }

    async fn scan_non_terminal(&self) -> Result<Vec<BackgroundTaskRow>, CoreError> {
        // No user_id filter — see method doc on the trait.
        let rows: Vec<(
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
            String,
            Option<String>,
        )> = sqlx::query_as(
            "SELECT id, user_id, kind_json, task_status, parent_task_id, \
                    title, last_error, progress_hint, started_at, ended_at, \
                    owner_todo, spawn_marker \
             FROM background_tasks \
             WHERE task_status NOT IN ('completed', 'failed', 'cancelled')",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut out = Vec::with_capacity(rows.len());
        for (
            id,
            user_id,
            kind_json,
            status_key,
            parent_task_id,
            title,
            last_error,
            progress_hint,
            started_at,
            ended_at,
            owner_todo,
            spawn_marker,
        ) in rows
        {
            out.push(row_from_db(
                id,
                user_id,
                kind_json,
                status_key,
                parent_task_id,
                title,
                last_error,
                progress_hint,
                started_at,
                ended_at,
                owner_todo,
                spawn_marker,
            )?);
        }
        Ok(out)
    }
}
