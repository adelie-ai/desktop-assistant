//! SQLite adapter for [`TurnStateStore`] (issue #107).
//!
//! Mirrors `PgTurnStateStore`: `(user_id, …)` scoped throughout, cross-user
//! reads return `None`/"not found", and `scan_non_terminal` intentionally
//! bypasses the per-user scope because its caller is the startup sweep (a
//! system task with no JWT context). `state_json` (Postgres JSONB) is a TEXT
//! column holding JSON, read/written as `serde_json::Value`.

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
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

type TurnTuple = (
    String,
    String,
    String,
    String,
    serde_json::Value,
    Option<String>,
);

fn row_from_db(
    id: String,
    user_id: String,
    conversation_id: String,
    status_key: String,
    state_json: serde_json::Value,
    last_error: Option<String>,
) -> Result<TurnRow, CoreError> {
    let status = TurnStatus::from_key(&status_key).ok_or_else(|| {
        CoreError::Storage(format!(
            "unknown turn status in DB: {status_key} for turn {id}"
        ))
    })?;
    let state: TurnStateJson = serde_json::from_value(state_json)
        .map_err(|e| CoreError::Storage(format!("malformed turn state_json for {id}: {e}")))?;
    Ok(TurnRow {
        id,
        user_id,
        conversation_id,
        status,
        state,
        last_error,
    })
}

#[async_trait]
impl TurnStateStore for SqliteTurnStateStore {
    async fn create_turn(&self, row: TurnRow) -> Result<(), CoreError> {
        let state_json = serde_json::to_value(&row.state)
            .map_err(|e| CoreError::Storage(format!("serialize turn state_json: {e}")))?;
        // The caller bakes `current_user_id()` into the row before calling
        // (mirrors `PgTurnStateStore::create_turn`).
        let result = sqlx::query(
            "INSERT INTO turns (id, user_id, conversation_id, status, state_json, last_error) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&row.id)
        .bind(&row.user_id)
        .bind(&row.conversation_id)
        .bind(row.status.as_key())
        .bind(state_json)
        .bind(row.last_error.as_deref())
        .execute(&self.pool)
        .await;
        match result {
            Ok(_) => Ok(()),
            Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => Err(
                CoreError::Storage(format!("turn id already exists: {}", row.id)),
            ),
            Err(e) => Err(CoreError::Storage(e.to_string())),
        }
    }

    async fn get_turn(&self, id: &str) -> Result<Option<TurnRow>, CoreError> {
        let user_id = current_user_id();
        let row: Option<TurnTuple> = sqlx::query_as(
            "SELECT id, user_id, conversation_id, status, state_json, last_error \
             FROM turns WHERE user_id = ? AND id = ?",
        )
        .bind(user_id.as_str())
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        match row {
            None => Ok(None),
            Some((id, user_id, conversation_id, status_key, state_json, last_error)) => {
                Ok(Some(row_from_db(
                    id,
                    user_id,
                    conversation_id,
                    status_key,
                    state_json,
                    last_error,
                )?))
            }
        }
    }

    async fn update_turn(
        &self,
        id: &str,
        status: TurnStatus,
        state: &TurnStateJson,
        last_error: Option<&str>,
    ) -> Result<(), CoreError> {
        let user_id = current_user_id();
        let state_json = serde_json::to_value(state)
            .map_err(|e| CoreError::Storage(format!("serialize turn state_json: {e}")))?;
        let result = sqlx::query(
            "UPDATE turns \
             SET status = ?, state_json = ?, last_error = ?, updated_at = CURRENT_TIMESTAMP \
             WHERE user_id = ? AND id = ?",
        )
        .bind(status.as_key())
        .bind(state_json)
        .bind(last_error)
        .bind(user_id.as_str())
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        if result.rows_affected() == 0 {
            return Err(CoreError::Storage(format!(
                "turn not found for current user: {id}"
            )));
        }
        Ok(())
    }

    async fn scan_non_terminal(&self) -> Result<Vec<TurnRow>, CoreError> {
        // No user_id filter — see the method doc on the trait.
        let rows: Vec<TurnTuple> = sqlx::query_as(
            "SELECT id, user_id, conversation_id, status, state_json, last_error \
             FROM turns WHERE status NOT IN ('complete', 'failed')",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut out = Vec::with_capacity(rows.len());
        for (id, user_id, conversation_id, status_key, state_json, last_error) in rows {
            out.push(row_from_db(
                id,
                user_id,
                conversation_id,
                status_key,
                state_json,
                last_error,
            )?);
        }
        Ok(out)
    }
}
