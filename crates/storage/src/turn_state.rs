//! Postgres-backed adapter for `TurnStateStore` (issue #107).
//!
//! Mirrors the patterns established by `PgConversationStore`:
//! - `(user_id, …)` scoped queries throughout
//! - Cross-user reads return `None` (or "not found"), not an error — see
//!   `desktop-assistant-core::ports::auth` and the #105 PR for the rule
//! - `current_user_id()` from a task-local; nothing in this adapter
//!   takes a `UserId` parameter
//!
//! The `scan_non_terminal` method intentionally bypasses the task-local
//! because the caller is a system task — `client_tools::sweep_non_terminal_turns_on_startup`
//! runs once at daemon boot before any JWT context exists. See the
//! method's doc comment in `core::ports::store` for the rationale.

use async_trait::async_trait;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::store::{
    TurnRow, TurnStateJson, TurnStateStore, TurnStatus,
};
use sqlx::PgPool;

/// Postgres adapter for the turn state table.
pub struct PgTurnStateStore {
    pool: PgPool,
}

impl PgTurnStateStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

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
    let state: TurnStateJson = serde_json::from_value(state_json).map_err(|e| {
        CoreError::Storage(format!("malformed turn state_json for {id}: {e}"))
    })?;
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
impl TurnStateStore for PgTurnStateStore {
    async fn create_turn(&self, row: TurnRow) -> Result<(), CoreError> {
        let state_json = serde_json::to_value(&row.state).map_err(|e| {
            CoreError::Storage(format!("serialize turn state_json: {e}"))
        })?;
        // We deliberately use the row's user_id verbatim instead of
        // overwriting it with `current_user_id()`. The caller owns the
        // identity decision — typically the application layer reads
        // `current_user_id()` and bakes it into the row before
        // calling. This mirrors how `PgConversationStore::create`
        // works, and matches the multi-tenant pattern in #105.
        let result = sqlx::query(
            "INSERT INTO turns (id, user_id, conversation_id, status, state_json, last_error) \
             VALUES ($1, $2, $3, $4, $5, $6)",
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
            Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => {
                Err(CoreError::Storage(format!(
                    "turn id already exists: {}",
                    row.id
                )))
            }
            Err(e) => Err(CoreError::Storage(e.to_string())),
        }
    }

    async fn get_turn(&self, id: &str) -> Result<Option<TurnRow>, CoreError> {
        let user_id = current_user_id();
        // (user_id, id) — the (user_id, …) composite filter keeps cross-
        // user probes from leaking existence.
        let row: Option<(String, String, String, String, serde_json::Value, Option<String>)> =
            sqlx::query_as(
                "SELECT id, user_id, conversation_id, status, state_json, last_error \
                 FROM turns \
                 WHERE user_id = $1 AND id = $2",
            )
            .bind(user_id.as_str())
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match row {
            None => Ok(None),
            Some((id, user_id, conversation_id, status_key, state_json, last_error)) => Ok(Some(
                row_from_db(id, user_id, conversation_id, status_key, state_json, last_error)?,
            )),
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
        let state_json = serde_json::to_value(state).map_err(|e| {
            CoreError::Storage(format!("serialize turn state_json: {e}"))
        })?;
        let result = sqlx::query(
            "UPDATE turns \
             SET status = $3, state_json = $4, last_error = $5, updated_at = now() \
             WHERE user_id = $1 AND id = $2",
        )
        .bind(user_id.as_str())
        .bind(id)
        .bind(status.as_key())
        .bind(state_json)
        .bind(last_error)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        if result.rows_affected() == 0 {
            // Same opacity rule as `get_turn`: don't distinguish
            // "doesn't exist" from "not yours".
            return Err(CoreError::Storage(format!(
                "turn not found for current user: {id}"
            )));
        }
        Ok(())
    }

    async fn scan_non_terminal(&self) -> Result<Vec<TurnRow>, CoreError> {
        // No user_id filter — see method doc on the trait.
        let rows: Vec<(String, String, String, String, serde_json::Value, Option<String>)> =
            sqlx::query_as(
                "SELECT id, user_id, conversation_id, status, state_json, last_error \
                 FROM turns \
                 WHERE status NOT IN ('complete', 'failed')",
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
