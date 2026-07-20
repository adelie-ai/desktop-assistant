//! SQLite adapter for [`LearnedWindowStore`] (issues #343 / #425).
//!
//! Global (no `user_id`): connector/model knowledge (how large a window a
//! hosted model actually accepts), not personal data.
//!
//! `record_overflow` enforces the DOWN-ONLY ratchet at the SQL level; a
//! deliberate configured-window change replaces the observation wholesale.
//! `record_success` keeps the LARGEST measured input (high-water) independently
//! of any overflow observation. Postgres-ism translations: `GREATEST` -> the
//! scalar `MAX(a, b)`, `IS DISTINCT FROM` -> the null-safe `IS NOT`, `now()` ->
//! `CURRENT_TIMESTAMP`; `excluded` is SQLite's spelling of `EXCLUDED`.

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
        let row = sqlx::query_as::<_, (Option<i64>, Option<i64>, Option<i64>)>(
            "SELECT observed_limit, configured_window, max_success_input \
               FROM context_window_observations \
              WHERE connector = ? AND model = ?",
        )
        .bind(connector)
        .bind(model)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Stored as INTEGER (i64) but always non-negative on the write path;
        // clamp defensively so a hand-edited negative row can't underflow into a
        // huge u64. A NULL column stays `None`.
        let non_neg = |v: Option<i64>| v.map(|n| n.max(0) as u64);
        Ok(row.map(|(observed, configured, success)| LearnedWindow {
            observed_limit: non_neg(observed),
            configured_window: non_neg(configured),
            max_success_input: non_neg(success),
        }))
    }

    async fn record_overflow(
        &self,
        connector: &str,
        model: &str,
        observed_limit: u64,
        configured_window: u64,
    ) -> Result<(), CoreError> {
        // Down-only ratchet in one statement, leaving `max_success_input`
        // untouched: INSERT the observation; ON CONFLICT, overwrite only when
        // the configured window CHANGED (stale -> start fresh), the stored
        // observation is NULL (success-only row), or the new observed limit is
        // strictly smaller (ratchet down). `IS NOT` is null-safe.
        sqlx::query(
            "INSERT INTO context_window_observations \
                 (connector, model, observed_limit, configured_window, updated_at) \
             VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) \
             ON CONFLICT (connector, model) DO UPDATE SET \
                 observed_limit = excluded.observed_limit, \
                 configured_window = excluded.configured_window, \
                 updated_at = CURRENT_TIMESTAMP \
             WHERE context_window_observations.configured_window \
                       IS NOT excluded.configured_window \
                OR context_window_observations.observed_limit IS NULL \
                OR excluded.observed_limit < context_window_observations.observed_limit",
        )
        .bind(connector)
        .bind(model)
        .bind(observed_limit as i64)
        .bind(configured_window as i64)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn record_success(
        &self,
        connector: &str,
        model: &str,
        input_tokens: u64,
    ) -> Result<(), CoreError> {
        // High-water: keep the LARGEST measured input we've seen succeed,
        // leaving any overflow observation on the row untouched. A brand-new row
        // has NULL observed_limit/configured_window (both nullable, #425).
        sqlx::query(
            "INSERT INTO context_window_observations \
                 (connector, model, max_success_input, updated_at) \
             VALUES (?, ?, ?, CURRENT_TIMESTAMP) \
             ON CONFLICT (connector, model) DO UPDATE SET \
                 max_success_input = MAX( \
                     COALESCE(context_window_observations.max_success_input, 0), \
                     excluded.max_success_input), \
                 updated_at = CURRENT_TIMESTAMP",
        )
        .bind(connector)
        .bind(model)
        .bind(input_tokens as i64)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }
}
