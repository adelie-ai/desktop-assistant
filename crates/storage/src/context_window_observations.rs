//! Postgres-backed [`LearnedWindowStore`] — the learned effective
//! context-window cache (issue #343), the reactive safety net that complements
//! #342.
//!
//! Global (no `user_id`): this is connector/model knowledge (how large a window
//! a hosted model actually accepts), not personal data.
//!
//! `record` enforces the DOWN-ONLY ratchet at the SQL level:
//!   - a fresh `(connector, model)` inserts the observation;
//!   - an existing row with the SAME `configured_window` is overwritten only
//!     when the new `observed_limit` is strictly smaller (ratchet down);
//!   - an existing row with a DIFFERENT `configured_window` is replaced
//!     wholesale — a deliberate window change (#342) invalidates the old
//!     observation and starts fresh.
//!
//! `lookup` is a plain key read; invalidation (comparing `configured_window` to
//! the current effective budget) is the caller's job, in `apply_learned_cap`.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{LearnedWindow, LearnedWindowStore};
use sqlx::PgPool;

pub struct PgLearnedWindowStore {
    pool: PgPool,
}

impl PgLearnedWindowStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait::async_trait]
impl LearnedWindowStore for PgLearnedWindowStore {
    async fn lookup(
        &self,
        connector: &str,
        model: &str,
    ) -> Result<Option<LearnedWindow>, CoreError> {
        let row = sqlx::query_as::<_, (Option<i64>, Option<i64>, Option<i64>)>(
            "SELECT observed_limit, configured_window, max_success_input \
               FROM context_window_observations \
              WHERE connector = $1 AND model = $2",
        )
        .bind(connector)
        .bind(model)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Stored as BIGINT (i64) but always non-negative on the write path;
        // clamp defensively so a manually-edited negative row can't underflow
        // into a huge u64. A NULL column stays `None`.
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
        // untouched (the UPDATE SET omits it):
        //   - INSERT the new observation (fresh rows have NULL success);
        //   - ON CONFLICT, replace when the configured window CHANGED (stale →
        //     start fresh) OR the new observed limit is strictly smaller than
        //     the stored one for the same configured window (ratchet down).
        //     `IS DISTINCT FROM` handles a success-only row whose
        //     `configured_window`/`observed_limit` are still NULL.
        sqlx::query(
            "INSERT INTO context_window_observations \
                 (connector, model, observed_limit, configured_window, updated_at) \
             VALUES ($1, $2, $3, $4, now()) \
             ON CONFLICT (connector, model) DO UPDATE SET \
                 observed_limit = EXCLUDED.observed_limit, \
                 configured_window = EXCLUDED.configured_window, \
                 updated_at = now() \
             WHERE context_window_observations.configured_window \
                       IS DISTINCT FROM EXCLUDED.configured_window \
                OR context_window_observations.observed_limit IS NULL \
                OR EXCLUDED.observed_limit < context_window_observations.observed_limit",
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
        // leaving any overflow observation on the row untouched.
        sqlx::query(
            "INSERT INTO context_window_observations \
                 (connector, model, max_success_input, updated_at) \
             VALUES ($1, $2, $3, now()) \
             ON CONFLICT (connector, model) DO UPDATE SET \
                 max_success_input = GREATEST( \
                     COALESCE(context_window_observations.max_success_input, 0), \
                     EXCLUDED.max_success_input), \
                 updated_at = now()",
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
