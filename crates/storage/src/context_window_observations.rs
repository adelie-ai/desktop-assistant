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
        let row = sqlx::query_as::<_, (i64, i64)>(
            "SELECT observed_limit, configured_window \
               FROM context_window_observations \
              WHERE connector = $1 AND model = $2",
        )
        .bind(connector)
        .bind(model)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Stored as BIGINT (i64) but always non-negative on the write path
        // (`record` rejects values below the sanity floor); clamp defensively
        // so a manually-edited negative row can't underflow into a huge u64.
        Ok(row.map(|(observed, configured)| LearnedWindow {
            observed_limit: observed.max(0) as u64,
            configured_window: configured.max(0) as u64,
        }))
    }

    async fn record(
        &self,
        connector: &str,
        model: &str,
        observed_limit: u64,
        configured_window: u64,
    ) -> Result<(), CoreError> {
        // Down-only ratchet in one statement:
        //   - INSERT the new observation;
        //   - ON CONFLICT, replace when the configured window CHANGED (stale →
        //     start fresh) OR the new observed limit is strictly smaller than
        //     the stored one for the same configured window (ratchet down).
        //     Otherwise keep the existing (smaller-or-equal) row.
        sqlx::query(
            "INSERT INTO context_window_observations \
                 (connector, model, observed_limit, configured_window, updated_at) \
             VALUES ($1, $2, $3, $4, now()) \
             ON CONFLICT (connector, model) DO UPDATE SET \
                 observed_limit = EXCLUDED.observed_limit, \
                 configured_window = EXCLUDED.configured_window, \
                 updated_at = now() \
             WHERE context_window_observations.configured_window <> EXCLUDED.configured_window \
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
}
