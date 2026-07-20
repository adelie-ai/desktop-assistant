//! SQLite adapter for [`ErrorClassificationStore`] (epic #178, tier 2).
//!
//! Global (no `user_id`): connector knowledge, not personal data. `lookup`
//! matches a stored signature as a case-insensitive literal substring of the
//! incoming message and bumps the hit counter; the most specific (longest
//! signature) match wins. `record` is an idempotent upsert on
//! `(connector, signature)`.
//!
//! Translation of the Postgres form: `strpos(a, b) > 0` -> `instr(a, b) > 0`,
//! `now()` -> `CURRENT_TIMESTAMP`. SQLite supports `UPDATE … RETURNING` (3.35+).

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{ErrorClassificationStore, LearnedClassification};
use sqlx::SqlitePool;

/// SQLite adapter for the `error_classifications` table.
pub struct SqliteErrorClassificationStore {
    pool: SqlitePool,
}

impl SqliteErrorClassificationStore {
    /// Construct a store over the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait::async_trait]
impl ErrorClassificationStore for SqliteErrorClassificationStore {
    async fn lookup(
        &self,
        connector: &str,
        message: &str,
    ) -> Result<Option<LearnedClassification>, CoreError> {
        // Most specific (longest signature) match wins; the UPDATE ... RETURNING
        // does the match, the hit-stats bump, and the read in one round trip.
        let row = sqlx::query_as::<_, (String, String)>(
            "UPDATE error_classifications \
                SET hit_count = hit_count + 1, last_matched_at = CURRENT_TIMESTAMP \
              WHERE id = ( \
                    SELECT id FROM error_classifications \
                     WHERE connector = ? \
                       AND instr(lower(?), lower(signature)) > 0 \
                     ORDER BY length(signature) DESC \
                     LIMIT 1) \
          RETURNING signature, cause",
        )
        .bind(connector)
        .bind(message)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(row.map(|(signature, cause)| LearnedClassification { signature, cause }))
    }

    async fn record(&self, connector: &str, signature: &str, cause: &str) -> Result<(), CoreError> {
        sqlx::query(
            "INSERT INTO error_classifications (connector, signature, cause) \
             VALUES (?, ?, ?) \
             ON CONFLICT (connector, signature) DO UPDATE SET cause = excluded.cause",
        )
        .bind(connector)
        .bind(signature)
        .bind(cause)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }
}
