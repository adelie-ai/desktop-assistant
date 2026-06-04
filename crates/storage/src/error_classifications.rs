//! Postgres-backed [`ErrorClassificationStore`] — the learned tier-2 cache
//! for the backend-error classifier (epic #178).
//!
//! Global (no `user_id`): this is connector knowledge, not personal data.
//! `lookup` matches a stored signature as a literal, case-insensitive
//! substring of the incoming message (via `strpos`, so the signature is never
//! interpreted as a `LIKE` pattern) and bumps the hit counter; `record` is an
//! idempotent upsert on `(connector, signature)`.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::store::{ErrorClassificationStore, LearnedClassification};
use sqlx::PgPool;

pub struct PgErrorClassificationStore {
    pool: PgPool,
}

impl PgErrorClassificationStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait::async_trait]
impl ErrorClassificationStore for PgErrorClassificationStore {
    async fn lookup(
        &self,
        connector: &str,
        message: &str,
    ) -> Result<Option<LearnedClassification>, CoreError> {
        // Most specific (longest signature) match wins; record the hit. The
        // UPDATE ... RETURNING does the match, the stats bump, and the read in
        // one round trip.
        let row = sqlx::query_as::<_, (String, String)>(
            "UPDATE error_classifications \
                SET hit_count = hit_count + 1, last_matched_at = now() \
              WHERE id = ( \
                    SELECT id FROM error_classifications \
                     WHERE connector = $1 \
                       AND strpos(lower($2), lower(signature)) > 0 \
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
             VALUES ($1, $2, $3) \
             ON CONFLICT (connector, signature) DO UPDATE SET cause = EXCLUDED.cause",
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
