//! Phase 3: archive old conversations.
//!
//! Marks conversations whose `updated_at` is older than `days` days ago
//! as archived. Hard-deletion of the underlying messages is a separate
//! concern. Consolidation tolerates the case where a fact's
//! `source_conversation_id` no longer resolves.

use sqlx::PgPool;

/// Archive conversations not touched in `days` days. Returns rows archived.
pub async fn run_archival_phase(pool: &PgPool, days: u32) -> Result<usize, String> {
    let result = sqlx::query(
        "UPDATE conversations
         SET archived_at = NOW()
         WHERE archived_at IS NULL
           AND updated_at < NOW() - make_interval(days => $1)",
    )
    .bind(days as i32)
    .execute(pool)
    .await
    .map_err(|e| format!("dreaming: archival query failed: {e}"))?;

    Ok(result.rows_affected() as usize)
}
