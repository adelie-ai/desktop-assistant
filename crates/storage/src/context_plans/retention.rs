//! Retention for the context-plan store (#1327).
//!
//! Separate from the store for the same reason
//! `crates/storage/src/turn_records/retention.rs` is: the sweep is the one
//! statement here that deliberately crosses every tenant, deleting by age on
//! behalf of the daemon rather than of a caller. Keeping it in its own file
//! keeps the cross-user exemption in `tests/audit_user_id_scoping.rs`
//! narrow, because a whole-file exemption on the store itself would stop
//! the audit checking the reads and writes that matter.

use chrono::{Duration, Utc};
use desktop_assistant_core::CoreError;
use sqlx::PgPool;

/// Delete every context plan older than `retention_days`, for every user,
/// and answer how many plans went.
///
/// `retention_days` is the daemon's resolved `[inspector] retention_days` -
/// the same knob `sweep_expired_turn_records` reads, because the plan and
/// the turn text answer the same forensic question and expire together.
/// That config already carries its own floor of one day; this function
/// applies the same floor again so a zero passed directly (a test, a future
/// caller that skips the daemon's resolution) cannot empty the table on the
/// next pass.
pub async fn sweep_expired_context_plans(
    pool: &PgPool,
    retention_days: u32,
) -> Result<usize, CoreError> {
    let days = i64::from(retention_days.max(1));
    let cutoff = Utc::now() - Duration::days(days);
    let deleted = sqlx::query("DELETE FROM context_plans WHERE recorded_at < $1")
        .bind(cutoff)
        .execute(pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
    Ok(deleted.rows_affected() as usize)
}
