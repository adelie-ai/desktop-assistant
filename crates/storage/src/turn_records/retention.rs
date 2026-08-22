//! Retention for the turn-record store (issue #1252).
//!
//! Separate from the store because the sweep is the one statement here that
//! deliberately crosses every tenant: it deletes by age, on behalf of the
//! daemon rather than of a caller. Keeping it in its own file keeps the
//! cross-user exemption in `tests/audit_user_id_scoping.rs` narrow enough to
//! be worth having - a whole-file exemption on the store itself would stop the
//! audit checking the reads and writes that matter.

use chrono::{Duration, Utc};
use desktop_assistant_core::CoreError;
use sqlx::PgPool;

/// Delete every turn record older than `retention_days`, for every user, and
/// answer how many turns went.
///
/// The rounds go with their turn, by the foreign key's cascade: they hold the
/// content, so a sweep that left them would report a window it does not keep.
///
/// `retention_days` is the daemon's resolved `[inspector] retention_days`,
/// which carries its own floor of one day. A zero here would delete every
/// record ever written on the next pass, so it is read as that floor rather
/// than obeyed - a store that empties itself is indistinguishable, to the
/// person reading it, from one that was never written.
pub async fn sweep_expired_turn_records(
    pool: &PgPool,
    retention_days: u32,
) -> Result<usize, CoreError> {
    let days = i64::from(retention_days.max(1));
    let cutoff = Utc::now() - Duration::days(days);
    let deleted = sqlx::query("DELETE FROM turn_records WHERE started_at < $1")
        .bind(cutoff)
        .execute(pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;
    Ok(deleted.rows_affected() as usize)
}
