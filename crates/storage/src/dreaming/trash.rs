//! Knowledge-base trash: retention, reaping, and the explicit empty-trash
//! control (issue #657).
//!
//! Consolidation retires an entry by stamping `deleted_at` rather than deleting
//! the row, so a bad run can be inspected and the entry is merely invisible to
//! every read path. What happens to the tombstone afterwards lives here:
//!
//! - [`reap_expired_trash`] frees the current user's tombstones once they are
//!   past the retention window.
//! - [`sweep_expired_trash`] does the same across every user; it is the entry
//!   point for the daemon's periodic sweep, so reaping no longer depends on
//!   whether the LLM-driven consolidation cycle ran. An instance with dreaming
//!   disabled used to accumulate tombstones forever — never searched, never
//!   freed.
//! - [`empty_trash`] and [`trash_count`] back the explicit user-facing
//!   controls: what is in the trash, and empty it now instead of waiting out
//!   the window.
//!
//! Every operation is scoped to a single user's partition. The one cross-user
//! query is the sweep's "which users have tombstones" scan, which immediately
//! installs a per-user scope before deleting anything.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::{UserId, current_user_id, with_user_id};
use sqlx::{PgExecutor, PgPool};

use super::common::is_total_failure;

/// Upper bound on a configured retention, in days (~1000 years).
///
/// Why: the reap compares against `NOW() - make_interval(days => $1)`. An
/// absurd configured value would push that timestamp outside the range
/// Postgres can represent and error the whole sweep, so clamp instead — a
/// retention this long already means "effectively never reap".
const MAX_RETENTION_DAYS: u32 = 365_000;

/// Delete the current user's soft-deleted entries whose `deleted_at` is older
/// than `retention_days`. Returns how many rows were freed.
///
/// A retention of 0 reaps every tombstone written before this call — the
/// documented "do not retain" setting.
pub async fn reap_expired_trash(pool: &PgPool, retention_days: u32) -> Result<usize, CoreError> {
    let user_id = current_user_id();
    let removed = reap_expired_for_user(pool, user_id.as_str(), retention_days).await?;
    Ok(removed as usize)
}

/// Shared reap statement, so the periodic sweep and the consolidation
/// transaction delete by exactly the same rule. Generic over the executor
/// because the consolidation call site runs inside an open transaction.
pub(super) async fn reap_expired_for_user<'e, E>(
    executor: E,
    user_id: &str,
    retention_days: u32,
) -> Result<u64, CoreError>
where
    E: PgExecutor<'e>,
{
    let days = i32::try_from(retention_days.min(MAX_RETENTION_DAYS))
        .expect("clamped retention always fits in i32");
    let result = sqlx::query(
        "DELETE FROM knowledge_base \
         WHERE user_id = $2 \
           AND deleted_at IS NOT NULL \
           AND deleted_at < NOW() - make_interval(days => $1)",
    )
    .bind(days)
    .bind(user_id)
    .execute(executor)
    .await
    .map_err(|e| CoreError::Storage(format!("knowledge trash: TTL reap failed: {e}")))?;
    Ok(result.rows_affected())
}

/// Reap every user's expired trash. The daemon's periodic backend task calls
/// this, which is what makes the TTL independent of consolidation.
///
/// A failure for one user is logged and the sweep continues, so one bad
/// partition cannot stop the rest; if *every* user failed the error is
/// surfaced, since that means the database itself is unhappy rather than one
/// tenant. Returns the total number of rows freed.
pub async fn sweep_expired_trash(pool: &PgPool, retention_days: u32) -> Result<usize, CoreError> {
    let user_ids = load_user_ids_with_trash(pool).await?;
    if user_ids.is_empty() {
        tracing::debug!("knowledge trash: nothing to sweep");
        return Ok(0);
    }

    let attempted = user_ids.len();
    let mut failed = 0usize;
    let mut last_failure: Option<String> = None;
    let mut total = 0usize;
    for user_id in user_ids {
        let scoped = UserId::new(user_id.clone());
        match with_user_id(scoped, async {
            reap_expired_trash(pool, retention_days).await
        })
        .await
        {
            Ok(0) => {}
            Ok(n) => {
                total += n;
                tracing::info!(
                    "knowledge trash: reaped {n} expired entr{} for user {user_id}",
                    if n == 1 { "y" } else { "ies" }
                );
            }
            Err(e) => {
                failed += 1;
                last_failure = Some(e.to_string());
                tracing::warn!("knowledge trash: sweep failed for user {user_id}: {e}");
            }
        }
    }

    if is_total_failure(attempted, failed, false) {
        return Err(CoreError::Storage(format!(
            "knowledge trash: sweep failed for all {attempted} user(s); last error: {}",
            last_failure.as_deref().unwrap_or("unknown")
        )));
    }

    Ok(total)
}

/// Permanently delete every soft-deleted entry belonging to the current user,
/// ignoring the retention window. Returns how many rows were freed; an already
/// empty trash is a successful `0`, not an error.
pub async fn empty_trash(pool: &PgPool) -> Result<usize, CoreError> {
    let user_id = current_user_id();
    let result =
        sqlx::query("DELETE FROM knowledge_base WHERE user_id = $1 AND deleted_at IS NOT NULL")
            .bind(user_id.as_str())
            .execute(pool)
            .await
            .map_err(|e| CoreError::Storage(format!("knowledge trash: empty failed: {e}")))?;
    Ok(result.rows_affected() as usize)
}

/// How many soft-deleted entries the current user has — what a panel shows as
/// "in the trash".
pub async fn trash_count(pool: &PgPool) -> Result<usize, CoreError> {
    let user_id = current_user_id();
    let (count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM knowledge_base WHERE user_id = $1 AND deleted_at IS NOT NULL",
    )
    .bind(user_id.as_str())
    .fetch_one(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("knowledge trash: count failed: {e}")))?;
    Ok(count.max(0) as usize)
}

/// Distinct users holding at least one tombstone. The one deliberately
/// cross-user statement in this module (a background sweep has no single
/// caller to scope to); it only reads `user_id`, and [`sweep_expired_trash`]
/// installs a per-user scope before any row is deleted.
async fn load_user_ids_with_trash(pool: &PgPool) -> Result<Vec<String>, CoreError> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT user_id FROM knowledge_base \
         WHERE deleted_at IS NOT NULL ORDER BY user_id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("knowledge trash: load user ids failed: {e}")))?;
    Ok(rows.into_iter().map(|(u,)| u).collect())
}
