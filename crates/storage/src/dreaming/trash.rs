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
//!
//! No statement here removes a row by itself. Every reap goes through
//! [`crate::knowledge_delete::hard_delete_knowledge`], which reads who asked
//! for the delete and applies the configured policy, so an instance can keep
//! its tombstones until a person frees them.

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::{UserId, current_user_id, with_user_id};
use sqlx::{PgExecutor, PgPool};

use super::common::is_total_failure;
use crate::knowledge_delete::{HardDeleteTarget, KnowledgeDeletePolicy, hard_delete_knowledge};

/// Delete the current user's soft-deleted entries whose `deleted_at` is older
/// than the policy's retention window. Returns how many rows were freed.
///
/// A retention of 0 reaps every tombstone written before this call — the
/// documented "do not retain" setting. A policy that reserves hard deletes to
/// a person frees nothing here and logs what it spared.
pub async fn reap_expired_trash(
    pool: &PgPool,
    policy: KnowledgeDeletePolicy,
) -> Result<usize, CoreError> {
    let user_id = current_user_id();
    let removed = reap_expired_for_user(pool, user_id.as_str(), policy).await?;
    Ok(removed as usize)
}

/// Shared reap, so the periodic sweep and the consolidation transaction delete
/// by exactly the same rule. Generic over the executor because the
/// consolidation call site runs inside an open transaction.
pub(super) async fn reap_expired_for_user<'e, E>(
    executor: E,
    user_id: &str,
    policy: KnowledgeDeletePolicy,
) -> Result<u64, CoreError>
where
    E: PgExecutor<'e>,
{
    let outcome = hard_delete_knowledge(
        executor,
        user_id,
        HardDeleteTarget::ExpiredTombstones,
        policy,
        "dreaming::trash::reap_expired_for_user",
    )
    .await?;
    Ok(outcome.into_removed_or_log())
}

/// Reap every user's expired trash. The daemon's periodic backend task calls
/// this, which is what makes the TTL independent of consolidation.
///
/// A failure for one user is logged and the sweep continues, so one bad
/// partition cannot stop the rest; if *every* user failed the error is
/// surfaced, since that means the database itself is unhappy rather than one
/// tenant. Returns the total number of rows freed.
pub async fn sweep_expired_trash(
    pool: &PgPool,
    policy: KnowledgeDeletePolicy,
) -> Result<usize, CoreError> {
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
        match with_user_id(scoped, async { reap_expired_trash(pool, policy).await }).await {
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
///
/// This is a person's own control, so a caller installs
/// `DeleteInitiator::Person` and the safety flag never stands in its way. A
/// caller that does not is refused, and is told why.
pub async fn empty_trash(pool: &PgPool, policy: KnowledgeDeletePolicy) -> Result<usize, CoreError> {
    let user_id = current_user_id();
    let removed = hard_delete_knowledge(
        pool,
        user_id.as_str(),
        HardDeleteTarget::AllTombstones,
        policy,
        "dreaming::trash::empty_trash",
    )
    .await?
    .into_removed_or_refusal()?;
    Ok(removed as usize)
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
