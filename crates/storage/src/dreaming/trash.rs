//! Knowledge-base trash: retention, reaping, restore, and the explicit
//! empty-trash control (issues #657, #710).
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
//! - [`search_trash`] finds a tombstone by full text, for a person who knows
//!   roughly what they lost but not its id, and [`restore_entry`] brings one
//!   back. Both back `builtin_knowledge_base_restore` (#710): until this pair
//!   existed, nothing in the daemon, a client, or a tool could ever clear
//!   `deleted_at`, and 30 days after a bad consolidation run every one of its
//!   tombstones was gone for good.
//! - [`set_disposition`] sets a live entry's disposition directly, for a
//!   person's correction arriving through `builtin_knowledge_base_write`
//!   rather than through consolidation's own judgement.
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
use desktop_assistant_core::domain::{Disposition, SUMMARY_MAX_CHARS};
use desktop_assistant_core::ports::auth::{UserId, current_user_id, with_user_id};
use desktop_assistant_core::ports::knowledge::{RestoreOutcome, TrashEntry};
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

/// Bring a tombstoned entry back: live again, as if it had never been
/// retired - with one exception (#710).
///
/// **Five of the six dispositions are curation judgements about the entry
/// and reset to [`Disposition::Active`]; `refuted` is preserved.**
/// `trivial`, `redundant`, `superseded`, `obsolete` and `active` all say how
/// this *record* should be handled - whether it is worth surfacing, still
/// current, or superseded by another row - and a restore is a person
/// overriding that handling decision (design doc section 3.2: "a person may
/// set or clear anything"), so all five reset together with their reason and
/// `superseded_by` cleared. `refuted` is different in kind: it is a claim
/// about the *world*, that what the entry asserts was established untrue.
/// That claim does not stop being true because the row carrying it was
/// undeleted. A non-person soft delete only ever touches `deleted_at`
/// (`hard_delete_knowledge`'s `soft_delete_ids` path), so a `refuted` entry
/// can end up in the trash with its refutation intact, and resetting it to
/// `active` on restore would silently erase a person's own correction - the
/// exact forgetting this wave's disposition vocabulary exists to prevent
/// (design doc section 1: "a negative is instructive"), and worse here
/// because it destroys the *person's* correction rather than the model's
/// guess. So a `refuted` tombstone keeps its disposition and its reason;
/// every other disposition resets to `active`.
///
/// `superseded_by` clears only for the two dispositions that resolve
/// through it, `superseded` and `redundant` - restore overrides exactly the
/// judgement that made the link meaningful, so the link goes with it. Every
/// other disposition keeps whatever `superseded_by` it carries:
/// `knowledge_base_superseded_by_chk` is a one-way implication, not a
/// biconditional (#1345), so the schema does not forbid a successor id on a
/// disposition other than those two, and migration 056 deliberately
/// preserves such a link, for example on a `trivial` tombstone that also
/// names a successor. An inert link costs nothing to keep; a cleared one
/// cannot be recovered, so restore must not destroy it. Pinned by
/// `restoring_an_entry_keeps_a_successor_link_that_does_not_resolve_through_it`
/// and `restoring_a_refuted_entry_keeps_the_refutation` - read both before
/// "simplifying" this back to an unconditional reset.
///
/// User-scoped by the ambient [`current_user_id`]. Zero rows touched is not
/// one outcome: an id another user's tombstone holds, an id a live row
/// already holds, and an id nobody has ever written all reach this call with
/// nothing to update, and a person asking "bring back X" needs to be told
/// which. The `UPDATE` runs first because it is the common case — an id
/// naming this user's own tombstone — so an ordinary restore costs one
/// statement; the existence check only runs when it touched nothing.
pub async fn restore_entry(pool: &PgPool, id: &str) -> Result<RestoreOutcome, CoreError> {
    let user_id = current_user_id();
    let restored: Option<(String,)> = sqlx::query_as(
        "UPDATE knowledge_base \
         SET deleted_at = NULL, \
             disposition = CASE WHEN disposition = 'refuted' THEN disposition ELSE 'active' END, \
             disposition_reason = CASE WHEN disposition = 'refuted' THEN disposition_reason \
                                        ELSE NULL END, \
             superseded_by = CASE WHEN disposition IN ('superseded', 'redundant') \
                                   THEN NULL ELSE superseded_by END, \
             updated_at = NOW() \
         WHERE user_id = $1 AND id = $2 AND deleted_at IS NOT NULL \
         RETURNING id",
    )
    .bind(user_id.as_str())
    .bind(id)
    .fetch_optional(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("knowledge trash: restore failed: {e}")))?;

    if restored.is_some() {
        return Ok(RestoreOutcome::Restored);
    }

    // Nothing matched. Tell apart "this id names a live row" from "this id
    // names nothing at all" — the first is a mistaken request, the second is
    // evidence that is actually gone, and #710 asks that the two read
    // differently rather than both landing on the same opaque zero.
    let exists: Option<(String,)> =
        sqlx::query_as("SELECT id FROM knowledge_base WHERE user_id = $1 AND id = $2")
            .bind(user_id.as_str())
            .bind(id)
            .fetch_optional(pool)
            .await
            .map_err(|e| {
                CoreError::Storage(format!(
                    "knowledge trash: restore existence check failed: {e}"
                ))
            })?;

    Ok(if exists.is_some() {
        RestoreOutcome::NotInTrash
    } else {
        RestoreOutcome::NoLongerExists
    })
}

/// One tombstone row as [`search_trash`] reads it off the query, before it is
/// shaped into the port's [`TrashEntry`].
#[derive(sqlx::FromRow)]
struct TombstoneRow {
    id: String,
    content: String,
    disposition: String,
    disposition_reason: Option<String>,
    deleted_at: chrono::DateTime<chrono::Utc>,
}

/// Find tombstones by full text, for a person who knows roughly what they
/// lost but not its id (#710).
///
/// FTS over the same `tsv` generated column every other knowledge search
/// reads (migration 002), restricted to `deleted_at IS NOT NULL` — the live
/// half of the store already has `builtin_knowledge_base_search`, and this
/// exists so the trash is reachable by more than a remembered id.
///
/// User-scoped by the ambient [`current_user_id`], the same as every other
/// read in this module.
pub async fn search_trash(
    pool: &PgPool,
    query: &str,
    limit: usize,
) -> Result<Vec<TrashEntry>, CoreError> {
    let user_id = current_user_id();
    let rows: Vec<TombstoneRow> = sqlx::query_as(
        "SELECT id, content, disposition, disposition_reason, deleted_at \
         FROM knowledge_base \
         WHERE user_id = $1 \
           AND deleted_at IS NOT NULL \
           AND tsv @@ plainto_tsquery('english', $2) \
         ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $2)) DESC, deleted_at DESC \
         LIMIT $3",
    )
    .bind(user_id.as_str())
    .bind(query)
    .bind(limit as i64)
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("knowledge trash: search failed: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|r| TrashEntry {
            id: r.id,
            content_excerpt: desktop_assistant_protocol::one_line(&r.content, SUMMARY_MAX_CHARS),
            // The CHECK constraint means a row read back here can never
            // actually carry an unrecognized spelling; `Active` is the parser's
            // own documented fallback for a spelling nothing else can produce.
            disposition: Disposition::parse(&r.disposition).unwrap_or_default(),
            disposition_reason: r.disposition_reason,
            deleted_at: r.deleted_at.format("%Y-%m-%d %H:%M:%S").to_string(),
        })
        .collect())
}

/// Set a live entry's disposition directly — the "a person may set or clear
/// anything" half of the vocabulary (design doc section 3.2), reached from
/// `builtin_knowledge_base_write`'s optional `disposition` argument rather
/// than from consolidation's own judgement, so "no, that is wrong" in
/// conversation can land as [`Disposition::Refuted`] when the user says it.
///
/// Confined to the four dispositions that name no successor —
/// [`Disposition::Active`], [`Disposition::Refuted`],
/// [`Disposition::Obsolete`], [`Disposition::Trivial`]. [`Disposition::Superseded`]
/// and [`Disposition::Redundant`] both require `superseded_by`, and this call
/// carries no argument for one; consolidation's `merge_new`/`disposition` ops
/// are what set those two, with the successor already in hand.
///
/// Live rows only (`deleted_at IS NULL`) — a tombstone's disposition is what
/// [`restore_entry`] resets, not what this call reaches.
pub async fn set_disposition(
    pool: &PgPool,
    id: &str,
    disposition: Disposition,
    reason: Option<&str>,
) -> Result<(), CoreError> {
    if matches!(
        disposition,
        Disposition::Superseded | Disposition::Redundant
    ) {
        return Err(CoreError::InvalidInput {
            code: "knowledge_disposition_needs_a_successor",
            description: format!(
                "disposition '{}' names a successor entry via superseded_by, and this call \
                 carries no argument for one",
                disposition.as_str()
            ),
            message: "That disposition needs to name the entry that replaced this one, which \
                      this write cannot do. Use 'refuted' or 'obsolete' instead."
                .to_string(),
        });
    }

    let user_id = current_user_id();
    let touched = sqlx::query(
        "UPDATE knowledge_base \
         SET disposition = $1, disposition_reason = $2, updated_at = NOW() \
         WHERE user_id = $3 AND id = $4 AND deleted_at IS NULL",
    )
    .bind(disposition.as_str())
    .bind(reason)
    .bind(user_id.as_str())
    .bind(id)
    .execute(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("knowledge disposition: set failed: {e}")))?;

    if touched.rows_affected() == 0 {
        return Err(CoreError::InvalidInput {
            code: "knowledge_entry_not_found",
            description: format!("no live knowledge entry {id} for this user"),
            message: "That entry does not exist, or it is not currently live.".to_string(),
        });
    }
    Ok(())
}
