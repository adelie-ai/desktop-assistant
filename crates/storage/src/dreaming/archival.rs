//! Phase 3: archive old conversations.
//!
//! Marks conversations whose `updated_at` is older than `days` days ago
//! as archived. This is a non-destructive soft flag (`archived_at`); the
//! messages and the conversation row remain, so a fact's
//! `source_conversation_id` still resolves after archival. (The only hard
//! deletion of a conversation is an explicit user-initiated delete.)
//!
//! Ordering is not a concern for the dream cycle: extraction processes a
//! conversation's messages within the hour they arrive — long before the
//! multi-day archival window — and holistic consolidation operates purely on
//! the (self-contained) knowledge entries, never re-reading transcripts. So
//! `source_conversation_id` is now informational provenance only.
//!
//! This phase iterates implicitly over all users — each row carries its
//! own `user_id`, and the archival flag is set in place, so the
//! per-row mutation preserves tenancy. We still scope by `user_id`
//! whenever a task-local user is installed (the consolidation loop
//! installs one per user); when archival runs at the top level (no
//! scope), it processes all users uniformly. The query is
//! audit-allowlisted because the cross-user form is intentional.

use sqlx::PgPool;

use desktop_assistant_auth_jwt::DEFAULT_USER_ID;
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;

/// Archive conversations not touched in `days` days. Returns rows archived.
///
/// When the task-local user-id is the schema sentinel (`"default"`) the
/// archival sweep operates over the sentinel partition — which, in
/// single-tenant deploys, IS the whole table. In multi-tenant deploys
/// archival is normally driven from inside a per-user consolidation
/// cycle, so the scope picks the right partition automatically.
pub async fn run_archival_phase(pool: &PgPool, days: u32) -> Result<usize, CoreError> {
    let user_id = current_user_id();
    if user_id.as_str() == DEFAULT_USER_ID {
        // Audit-allowlisted: when no per-user scope is installed (e.g.
        // a daemon-wide archival sweep), the worker archives across
        // all users. Single-tenant installs degenerate to the
        // sentinel partition.
        let result = sqlx::query(
            "UPDATE conversations \
             SET archived_at = NOW() \
             WHERE archived_at IS NULL \
               AND updated_at < NOW() - make_interval(days => $1)",
        )
        .bind(days as i32)
        .execute(pool)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: archival query failed: {e}")))?;
        Ok(result.rows_affected() as usize)
    } else {
        let result = sqlx::query(
            "UPDATE conversations \
             SET archived_at = NOW() \
             WHERE user_id = $2 \
               AND archived_at IS NULL \
               AND updated_at < NOW() - make_interval(days => $1)",
        )
        .bind(days as i32)
        .bind(user_id.as_str())
        .execute(pool)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: archival query failed: {e}")))?;
        Ok(result.rows_affected() as usize)
    }
}
