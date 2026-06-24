//! Transcript loading, watermark management, and other shared helpers.
//!
//! The dreaming worker is a daemon-wide background task (rule #6 in
//! `docs/architecture-evolution.md`): it iterates over every user's
//! conversations and processes each one. The cross-user iteration is
//! deliberate — there's only one worker per daemon, not per-user — but
//! the per-conversation sub-queries must still scope to the owning
//! user (#105). The pattern used by [`super::run_dreaming_scan`] is:
//!
//! 1. Call [`find_conversations_with_new_messages`] (cross-user) to get
//!    `(conversation_id, user_id, watermark, context_summary)` rows.
//! 2. For each row, install the user's `UserId` via
//!    [`desktop_assistant_core::ports::auth::with_user_id`] and run the
//!    rest of the per-conversation work inside that scope. Every helper
//!    in this module then reads `current_user_id()` and scopes its SQL
//!    by it.
//!
//! The cross-user scan in step 1 is the only query that intentionally
//! crosses tenancy; it's documented in the audit allowlist.

use sqlx::PgPool;

use super::types::{MAX_CONVERSATIONS_PER_SCAN, MAX_MESSAGE_CHARS};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;

/// Conversations that have messages beyond their watermark, with their
/// owning user. Returns `(conversation_id, user_id, watermark,
/// context_summary)`. The caller wraps the per-row work in
/// [`with_user_id`] so downstream helpers scope correctly.
///
/// This query is intentionally cross-user — the dreaming worker is one
/// process per daemon, not one per tenant — and is listed in the audit
/// allowlist.
pub async fn find_conversations_with_new_messages(
    pool: &PgPool,
) -> Result<Vec<(String, String, i32, String)>, CoreError> {
    // Audit-allowlisted: cross-user scan in the dreaming background
    // worker. The returned `user_id` column is consumed by the worker
    // to install a per-user scope before any subsequent query runs.
    let rows: Vec<(String, String, i32, String)> = sqlx::query_as(
        "SELECT c.id, \
                c.user_id, \
                COALESCE(w.last_processed_ordinal, 0) AS watermark, \
                c.context_summary \
         FROM conversations c \
         LEFT JOIN dreaming_watermarks w \
                ON w.conversation_id = c.id AND w.user_id = c.user_id \
         WHERE EXISTS ( \
             SELECT 1 FROM messages m \
             WHERE m.conversation_id = c.id \
               AND m.user_id = c.user_id \
               AND m.ordinal > COALESCE(w.last_processed_ordinal, 0) \
         ) \
         ORDER BY c.updated_at DESC \
         LIMIT $1",
    )
    .bind(MAX_CONVERSATIONS_PER_SCAN)
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: failed to find conversations: {e}")))?;

    Ok(rows)
}

/// Load user and assistant messages after `from_ordinal` as a formatted
/// transcript. Scoped to the task-local user id (the dreaming worker
/// installs it before calling this).
pub async fn load_new_transcript(
    pool: &PgPool,
    conversation_id: &str,
    from_ordinal: i32,
) -> Result<String, CoreError> {
    let user_id = current_user_id();
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT role, content FROM messages \
         WHERE user_id = $1 AND conversation_id = $2 AND ordinal > $3 \
           AND role IN ('user', 'assistant') \
         ORDER BY ordinal ASC",
    )
    .bind(user_id.as_str())
    .bind(conversation_id)
    .bind(from_ordinal)
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: failed to load transcript: {e}")))?;

    let mut transcript = String::new();
    for (role, content) in rows {
        let truncated = if content.len() > MAX_MESSAGE_CHARS {
            let end = content.floor_char_boundary(MAX_MESSAGE_CHARS);
            format!("{}…", &content[..end])
        } else {
            content
        };
        transcript.push_str(&format!("[{role}]: {truncated}\n\n"));
    }

    Ok(transcript)
}

/// Get the maximum message ordinal for a conversation. Scoped to the
/// task-local user id.
pub async fn get_max_ordinal(pool: &PgPool, conversation_id: &str) -> Result<i32, CoreError> {
    let user_id = current_user_id();
    let row: (Option<i32>,) = sqlx::query_as(
        "SELECT MAX(ordinal) FROM messages \
         WHERE user_id = $1 AND conversation_id = $2",
    )
    .bind(user_id.as_str())
    .bind(conversation_id)
    .fetch_one(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: failed to get max ordinal: {e}")))?;

    Ok(row.0.unwrap_or(0))
}

/// UPSERT the watermark for a conversation. Scoped to the task-local
/// user id. The `(user_id, conversation_id)` pair is the natural key on
/// `dreaming_watermarks` after #102's migration.
pub async fn update_watermark(
    pool: &PgPool,
    conversation_id: &str,
    ordinal: i32,
) -> Result<(), CoreError> {
    let user_id = current_user_id();
    sqlx::query(
        "INSERT INTO dreaming_watermarks \
            (user_id, conversation_id, last_processed_ordinal, last_scanned_at) \
         VALUES ($1, $2, $3, NOW()) \
         ON CONFLICT (conversation_id) DO UPDATE \
            SET last_processed_ordinal = EXCLUDED.last_processed_ordinal, \
                last_scanned_at = NOW() \
            WHERE dreaming_watermarks.user_id = $1",
    )
    .bind(user_id.as_str())
    .bind(conversation_id)
    .bind(ordinal)
    .execute(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: failed to update watermark: {e}")))?;

    Ok(())
}

/// Extract a JSON value from a response that may contain code fences or preamble.
///
/// Tries (in order): \`\`\`json fenced block, generic \`\`\` fence, then the
/// outermost JSON value — whichever of `{` or `[` appears first wins, paired
/// with its matching last `}` or `]`. This keeps `{"confirmed": [...]}`
/// intact instead of greedily snipping out the inner array.
pub fn extract_json_payload(text: &str) -> String {
    if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        let content_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let after_tag = &after[content_start..];
        if let Some(end) = after_tag.find("```") {
            return after_tag[..end].trim().to_string();
        }
    }

    let first_array = text.find('[');
    let first_object = text.find('{');
    let object_first = match (first_array, first_object) {
        (Some(a), Some(o)) => o < a,
        (Some(_), None) => false,
        (None, Some(_)) => true,
        (None, None) => return text.to_string(),
    };

    if object_first {
        if let Some(start) = first_object
            && let Some(end) = text.rfind('}')
            && end > start
        {
            return text[start..=end].to_string();
        }
    } else if let Some(start) = first_array
        && let Some(end) = text.rfind(']')
        && end > start
    {
        return text[start..=end].to_string();
    }

    text.to_string()
}

/// Whether a maintenance pass that attempted `attempted` independent units
/// (conversations / users / prompt slices), of which `failed` errored, should
/// be treated as a hard failure rather than a no-op "0 changes" success.
///
/// The distinction matters for the maintenance task UI: a pass where the model
/// legitimately kept everything (0 operations) must finalize as `Completed`,
/// but a pass where **every** LLM call failed — e.g. the configured model is
/// unauthorized (HTTP 401) or unreachable — must finalize as `Failed`.
/// Otherwise a broken configuration is indistinguishable from a successful
/// no-op, which is exactly what made a 401 on the consolidation model look like
/// "consolidation did nothing." A cancelled pass is never a failure: it may
/// have intentionally attempted nothing.
pub fn is_total_failure(attempted: usize, failed: usize, cancelled: bool) -> bool {
    !cancelled && attempted > 0 && failed == attempted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_failure_only_when_all_attempts_failed_and_not_cancelled() {
        assert!(is_total_failure(3, 3, false)); // every unit failed → failure
        assert!(!is_total_failure(3, 2, false)); // one succeeded → not a failure
        assert!(!is_total_failure(0, 0, false)); // nothing attempted → not a failure
        assert!(!is_total_failure(3, 3, true)); // cancelled → never a failure
        assert!(is_total_failure(1, 1, false)); // single unit, failed → failure
        assert!(!is_total_failure(1, 0, false)); // single unit, succeeded → ok
    }

    #[test]
    fn extract_json_array_from_code_fence() {
        let got = extract_json_payload("```json\n[1,2,3]\n```");
        assert_eq!(got, "[1,2,3]");
    }

    #[test]
    fn extract_json_object_from_generic_fence() {
        let got = extract_json_payload("```\n{\"a\":1}\n```");
        assert_eq!(got, "{\"a\":1}");
    }

    #[test]
    fn extract_json_object_bare() {
        let got = extract_json_payload("preamble\n{\"a\":1}\ntrailing");
        assert_eq!(got, "{\"a\":1}");
    }

    #[test]
    fn extract_prefers_array_when_both_present() {
        let got = extract_json_payload("[1,2,3]\n{\"a\":1}");
        assert_eq!(got, "[1,2,3]");
    }
}
