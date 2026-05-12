//! Transcript loading, watermark management, and other shared helpers.

use sqlx::PgPool;

use super::types::{MAX_CONVERSATIONS_PER_SCAN, MAX_MESSAGE_CHARS};

/// Conversations that have messages beyond their watermark.
/// Returns `(conversation_id, last_processed_ordinal, context_summary)`.
pub async fn find_conversations_with_new_messages(
    pool: &PgPool,
) -> Result<Vec<(String, i32, String)>, String> {
    let rows: Vec<(String, i32, String)> = sqlx::query_as(
        "SELECT c.id,
                COALESCE(w.last_processed_ordinal, 0) AS watermark,
                c.context_summary
         FROM conversations c
         LEFT JOIN dreaming_watermarks w ON w.conversation_id = c.id
         WHERE EXISTS (
             SELECT 1 FROM messages m
             WHERE m.conversation_id = c.id
               AND m.ordinal > COALESCE(w.last_processed_ordinal, 0)
         )
         ORDER BY c.updated_at DESC
         LIMIT $1",
    )
    .bind(MAX_CONVERSATIONS_PER_SCAN)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("dreaming: failed to find conversations: {e}"))?;

    Ok(rows)
}

/// Load user and assistant messages after `from_ordinal` as a formatted transcript.
pub async fn load_new_transcript(
    pool: &PgPool,
    conversation_id: &str,
    from_ordinal: i32,
) -> Result<String, String> {
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT role, content FROM messages
         WHERE conversation_id = $1 AND ordinal > $2
           AND role IN ('user', 'assistant')
         ORDER BY ordinal ASC",
    )
    .bind(conversation_id)
    .bind(from_ordinal)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("dreaming: failed to load transcript: {e}"))?;

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

/// Load the full message history for a conversation (used by consolidation
/// when the LLM requests source disambiguation).
///
/// Returns an empty string if the conversation has been hard-deleted —
/// callers must handle that case (consolidation falls through to KB-only
/// judgment).
pub async fn load_full_transcript(
    pool: &PgPool,
    conversation_id: &str,
) -> Result<String, String> {
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT role, content FROM messages
         WHERE conversation_id = $1 AND role IN ('user', 'assistant')
         ORDER BY ordinal ASC",
    )
    .bind(conversation_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("dreaming: failed to load full transcript: {e}"))?;

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

/// Get the maximum message ordinal for a conversation.
pub async fn get_max_ordinal(pool: &PgPool, conversation_id: &str) -> Result<i32, String> {
    let row: (Option<i32>,) =
        sqlx::query_as("SELECT MAX(ordinal) FROM messages WHERE conversation_id = $1")
            .bind(conversation_id)
            .fetch_one(pool)
            .await
            .map_err(|e| format!("dreaming: failed to get max ordinal: {e}"))?;

    Ok(row.0.unwrap_or(0))
}

/// UPSERT the watermark for a conversation.
pub async fn update_watermark(
    pool: &PgPool,
    conversation_id: &str,
    ordinal: i32,
) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO dreaming_watermarks (conversation_id, last_processed_ordinal, last_scanned_at)
         VALUES ($1, $2, NOW())
         ON CONFLICT (conversation_id) DO UPDATE
            SET last_processed_ordinal = EXCLUDED.last_processed_ordinal,
                last_scanned_at = NOW()",
    )
    .bind(conversation_id)
    .bind(ordinal)
    .execute(pool)
    .await
    .map_err(|e| format!("dreaming: failed to update watermark: {e}"))?;

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

#[cfg(test)]
mod tests {
    use super::*;

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
