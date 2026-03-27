//! Periodic extraction of long-term facts from conversations ("cat-napping").
//!
//! Scans conversations for new messages beyond their high-water mark, sends
//! transcripts to a background LLM for fact extraction, deduplicates against
//! the existing knowledge base via vector similarity, and writes novel facts.
//!
//! Also performs memory consolidation: reviews existing KB entries to merge
//! redundant facts, add missing context to overly-general entries, and remove
//! entries that have been superseded by newer information.

use std::future::Future;
use std::pin::Pin;

use desktop_assistant_core::chunking::{CHUNK_MAX_CHARS, CHUNK_OVERLAP, chunk_text};
use pgvector::Vector;
use sqlx::PgPool;

/// Boxed async LLM function: `(system_prompt, user_prompt) → Result<response, error>`.
pub type DreamingLlmFn = Box<
    dyn Fn(String, String) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

/// Re-use the same embed function type as embedding_backfill.
pub use crate::embedding_backfill::BackfillEmbedFn;

/// Cosine-distance threshold below which a fact is considered a duplicate.
/// pgvector `<=>` returns cosine distance in [0, 2]; lower = more similar.
const DEDUP_DISTANCE_THRESHOLD: f64 = 0.15;

/// Maximum characters per message when building transcripts.
const MAX_MESSAGE_CHARS: usize = 2000;

/// Maximum number of conversations to process in a single scan cycle.
const MAX_CONVERSATIONS_PER_SCAN: i64 = 10;

/// Maximum number of KB entries to include in a consolidation review.
const MAX_ENTRIES_FOR_CONSOLIDATION: i64 = 50;

/// Run one dreaming scan cycle: extract new facts, consolidate existing memories,
/// and archive old conversations.
///
/// Returns the total number of new facts written to the knowledge base.
pub async fn run_dreaming_scan(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    archive_after_days: u32,
) -> Result<usize, String> {
    // Phase 1: Extract new facts from conversations
    let new_facts = run_extraction_phase(pool, llm_fn, embed_fn, embedding_model).await?;

    // Phase 2: Consolidate existing memories
    match run_consolidation_phase(pool, llm_fn, embed_fn, embedding_model).await {
        Ok(stats) => {
            if stats.updated > 0 || stats.deleted > 0 {
                tracing::info!(
                    "dreaming: consolidation updated {} and deleted {} memor{}",
                    stats.updated,
                    stats.deleted,
                    if stats.updated + stats.deleted == 1 {
                        "y"
                    } else {
                        "ies"
                    }
                );
            } else {
                tracing::debug!("dreaming: consolidation found no changes needed");
            }
        }
        Err(e) => {
            tracing::warn!("dreaming: consolidation phase failed: {e}");
        }
    }

    // Phase 3: Archive old conversations
    if archive_after_days > 0 {
        match run_archival_phase(pool, archive_after_days).await {
            Ok(archived) => {
                if archived > 0 {
                    tracing::info!(
                        "dreaming: archived {archived} conversation(s) older than {archive_after_days} day(s)"
                    );
                } else {
                    tracing::debug!("dreaming: no conversations to archive");
                }
            }
            Err(e) => {
                tracing::warn!("dreaming: archival phase failed: {e}");
            }
        }
    }

    Ok(new_facts)
}

/// Phase 1: Extract new facts from recent conversation messages.
async fn run_extraction_phase(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
) -> Result<usize, String> {
    let conversations = find_conversations_with_new_messages(pool).await?;

    if conversations.is_empty() {
        tracing::debug!("dreaming: no conversations with new messages");
        return Ok(0);
    }

    tracing::info!(
        "dreaming: found {} conversation(s) with new messages",
        conversations.len()
    );

    let mut total_facts = 0usize;

    for (conv_id, watermark, context_summary) in conversations {
        let max_ordinal = get_max_ordinal(pool, &conv_id).await?;
        if max_ordinal <= watermark {
            continue;
        }

        let transcript = load_new_transcript(pool, &conv_id, watermark).await?;
        if transcript.is_empty() {
            update_watermark(pool, &conv_id, max_ordinal).await?;
            continue;
        }

        let system_prompt = build_extraction_system_prompt();
        let user_prompt = build_extraction_user_prompt(&context_summary, &transcript);

        let response = match llm_fn(system_prompt, user_prompt).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("dreaming: LLM call failed for conversation {conv_id}: {e}");
                continue;
            }
        };

        let facts = parse_extracted_facts(&response);

        for fact in &facts {
            match dedup_and_write_fact(pool, embed_fn, embedding_model, &fact.content, &fact.tags)
                .await
            {
                Ok(true) => total_facts += 1,
                Ok(false) => {} // duplicate, skipped
                Err(e) => {
                    tracing::warn!("dreaming: failed to write fact: {e}");
                }
            }
        }

        update_watermark(pool, &conv_id, max_ordinal).await?;

        tracing::info!(
            "dreaming: conversation {conv_id}: extracted {} fact(s), {} new",
            facts.len(),
            total_facts
        );
    }

    Ok(total_facts)
}

// ── Phase 2: Memory Consolidation ──────────────────────────────────────────

struct ConsolidationStats {
    updated: usize,
    deleted: usize,
}

/// Phase 2: Review existing KB entries and consolidate, correct, or refine them.
///
/// Loads existing memories, asks the LLM to identify entries that should be
/// updated (to add context, correct, or merge with another), or deleted
/// (superseded, redundant). Applies the operations.
async fn run_consolidation_phase(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
) -> Result<ConsolidationStats, String> {
    let entries = load_kb_entries_for_review(pool).await?;

    if entries.is_empty() {
        return Ok(ConsolidationStats {
            updated: 0,
            deleted: 0,
        });
    }

    let system_prompt = build_consolidation_system_prompt();
    let user_prompt = build_consolidation_user_prompt(&entries);

    let response = llm_fn(system_prompt, user_prompt)
        .await
        .map_err(|e| format!("dreaming: consolidation LLM call failed: {e}"))?;

    let operations = parse_consolidation_operations(&response);

    let mut updated = 0usize;
    let mut deleted = 0usize;

    for op in operations {
        match op {
            ConsolidationOp::Update {
                id,
                new_content,
                new_tags,
            } => {
                match apply_update(
                    pool,
                    embed_fn,
                    embedding_model,
                    &id,
                    &new_content,
                    &new_tags,
                )
                .await
                {
                    Ok(()) => {
                        tracing::info!(
                            "dreaming: consolidated memory {id}: {}",
                            &new_content[..new_content.len().min(80)]
                        );
                        updated += 1;
                    }
                    Err(e) => tracing::warn!("dreaming: failed to update memory {id}: {e}"),
                }
            }
            ConsolidationOp::Delete { id, reason } => match apply_delete(pool, &id).await {
                Ok(()) => {
                    tracing::info!("dreaming: removed memory {id}: {reason}");
                    deleted += 1;
                }
                Err(e) => tracing::warn!("dreaming: failed to delete memory {id}: {e}"),
            },
        }
    }

    Ok(ConsolidationStats { updated, deleted })
}

/// Load KB entries for consolidation review.
/// Returns `(id, content, tags)` tuples.
async fn load_kb_entries_for_review(
    pool: &PgPool,
) -> Result<Vec<(String, String, Vec<String>)>, String> {
    let rows: Vec<(String, String, Vec<String>)> = sqlx::query_as(
        "SELECT id, content, tags FROM knowledge_base
         ORDER BY updated_at DESC
         LIMIT $1",
    )
    .bind(MAX_ENTRIES_FOR_CONSOLIDATION)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("dreaming: failed to load KB entries for review: {e}"))?;

    Ok(rows)
}

fn build_consolidation_system_prompt() -> String {
    r#"You are a memory maintenance assistant. You review a knowledge base of long-term facts and identify entries that need consolidation, correction, or removal.

Look for:
1. **Redundant entries** — two or more entries that say essentially the same thing. Keep the best version, delete the others.
2. **Superseded entries** — an older entry contradicted or updated by a newer one. Update the older or delete it.
3. **Overly-general entries** — entries that lack important context, causing ambiguity. For example, "the project directory is /home/user/foo" when there are multiple projects — add the project name for clarity.
4. **Mergeable entries** — related entries that would be more useful as a single combined entry.
5. **Stale or incorrect entries** — entries that appear wrong based on other entries in the knowledge base.

For each change, return a JSON array of operations:
[
  {"op": "update", "id": "entry-id", "new_content": "Updated fact with better context", "new_tags": ["tag1", "tag2"]},
  {"op": "delete", "id": "entry-id", "reason": "Superseded by entry xyz"}
]

Rules:
- Only propose changes when you are confident they improve the knowledge base.
- When merging entries, update one and delete the others.
- Preserve the "source:dreaming" tag in new_tags when it was present in the original.
- Keep entries concise and standalone — each should make sense without the others.
- If no changes are needed, return an empty array: []"#.to_string()
}

fn build_consolidation_user_prompt(entries: &[(String, String, Vec<String>)]) -> String {
    let mut prompt = String::from("## Current knowledge base entries\n\n");

    for (id, content, tags) in entries {
        let tags_str = if tags.is_empty() {
            String::new()
        } else {
            format!(" [tags: {}]", tags.join(", "))
        };
        prompt.push_str(&format!("- **{id}**: {content}{tags_str}\n"));
    }

    prompt.push_str("\nReview these entries and return a JSON array of consolidation operations. Return [] if no changes are needed.");

    prompt
}

enum ConsolidationOp {
    Update {
        id: String,
        new_content: String,
        new_tags: Vec<String>,
    },
    Delete {
        id: String,
        reason: String,
    },
}

/// Parse the LLM consolidation response into operations.
fn parse_consolidation_operations(response: &str) -> Vec<ConsolidationOp> {
    let trimmed = response.trim();
    let json_str = extract_json_array(trimmed);

    let parsed: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("dreaming: failed to parse consolidation response as JSON: {e}");
            tracing::debug!("dreaming: raw consolidation response: {trimmed}");
            return Vec::new();
        }
    };

    parsed
        .into_iter()
        .filter_map(|v| {
            let op = v.get("op")?.as_str()?;
            let id = v.get("id")?.as_str()?.trim().to_string();
            if id.is_empty() {
                return None;
            }

            match op {
                "update" => {
                    let new_content = v.get("new_content")?.as_str()?.trim().to_string();
                    if new_content.is_empty() {
                        return None;
                    }
                    let new_tags: Vec<String> = v
                        .get("new_tags")
                        .and_then(|t| t.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|t| t.as_str().map(|s| s.trim().to_string()))
                                .filter(|s| !s.is_empty())
                                .collect()
                        })
                        .unwrap_or_default();
                    Some(ConsolidationOp::Update {
                        id,
                        new_content,
                        new_tags,
                    })
                }
                "delete" => {
                    let reason = v
                        .get("reason")
                        .and_then(|r| r.as_str())
                        .unwrap_or("no reason given")
                        .to_string();
                    Some(ConsolidationOp::Delete { id, reason })
                }
                _ => {
                    tracing::warn!("dreaming: unknown consolidation op '{op}'");
                    None
                }
            }
        })
        .collect()
}

/// Apply an update operation: update content, tags, and re-embed.
async fn apply_update(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    id: &str,
    new_content: &str,
    new_tags: &[String],
) -> Result<(), String> {
    // Chunk and re-embed the updated content
    let chunks = chunk_text(new_content, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
    let embeddings = embed_fn(chunks).await?;
    if embeddings.is_empty() {
        return Err("dreaming: embedding returned no vectors".to_string());
    }
    let embedding_vecs: Vec<Vector> = embeddings.into_iter().map(Vector::from).collect();

    sqlx::query(
        "UPDATE knowledge_base
         SET content = $1, tags = $2, embedding = $3::vector[], embedding_model = $4, updated_at = NOW()
         WHERE id = $5",
    )
    .bind(new_content)
    .bind(new_tags)
    .bind(&embedding_vecs)
    .bind(embedding_model)
    .bind(id)
    .execute(pool)
    .await
    .map_err(|e| format!("dreaming: update failed: {e}"))?;

    Ok(())
}

/// Apply a delete operation.
async fn apply_delete(pool: &PgPool, id: &str) -> Result<(), String> {
    sqlx::query("DELETE FROM knowledge_base WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await
        .map_err(|e| format!("dreaming: delete failed: {e}"))?;

    Ok(())
}

// ── Phase 3: Conversation Archival ────────────────────────────────────────

/// Archive conversations whose `updated_at` is older than `days` days ago
/// and that are not already archived.
///
/// Returns the number of conversations archived.
async fn run_archival_phase(pool: &PgPool, days: u32) -> Result<usize, String> {
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

/// Conversations that have messages beyond their watermark.
/// Returns `(conversation_id, last_processed_ordinal, context_summary)`.
async fn find_conversations_with_new_messages(
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
async fn load_new_transcript(
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

/// Get the maximum message ordinal for a conversation.
async fn get_max_ordinal(pool: &PgPool, conversation_id: &str) -> Result<i32, String> {
    let row: (Option<i32>,) =
        sqlx::query_as("SELECT MAX(ordinal) FROM messages WHERE conversation_id = $1")
            .bind(conversation_id)
            .fetch_one(pool)
            .await
            .map_err(|e| format!("dreaming: failed to get max ordinal: {e}"))?;

    Ok(row.0.unwrap_or(0))
}

/// UPSERT the watermark for a conversation.
async fn update_watermark(
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

fn build_extraction_system_prompt() -> String {
    r#"You are a fact-extraction assistant. Your job is to identify important long-term facts, preferences, and knowledge from conversation transcripts.

Extract facts that would be useful to remember across future conversations. Focus on:
- User preferences (tools, workflows, communication style)
- Technical decisions and architectural choices
- Project-specific knowledge (file paths, patterns, conventions)
- Personal context the user has shared
- Recurring problems and their solutions

Do NOT extract:
- Transient task details (what the user is working on right now)
- Obvious or generic information
- Information that is only relevant to the current session
- Code snippets or implementation details

Return a JSON array of objects with "content" and "tags" fields:
[
  {"content": "The user prefers dark mode in all editors", "tags": ["preference", "editor"]},
  {"content": "Project uses PostgreSQL with pgvector for semantic search", "tags": ["architecture", "database"]}
]

Write each fact as a concise, standalone prose sentence. Use descriptive topic tags (not "source:dreaming" — that is added automatically). If there are no facts worth extracting, return an empty array: []"#.to_string()
}

fn build_extraction_user_prompt(context_summary: &str, transcript: &str) -> String {
    let mut prompt = String::new();

    if !context_summary.is_empty() {
        prompt.push_str("## Conversation context summary\n\n");
        prompt.push_str(context_summary);
        prompt.push_str("\n\n");
    }

    prompt.push_str("## Recent messages\n\n");
    prompt.push_str(transcript);
    prompt.push_str(
        "\n\nExtract any important long-term facts from the above transcript. Return a JSON array.",
    );

    prompt
}

struct ExtractedFact {
    content: String,
    tags: Vec<String>,
}

/// Parse the LLM response into extracted facts.
/// Handles code fences, preamble text, and various JSON formatting quirks.
fn parse_extracted_facts(response: &str) -> Vec<ExtractedFact> {
    let trimmed = response.trim();

    // Try to find JSON array in the response (may be wrapped in code fences or have preamble)
    let json_str = extract_json_array(trimmed);

    let parsed: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("dreaming: failed to parse LLM response as JSON: {e}");
            tracing::debug!("dreaming: raw response: {trimmed}");
            return Vec::new();
        }
    };

    parsed
        .into_iter()
        .filter_map(|v| {
            let content = v.get("content")?.as_str()?.trim().to_string();
            if content.is_empty() {
                return None;
            }

            let tags: Vec<String> = v
                .get("tags")
                .and_then(|t| t.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str().map(|s| s.trim().to_string()))
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default();

            Some(ExtractedFact { content, tags })
        })
        .collect()
}

/// Extract a JSON array from a response that may contain code fences or preamble.
fn extract_json_array(text: &str) -> String {
    // Try code-fenced JSON first
    if let Some(start) = text.find("```json") {
        let after_fence = &text[start + 7..];
        if let Some(end) = after_fence.find("```") {
            return after_fence[..end].trim().to_string();
        }
    }

    // Try generic code fence
    if let Some(start) = text.find("```") {
        let after_fence = &text[start + 3..];
        // Skip optional language tag on first line
        let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let after_tag = &after_fence[content_start..];
        if let Some(end) = after_tag.find("```") {
            return after_tag[..end].trim().to_string();
        }
    }

    // Try to find bare [ ... ] in the text
    if let Some(start) = text.find('[')
        && let Some(end) = text.rfind(']')
        && end > start
    {
        return text[start..=end].to_string();
    }

    text.to_string()
}

/// Embed a fact, check for duplicates via vector similarity, and write if novel.
/// Returns `Ok(true)` if the fact was written, `Ok(false)` if it was a duplicate.
async fn dedup_and_write_fact(
    pool: &PgPool,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    content: &str,
    tags: &[String],
) -> Result<bool, String> {
    // Chunk and embed the new fact
    let chunks = chunk_text(content, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
    let embeddings = embed_fn(chunks).await?;
    if embeddings.is_empty() {
        return Err("dreaming: embedding returned no vectors".to_string());
    }

    // Use the first chunk's embedding for dedup (representative of the fact)
    let query_vec = Vector::from(embeddings[0].clone());

    // Check for close matches — unnest existing vector arrays
    let closest_distance: Option<(f64,)> = sqlx::query_as(
        "SELECT MIN(chunk <=> $1) AS distance
         FROM knowledge_base, unnest(embedding) AS chunk
         WHERE embedding IS NOT NULL",
    )
    .bind(&query_vec)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("dreaming: dedup search failed: {e}"))?;

    if let Some((distance,)) = closest_distance
        && distance < DEDUP_DISTANCE_THRESHOLD
    {
        tracing::debug!(
            "dreaming: skipping duplicate fact (distance={distance:.4}): {}",
            &content[..content.len().min(80)]
        );
        return Ok(false);
    }

    // Write the new fact with chunked embeddings
    let id = uuid::Uuid::now_v7().to_string();
    let mut all_tags: Vec<String> = vec!["source:dreaming".to_string()];
    all_tags.extend(tags.iter().cloned());
    let metadata = serde_json::json!({});
    let embedding_vecs: Vec<Vector> = embeddings.into_iter().map(Vector::from).collect();

    sqlx::query(
        "INSERT INTO knowledge_base (id, content, tags, metadata, embedding, embedding_model)
         VALUES ($1, $2, $3, $4, $5::vector[], $6)",
    )
    .bind(&id)
    .bind(content)
    .bind(&all_tags)
    .bind(&metadata)
    .bind(&embedding_vecs)
    .bind(embedding_model)
    .execute(pool)
    .await
    .map_err(|e| format!("dreaming: failed to write fact: {e}"))?;

    tracing::info!(
        "dreaming: wrote new fact id={id}: {}",
        &content[..content.len().min(80)]
    );

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_clean_json_response() {
        let response = r#"[
            {"content": "User prefers vim keybindings", "tags": ["preference", "editor"]},
            {"content": "Project uses Rust with Tokio", "tags": ["tech-stack"]}
        ]"#;

        let facts = parse_extracted_facts(response);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].content, "User prefers vim keybindings");
        assert_eq!(facts[0].tags, vec!["preference", "editor"]);
        assert_eq!(facts[1].content, "Project uses Rust with Tokio");
    }

    #[test]
    fn parse_code_fenced_response() {
        let response = r#"Here are the facts I extracted:

```json
[
    {"content": "User likes dark mode", "tags": ["preference"]}
]
```

That's all I found."#;

        let facts = parse_extracted_facts(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "User likes dark mode");
    }

    #[test]
    fn parse_generic_code_fence() {
        let response = r#"```
[
    {"content": "Uses PostgreSQL", "tags": ["database"]}
]
```"#;

        let facts = parse_extracted_facts(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "Uses PostgreSQL");
    }

    #[test]
    fn parse_empty_array() {
        let facts = parse_extracted_facts("[]");
        assert!(facts.is_empty());
    }

    #[test]
    fn parse_response_with_preamble() {
        let response = "I found the following facts:\n[{\"content\": \"Fact one\", \"tags\": []}]";
        let facts = parse_extracted_facts(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "Fact one");
    }

    #[test]
    fn parse_skips_empty_content() {
        let response = r#"[
            {"content": "", "tags": ["empty"]},
            {"content": "Valid fact", "tags": []}
        ]"#;

        let facts = parse_extracted_facts(response);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "Valid fact");
    }

    #[test]
    fn parse_handles_missing_tags() {
        let response = r#"[{"content": "No tags here"}]"#;
        let facts = parse_extracted_facts(response);
        assert_eq!(facts.len(), 1);
        assert!(facts[0].tags.is_empty());
    }

    #[test]
    fn parse_handles_invalid_json() {
        let facts = parse_extracted_facts("this is not json at all");
        assert!(facts.is_empty());
    }

    #[test]
    fn extraction_prompt_includes_context() {
        let prompt = build_extraction_user_prompt("Some context", "Some transcript");
        assert!(prompt.contains("Some context"));
        assert!(prompt.contains("Some transcript"));
        assert!(prompt.contains("## Conversation context summary"));
    }

    #[test]
    fn extraction_prompt_skips_empty_context() {
        let prompt = build_extraction_user_prompt("", "Some transcript");
        assert!(!prompt.contains("## Conversation context summary"));
        assert!(prompt.contains("Some transcript"));
    }

    // ── Consolidation parsing tests ──

    #[test]
    fn parse_consolidation_update_op() {
        let response = r#"[
            {"op": "update", "id": "abc-123", "new_content": "Updated fact", "new_tags": ["source:dreaming", "project"]}
        ]"#;

        let ops = parse_consolidation_operations(response);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            ConsolidationOp::Update {
                id,
                new_content,
                new_tags,
            } => {
                assert_eq!(id, "abc-123");
                assert_eq!(new_content, "Updated fact");
                assert_eq!(new_tags, &["source:dreaming", "project"]);
            }
            _ => panic!("expected Update op"),
        }
    }

    #[test]
    fn parse_consolidation_delete_op() {
        let response =
            r#"[{"op": "delete", "id": "old-entry", "reason": "Superseded by newer entry"}]"#;

        let ops = parse_consolidation_operations(response);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            ConsolidationOp::Delete { id, reason } => {
                assert_eq!(id, "old-entry");
                assert_eq!(reason, "Superseded by newer entry");
            }
            _ => panic!("expected Delete op"),
        }
    }

    #[test]
    fn parse_consolidation_mixed_ops() {
        let response = r#"[
            {"op": "update", "id": "a", "new_content": "Merged fact", "new_tags": ["merged"]},
            {"op": "delete", "id": "b", "reason": "Merged into a"}
        ]"#;

        let ops = parse_consolidation_operations(response);
        assert_eq!(ops.len(), 2);
        assert!(matches!(&ops[0], ConsolidationOp::Update { .. }));
        assert!(matches!(&ops[1], ConsolidationOp::Delete { .. }));
    }

    #[test]
    fn parse_consolidation_empty_array() {
        let ops = parse_consolidation_operations("[]");
        assert!(ops.is_empty());
    }

    #[test]
    fn parse_consolidation_skips_unknown_op() {
        let response = r#"[{"op": "merge", "id": "x"}]"#;
        let ops = parse_consolidation_operations(response);
        assert!(ops.is_empty());
    }

    #[test]
    fn parse_consolidation_skips_empty_id() {
        let response = r#"[{"op": "delete", "id": "", "reason": "test"}]"#;
        let ops = parse_consolidation_operations(response);
        assert!(ops.is_empty());
    }

    #[test]
    fn parse_consolidation_delete_without_reason() {
        let response = r#"[{"op": "delete", "id": "abc"}]"#;
        let ops = parse_consolidation_operations(response);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            ConsolidationOp::Delete { reason, .. } => {
                assert_eq!(reason, "no reason given");
            }
            _ => panic!("expected Delete op"),
        }
    }

    #[test]
    fn consolidation_prompt_lists_entries() {
        let entries = vec![
            (
                "id-1".to_string(),
                "Fact one".to_string(),
                vec!["tag1".to_string()],
            ),
            ("id-2".to_string(), "Fact two".to_string(), vec![]),
        ];
        let prompt = build_consolidation_user_prompt(&entries);
        assert!(prompt.contains("**id-1**: Fact one [tags: tag1]"));
        assert!(prompt.contains("**id-2**: Fact two"));
        assert!(!prompt.contains("[tags: ]"));
    }
}
