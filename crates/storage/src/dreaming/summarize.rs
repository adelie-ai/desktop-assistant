//! Phase 4: write the one-line summaries the knowledge base is missing (#1099).
//!
//! `knowledge_base.summary` is nullable and unenforced at the write boundary,
//! because refusing a write that omits it would lose the fact. So two
//! populations of rows carry none: every entry stored before the column existed,
//! and any later write that named no summary. The `[Recall]` block renders one
//! line per candidate entry and that line is the summary, so an entry without
//! one is offered back to the model as a cut-down prefix of its body.
//!
//! This pass reads those rows, asks the model for one line each, and writes the
//! line back. Four rules bound it:
//!
//! 1. **The body is never rewritten.** #694 is the standing concern that the
//!    store is becoming model-rewritten prose rather than accumulated evidence.
//!    A summarising pass that edits the body is exactly that failure, so the
//!    write statement names `summary` and nothing else.
//! 2. **A cycle takes at most [`MAX_SUMMARIES_PER_CYCLE`] rows**, shared across
//!    the users that have work. It is a backfill, not a deadline: what is left
//!    over is logged and taken next time.
//! 3. **Rows are asked about in batches.** One call per row is the expensive way
//!    to spend a backfill of hundreds of rows.
//! 4. **A row that fails keeps no summary.** Nothing is stamped, so it is in the
//!    worklist again next cycle. This is deliberately unlike
//!    [`crate::embedding_backfill`], which stamps a failed row to stop a tight
//!    retry loop: a missing summary costs one line in a batched prompt to
//!    re-attempt, while a missing embedding costs a metered vector call.
//!
//! ## Drift
//!
//! A summary is a condensation of the content, so an edit to the body makes the
//! stored line describe something the entry no longer says - and a confidently
//! wrong line is worse than none, because a reader believes it and never opens
//! the entry. The write path preserves a stored summary when an update names
//! none (#1098), which is exactly how that drift arises, and holistic
//! consolidation rewrites content without touching the summary at all.
//!
//! So the worklist is not `summary IS NULL`. It is the same shape
//! `embedding_backfill` already uses: work is due when the stamp is absent or
//! older than `updated_at`.

use std::collections::{BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::SUMMARY_MAX_CHARS;
use desktop_assistant_core::ports::auth::{UserId, current_user_id, with_user_id};
use desktop_assistant_protocol::one_line;
use serde::Deserialize;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

use super::common::{extract_json_payload, is_total_failure};
use super::types::{
    DreamingLlmFn, KnowledgeChangeFn, MAX_SUMMARIES_PER_CYCLE, MAX_SUMMARY_BATCH_ROWS,
    MAX_SUMMARY_PROMPT_CHARS, MAX_SUMMARY_SOURCE_CHARS,
};

/// One entry the pass owes a line.
struct SummaryTarget {
    id: String,
    content: String,
    tags: Vec<String>,
    /// The body's last-modified time as this pass read it. Carried so the write
    /// can refuse a line that was written for a body the row no longer holds.
    read_at: DateTime<Utc>,
}

/// What one summary pass did, and what it left behind.
#[derive(Debug, Default, Clone, Copy)]
pub struct SummaryStats {
    /// Rows that now carry a freshly written summary.
    pub written: usize,
    /// Rows the pass took from the worklist and showed to the model. Larger
    /// than `written` when the model skipped a row or a call failed.
    pub attempted: usize,
    /// Rows still without a current summary once this pass finished. Reported
    /// rather than inferred, so the per-cycle cap is never a silent truncation.
    pub remaining: usize,
}

/// Write the missing and stale summaries for every user that has some.
///
/// Cross-user iteration is audit-allowlisted (a background-worker entry point);
/// every per-user pass installs a `with_user_id` scope so all sub-queries land
/// in the right partition. `on_change`, when set, fires per user whose entries
/// changed, so connected knowledge panels refetch live.
pub async fn run_summary_phase(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    cancellation: &CancellationToken,
    on_change: Option<&KnowledgeChangeFn>,
) -> Result<SummaryStats, CoreError> {
    let user_ids = load_user_ids_needing_summaries(pool).await?;
    if user_ids.is_empty() {
        return Ok(SummaryStats::default());
    }

    // Share the cycle's budget across the users that have work, rather than
    // letting the first one spend it. One user with a large backlog would
    // otherwise starve every user behind it, every cycle, for as long as the
    // backlog lasted.
    let user_count = user_ids.len();
    let per_user = MAX_SUMMARIES_PER_CYCLE.div_ceil(user_count).max(1);
    let mut budget = MAX_SUMMARIES_PER_CYCLE;

    let mut total = SummaryStats::default();
    let mut failed_users = 0usize;
    let mut last_failure: Option<String> = None;

    for user_id_str in user_ids {
        // Each user is at least one LLM call, so stop promptly between them.
        if cancellation.is_cancelled() {
            tracing::info!("dreaming: summary pass cancelled; stopping scan");
            break;
        }
        if budget == 0 {
            break;
        }

        let take = per_user.min(budget);
        let result = with_user_id(UserId::new(user_id_str.clone()), async {
            summarize_one_user(pool, llm_fn, take, cancellation).await
        })
        .await;

        match result {
            Ok(stats) => {
                budget = budget.saturating_sub(stats.attempted);
                total.written += stats.written;
                total.attempted += stats.attempted;
                total.remaining += stats.remaining;
                if stats.written > 0
                    && let Some(notify) = on_change
                {
                    notify(&UserId::new(user_id_str.clone()));
                }
            }
            Err(e) => {
                failed_users += 1;
                last_failure = Some(e.to_string());
                tracing::warn!("dreaming: summary pass failed for user {user_id_str}: {e}");
            }
        }
    }

    if is_total_failure(user_count, failed_users, cancellation.is_cancelled()) {
        return Err(CoreError::Storage(format!(
            "summary pass failed for all {user_count} user(s); last error: {}",
            last_failure.as_deref().unwrap_or("unknown")
        )));
    }

    Ok(total)
}

/// Summarise up to `take` of the current user's entries that need a line.
async fn summarize_one_user(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    take: usize,
    cancellation: &CancellationToken,
) -> Result<SummaryStats, CoreError> {
    let outstanding = count_entries_needing_a_summary(pool).await?;
    let targets = load_entries_needing_a_summary(pool, take).await?;
    if targets.is_empty() {
        return Ok(SummaryStats::default());
    }
    let attempted = targets.len();

    let batches = batch_targets(targets);
    let batch_count = batches.len();
    let mut written = 0usize;
    let mut failed_batches = 0usize;
    let mut last_failure: Option<String> = None;

    for batch in &batches {
        // Each batch is its own LLM call.
        if cancellation.is_cancelled() {
            break;
        }
        match summaries_for_batch(llm_fn, batch).await {
            Ok(lines) => {
                for (target, summary) in lines {
                    match write_summary(pool, target, &summary).await {
                        Ok(true) => written += 1,
                        // The row was retired, or its body changed, between the
                        // read and the write. Not a failure: the line describes
                        // a body that is gone, and the row is still in the
                        // worklist for the next cycle.
                        Ok(false) => tracing::debug!(
                            "dreaming: knowledge entry {} moved on before its summary \
                             landed; the line was discarded",
                            target.id
                        ),
                        Err(e) => tracing::warn!(
                            "dreaming: storing summary for {} failed: {e}",
                            target.id
                        ),
                    }
                }
            }
            Err(e) => {
                tracing::warn!("dreaming: summary batch failed: {e}");
                failed_batches += 1;
                last_failure = Some(e);
            }
        }
    }

    if is_total_failure(batch_count, failed_batches, cancellation.is_cancelled()) {
        return Err(CoreError::Storage(format!(
            "summary pass failed: all {batch_count} batch(es) failed; last error: {}",
            last_failure.as_deref().unwrap_or("unknown")
        )));
    }

    Ok(SummaryStats {
        written,
        attempted,
        remaining: outstanding.saturating_sub(written),
    })
}

/// Ask the model for one line per entry in `batch`, keyed by entry id.
///
/// Only ids the call actually named survive, so an answer for an entry this
/// batch did not show - a hallucinated id, or one belonging elsewhere - is
/// dropped rather than written. Each line is reduced to a single display line
/// bounded by [`SUMMARY_MAX_CHARS`], because nothing bounds what the model
/// returns and the line is rendered into a list row and a context block with a
/// fixed budget.
async fn summaries_for_batch<'a>(
    llm_fn: &DreamingLlmFn,
    batch: &'a [SummaryTarget],
) -> Result<Vec<(&'a SummaryTarget, String)>, String> {
    let response = llm_fn(build_system_prompt(), build_user_prompt(batch)).await?;
    let asked: BTreeSet<&str> = batch.iter().map(|t| t.id.as_str()).collect();
    let parsed = parse_summaries(&response)?;

    let mut lines: HashMap<String, String> = HashMap::new();
    for raw in parsed {
        if !asked.contains(raw.id.as_str()) {
            tracing::debug!("dreaming: ignoring summary for unasked entry {}", raw.id);
            continue;
        }
        let line = one_line(&raw.summary, SUMMARY_MAX_CHARS);
        // A blank answer is not a summary. Storing it would leave an empty
        // string, which renders as a blank row and takes the entry permanently
        // out of a `summary IS NULL` worklist.
        if line.is_empty() {
            tracing::debug!("dreaming: model returned a blank summary for {}", raw.id);
            continue;
        }
        lines.insert(raw.id, line);
    }

    // Answer in the order the batch was asked, so a partial write is a prefix
    // of the batch rather than an arbitrary subset.
    Ok(batch
        .iter()
        .filter_map(|t| lines.remove(&t.id).map(|line| (t, line)))
        .collect())
}

/// One line the model wrote for one entry.
#[derive(Debug, Deserialize)]
struct RawSummary {
    id: String,
    #[serde(default)]
    summary: String,
}

#[derive(Debug, Deserialize)]
struct SummariesEnvelope {
    #[serde(default)]
    summaries: Vec<RawSummary>,
}

fn parse_summaries(response: &str) -> Result<Vec<RawSummary>, String> {
    let payload = extract_json_payload(response);
    serde_json::from_str::<SummariesEnvelope>(&payload)
        .map(|env| env.summaries)
        .map_err(|e| format!("dreaming: bad summary JSON: {e}"))
}

/// Greedily pack targets into batches under both the row cap and the character
/// budget. The row cap sizes the answer; the character budget sizes the
/// question, because nothing bounds how long an entry's content is.
fn batch_targets(targets: Vec<SummaryTarget>) -> Vec<Vec<SummaryTarget>> {
    const PER_ENTRY_OVERHEAD: usize = 100;
    let mut batches: Vec<Vec<SummaryTarget>> = Vec::new();
    let mut current: Vec<SummaryTarget> = Vec::new();
    let mut current_chars = 0usize;

    for target in targets {
        // Counted in characters, because the budget is stated in characters.
        // `len()` is bytes, which under-fills a batch for any non-ASCII entry.
        let cost = target.content.chars().count().min(MAX_SUMMARY_SOURCE_CHARS)
            + target
                .tags
                .iter()
                .map(|t| t.chars().count() + 2)
                .sum::<usize>()
            + target.id.chars().count()
            + PER_ENTRY_OVERHEAD;
        let full = current.len() >= MAX_SUMMARY_BATCH_ROWS
            || (!current.is_empty() && current_chars + cost > MAX_SUMMARY_PROMPT_CHARS);
        if full {
            batches.push(std::mem::take(&mut current));
            current_chars = 0;
        }
        current.push(target);
        current_chars += cost;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

fn build_system_prompt() -> String {
    format!(
        "You write the one-line summary that stands for an entry in a personal long-term \
         knowledge base. The line is how the entry is offered back later, as a candidate the \
         reader may or may not open.\n\
         \n\
         A summary says what the entry STATES. It is not a topic label: a reader must learn \
         the fact from the line alone. Write \"keeps the facet colon in tag names\", not \
         \"tag naming\".\n\
         \n\
         Rules:\n\
         - One line per entry, at most {SUMMARY_MAX_CHARS} characters. No line breaks, no \
           markdown, no surrounding quotes.\n\
         - Keep the entry's own subject and specifics - the person, project, tool, file or \
           value it names. Never replace one with a category word.\n\
         - State only what the entry says. Add nothing, correct nothing, and judge nothing.\n\
         - Each entry lists its tags. Read them as context for what kind of fact it is; do \
           not restate them as the summary.\n\
         - Write about the entry, not about yourself: no \"this entry says\", no \"the note \
           records\".\n\
         \n\
         ## Output format\n\
         \n\
         Return a JSON object with a `summaries` array, one object per entry shown:\n\
         {{\"summaries\":[{{\"id\":\"<id>\",\"summary\":\"<one line>\"}}]}}\n\
         \n\
         Refer to entries ONLY by the ids shown. Do not invent ids. Omit an entry you cannot \
         summarise rather than guessing at it. Output ONLY the JSON object."
    )
}

fn build_user_prompt(batch: &[SummaryTarget]) -> String {
    let mut prompt = String::with_capacity(batch.len() * 256);
    prompt.push_str("# Knowledge base entries\n\n");
    for target in batch {
        prompt.push_str("## ");
        prompt.push_str(&target.id);
        prompt.push('\n');

        prompt.push_str("tags: ");
        if target.tags.is_empty() {
            prompt.push_str("(none)");
        } else {
            prompt.push_str(&target.tags.join(", "));
        }
        prompt.push('\n');

        // Reduced to one physical line, which bounds the excerpt on a character
        // boundary and - because every run of whitespace collapses - makes it
        // impossible for a body to forge the `## <id>` heading that separates
        // the entries.
        prompt.push_str(&one_line(&target.content, MAX_SUMMARY_SOURCE_CHARS));
        prompt.push_str("\n\n");
    }
    prompt.push_str("Write one summary line for each entry above.");
    prompt
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Distinct users holding at least one entry that needs a summary written.
/// Audit-allowlisted cross-user scan (background worker); the caller
/// immediately scopes per user.
async fn load_user_ids_needing_summaries(pool: &PgPool) -> Result<Vec<String>, CoreError> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT user_id FROM knowledge_base \
         WHERE deleted_at IS NULL \
           AND (summary IS NULL \
             OR summary_updated_at IS NULL \
             OR summary_updated_at < updated_at) \
         ORDER BY user_id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: load summary user ids failed: {e}")))?;
    Ok(rows.into_iter().map(|(u,)| u).collect())
}

/// The current user's entries that need a summary, oldest first, capped.
async fn load_entries_needing_a_summary(
    pool: &PgPool,
    limit: usize,
) -> Result<Vec<SummaryTarget>, CoreError> {
    let user_id = current_user_id();
    type Row = (String, String, Vec<String>, DateTime<Utc>);
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT id, content, tags, updated_at FROM knowledge_base \
         WHERE user_id = $1 AND deleted_at IS NULL \
           AND (summary IS NULL \
             OR summary_updated_at IS NULL \
             OR summary_updated_at < updated_at) \
         ORDER BY created_at ASC, id ASC \
         LIMIT $2",
    )
    .bind(user_id.as_str())
    .bind(limit as i64)
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: load unsummarised entries failed: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|(id, content, tags, read_at)| SummaryTarget {
            id,
            content,
            tags,
            read_at,
        })
        .collect())
}

/// How many of the current user's entries need a summary, before this pass took
/// its share. Read so the leftover can be reported rather than guessed at.
async fn count_entries_needing_a_summary(pool: &PgPool) -> Result<usize, CoreError> {
    let user_id = current_user_id();
    let (count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM knowledge_base \
         WHERE user_id = $1 AND deleted_at IS NULL \
           AND (summary IS NULL \
             OR summary_updated_at IS NULL \
             OR summary_updated_at < updated_at)",
    )
    .bind(user_id.as_str())
    .fetch_one(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: count unsummarised entries failed: {e}")))?;
    Ok(count.max(0) as usize)
}

/// Store one entry's summary. Returns whether a row was written.
///
/// The statement names `summary` and its freshness stamp, and nothing else.
/// `content` is evidence and is not this pass's to edit, and `updated_at` is
/// deliberately left alone: bumping it would mark the row's embedding stale and
/// send the whole backfilled store back through the embedding backfill for a
/// change that never touched the embedded text.
///
/// `updated_at` is instead a precondition. The pass reads a row, spends a model
/// call, then writes, and a content write can land in that window - so the line
/// would describe the body the pass read while the freshness stamp declared it
/// current, and nothing would revisit it. Matching the body's modified time
/// makes the write a no-op in that case, and the row stays in the worklist.
async fn write_summary(
    pool: &PgPool,
    target: &SummaryTarget,
    summary: &str,
) -> Result<bool, CoreError> {
    let user_id = current_user_id();
    let result = sqlx::query(
        "UPDATE knowledge_base \
         SET summary = $1, summary_updated_at = NOW() \
         WHERE user_id = $2 AND id = $3 AND deleted_at IS NULL \
           AND updated_at = $4",
    )
    .bind(summary)
    .bind(user_id.as_str())
    .bind(&target.id)
    .bind(target.read_at)
    .execute(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: summary update failed: {e}")))?;
    Ok(result.rows_affected() > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(id: &str, content: &str, tags: &[&str]) -> SummaryTarget {
        SummaryTarget {
            id: id.to_string(),
            content: content.to_string(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            read_at: DateTime::UNIX_EPOCH,
        }
    }

    /// An LLM that answers with a fixed response, whatever it is asked.
    fn llm_returning(response: &str) -> DreamingLlmFn {
        let response = response.to_string();
        Box::new(move |_system, _user| {
            let response = response.clone();
            Box::pin(async move { Ok(response) })
        })
    }

    #[test]
    fn parses_a_summaries_envelope() {
        let parsed = parse_summaries(
            r#"```json
            {"summaries":[{"id":"a","summary":"one line"},{"id":"b","summary":"another"}]}
            ```"#,
        )
        .expect("a fenced envelope parses");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].id, "a");
        assert_eq!(parsed[1].summary, "another");
    }

    #[test]
    fn a_missing_summaries_key_is_no_lines_rather_than_an_error() {
        let parsed = parse_summaries("{}").expect("a valid object parses");
        assert!(parsed.is_empty());
    }

    #[test]
    fn malformed_json_is_reported_as_a_failure() {
        parse_summaries("not json at all").expect_err("a non-JSON answer is a failed batch");
    }

    #[tokio::test]
    async fn a_line_for_an_entry_the_batch_did_not_show_is_dropped() {
        let llm = llm_returning(
            r#"{"summaries":[{"id":"asked","summary":"kept"},{"id":"other","summary":"dropped"}]}"#,
        );
        let batch = vec![target("asked", "a fact", &[])];
        let lines = summaries_for_batch(&llm, &batch)
            .await
            .expect("the batch answered");
        let kept: Vec<(&str, &str)> = lines
            .iter()
            .map(|(t, line)| (t.id.as_str(), line.as_str()))
            .collect();
        assert_eq!(kept, vec![("asked", "kept")]);
    }

    #[tokio::test]
    async fn a_blank_line_leaves_the_entry_unsummarised() {
        let llm = llm_returning(r#"{"summaries":[{"id":"a","summary":"   \n  "}]}"#);
        let batch = vec![target("a", "a fact", &[])];
        let lines = summaries_for_batch(&llm, &batch)
            .await
            .expect("the batch answered");
        assert!(
            lines.is_empty(),
            "a blank answer must not be stored as an empty summary"
        );
    }

    #[tokio::test]
    async fn an_over_long_line_is_cut_to_the_display_limit() {
        // Multi-byte, so a byte-indexed cut would panic rather than truncate.
        let over_long = "é".repeat(SUMMARY_MAX_CHARS * 3);
        let llm = llm_returning(&format!(
            r#"{{"summaries":[{{"id":"a","summary":"{over_long}"}}]}}"#
        ));
        let batch = vec![target("a", "a fact", &[])];
        let lines = summaries_for_batch(&llm, &batch)
            .await
            .expect("the batch answered");
        assert_eq!(lines[0].1.chars().count(), SUMMARY_MAX_CHARS);
    }

    #[tokio::test]
    async fn a_multi_line_answer_is_reduced_to_one_display_line() {
        let llm =
            llm_returning(r#"{"summaries":[{"id":"a","summary":"first part\n\n  second part"}]}"#);
        let batch = vec![target("a", "a fact", &[])];
        let lines = summaries_for_batch(&llm, &batch)
            .await
            .expect("the batch answered");
        assert_eq!(lines[0].1, "first part second part");
    }

    #[test]
    fn batches_are_capped_at_the_row_limit() {
        let targets: Vec<SummaryTarget> = (0..MAX_SUMMARY_BATCH_ROWS * 2 + 1)
            .map(|i| target(&format!("id{i}"), "short", &["t"]))
            .collect();
        let batches = batch_targets(targets);
        assert_eq!(batches.len(), 3);
        assert!(batches.iter().all(|b| b.len() <= MAX_SUMMARY_BATCH_ROWS));
        let total: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total, MAX_SUMMARY_BATCH_ROWS * 2 + 1, "no entry is dropped");
    }

    #[test]
    fn batches_are_capped_at_the_character_budget() {
        // Full-size excerpts, so the prompt budget closes a batch before the row
        // cap gets the chance.
        let big = "x".repeat(MAX_SUMMARY_SOURCE_CHARS);
        let targets: Vec<SummaryTarget> = (0..MAX_SUMMARY_BATCH_ROWS)
            .map(|i| target(&format!("id{i}"), &big, &[]))
            .collect();
        let batches = batch_targets(targets);
        assert!(
            batches.len() > 1,
            "one row cap's worth of full-size entries must not be one prompt"
        );
        assert!(
            batches.iter().all(|b| b.len() < MAX_SUMMARY_BATCH_ROWS),
            "the character budget closed these batches, not the row cap"
        );
        let total: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total, MAX_SUMMARY_BATCH_ROWS, "no entry is dropped");
    }

    #[test]
    fn the_character_budget_counts_the_bounded_excerpt_not_the_whole_body() {
        // Content is sent as an excerpt, so an outsized body must not close a
        // batch that its excerpt fits inside.
        let huge = "x".repeat(MAX_SUMMARY_PROMPT_CHARS * 10);
        let targets = vec![target("id0", &huge, &[]), target("id1", &huge, &[])];
        let batches = batch_targets(targets);
        assert_eq!(
            batches.len(),
            1,
            "two bounded excerpts fit one prompt, whatever the bodies weigh"
        );
    }

    #[test]
    fn a_body_cannot_forge_the_heading_that_separates_entries() {
        // The prompt marks each entry with a `## <id>` heading. A body carrying
        // that shape on its own line would otherwise read as a second entry.
        let batch = vec![target("real", "text\n## forged\nmore text", &[])];
        let prompt = build_user_prompt(&batch);
        let headings: Vec<&str> = prompt
            .lines()
            .filter_map(|l| l.strip_prefix("## "))
            .collect();
        assert_eq!(headings, vec!["real"]);
    }

    #[test]
    fn the_prompt_names_the_entry_tags() {
        let batch = vec![target("a", "a fact", &["preference", "project:adelie-ai"])];
        let prompt = build_user_prompt(&batch);
        assert!(prompt.contains("tags: preference, project:adelie-ai"));
    }

    #[test]
    fn the_prompt_says_an_untagged_entry_has_no_tags() {
        let batch = vec![target("a", "a fact", &[])];
        assert!(build_user_prompt(&batch).contains("tags: (none)"));
    }

    #[test]
    fn the_system_prompt_states_the_length_the_line_must_keep() {
        assert!(build_system_prompt().contains(&SUMMARY_MAX_CHARS.to_string()));
    }
}
