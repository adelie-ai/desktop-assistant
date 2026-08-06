//! Phase 2: holistic knowledge-base consolidation (issue #394).
//!
//! Rather than reviewing entries one-by-one against a handful of neighbours,
//! this loads the user's entire active knowledge base and asks a strong model
//! to recompute what it should look like — pruning trivia, merging duplicates,
//! tightening verbose entries — emitting explicit operations against existing
//! ids. The operations are applied transactionally with soft-delete via
//! [`reconcile::apply_ops`].
//!
//! The plan is not applied verbatim. Three rules bound what one night's
//! judgment can do, because that judgment is formed from prose alone with no
//! signal about whether an entry was ever retrieved or cited:
//!
//! 1. A deliberately promoted entry ([`SOURCE_EXPLICIT`]) is never pruned. It
//!    may be rewritten or merged, and the provenance follows it, so the
//!    protection cannot be laundered away over successive runs.
//! 2. An entry already rewritten [`MAX_REVIEW_GENERATION`] times is settled:
//!    consolidation re-reads its own output every pass, so without a stop the
//!    store drifts from what was observed toward paraphrase of paraphrase. A
//!    settled entry stays prunable - the cap settles its prose, not the store.
//! 3. Outright prunes are capped at [`MAX_DELETE_FRACTION`] of the active set
//!    per run. Merges do not count: their content survives in a canonical row.
//!
//! When a user's KB is too large for a single prompt it is sliced into
//! tag-grouped chunks under a character budget and each chunk is recomputed
//! independently — redundancy clusters by tag, so near-duplicates stay in the
//! same slice. Slicing is logged so coverage is never silently bounded.

use std::collections::HashSet;

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::{UserId, current_user_id, with_user_id};
use serde::Deserialize;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

use super::common::{extract_json_payload, is_total_failure};
use super::reconcile::{OpBuffer, ProposedOp, SynthesizedMerge, apply_ops};
use super::types::{
    ConsolidationStats, DreamingLlmFn, KnowledgeChangeFn, MAX_DELETE_FRACTION,
    MAX_DELETE_REASON_CHARS, MAX_HOLISTIC_PROMPT_CHARS, MAX_REVIEW_GENERATION, SOURCE_EXPLICIT,
};
use crate::kb_metadata::{KbMetadata, KbScope};

/// One active KB entry loaded for holistic review.
struct KbEntry {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: KbMetadata,
    /// Provenance (`extraction` | `consolidation` | `explicit`, or NULL on rows
    /// written before the column existed). Gates the never-prune rule.
    source: Option<String>,
    /// How many times consolidation has already rewritten this entry.
    review_generation: i16,
}

impl KbEntry {
    /// Deliberately promoted by a person, so consolidation may rewrite or merge
    /// it but never prune it.
    fn is_protected(&self) -> bool {
        self.source.as_deref() == Some(SOURCE_EXPLICIT)
    }

    /// Rewritten as many times as the review cap allows. Its prose is settled:
    /// consolidation re-reads its own output every pass, so without a stop the
    /// entry drifts from what was observed toward paraphrase of paraphrase.
    fn is_settled(&self) -> bool {
        self.review_generation >= MAX_REVIEW_GENERATION
    }
}

/// Entry point for the consolidation scan. Recomputes each user's active
/// knowledge base holistically. Cross-user iteration is audit-allowlisted (a
/// background-worker entry point); every per-user pass installs a `with_user_id`
/// scope so all sub-queries land in the right partition.
pub async fn run_consolidation_phase(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    soft_delete_retention_days: u32,
    cancellation: &CancellationToken,
    on_change: Option<&KnowledgeChangeFn>,
) -> Result<ConsolidationStats, CoreError> {
    let user_ids = load_user_ids_with_active_entries(pool).await?;
    if user_ids.is_empty() {
        tracing::debug!("dreaming: no active knowledge entries to consolidate");
        return Ok(ConsolidationStats::default());
    }

    let mut total = ConsolidationStats::default();
    // If every user we attempt fails outright, the whole pass is broken (the
    // model is unauthorized/unreachable) — surface it rather than returning an
    // empty success.
    let user_count = user_ids.len();
    let mut failed_users = 0usize;
    let mut last_failure: Option<String> = None;
    for user_id_str in user_ids {
        // Stop promptly between users when cancelled (each user is a full
        // holistic recompute — potentially several LLM calls).
        if cancellation.is_cancelled() {
            tracing::info!("dreaming: consolidation cancelled; stopping scan");
            break;
        }
        let result = with_user_id(UserId::new(user_id_str.clone()), async {
            consolidate_user(pool, llm_fn, soft_delete_retention_days, cancellation).await
        })
        .await;

        match result {
            Ok(stats) => {
                total.reviewed += stats.reviewed;
                total.merged_clusters += stats.merged_clusters;
                total.updated += stats.updated;
                total.scope_added += stats.scope_added;
                total.soft_deleted += stats.soft_deleted;
                total.protected_from_delete += stats.protected_from_delete;
                total.settled_unchanged += stats.settled_unchanged;
                // Live refresh: if this user's KB actually changed, let connected
                // panels refetch as the scan progresses.
                if (stats.merged_clusters > 0
                    || stats.updated > 0
                    || stats.soft_deleted > 0
                    || stats.scope_added > 0)
                    && let Some(notify) = on_change
                {
                    notify(&UserId::new(user_id_str.clone()));
                }
            }
            Err(e) => {
                failed_users += 1;
                last_failure = Some(e.to_string());
                tracing::warn!(
                    "dreaming: holistic consolidation failed for user {user_id_str}: {e}"
                )
            }
        }
    }

    if is_total_failure(user_count, failed_users, cancellation.is_cancelled()) {
        return Err(CoreError::Storage(format!(
            "consolidation failed for all {user_count} user(s); last error: {}",
            last_failure.as_deref().unwrap_or("unknown")
        )));
    }

    Ok(total)
}

/// Holistically recompute the current user's active KB.
async fn consolidate_user(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    soft_delete_retention_days: u32,
    cancellation: &CancellationToken,
) -> Result<ConsolidationStats, CoreError> {
    let entries = load_active_entries(pool).await?;
    let total_entries = entries.len();
    if total_entries == 0 {
        return Ok(ConsolidationStats::default());
    }

    let slices = slice_entries(entries);
    if slices.len() > 1 {
        tracing::info!(
            "dreaming: KB ({total_entries} entries) exceeds the holistic prompt budget; \
             recomputing in {} tag-grouped slices",
            slices.len()
        );
    }

    let mut buffer = OpBuffer::new();
    // Merge groups are routed through the buffer's union-find (pairwise) so a
    // member can't also be edited/deleted standalone, and the model's
    // synthesized content is recorded keyed by the group's lowest id.
    let mut merge_content: std::collections::HashMap<String, (String, Option<KbScope>)> =
        std::collections::HashMap::new();
    // Deletes are collected across slices so the per-run deletion cap applies
    // to the user's whole KB, not each slice.
    let mut delete_ops: Vec<(String, Option<String>)> = Vec::new();
    // Refusals, reported so an operator can see what the model keeps asking
    // for and the guards keep declining.
    let mut protected_from_delete = 0usize;
    let mut settled_unchanged = 0usize;
    // Track per-slice LLM/parse failures so a pass where EVERY slice failed
    // (e.g. the consolidation model is unauthorized or unreachable) surfaces as
    // an error instead of a silent "0 changes" success.
    let slice_count = slices.len();
    let mut failed_slices = 0usize;
    let mut last_failure: Option<String> = None;

    for slice in &slices {
        // Bail between slices when cancelled — each slice is its own LLM call.
        if cancellation.is_cancelled() {
            break;
        }
        let valid: HashSet<&str> = slice.iter().map(|e| e.id.as_str()).collect();
        let protected: HashSet<&str> = slice
            .iter()
            .filter(|e| e.is_protected())
            .map(|e| e.id.as_str())
            .collect();
        let settled: HashSet<&str> = slice
            .iter()
            .filter(|e| e.is_settled())
            .map(|e| e.id.as_str())
            .collect();

        let response = match llm_fn(build_system_prompt(), build_user_prompt(slice)).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("dreaming: consolidation LLM call failed: {e}");
                failed_slices += 1;
                last_failure = Some(e);
                continue;
            }
        };

        let ops = match parse_operations(&response) {
            Ok(ops) => ops,
            Err(e) => {
                tracing::warn!("dreaming: could not parse consolidation operations: {e}");
                failed_slices += 1;
                last_failure = Some(e.to_string());
                continue;
            }
        };

        for op in ops {
            match op {
                RawOp::Delete { ids, id, reason } => {
                    for did in ids.into_iter().chain(id) {
                        if !valid.contains(did.as_str()) {
                            tracing::debug!("dreaming: ignoring delete of unknown id {did}");
                            continue;
                        }
                        // A fact someone entered on purpose is not the model's
                        // to remove. It may still be rewritten or merged, so
                        // this refuses the prune, not the entry.
                        if protected.contains(did.as_str()) {
                            protected_from_delete += 1;
                            tracing::debug!(
                                "dreaming: refusing to prune deliberately-entered entry {did}"
                            );
                            continue;
                        }
                        delete_ops.push((did, clamp_delete_reason(&reason)));
                    }
                }
                RawOp::Merge {
                    ids,
                    content,
                    scope,
                } => {
                    let members: Vec<String> = ids
                        .into_iter()
                        .filter(|i| valid.contains(i.as_str()))
                        .collect();
                    if members.len() < 2 {
                        tracing::debug!("dreaming: skipping merge with <2 valid members");
                        continue;
                    }
                    // The whole merge is dropped, not just the settled member:
                    // the synthesized content was written to stand for every
                    // member, so applying it while one of them stays live would
                    // duplicate that entry rather than unify it.
                    if let Some(id) = members.iter().find(|i| settled.contains(i.as_str())) {
                        tracing::debug!("dreaming: skipping merge touching settled entry {id}");
                        settled_unchanged += 1;
                        continue;
                    }
                    // Chain pairwise merges so the union-find groups the members;
                    // record the synthesized content under the lowest id.
                    let canonical = members.iter().min().cloned().unwrap();
                    for other in members.iter().skip(1) {
                        buffer.absorb(ProposedOp::Merge {
                            a: members[0].clone(),
                            b: other.clone(),
                        });
                    }
                    merge_content.insert(canonical, (content, scope.filter(|s| !s.is_empty())));
                }
                RawOp::Edit { id, content, scope } => {
                    if !valid.contains(id.as_str()) {
                        tracing::debug!("dreaming: ignoring edit of unknown id {id}");
                        continue;
                    }
                    if let Some(content) = content {
                        if settled.contains(id.as_str()) {
                            settled_unchanged += 1;
                            tracing::debug!("dreaming: skipping rewrite of settled entry {id}");
                        } else {
                            buffer.absorb(ProposedOp::Update {
                                id: id.clone(),
                                new_content: content,
                            });
                        }
                    }
                    // Attaching a scope is metadata, not paraphrase: it does not
                    // advance the review generation and cannot drift the prose,
                    // so a settled entry can still be filed more precisely.
                    if let Some(scope) = scope.filter(|s| !s.is_empty()) {
                        buffer.absorb(ProposedOp::AddScope { id, scope });
                    }
                }
                RawOp::Keep => {}
            }
        }
    }

    // Every slice we attempted failed (LLM call or parse) — this is a broken
    // pass, not a "model kept everything" success. Surface it instead of
    // applying an empty plan, so the maintenance task finalizes as Failed.
    if is_total_failure(slice_count, failed_slices, cancellation.is_cancelled()) {
        return Err(CoreError::Storage(format!(
            "consolidation failed: all {slice_count} slice(s) failed; last error: {}",
            last_failure.as_deref().unwrap_or("unknown")
        )));
    }

    // Mark every loaded entry reviewed so first-review timestamps advance even
    // for entries the model left untouched.
    for slice in &slices {
        for e in slice {
            buffer.mark_reviewed(&e.id);
        }
    }

    // Resolve merge clusters (union-find over the chained pairwise merges) into
    // synthesized merges, pulling the recorded content for each group.
    let mut synthesized: Vec<SynthesizedMerge> = Vec::new();
    for cluster in buffer.merge_clusters() {
        let Some((_, (new_content, new_scope))) = cluster
            .iter()
            .find_map(|id| merge_content.get(id).map(|c| (id, c)))
        else {
            tracing::warn!("dreaming: merge cluster without synthesized content; skipping");
            continue;
        };
        let canonical_id = OpBuffer::canonical_of(&cluster)
            .cloned()
            .expect("non-empty cluster has a canonical id");
        synthesized.push(SynthesizedMerge {
            canonical_id,
            member_ids: cluster.iter().cloned().collect(),
            new_content: new_content.clone(),
            new_scope: new_scope.clone(),
        });
    }

    // Deletion cap over the whole KB. Protected ids never reach `delete_ops`,
    // so refusing them does not consume the budget. The `.max(1)` floor keeps a
    // genuinely bad entry removable from a tiny store, where the fraction would
    // otherwise round to zero.
    let cap = ((total_entries as f64) * MAX_DELETE_FRACTION).ceil() as usize;
    let cap = cap.max(1);
    if delete_ops.len() > cap {
        tracing::warn!(
            "dreaming: holistic consolidation proposed {} deletes for {total_entries} entries; \
             capping at {cap} (excess dropped this run)",
            delete_ops.len()
        );
        delete_ops.truncate(cap);
    }
    for (id, reason) in &delete_ops {
        tracing::debug!(
            "dreaming: consolidation prune {id}: {}",
            reason.as_deref().unwrap_or("(no reason given)")
        );
        buffer.absorb(ProposedOp::Delete {
            id: id.clone(),
            reason: reason.clone(),
        });
    }

    tracing::info!(
        "dreaming: holistic consolidation plan for {total_entries} entries — \
         {} merge(s), {} edit(s)/scope-add(s), {} prune(s); \
         {protected_from_delete} protected, {settled_unchanged} settled",
        synthesized.len(),
        buffer.standalone_updates().len() + buffer.standalone_scope_adds().len(),
        delete_ops.len(),
    );

    let mut stats = apply_ops(pool, &buffer, &synthesized, soft_delete_retention_days).await?;
    stats.protected_from_delete = protected_from_delete;
    stats.settled_unchanged = settled_unchanged;
    Ok(stats)
}

/// The model's stated delete reason, normalized for storage: trimmed, bounded,
/// and `None` when it amounts to nothing.
///
/// Why: the reason is free text from the model that lands in a TEXT column and
/// a log line, neither of which limits it. Truncation is by characters, not
/// bytes, so a multi-byte reason cannot be cut mid-codepoint.
fn clamp_delete_reason(reason: &str) -> Option<String> {
    let trimmed = reason.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.chars().take(MAX_DELETE_REASON_CHARS).collect())
}

/// Distinct users that have at least one non-deleted KB entry. Audit-allowlisted
/// cross-user scan (background worker); callers immediately scope per user.
async fn load_user_ids_with_active_entries(pool: &PgPool) -> Result<Vec<String>, CoreError> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT user_id FROM knowledge_base WHERE deleted_at IS NULL ORDER BY user_id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: load user ids failed: {e}")))?;
    Ok(rows.into_iter().map(|(u,)| u).collect())
}

/// All active entries for the current user, ordered by tags so that slicing
/// (when needed) groups likely-related entries together.
async fn load_active_entries(pool: &PgPool) -> Result<Vec<KbEntry>, CoreError> {
    let user_id = current_user_id();
    type Row = (
        String,
        String,
        Vec<String>,
        serde_json::Value,
        Option<String>,
        i16,
    );
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT id, content, tags, metadata, source, review_generation \
         FROM knowledge_base \
         WHERE user_id = $1 AND deleted_at IS NULL \
         ORDER BY tags, created_at ASC",
    )
    .bind(user_id.as_str())
    .fetch_all(pool)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: load active entries failed: {e}")))?;

    Ok(rows
        .into_iter()
        .map(
            |(id, content, tags, md, source, review_generation)| KbEntry {
                id,
                content,
                tags,
                metadata: KbMetadata::from_json(&md),
                source,
                review_generation,
            },
        )
        .collect())
}

/// Greedily pack tag-ordered entries into slices under the prompt char budget.
fn slice_entries(entries: Vec<KbEntry>) -> Vec<Vec<KbEntry>> {
    const PER_ENTRY_OVERHEAD: usize = 200;
    let mut slices: Vec<Vec<KbEntry>> = Vec::new();
    let mut current: Vec<KbEntry> = Vec::new();
    let mut current_chars = 0usize;

    for e in entries {
        let cost = e.content.len()
            + e.tags.iter().map(|t| t.len() + 2).sum::<usize>()
            + PER_ENTRY_OVERHEAD;
        if !current.is_empty() && current_chars + cost > MAX_HOLISTIC_PROMPT_CHARS {
            slices.push(std::mem::take(&mut current));
            current_chars = 0;
        }
        current.push(e);
        current_chars += cost;
    }
    if !current.is_empty() {
        slices.push(current);
    }
    slices
}

fn build_system_prompt() -> String {
    String::from(
        "You are curating a personal long-term knowledge base. You are shown the COMPLETE set \
         of entries (or a self-contained slice of it). Recompute what this set SHOULD look like \
         and return the operations that get it there.\n\
         \n\
         Bias toward a lean, high-signal store:\n\
         - DELETE entries that are trivial, transient, or circumstantial — facts that mattered \
           only in the moment, are no longer useful going forward, or are obvious/generic.\n\
         - MERGE entries that are duplicates, near-duplicates, or that together describe one \
           thing, into a single clear entry. Only merge entries about the SAME subject and scope.\n\
         - EDIT entries that are correct but verbose, vague, or missing their scope: tighten the \
           prose and/or attach a scope.\n\
         - KEEP (do nothing) for entries that are already good, durable, and distinct.\n\
         \n\
         Preserve genuinely useful durable knowledge — preferences, decisions, project facts, \
         recurring solutions. When in doubt about a unique, useful fact, keep it. When in doubt \
         about a near-duplicate or a trivial note, prune it.\n\
         \n\
         Each entry shows its id, tags, scope, and content. Refer to entries ONLY by the ids \
         shown. Do not invent ids.\n\
         \n\
         ## Output format\n\
         \n\
         Return a JSON object with an `operations` array. Each operation is one of:\n\
         - {\"op\":\"delete\",\"ids\":[\"<id>\",...],\"reason\":\"<why, short>\"}\n\
         - {\"op\":\"merge\",\"ids\":[\"<id>\",\"<id>\",...],\"content\":\"<unified self-contained prose>\",\"scope\":{<dim>:<value>}|null}\n\
         - {\"op\":\"edit\",\"id\":\"<id>\",\"content\":\"<rewritten prose, optional>\",\"scope\":{<dim>:<value>}|null}\n\
         \n\
         Only emit operations for entries that should change; omit anything you would keep \
         as-is. `scope` is an object of string dimensions (e.g. {\"project\":\"adelie-ai\"}) or \
         null for universal facts. Output ONLY the JSON object.",
    )
}

fn build_user_prompt(entries: &[KbEntry]) -> String {
    let mut prompt = String::with_capacity(entries.len() * 256);
    prompt.push_str("# Knowledge base entries\n\n");
    for e in entries {
        prompt.push_str("## ");
        prompt.push_str(&e.id);
        prompt.push('\n');

        prompt.push_str("tags: ");
        if e.tags.is_empty() {
            prompt.push_str("(none)");
        } else {
            prompt.push_str(&e.tags.join(", "));
        }
        prompt.push('\n');

        prompt.push_str("scope: ");
        match e.metadata.effective_scope() {
            Some(scope) => {
                let dims: Vec<String> = scope.0.iter().map(|(k, v)| format!("{k}={v}")).collect();
                prompt.push_str(&dims.join(", "));
            }
            None => prompt.push_str("(universal)"),
        }
        prompt.push('\n');

        prompt.push_str(&e.content);
        prompt.push_str("\n\n");
    }
    prompt.push_str(
        "Return the operations (delete / merge / edit) that improve this set. \
         Omit entries you would keep unchanged.",
    );
    prompt
}

/// One operation in the model's recompute plan. `keep` (and any unrecognized
/// op) is a no-op via `#[serde(other)]`.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum RawOp {
    Delete {
        #[serde(default)]
        ids: Vec<String>,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        reason: String,
    },
    Merge {
        #[serde(default)]
        ids: Vec<String>,
        content: String,
        #[serde(default)]
        scope: Option<KbScope>,
    },
    Edit {
        id: String,
        #[serde(default)]
        content: Option<String>,
        #[serde(default)]
        scope: Option<KbScope>,
    },
    #[serde(other)]
    Keep,
}

#[derive(Debug, Deserialize)]
struct OpsEnvelope {
    #[serde(default)]
    operations: Vec<RawOp>,
}

fn parse_operations(response: &str) -> Result<Vec<RawOp>, CoreError> {
    let payload = extract_json_payload(response);
    let env: OpsEnvelope = serde_json::from_str(&payload)
        .map_err(|e| CoreError::Storage(format!("dreaming: bad consolidation JSON: {e}")))?;
    Ok(env.operations)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    fn entry(id: &str, content: &str, tags: &[&str]) -> KbEntry {
        KbEntry {
            id: id.to_string(),
            content: content.to_string(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            metadata: KbMetadata::default(),
            source: None,
            review_generation: 0,
        }
    }

    #[test]
    fn parses_all_op_kinds_and_ignores_keep() {
        let resp = r#"```json
        {"operations": [
            {"op": "delete", "ids": ["a", "b"], "reason": "trivial"},
            {"op": "merge", "ids": ["c", "d"], "content": "unified", "scope": {"project": "x"}},
            {"op": "edit", "id": "e", "content": "tighter"},
            {"op": "keep", "ids": ["f"]},
            {"op": "something_new", "id": "g"}
        ]}
        ```"#;
        let ops = parse_operations(resp).unwrap();
        assert_eq!(ops.len(), 5);
        assert!(matches!(&ops[0], RawOp::Delete { ids, reason, .. }
            if ids == &["a", "b"] && reason == "trivial"));
        assert!(matches!(&ops[1], RawOp::Merge { ids, content, scope }
            if ids == &["c", "d"] && content == "unified" && scope.is_some()));
        assert!(matches!(&ops[2], RawOp::Edit { id, content, .. }
            if id == "e" && content.as_deref() == Some("tighter")));
        // "keep" and unknown ops both fold into the Keep no-op variant.
        assert!(matches!(ops[3], RawOp::Keep));
        assert!(matches!(ops[4], RawOp::Keep));
    }

    #[test]
    fn missing_operations_key_is_empty() {
        let ops = parse_operations("{}").unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn slice_entries_splits_over_budget() {
        // Each entry ~ MAX/3 chars, so 4 entries span 2 slices.
        let big = "x".repeat(MAX_HOLISTIC_PROMPT_CHARS / 3);
        let entries: Vec<KbEntry> = (0..4)
            .map(|i| entry(&format!("id{i}"), &big, &[]))
            .collect();
        let slices = slice_entries(entries);
        assert!(
            slices.len() >= 2,
            "expected multiple slices, got {}",
            slices.len()
        );
        // Every entry is preserved across slices.
        let total: usize = slices.iter().map(|s| s.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn clamp_delete_reason_treats_blank_as_unstated() {
        assert_eq!(clamp_delete_reason(""), None);
        assert_eq!(clamp_delete_reason("   \n\t "), None);
        assert_eq!(
            clamp_delete_reason("  trivial  ").as_deref(),
            Some("trivial")
        );
    }

    #[test]
    fn clamp_delete_reason_bounds_length_on_char_boundaries() {
        let over_long = "é".repeat(MAX_DELETE_REASON_CHARS + 10);
        let clamped = clamp_delete_reason(&over_long).expect("a non-blank reason survives");
        assert_eq!(clamped.chars().count(), MAX_DELETE_REASON_CHARS);
        assert!(clamped.chars().all(|c| c == 'é'), "no partial codepoint");
    }

    #[test]
    fn slice_entries_keeps_small_kb_in_one_slice() {
        let entries: Vec<KbEntry> = (0..10)
            .map(|i| entry(&format!("id{i}"), "short", &["t"]))
            .collect();
        let slices = slice_entries(entries);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].len(), 10);
    }

    /// A response the model cut off at its output limit, ending mid-string.
    const TRUNCATED: &str = r#"{"operations": [
        {"op": "delete", "ids": ["a"], "reason": "trivial"},
        {"op": "edit", "id": "b", "content": "tigh"#;

    /// A response that is not valid JSON for a reason other than ending early.
    const MALFORMED: &str = r#"{"operations": [{"op": , "id": "a"}]}"#;

    fn ops_response(ids: &[&str]) -> String {
        let ops: Vec<String> = ids
            .iter()
            .map(|id| format!(r#"{{"op":"delete","ids":["{id}"],"reason":"trivial"}}"#))
            .collect();
        format!(r#"{{"operations": [{}]}}"#, ops.join(","))
    }

    /// How many entries a built user prompt describes.
    fn entries_in_prompt(prompt: &str) -> usize {
        prompt.lines().filter(|l| l.starts_with("## ")).count()
    }

    /// Records every prompt it is asked, and answers by rule.
    fn recording_llm(
        calls: Arc<Mutex<Vec<String>>>,
        answer: impl Fn(usize) -> Result<String, String> + Send + Sync + 'static,
    ) -> DreamingLlmFn {
        Box::new(move |_system, user| {
            let seen = entries_in_prompt(&user);
            calls.lock().expect("prompt log is not poisoned").push(user);
            let result = answer(seen);
            Box::pin(async move { result })
        })
    }

    fn slice_of(n: usize) -> Vec<KbEntry> {
        (0..n)
            .map(|i| entry(&format!("id{i}"), "some durable content", &["t"]))
            .collect()
    }

    #[test]
    fn truncated_response_reports_truncation_not_malformed_json() {
        let err = parse_operations(TRUNCATED).expect_err("a cut-off response cannot parse");
        assert!(
            matches!(err, ParseFailure::Truncated(_)),
            "expected a truncation verdict, got: {err:?}"
        );
        let message = err.to_string();
        assert!(
            !message.contains("bad consolidation JSON"),
            "a truncated response must not be reported as malformed: {message}"
        );
    }

    #[test]
    fn malformed_json_reports_a_parse_error() {
        let err = parse_operations(MALFORMED).expect_err("invalid JSON cannot parse");
        assert!(
            matches!(err, ParseFailure::Malformed(_)),
            "expected a malformed verdict, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn truncated_slice_is_split_and_retried() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        // Answers only once the slice has been cut down to 2 entries or fewer.
        let llm = recording_llm(calls.clone(), |seen| {
            if seen > 2 {
                Ok(TRUNCATED.to_string())
            } else {
                Ok(ops_response(&["a"]))
            }
        });

        let ops = operations_for_slice(&llm, &slice_of(4), &CancellationToken::new())
            .await
            .expect("splitting recovers the slice");

        assert!(!ops.is_empty(), "the retry must yield the recovered ops");
        let sizes: Vec<usize> = calls
            .lock()
            .expect("prompt log is not poisoned")
            .iter()
            .map(|p| entries_in_prompt(p))
            .collect();
        assert!(
            sizes.iter().any(|&n| n <= 2),
            "expected a retry with a smaller slice, saw sizes {sizes:?}"
        );
    }

    #[tokio::test]
    async fn split_retry_applies_operations_from_both_halves() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm = recording_llm(calls, |seen| {
            if seen > 2 {
                Ok(TRUNCATED.to_string())
            } else {
                // Each half names the ids it was shown, so a dropped half is visible.
                Ok(ops_response(&["half"]))
            }
        });

        let ops = operations_for_slice(&llm, &slice_of(4), &CancellationToken::new())
            .await
            .expect("splitting recovers the slice");

        let deletes = ops
            .iter()
            .filter(|o| matches!(o, RawOp::Delete { .. }))
            .count();
        assert_eq!(deletes, 2, "both halves' operations must be kept");
    }

    #[tokio::test]
    async fn malformed_json_is_not_retried() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm = recording_llm(calls.clone(), |_| Ok(MALFORMED.to_string()));

        operations_for_slice(&llm, &slice_of(8), &CancellationToken::new())
            .await
            .expect_err("malformed JSON is a real failure");

        assert_eq!(
            calls.lock().expect("prompt log is not poisoned").len(),
            1,
            "a syntax error is not a size problem, so it must not be split and retried"
        );
    }

    #[tokio::test]
    async fn split_retry_stops_at_the_depth_limit() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm = recording_llm(calls.clone(), |_| Ok(TRUNCATED.to_string()));

        operations_for_slice(&llm, &slice_of(64), &CancellationToken::new())
            .await
            .expect_err("a slice that never fits is a failure");

        // Bounded by the split depth: 1 + 2 + 4 + ... for MAX_SLICE_SPLIT_DEPTH levels.
        let made = calls.lock().expect("prompt log is not poisoned").len();
        let ceiling = (1usize << (MAX_SLICE_SPLIT_DEPTH + 1)) - 1;
        assert!(
            made <= ceiling,
            "expected at most {ceiling} calls at depth {MAX_SLICE_SPLIT_DEPTH}, made {made}"
        );
    }

    #[tokio::test]
    async fn single_entry_slice_that_truncates_fails_cleanly() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm = recording_llm(calls.clone(), |_| Ok(TRUNCATED.to_string()));

        operations_for_slice(&llm, &slice_of(1), &CancellationToken::new())
            .await
            .expect_err("one entry cannot be split any further");

        assert_eq!(
            calls.lock().expect("prompt log is not poisoned").len(),
            1,
            "a one-entry slice has nothing to split, so it must not retry"
        );
    }

    #[tokio::test]
    async fn a_half_that_fails_does_not_discard_the_half_that_worked() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        // The full slice truncates; afterwards one half answers and one stays broken.
        let llm = recording_llm(calls, |seen| {
            if seen > 2 {
                Ok(TRUNCATED.to_string())
            } else if seen == 2 {
                Ok(ops_response(&["kept"]))
            } else {
                Err("backend refused".to_string())
            }
        });

        let ops = operations_for_slice(&llm, &slice_of(4), &CancellationToken::new())
            .await
            .expect("a partial recovery still returns what worked");
        assert!(
            !ops.is_empty(),
            "the half that answered must not be thrown away"
        );
    }

    #[tokio::test]
    async fn every_part_failing_reports_the_slice_as_failed() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm = recording_llm(calls, |_| Err("backend unreachable".to_string()));

        operations_for_slice(&llm, &slice_of(4), &CancellationToken::new())
            .await
            .expect_err("a slice whose every part failed is a failure");
    }

    #[test]
    fn slice_entries_splits_a_reference_sized_kb_into_several_slices() {
        // The shape that fails in production: 756 entries averaging ~229 chars.
        let content = "x".repeat(229);
        let entries: Vec<KbEntry> = (0..756)
            .map(|i| entry(&format!("id{i}"), &content, &["tag"]))
            .collect();
        let slices = slice_entries(entries);
        assert!(
            slices.len() > 4,
            "a knowledge base this size must not be sent as 2 huge prompts, got {} slices",
            slices.len()
        );
        let total: usize = slices.iter().map(|s| s.len()).sum();
        assert_eq!(total, 756, "no entry may be dropped while slicing");
    }

    #[test]
    fn slice_entries_budgets_in_characters_not_bytes() {
        // Each 'e' with an accent is 2 bytes but 1 character. A budget counted in
        // bytes splits this set; a budget counted in characters does not.
        let half_budget = "é".repeat(MAX_HOLISTIC_PROMPT_CHARS / 2 - 500);
        let entries: Vec<KbEntry> = (0..2)
            .map(|i| entry(&format!("id{i}"), &half_budget, &[]))
            .collect();
        let slices = slice_entries(entries);
        assert_eq!(
            slices.len(),
            1,
            "two entries of half the character budget fit in one slice"
        );
    }
}
