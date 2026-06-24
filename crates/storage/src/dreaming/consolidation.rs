//! Phase 2: holistic knowledge-base consolidation (issue #394).
//!
//! Rather than reviewing entries one-by-one against a handful of neighbours,
//! this loads the user's entire active knowledge base and asks a strong model
//! to recompute what it should look like — pruning trivia, merging duplicates,
//! tightening verbose entries — emitting explicit operations against existing
//! ids. The operations are applied transactionally with soft-delete via
//! [`reconcile::apply_ops`]; a deletion cap and a logged op-diff guard against
//! a bad run gutting the store.
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

use super::common::extract_json_payload;
use super::reconcile::{OpBuffer, ProposedOp, SynthesizedMerge, apply_ops};
use super::types::{
    ConsolidationStats, DreamingLlmFn, KnowledgeChangeFn, MAX_DELETE_FRACTION,
    MAX_HOLISTIC_PROMPT_CHARS, SOFT_DELETE_TTL_DAYS,
};
use crate::kb_metadata::{KbMetadata, KbScope};

/// One active KB entry loaded for holistic review.
struct KbEntry {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: KbMetadata,
}

/// Entry point for the consolidation scan. Recomputes each user's active
/// knowledge base holistically. Cross-user iteration is audit-allowlisted (a
/// background-worker entry point); every per-user pass installs a `with_user_id`
/// scope so all sub-queries land in the right partition.
pub async fn run_consolidation_phase(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    cancellation: &CancellationToken,
    on_change: Option<&KnowledgeChangeFn>,
) -> Result<ConsolidationStats, CoreError> {
    let user_ids = load_user_ids_with_active_entries(pool).await?;
    if user_ids.is_empty() {
        tracing::debug!("dreaming: no active knowledge entries to consolidate");
        return Ok(ConsolidationStats::default());
    }

    let mut total = ConsolidationStats::default();
    for user_id_str in user_ids {
        // Stop promptly between users when cancelled (each user is a full
        // holistic recompute — potentially several LLM calls).
        if cancellation.is_cancelled() {
            tracing::info!("dreaming: consolidation cancelled; stopping scan");
            break;
        }
        let result = with_user_id(UserId::new(user_id_str.clone()), async {
            consolidate_user(pool, llm_fn, cancellation).await
        })
        .await;

        match result {
            Ok(stats) => {
                total.reviewed += stats.reviewed;
                total.merged_clusters += stats.merged_clusters;
                total.updated += stats.updated;
                total.scope_added += stats.scope_added;
                total.soft_deleted += stats.soft_deleted;
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
                tracing::warn!(
                    "dreaming: holistic consolidation failed for user {user_id_str}: {e}"
                )
            }
        }
    }

    Ok(total)
}

/// Holistically recompute the current user's active KB.
async fn consolidate_user(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
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
    let mut delete_ops: Vec<(String, String)> = Vec::new();

    for slice in &slices {
        // Bail between slices when cancelled — each slice is its own LLM call.
        if cancellation.is_cancelled() {
            break;
        }
        let valid: HashSet<&str> = slice.iter().map(|e| e.id.as_str()).collect();

        let response = match llm_fn(build_system_prompt(), build_user_prompt(slice)).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("dreaming: consolidation LLM call failed: {e}");
                continue;
            }
        };

        let ops = match parse_operations(&response) {
            Ok(ops) => ops,
            Err(e) => {
                tracing::warn!("dreaming: could not parse consolidation operations: {e}");
                continue;
            }
        };

        for op in ops {
            match op {
                RawOp::Delete { ids, id, reason } => {
                    for did in ids.into_iter().chain(id) {
                        if valid.contains(did.as_str()) {
                            delete_ops.push((did, reason.clone()));
                        } else {
                            tracing::debug!("dreaming: ignoring delete of unknown id {did}");
                        }
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
                        buffer.absorb(ProposedOp::Update {
                            id: id.clone(),
                            new_content: content,
                        });
                    }
                    if let Some(scope) = scope.filter(|s| !s.is_empty()) {
                        buffer.absorb(ProposedOp::AddScope { id, scope });
                    }
                }
                RawOp::Keep => {}
            }
        }
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

    // Deletion cap over the whole KB.
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
        tracing::debug!("dreaming: consolidation delete {id}: {reason}");
        buffer.absorb(ProposedOp::Delete {
            id: id.clone(),
            reason: reason.clone(),
        });
    }

    tracing::info!(
        "dreaming: holistic consolidation plan for {total_entries} entries — \
         {} merge(s), {} edit(s)/scope-add(s), {} delete(s)",
        synthesized.len(),
        buffer.standalone_updates().len() + buffer.standalone_scope_adds().len(),
        delete_ops.len(),
    );

    apply_ops(pool, &buffer, &synthesized, SOFT_DELETE_TTL_DAYS).await
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
    let rows: Vec<(String, String, Vec<String>, serde_json::Value)> = sqlx::query_as(
        "SELECT id, content, tags, metadata \
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
        .map(|(id, content, tags, md)| KbEntry {
            id,
            content,
            tags,
            metadata: KbMetadata::from_json(&md),
        })
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
    use super::*;

    fn entry(id: &str, content: &str, tags: &[&str]) -> KbEntry {
        KbEntry {
            id: id.to_string(),
            content: content.to_string(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            metadata: KbMetadata::default(),
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
    fn slice_entries_keeps_small_kb_in_one_slice() {
        let entries: Vec<KbEntry> = (0..10)
            .map(|i| entry(&format!("id{i}"), "short", &["t"]))
            .collect();
        let slices = slice_entries(entries);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].len(), 10);
    }
}
