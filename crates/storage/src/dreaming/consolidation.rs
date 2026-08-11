//! Phase 2: holistic knowledge-base consolidation (issue #394).
//!
//! Rather than reviewing entries one-by-one against a handful of neighbours,
//! this loads the user's entire active knowledge base and asks a strong model
//! to recompute what it should look like — pruning trivia, merging duplicates,
//! tightening verbose entries — emitting explicit operations against existing
//! ids. The operations are applied transactionally with soft-delete via
//! [`reconcile::apply_ops`].
//!
//! The plan is not applied verbatim. Four rules bound what one run's judgment
//! can do, because that judgment is formed from prose alone with no signal
//! about whether an entry was ever retrieved or cited:
//!
//! 1. A deliberately promoted entry ([`SOURCE_EXPLICIT`]) is never pruned. It
//!    may be rewritten or merged, and the provenance follows it, so the
//!    protection cannot be laundered away over successive runs.
//! 2. An entry already rewritten [`MAX_REVIEW_GENERATION`] times is settled:
//!    consolidation re-reads its own output every pass, so without a stop the
//!    store drifts from what was observed toward paraphrase of paraphrase. A
//!    settled entry stays prunable - the cap settles its prose, not the store.
//! 3. Outright prunes are capped at the configured share of the active set per
//!    run ([`KnowledgeDeletePolicy::prune_cap`]). Merges do not count: their
//!    content survives in a canonical row. A configured share of zero applies
//!    the merges and the edits and retires nothing.
//! 4. Rewrites are capped the same way
//!    ([`KnowledgeDeletePolicy::rewrite_cap`]). An edit and a merge both
//!    overwrite content with no prior version kept, so one degraded answer
//!    must not reach the whole store. A merge costs one whatever its cluster
//!    size, because only the canonical row's content is replaced.
//!
//! Both caps defer rather than discard: the entries are still there for the
//! next run, and the counts are reported.
//!
//! When a user's KB is too large for a single prompt it is sliced into
//! tag-grouped chunks under a character budget and each chunk is recomputed
//! independently — redundancy clusters by tag, so near-duplicates stay in the
//! same slice. Slicing is logged so coverage is never silently bounded.
//!
//! The budget is sized from the answer, not from the model's context window.
//! One operation comes back per entry the model changes, so a slice of several
//! hundred entries overruns the output allowance and the answer arrives cut off
//! mid-array. That is a size fault, not a formatting fault, so it is told apart
//! from malformed JSON and the slice is halved and recomputed rather than lost.
//!
//! One malformed detail must not discard the whole answer either, because a
//! slice is an expensive call and a lost slice waits for the next run. So the
//! answer is read leniently in its encoding and strictly in its values:
//!
//! - A repeated `operations` key is an encoding accident. The arrays are joined
//!   rather than the answer rejected.
//! - Each element is read on its own. One element that is not an operation is
//!   set aside; the operations around it still apply, and each is still
//!   validated against the slice downstream.
//! - What was kept and what was set aside are both counted and logged. Salvage
//!   that quietly returns less than the model proposed is the same loss in a
//!   smaller number.
//!
//! Two shapes stay unrecoverable, and both keep their own verdict. An answer
//! that ended early has no complete envelope to read elements out of, so it
//! stays a truncation and takes the halve-and-recompute path. An answer whose
//! JSON does not parse, or every one of whose operations is unreadable, stays a
//! failure: a plan that produced no work must not look like a model that kept
//! everything.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::activation::ActivationWeights;
use desktop_assistant_core::domain::knowledge_use::KnowledgeUseRecord;
use desktop_assistant_core::domain::replay::replay_priority;
use desktop_assistant_core::domain::salience::{SalienceReading, SalienceSource};
use desktop_assistant_core::ports::auth::{UserId, current_user_id, with_user_id};
use serde::Deserialize;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

use crate::knowledge_use::all_use_records;

use super::common::{extract_json_payload, is_total_failure};
use super::reconcile::{OpBuffer, ProposedOp, SynthesizedMerge, apply_ops};
use super::types::{
    ConsolidationStats, DreamingLlmFn, KnowledgeChangeFn, MAX_DELETE_REASON_CHARS,
    MAX_DROPPED_OP_EXCERPT_CHARS, MAX_HOLISTIC_PROMPT_CHARS, MAX_REVIEW_GENERATION,
    MAX_SLICE_SPLIT_DEPTH, SOURCE_EXPLICIT,
};
use crate::kb_metadata::{KbMetadata, KbScope};
use crate::knowledge_delete::KnowledgeDeletePolicy;

/// One active KB entry loaded for holistic review.
struct KbEntry {
    id: String,
    content: String,
    tags: Vec<String>,
    metadata: KbMetadata,
    /// Provenance (`extraction` | `consolidation` | `explicit`, or NULL on rows
    /// written before the column existed). Gates the never-prune rule, and is
    /// the one salience signal read from provenance rather than from text.
    source: Option<String>,
    /// How many times consolidation has already rewritten this entry.
    review_generation: i16,
    /// The entry's one-line summary, where it has one. Read for its salience
    /// signals and for nothing else - the prompt still shows the body.
    summary: Option<String>,
}

impl KbEntry {
    /// What re-examining this entry is worth today (#1127).
    ///
    /// [`replay_priority`] holds the whole rule: what was retrieved, what was
    /// contradicted, and what is salient, on the scale the `[Recall]` block
    /// already scores in.
    fn replay_priority(
        &self,
        records: &HashMap<&str, &KnowledgeUseRecord>,
        now: DateTime<Utc>,
        weights: &ActivationWeights,
    ) -> f64 {
        let share = SalienceReading::read(&SalienceSource {
            provenance: self.source.as_deref(),
            content: &self.content,
            summary: self.summary.as_deref(),
            tags: &self.tags,
        })
        .share();
        replay_priority(records.get(self.id.as_str()).copied(), share, now, weights)
    }

    /// Written during a live turn, so consolidation may rewrite or merge it but
    /// never prune it.
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
    policy: KnowledgeDeletePolicy,
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
            consolidate_user(pool, llm_fn, policy, cancellation).await
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
                total.prunes_over_cap += stats.prunes_over_cap;
                total.rewrites_over_cap += stats.rewrites_over_cap;
                total.dropped_operations += stats.dropped_operations;
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
    policy: KnowledgeDeletePolicy,
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

    // What the use log knows, read once for the whole store (#1127). A read
    // that fails costs two of the three terms and not the pass: salience is read
    // from the entries themselves, so the slices are still ordered, on that term
    // alone. Every slice is still examined either way, and a night skipped
    // because the log was unreadable would be the worse answer.
    //
    // Read only where it can change something. One slice is one call that shows
    // the model everything, so there is no order to decide and no reason to
    // spend the query.
    let slices = if slices.len() > 1 {
        let records = match all_use_records(pool, current_user_id().as_str()).await {
            Ok(records) => records,
            Err(e) => {
                tracing::warn!(
                    "dreaming: the use log could not be read, so this pass is ordered on \
                     salience alone - what was retrieved and what was contradicted are both \
                     unknown to it: {e}"
                );
                Vec::new()
            }
        };
        order_slices_by_replay_priority(slices, &records, Utc::now())
    } else {
        slices
    };

    let mut buffer = OpBuffer::new();
    // Every op the model proposes is collected across all slices and only then
    // absorbed, because both caps apply to the user's whole KB rather than to
    // each slice. Order is the order the model answered in, so a capped run is
    // deterministic.
    let mut merge_ops: Vec<MergeOp> = Vec::new();
    let mut update_ops: Vec<(String, String)> = Vec::new();
    let mut scope_ops: Vec<(String, KbScope)> = Vec::new();
    let mut delete_ops: Vec<(String, Option<String>)> = Vec::new();
    // Refusals, reported so an operator can see what the model keeps asking
    // for and the guards keep declining.
    let mut protected_from_delete = 0usize;
    let mut settled_unchanged = 0usize;
    // Operations the model proposed that could not be read back. Counted so a
    // repaired answer is never quietly smaller than the one the model sent.
    let mut dropped_operations = 0usize;
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

        let answer = match operations_for_slice(llm_fn, slice, cancellation).await {
            Ok(answer) => answer,
            Err(e) => {
                tracing::warn!("dreaming: consolidation slice failed: {}", e.message);
                dropped_operations += e.dropped;
                failed_slices += 1;
                last_failure = Some(e.message);
                continue;
            }
        };
        dropped_operations += answer.dropped;

        for op in answer.ops {
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
                    merge_ops.push(MergeOp {
                        members,
                        content,
                        scope: scope.filter(|s| !s.is_empty()),
                    });
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
                            update_ops.push((id.clone(), content));
                        }
                    }
                    // Attaching a scope is metadata, not paraphrase: it does not
                    // advance the review generation and cannot drift the prose,
                    // so a settled entry can still be filed more precisely.
                    if let Some(scope) = scope.filter(|s| !s.is_empty()) {
                        scope_ops.push((id, scope));
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

    // Rewrite cap over the whole KB. A merge and an edit both overwrite
    // `content` with no prior version kept, so the prune cap says nothing
    // about them and this one does.
    let rewrite_cap = policy.rewrite_cap(total_entries);
    let proposed_rewrites = merge_ops.len() + update_ops.len();
    let (merge_ops, update_ops) = take_within_rewrite_cap(merge_ops, update_ops, rewrite_cap);
    let rewrites_over_cap = proposed_rewrites - (merge_ops.len() + update_ops.len());
    if rewrites_over_cap > 0 {
        tracing::warn!(
            "dreaming: holistic consolidation proposed {proposed_rewrites} rewrite(s) for \
             {total_entries} entries; capping at {rewrite_cap} ({rewrites_over_cap} deferred to \
             a later run)"
        );
    }

    // Chain each surviving group's pairwise merges so the union-find collects
    // the members, and record the synthesized content under the group's lowest
    // id.
    let mut merge_content: std::collections::HashMap<String, (String, Option<KbScope>)> =
        std::collections::HashMap::new();
    for merge in merge_ops {
        let canonical = merge
            .members
            .iter()
            .min()
            .cloned()
            .expect("a merge op holds at least two members");
        for other in merge.members.iter().skip(1) {
            buffer.absorb(ProposedOp::Merge {
                a: merge.members[0].clone(),
                b: other.clone(),
            });
        }
        merge_content.insert(canonical, (merge.content, merge.scope));
    }
    for (id, new_content) in update_ops {
        buffer.absorb(ProposedOp::Update { id, new_content });
    }
    // Attaching a scope is metadata, not a rewrite, so it is uncapped.
    for (id, scope) in scope_ops {
        buffer.absorb(ProposedOp::AddScope { id, scope });
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

    // Prune cap over the whole KB. Protected ids never reach `delete_ops`, so
    // refusing them does not consume the budget. A configured fraction of zero
    // yields a cap of zero, which is how a deployment keeps consolidation's
    // merges and declines its deletes.
    let prune_cap = policy.prune_cap(total_entries);
    let prunes_over_cap = delete_ops.len().saturating_sub(prune_cap);
    if prunes_over_cap > 0 {
        tracing::warn!(
            "dreaming: holistic consolidation proposed {} prune(s) for {total_entries} entries; \
             capping at {prune_cap} ({prunes_over_cap} deferred to a later run)",
            delete_ops.len()
        );
        delete_ops.truncate(prune_cap);
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
         {protected_from_delete} protected, {settled_unchanged} settled, \
         {rewrites_over_cap} rewrite(s) and {prunes_over_cap} prune(s) over the configured share, \
         {dropped_operations} operation(s) unreadable and dropped",
        synthesized.len(),
        buffer.standalone_updates().len() + buffer.standalone_scope_adds().len(),
        delete_ops.len(),
    );

    let mut stats = apply_ops(pool, &buffer, &synthesized, policy).await?;
    stats.protected_from_delete = protected_from_delete;
    stats.settled_unchanged = settled_unchanged;
    stats.prunes_over_cap = prunes_over_cap;
    stats.rewrites_over_cap = rewrites_over_cap;
    stats.dropped_operations = dropped_operations;
    Ok(stats)
}

/// One merge the model proposed, before the rewrite cap decides whether it is
/// applied this run.
struct MergeOp {
    members: Vec<String>,
    content: String,
    scope: Option<KbScope>,
}

/// Keep as many proposed rewrites as the cap allows.
///
/// A merge costs one, whatever the size of its cluster: it overwrites the
/// canonical row's content, and every other member keeps its own text on its
/// tombstone. Counting members instead would put a large cluster permanently
/// out of reach on a small store, where the cap is smaller than the cluster.
///
/// Merges are taken before edits. Merging duplicates is the work consolidation
/// exists to do, and an edit only tightens prose that is already correct.
fn take_within_rewrite_cap(
    merges: Vec<MergeOp>,
    updates: Vec<(String, String)>,
    cap: usize,
) -> (Vec<MergeOp>, Vec<(String, String)>) {
    let kept_merges: Vec<MergeOp> = merges.into_iter().take(cap).collect();
    let remaining = cap - kept_merges.len();
    let kept_updates: Vec<(String, String)> = updates.into_iter().take(remaining).collect();
    (kept_merges, kept_updates)
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
        Option<String>,
    );
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT id, content, tags, metadata, source, review_generation, summary \
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
            |(id, content, tags, md, source, review_generation, summary)| KbEntry {
                id,
                content,
                tags,
                metadata: KbMetadata::from_json(&md),
                source,
                review_generation,
                summary,
            },
        )
        .collect())
}

/// Examine the slices most worth re-examining first (#1127).
///
/// The pass examines every entry either way. What this decides is which slice
/// the day's first expensive call is spent on, and - because the pass stops
/// between slices when it is cancelled and continues past a slice whose call
/// failed - which entries a pass that did not finish actually reached.
///
/// **Slices, not entries, and not tag groups.** The loader's `ORDER BY tags`
/// is what puts near-duplicates side by side, and finding those is the work the
/// pass exists to do. Ordering anything smaller than a slice moves entries
/// across slice boundaries and can separate a pair the packing had together -
/// `{invoice}` and `{invoices}` are adjacent under a tag sort and are exactly
/// the pair a merge is wanted for. Sorting whole slices leaves every slice's
/// membership byte for byte what it was, so nothing that was examined together
/// stops being examined together.
///
/// It follows that a store small enough for one slice is unaffected, which is
/// correct rather than a gap: such a pass shows the model everything in one
/// call, so it has no order to get wrong.
///
/// A slice's priority is that of its best entry. **The sort is stable**, so
/// slices of equal priority keep the order the loader gave them.
fn order_slices_by_replay_priority(
    slices: Vec<Vec<KbEntry>>,
    records: &[KnowledgeUseRecord],
    now: DateTime<Utc>,
) -> Vec<Vec<KbEntry>> {
    let weights = ActivationWeights::default();
    let by_id: HashMap<&str, &KnowledgeUseRecord> = records
        .iter()
        .map(|record| (record.entry_id.as_str(), record))
        .collect();

    let mut scored: Vec<(f64, Vec<KbEntry>)> = slices
        .into_iter()
        .map(|slice| {
            let best = slice
                .iter()
                .map(|entry| entry.replay_priority(&by_id, now, &weights))
                .fold(0.0_f64, f64::max);
            (best, slice)
        })
        .collect();

    // `total_cmp` rather than `partial_cmp`, so the comparator is a total order
    // and the sort cannot depend on which pair it happened to visit first. A
    // priority that is not a number cannot reach here - every term is bounded
    // arithmetic over stored timestamps - and if one did, `total_cmp` still
    // orders it rather than making the sort incoherent.
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    scored.into_iter().map(|(_, slice)| slice).collect()
}

/// Greedily pack tag-ordered entries into slices under the prompt char budget.
fn slice_entries(entries: Vec<KbEntry>) -> Vec<Vec<KbEntry>> {
    const PER_ENTRY_OVERHEAD: usize = 200;
    let mut slices: Vec<Vec<KbEntry>> = Vec::new();
    let mut current: Vec<KbEntry> = Vec::new();
    let mut current_chars = 0usize;

    for e in entries {
        // Counted in characters, because the budget is stated in characters.
        // `len()` is bytes, which under-fills a slice for any non-ASCII entry.
        let cost = e.content.chars().count()
            + e.tags.iter().map(|t| t.chars().count() + 2).sum::<usize>()
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
         Return a JSON object holding exactly one `operations` array. Put every operation in \
         that one array; do not give the `operations` key more than once. Each operation is \
         one of:\n\
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

/// The object the model wraps its plan in, read one key at a time.
///
/// Not derived, for two reasons. Serde rejects a repeated field, and a model
/// that gives `operations` twice is making an encoding mistake, not proposing
/// nothing - so the arrays are joined instead of the answer discarded. And the
/// elements are held as raw JSON values rather than as [`RawOp`], so one
/// element that is not an operation is set aside on its own instead of failing
/// the whole document.
#[derive(Debug, Default)]
struct OpsEnvelope {
    /// Every element of every `operations` array, in the order they were read.
    elements: Vec<serde_json::Value>,
    /// How many `operations` keys the object carried. More than one is the
    /// encoding mistake above.
    operations_keys: usize,
}

impl<'de> Deserialize<'de> for OpsEnvelope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct EnvelopeVisitor;

        impl<'de> serde::de::Visitor<'de> for EnvelopeVisitor {
            type Value = OpsEnvelope;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object holding an `operations` array")
            }

            fn visit_map<A>(self, mut map: A) -> Result<OpsEnvelope, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut envelope = OpsEnvelope::default();
                while let Some(key) = map.next_key::<String>()? {
                    if key != "operations" {
                        map.next_value::<serde::de::IgnoredAny>()?;
                        continue;
                    }
                    envelope.operations_keys += 1;
                    match map.next_value::<serde_json::Value>()? {
                        serde_json::Value::Array(items) => envelope.elements.extend(items),
                        // Anything else under this key is not a list of
                        // operations. It becomes one unreadable element rather
                        // than a hard failure, so it is reported by the same
                        // path as every other shape the parser cannot use.
                        other => envelope.elements.push(other),
                    }
                }
                Ok(envelope)
            }
        }

        deserializer.deserialize_map(EnvelopeVisitor)
    }
}

/// Why a consolidation response could not be turned into operations.
///
/// The two cases need opposite responses, which is why they are told apart. A
/// response that ended early is a size problem: the model stopped at its output
/// limit, and a smaller slice gets a complete answer. A response that is not
/// valid JSON is not a size problem, so asking again with fewer entries only
/// spends calls.
#[derive(Debug)]
enum ParseFailure {
    /// The response ended before the JSON did.
    Truncated(String),
    /// The response is not valid JSON, or holds no operation that could be
    /// read, for a reason other than ending early.
    Malformed {
        detail: String,
        /// How many proposed operations were read as elements of the answer but
        /// could not be read as operations. Zero when the JSON itself did not
        /// parse, because then no element was ever separated out to count.
        dropped: usize,
    },
}

impl ParseFailure {
    /// Proposed operations this failure lost, for the caller's running count.
    ///
    /// Truncation loses none. The slice is halved and recomputed, so the same
    /// operations come back in the retry and counting them here would count
    /// them twice; a retry that never succeeds is reported as unreviewed
    /// entries instead.
    fn dropped(&self) -> usize {
        match self {
            Self::Truncated(_) => 0,
            Self::Malformed { dropped, .. } => *dropped,
        }
    }
}

impl std::fmt::Display for ParseFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated(detail) => write!(
                f,
                "dreaming: the consolidation answer stopped before the JSON ended, \
                 which means the model reached its output limit: {detail}"
            ),
            Self::Malformed { detail, .. } => {
                write!(f, "dreaming: bad consolidation JSON: {detail}")
            }
        }
    }
}

/// One array element the parser could not read as an operation, kept so the
/// caller can report what came back instead of dropping it without a trace.
#[derive(Debug)]
struct DroppedOperation {
    /// Position in the answer's operations array, counting from zero.
    index: usize,
    /// The element's `op` tag, when it has one. Bounded where it is rendered.
    ///
    /// Named on its own rather than left to the excerpt: `serde_json` writes an
    /// object's keys in alphabetical order, so a long field ahead of `op` would
    /// push the one thing that says what the model was attempting out of a
    /// bounded quote.
    ///
    /// It is the one part of the element that reaches the default log stream as
    /// the model wrote it. Today it can only be `delete`, `merge` or `edit`:
    /// `#[serde(other)]` on [`RawOp`] folds every other tag into `Keep`, which
    /// reads cleanly and so never arrives here. That is a coupling between two
    /// distant decisions, not a guarantee - drop `#[serde(other)]` to reject an
    /// unknown op and a tag of arbitrary length arrives at once. The bound in
    /// [`Display`](std::fmt::Display) holds either way.
    op: Option<String>,
    /// Why the element did not read, taken from serde but with every value the
    /// model wrote replaced first. See [`reason_for`].
    reason: String,
    /// The element itself, bounded so one long merge body cannot fill the log.
    excerpt: String,
}

impl DroppedOperation {
    /// The diagnosis with a bounded quote of the element itself.
    ///
    /// The quote can hold a fragment of the user's own knowledge base, because
    /// a rejected merge or edit carries the prose the model wrote for it. So
    /// this belongs at `DEBUG`, beside the delete reasons this file already
    /// keeps there: the journal is world-readable and is shipped on. The
    /// [`Display`](std::fmt::Display) form carries the diagnosis without the
    /// quote and is what reaches the default log stream.
    fn with_element(&self) -> String {
        format!("{self} -- {}", self.excerpt)
    }
}

/// The diagnosis without the element: where it sat, what the model said it was
/// attempting, and why it did not read.
///
/// Every value the model wrote has been replaced by the time this is built (see
/// [`reason_for`]), and the field names serde can name are this code's own, so
/// nothing here carries the user's knowledge base. That is what makes it safe
/// for the default log stream.
impl std::fmt::Display for DroppedOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            index, op, reason, ..
        } = self;
        // Bounded here, not where the record was built, so the guarantee this
        // doc comment states is held by the code that states it.
        let op = op.as_deref().map(bound_chars);
        let op = op.as_deref().unwrap_or("unstated");
        write!(f, "operation #{index} [op={op}]: {reason}")
    }
}

/// What one consolidation answer yielded, after the elements that could not be
/// read were set aside.
#[derive(Debug, Default)]
struct ParsedOperations {
    /// The operations that read cleanly, in the order the model gave them.
    ops: Vec<RawOp>,
    /// One record per element that did not read as an operation.
    dropped: Vec<DroppedOperation>,
    /// How many `operations` keys the answer carried.
    operations_keys: usize,
}

/// Stands in for a value the model wrote, in a diagnosis that must not carry
/// one. Short, so it cannot crowd out the rest of the message.
const REDACTED_VALUE: &str = "<value>";

/// A copy of `element` with every string the model wrote replaced by
/// [`REDACTED_VALUE`], except the `op` tag.
///
/// Why: serde reports a wrong type as ``invalid type: string "<the whole
/// value>", expected a sequence``, with no bound on the value. A model that
/// string-encodes an array instead of nesting it is an ordinary mistake, and
/// for a rejected merge or edit that value is the prose the model wrote for a
/// knowledge-base entry. So the reason is taken from a re-read of this copy
/// instead. Cutting the value back out of serde's message would mean
/// pattern-matching on an error string, which this codebase does not do.
///
/// The verdict does not change, because the `op` tag is the only content the
/// shape depends on and it is kept: replacing a string with another string
/// leaves every type the same. [`reason_for`] handles the case where it changes
/// anyway rather than trusting that argument.
fn redact_values(element: &serde_json::Value) -> serde_json::Value {
    match element {
        serde_json::Value::String(_) => serde_json::Value::String(REDACTED_VALUE.to_string()),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(redact_values).collect())
        }
        serde_json::Value::Object(fields) => serde_json::Value::Object(
            fields
                .iter()
                .map(|(name, value)| {
                    let value = if name == "op" {
                        value.clone()
                    } else {
                        redact_values(value)
                    };
                    (name.clone(), value)
                })
                .collect(),
        ),
        number_or_bool_or_null => number_or_bool_or_null.clone(),
    }
}

/// Why an answer could not be read as the envelope at all, in words that carry
/// nothing the model wrote.
///
/// Serde reports a wrong top-level type the way it reports a wrong field type -
/// by quoting the value - and here the value is the whole answer. A payload
/// that is one JSON string comes back as ``invalid type: string "<the entire
/// answer>", expected ...``, and both routes into that are reachable: a fenced
/// `json` block holding a bare quoted string, and a bare quoted string with no
/// brackets anywhere. Bounding the message is not enough, because a bounded
/// quote is still a quote. So a wrong type is named by its JSON type alone,
/// read back from the payload, which carries no content at all.
///
/// A payload that holds no JSON value keeps serde's own message. Those state a
/// reason and a position - `expected value at line 1 column 1` - and never
/// quote what they found.
///
/// The value is read with a streaming deserializer rather than `from_str`,
/// which would reject anything after the first value: `"a" and then some prose`
/// is a wrong type followed by trailing characters, and it must still be
/// reported as a string rather than fall through to the message that quotes it.
fn envelope_fault(payload: &str, error: &serde_json::Error) -> String {
    let mut reader = serde_json::Deserializer::from_str(payload);
    match serde_json::Value::deserialize(&mut reader) {
        Ok(value) => format!(
            "the answer is {}, not a JSON object holding an `operations` array",
            json_type_of(&value)
        ),
        Err(_) => bound_chars(&error.to_string()),
    }
}

/// The JSON type of `value` as a phrase, and nothing of its content.
fn json_type_of(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Why an element did not read, in words that carry no value the model wrote
/// and that are bounded in length.
fn reason_for(element: &serde_json::Value) -> String {
    match RawOp::deserialize(&redact_values(element)) {
        // Bounding a redacted message is defence in depth, not a live need: no
        // input reaches this with a long one today, because the only value
        // serde quotes without a length of its own is a string and every string
        // has been replaced. It costs one call and it survives a serde whose
        // messages grow.
        Err(e) => bound_chars(&e.to_string()),
        // Redaction changed the verdict, which the argument above says it
        // cannot. Say that plainly rather than fall back to a message that
        // would carry the value.
        Ok(_) => "could not be read as an operation".to_string(),
    }
}

/// `text`, cut to [`MAX_DROPPED_OP_EXCERPT_CHARS`] characters.
///
/// By characters, not bytes, so multi-byte text cannot be cut mid-codepoint.
fn bound_chars(text: &str) -> String {
    if text.chars().count() <= MAX_DROPPED_OP_EXCERPT_CHARS {
        return text.to_string();
    }
    let mut cut: String = text.chars().take(MAX_DROPPED_OP_EXCERPT_CHARS).collect();
    cut.push_str("...");
    cut
}

/// The first few dropped operations as one line, so an error or a log names
/// what came back without listing every element of a long answer.
///
/// `render` chooses how much of each one is shown:
/// [`Display`](std::fmt::Display) for the default log stream, and
/// [`DroppedOperation::with_element`] for the `DEBUG` line that quotes the
/// element.
fn summarize_dropped(
    dropped: &[DroppedOperation],
    render: impl Fn(&DroppedOperation) -> String,
) -> String {
    const SHOWN: usize = 3;
    let mut line = dropped
        .iter()
        .take(SHOWN)
        .map(render)
        .collect::<Vec<_>>()
        .join("; ");
    if let Some(rest) = dropped.len().checked_sub(SHOWN).filter(|r| *r > 0) {
        line.push_str(&format!("; and {rest} more"));
    }
    line
}

/// A bounded rendering of one element, so the report names the shape that came
/// back without carrying a whole merge body with it.
fn excerpt_of(element: &serde_json::Value) -> String {
    bound_chars(&element.to_string())
}

/// Read one consolidation answer into the operations it proposes.
///
/// Pure: it reports what it could not read rather than logging it, so the
/// caller decides how loud that is.
///
/// Three faults are told apart, because they need three different responses.
/// An answer that ended early is a size fault and the slice is halved. An
/// answer whose JSON does not parse at all is a formatting fault and repeating
/// it smaller only spends calls. An answer that parses but holds an element
/// that is not an operation is neither: the rest of it is still the work of an
/// expensive call, so the readable operations are kept and the others are
/// reported. An answer where nothing at all was readable stays a failure - a
/// plan that produced no work must not look like a model that kept everything.
fn parse_operations(response: &str) -> Result<ParsedOperations, ParseFailure> {
    let payload = extract_json_payload(response);
    let envelope = match serde_json::from_str::<OpsEnvelope>(&payload) {
        Ok(envelope) => envelope,
        // `classify` is the structured signal for "the input ended early".
        // Reading it off the message text would break on any serde_json wording
        // change.
        Err(e) if e.classify() == serde_json::error::Category::Eof => {
            return Err(ParseFailure::Truncated(e.to_string()));
        }
        Err(e) => {
            return Err(ParseFailure::Malformed {
                detail: envelope_fault(&payload, &e),
                dropped: 0,
            });
        }
    };

    let mut parsed = ParsedOperations {
        operations_keys: envelope.operations_keys,
        ..ParsedOperations::default()
    };
    for (index, element) in envelope.elements.iter().enumerate() {
        // Deserializing from `&Value` rather than from the owned value leaves
        // the element in hand for the excerpt when it does not read.
        match RawOp::deserialize(element) {
            Ok(op) => parsed.ops.push(op),
            Err(_) => parsed.dropped.push(DroppedOperation {
                index,
                op: element
                    .get("op")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                reason: reason_for(element),
                excerpt: excerpt_of(element),
            }),
        }
    }

    if parsed.ops.is_empty() && !parsed.dropped.is_empty() {
        return Err(ParseFailure::Malformed {
            detail: format!(
                "none of the {} proposed operation(s) could be read: {}",
                parsed.dropped.len(),
                summarize_dropped(&parsed.dropped, DroppedOperation::to_string)
            ),
            dropped: parsed.dropped.len(),
        });
    }
    Ok(parsed)
}

/// Log what one answer needed doing to it, so a repaired answer is visible
/// rather than quietly smaller than the one the model sent.
///
/// Both cases are warnings, not errors. The answer was usable, so the run
/// carries on; but a model that keeps mis-encoding its plan is losing work, and
/// nobody can see that from an info line that only counts what applied.
fn report_salvage(parsed: &ParsedOperations) {
    if parsed.operations_keys > 1 {
        tracing::warn!(
            "dreaming: the consolidation answer gave `operations` {} times; the arrays were \
             joined into one plan of {} operation(s) rather than the slice discarded",
            parsed.operations_keys,
            parsed.ops.len()
        );
    }
    if !parsed.dropped.is_empty() {
        tracing::warn!(
            "dreaming: consolidation salvaged {} operation(s) from the answer and dropped {} \
             it could not read; the entries those named stay unchanged until the next run: {}",
            parsed.ops.len(),
            parsed.dropped.len(),
            summarize_dropped(&parsed.dropped, DroppedOperation::to_string)
        );
        // What the model actually emitted, which is the next question anyone
        // asks. Held at DEBUG because the quote can carry the user's own
        // knowledge-base prose, the same level this file keeps a model-supplied
        // delete reason at.
        tracing::debug!(
            "dreaming: the operations consolidation could not read: {}",
            summarize_dropped(&parsed.dropped, DroppedOperation::with_element)
        );
    }
}

/// Recompute one slice, halving it when the model's answer comes back cut off.
///
/// A truncated answer means the slice asked for more output than the model
/// would return, so each half is recomputed instead, to
/// [`MAX_SLICE_SPLIT_DEPTH`] levels. Any other failure is returned as it is:
/// repeating a malformed answer with fewer entries only spends calls.
///
/// Returns the operations from every part that answered. This is an error only
/// when no part answered at all, so one failed half never discards the half
/// that worked - that loss is logged instead, because the entries in the failed
/// half simply go unreviewed until the next run.
async fn operations_for_slice(
    llm_fn: &DreamingLlmFn,
    slice: &[KbEntry],
    cancellation: &CancellationToken,
) -> Result<SliceAnswer, SliceFailure> {
    let mut ops: Vec<RawOp> = Vec::new();
    let mut dropped = 0usize;
    let mut pending: Vec<(&[KbEntry], usize)> = vec![(slice, 0)];
    let mut answered = 0usize;
    let mut unreviewed = 0usize;
    let mut last_failure: Option<String> = None;

    while let Some((chunk, depth)) = pending.pop() {
        // Each part is its own LLM call, so stop promptly between them.
        if cancellation.is_cancelled() {
            break;
        }

        let failure = match llm_fn(build_system_prompt(), build_user_prompt(chunk)).await {
            Ok(response) => match parse_operations(&response) {
                Ok(mut parsed) => {
                    report_salvage(&parsed);
                    dropped += parsed.dropped.len();
                    ops.append(&mut parsed.ops);
                    answered += 1;
                    continue;
                }
                Err(ParseFailure::Truncated(_))
                    if depth < MAX_SLICE_SPLIT_DEPTH && chunk.len() > 1 =>
                {
                    let (head, tail) = chunk.split_at(chunk.len() / 2);
                    tracing::debug!(
                        "dreaming: consolidation answer was cut off for {} entries; \
                         recomputing as {} + {}",
                        chunk.len(),
                        head.len(),
                        tail.len()
                    );
                    // Tail first: this is a stack, and the deletion cap keeps
                    // the operations it sees first, so the halves must come
                    // back in entry order.
                    pending.push((tail, depth + 1));
                    pending.push((head, depth + 1));
                    continue;
                }
                Err(e) => {
                    // An answer nothing could be read out of still lost the
                    // operations it proposed, so the count leaves with the
                    // failure rather than only as prose inside it.
                    dropped += e.dropped();
                    e.to_string()
                }
            },
            Err(e) => e,
        };

        unreviewed += chunk.len();
        last_failure = Some(failure);
    }

    if answered == 0 {
        return Err(SliceFailure {
            message: last_failure
                .unwrap_or_else(|| "dreaming: consolidation produced no answer".to_string()),
            dropped,
        });
    }

    if unreviewed > 0 {
        tracing::warn!(
            "dreaming: consolidation kept what it recovered, but {unreviewed} entr(ies) of this \
             slice were not recomputed and stay unchanged until the next run: {}",
            last_failure.as_deref().unwrap_or("unknown failure")
        );
    }

    Ok(SliceAnswer { ops, dropped })
}

/// Why a slice produced nothing at all, and what it lost on the way.
#[derive(Debug)]
struct SliceFailure {
    /// The last failure the slice hit, for the log and for the run's error.
    message: String,
    /// Proposed operations that were read but could not be used before the
    /// slice gave up. Counted here as well, so a loss is never invisible just
    /// because the slice as a whole failed.
    dropped: usize,
}

/// What one slice yielded, across every part of it that answered.
#[derive(Debug)]
struct SliceAnswer {
    /// Every operation recovered from the parts that answered, in the order
    /// the model gave them.
    ops: Vec<RawOp>,
    /// How many proposed operations could not be read and were set aside. The
    /// entries they named stay unchanged until the next run.
    dropped: usize,
}

#[cfg(test)]
mod tests {

    /// The same rule reaches the consolidation prompt.
    ///
    /// Consolidation rewrites and merges entries that already exist, so without
    /// it the pass tightens a mis-filed procedure into a better-written fact
    /// and cements it - the one pass in the cycle whose whole job is to decide
    /// what an entry should look like, deciding it without the rule that says
    /// this one should not be an entry at all.
    #[test]
    fn the_consolidation_prompt_states_the_method_is_not_a_fact_rule() {
        let prompt = build_system_prompt();
        assert!(
            prompt.contains(desktop_assistant_core::skill_promotion::METHOD_IS_NOT_A_FACT),
            "consolidation decides what an entry should be, so it has to be told what is not \
             one: {prompt}"
        );
    }
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex, Once};

    use super::*;

    /// An `io::Write` sink that appends into a shared buffer, so every writer
    /// handle a `fmt` layer builds lands in the same place.
    #[derive(Clone)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuf {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("the capture buffer is not poisoned")
                .extend_from_slice(data);
            Ok(data.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    static PERMISSIVE_GLOBAL_DEFAULT: Once = Once::new();

    /// Run `f` under a subscriber that writes every event into a buffer, and
    /// return `f`'s result with the captured text. Used to hold a count to
    /// reaching the log, rather than to the value the log line is built from.
    ///
    /// `tracing` caches each callsite's interest process-wide, so a callsite
    /// first evaluated on a thread with no subscriber can latch "never" for the
    /// whole test binary and a scoped subscriber then sees nothing. Installing a
    /// permissive process-wide default once keeps every callsite reachable.
    ///
    /// Safe to hold across `.await`: `#[tokio::test]`'s `current_thread` flavor
    /// never migrates a task to another thread mid-poll, so the thread-local
    /// default stays in force for `f`'s whole run.
    async fn capture_tracing<F, Fut, T>(f: F) -> (T, String)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        PERMISSIVE_GLOBAL_DEFAULT.call_once(|| {
            let _ = tracing::subscriber::set_global_default(
                tracing_subscriber::fmt()
                    .with_max_level(tracing::Level::TRACE)
                    .with_writer(io::sink)
                    .finish(),
            );
        });
        let buf = SharedBuf(Arc::new(Mutex::new(Vec::new())));
        let for_writer = buf.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_writer(move || for_writer.clone())
            .with_ansi(false)
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        let result = f().await;
        drop(guard);
        let bytes = buf
            .0
            .lock()
            .expect("the capture buffer is not poisoned")
            .clone();
        (
            result,
            String::from_utf8(bytes).expect("captured log output is UTF-8"),
        )
    }

    fn entry(id: &str, content: &str, tags: &[&str]) -> KbEntry {
        KbEntry {
            id: id.to_string(),
            content: content.to_string(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            metadata: KbMetadata::default(),
            source: None,
            review_generation: 0,
            summary: None,
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
        let ops = parse_operations(resp).unwrap().ops;
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
        let ops = parse_operations("{}").unwrap().ops;
        assert!(ops.is_empty());
    }

    // --- Replay: what the pass looks at first (#1127) ------------------------

    /// The instant every ordering test runs at. Fixed, so a use record's age is
    /// the number the test wrote and not the number the clock gave.
    fn replay_now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-08-07T12:00:00Z")
            .expect("a fixed clock parses")
            .with_timezone(&Utc)
    }

    /// A use record for `id`, opened at each of `ages` seconds ago.
    fn retrieved(id: &str, ages: &[i64]) -> KnowledgeUseRecord {
        let now = replay_now();
        KnowledgeUseRecord {
            entry_id: id.to_string(),
            offered_count: ages.len() as u64,
            opened_count: ages.len() as u64,
            marked_count: 0,
            first_seen_at: now
                - chrono::TimeDelta::seconds(ages.iter().copied().max().unwrap_or(1)),
            last_offered_at: Some(
                now - chrono::TimeDelta::seconds(ages.iter().copied().min().unwrap_or(1)),
            ),
            recent_uses: ages
                .iter()
                .map(|a| now - chrono::TimeDelta::seconds(*a))
                .collect(),
            marks: Vec::new(),
        }
    }

    /// The same record, plus a standing negative mark set `age` seconds ago.
    ///
    /// As the writer stores it: the same statement that moves `marked_count`
    /// prepends the stamp to the recent window, because a mark is a use
    /// whichever way it points. A fixture that moved only the counter would
    /// describe a record no store holds.
    fn contradicted(mut record: KnowledgeUseRecord, age: i64) -> KnowledgeUseRecord {
        record.marked_count += 1;
        record
            .recent_uses
            .push(replay_now() - chrono::TimeDelta::seconds(age));
        record.recent_uses.sort_unstable_by(|a, b| b.cmp(a));
        record.marks.push(
            desktop_assistant_core::domain::knowledge_use::KnowledgeMark {
                source: desktop_assistant_core::domain::MarkSource::Model,
                polarity: desktop_assistant_core::domain::MarkPolarity::Negative,
                reason: Some("the fact it states was withdrawn".to_string()),
                marked_at: replay_now() - chrono::TimeDelta::seconds(age),
            },
        );
        record
    }

    /// The ids of each slice, in the order the pass would examine them.
    fn examined(slices: Vec<Vec<KbEntry>>, records: &[KnowledgeUseRecord]) -> Vec<Vec<String>> {
        order_slices_by_replay_priority(slices, records, replay_now())
            .into_iter()
            .map(|slice| slice.into_iter().map(|e| e.id).collect())
            .collect()
    }

    /// Two slices, each holding one entry, in the order the loader gave them.
    fn two_slices(first: KbEntry, second: KbEntry) -> Vec<Vec<KbEntry>> {
        vec![vec![first], vec![second]]
    }

    /// Acceptance (#1127): the daily pass examines a recently retrieved entry
    /// before one that was only recently written.
    ///
    /// The entry nothing has reached for is the newer of the two and is sliced
    /// first, so a pass that ignored the log would spend its first call on it.
    /// Write activity alone cannot tell the two apart; the use log can.
    #[test]
    fn the_daily_pass_examines_a_recently_retrieved_entry_before_a_recently_written_one() {
        let slices = two_slices(
            entry("kb-written", "a fact nobody has reached for", &["a-topic"]),
            entry(
                "kb-retrieved",
                "a fact the work keeps needing",
                &["b-topic"],
            ),
        );
        let records = vec![retrieved("kb-retrieved", &[3_600, 7_200])];

        assert_eq!(
            examined(slices, &records),
            vec![
                vec!["kb-retrieved".to_string()],
                vec!["kb-written".to_string()]
            ]
        );
    }

    /// Acceptance (#1127): a fact that was retrieved and then contradicted is
    /// examined before one that was merely retrieved.
    ///
    /// Identical retrieval histories, so the contradiction is the only thing
    /// separating them.
    #[test]
    fn the_daily_pass_examines_a_contradicted_entry_before_a_merely_retrieved_one() {
        let slices = two_slices(
            entry("kb-fine", "a fact nobody has disputed", &["a-topic"]),
            entry("kb-wrong", "a fact somebody said was wrong", &["b-topic"]),
        );
        let records = vec![
            retrieved("kb-fine", &[600, 6_000]),
            contradicted(retrieved("kb-wrong", &[600, 6_000]), 600),
        ];

        assert_eq!(
            examined(slices, &records),
            vec![vec!["kb-wrong".to_string()], vec!["kb-fine".to_string()]]
        );
    }

    /// Acceptance (#1127): a salient entry is examined before a non-salient
    /// entry of the same age.
    ///
    /// Neither has ever been retrieved, which is the state most of a store is
    /// in, so salience is the only thing separating them.
    #[test]
    fn the_daily_pass_examines_a_salient_entry_before_a_non_salient_one_of_the_same_age() {
        let slices = two_slices(
            entry(
                "kb-plain",
                "the kitchen tap turns the wrong way",
                &["a-topic"],
            ),
            entry(
                "kb-salient",
                "the insurance renewal is due by the end of March",
                &["b-topic"],
            ),
        );

        assert_eq!(
            examined(slices, &[]),
            vec![vec!["kb-salient".to_string()], vec!["kb-plain".to_string()]]
        );
    }

    /// A slice's membership is exactly what the packing made it. Ordering moves
    /// whole slices, so a pair the tag sort put together - `{invoice}` beside
    /// `{invoices}`, which is the pair a merge is wanted for - is still shown to
    /// the model together, whatever their priorities.
    #[test]
    fn ordering_never_moves_an_entry_between_slices() {
        let slices = vec![
            vec![
                entry("kb-invoice", "the invoice went out on Monday", &["invoice"]),
                entry("kb-invoices", "invoices go out on Mondays", &["invoices"]),
            ],
            vec![entry("kb-zebra", "an unrelated fact", &["zebra"])],
        ];
        // The second slice is the retrieved one, so it leads - and the pair in
        // the first slice must still travel together behind it.
        let records = vec![retrieved("kb-zebra", &[60])];

        assert_eq!(
            examined(slices, &records),
            vec![
                vec!["kb-zebra".to_string()],
                vec!["kb-invoice".to_string(), "kb-invoices".to_string()],
            ]
        );
    }

    /// The pass still examines everything. Ordering decides what it reaches
    /// first, and a cancelled or partly-failed pass is what makes that matter -
    /// it must never decide what it reaches at all.
    #[test]
    fn ordering_examines_every_slice_and_every_entry_it_was_given() {
        let slices: Vec<Vec<KbEntry>> = (0..8)
            .map(|s| {
                (0..5)
                    .map(|i| {
                        entry(
                            &format!("kb-{s}-{i}"),
                            "a stored fact",
                            &[if s % 2 == 0 { "one" } else { "two" }],
                        )
                    })
                    .collect()
            })
            .collect();
        let records = vec![retrieved("kb-3-2", &[600]), retrieved("kb-6-0", &[60])];

        let ordered = examined(slices, &records);
        assert_eq!(ordered.len(), 8, "no slice may be dropped");
        let mut ids: Vec<String> = ordered.into_iter().flatten().collect();
        assert_eq!(ids.len(), 40);
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 40, "no entry may be dropped or duplicated");
    }

    /// Slices nothing separates keep the order the packing gave them, because
    /// the sort is stable.
    ///
    /// "Nothing separates them" is the load-bearing part and it is more than an
    /// empty use log: salience is read whether or not the log can be, so the
    /// entries here carry no cue either. A store with a use history of nothing
    /// and a deadline in one slice is **not** ordered as it was, and
    /// `the_daily_pass_examines_a_salient_entry_before_a_non_salient_one_of_the_same_age`
    /// is where that is stated.
    #[test]
    fn slices_with_nothing_to_separate_them_keep_the_order_they_were_packed_in() {
        let slices: Vec<Vec<KbEntry>> = (0..6)
            .map(|s| vec![entry(&format!("kb-{s}"), "a stored fact", &["topic"])])
            .collect();
        let expected: Vec<Vec<String>> = (0..6).map(|s| vec![format!("kb-{s}")]).collect();

        assert_eq!(examined(slices, &[]), expected);
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
    fn a_wholly_unreadable_answer_still_fails_and_reports_why() {
        let err = parse_operations(MALFORMED).expect_err("invalid JSON cannot parse");
        assert!(
            matches!(err, ParseFailure::Malformed { .. }),
            "expected a malformed verdict, got: {err:?}"
        );
        let message = err.to_string();
        assert!(
            message.len() > "dreaming: bad consolidation JSON: ".len(),
            "the failure must carry serde's reason, not only its own label: {message}"
        );
    }

    #[test]
    fn truncation_is_reported_as_truncation_even_when_whole_operations_precede_the_cut() {
        // Salvage reads complete elements out of a complete envelope. A cut-off
        // answer has no complete envelope, so the operations before the cut are
        // NOT salvaged - the answer keeps its truncation verdict and reaches the
        // halve-and-recompute path, which is the fix that actually fits.
        // The premise this test rests on: the fixture really does carry one
        // complete, valid operation before the cut, so salvage would be
        // tempting here and is deliberately not attempted.
        assert!(
            TRUNCATED.contains(r#"{"op": "delete", "ids": ["a"], "reason": "trivial"}"#),
            "the fixture must hold a complete operation ahead of the cut"
        );
        let err = parse_operations(TRUNCATED).expect_err("a cut-off response cannot parse");
        assert!(
            matches!(err, ParseFailure::Truncated(_)),
            "a whole operation before the cut must not turn truncation into salvage: {err:?}"
        );
        assert_eq!(
            err.dropped(),
            0,
            "a retried slice must not have its operations counted as lost as well"
        );
    }

    /// One realistic rewritten-entry body. A dozen of these push a repeated key
    /// several thousand characters into the answer, which is the shape the
    /// observed failure had - a repeat at position ten is a different test.
    fn rewritten_entry(index: usize) -> String {
        format!(
            "Preference {index}: the workspace keeps a dark theme in the editor and in the \
             terminal, at a 13 point font. Stated during setup and held across later sessions, \
             so it is durable rather than a one-off request. Merged from three near-duplicate \
             notes that each recorded one half of the same setting, and tightened so the entry \
             states the setting once."
        )
    }

    /// A consolidation answer that gives `operations` twice: a long first array,
    /// then a second copy of the key deep in the document.
    fn answer_with_a_repeated_operations_key() -> String {
        let first: Vec<String> = (0..14)
            .map(|i| {
                format!(
                    r#"{{"op":"edit","id":"kb-{i:03}","content":"{}"}}"#,
                    rewritten_entry(i)
                )
            })
            .collect();
        let second: Vec<String> = (0..3)
            .map(|i| format!(r#"{{"op":"delete","ids":["kb-9{i:02}"],"reason":"transient"}}"#))
            .collect();
        format!(
            r#"{{"operations":[{}],"operations":[{}]}}"#,
            first.join(","),
            second.join(",")
        )
    }

    #[test]
    fn a_repeated_operations_key_yields_the_union_of_both_arrays() {
        let answer = answer_with_a_repeated_operations_key();
        let repeat_at = answer
            .rfind(r#""operations""#)
            .expect("the fixture repeats the key");
        assert!(
            repeat_at > 4_000,
            "the repeat must sit deep in the answer, not at position {repeat_at}"
        );

        let parsed =
            parse_operations(&answer).expect("a repeated key is an encoding accident, not a loss");
        assert_eq!(
            parsed.operations_keys, 2,
            "the answer carried the key twice and that must be visible"
        );
        assert!(
            parsed.dropped.is_empty(),
            "every operation in both arrays is individually valid: {:?}",
            parsed.dropped
        );

        let edits: Vec<&str> = parsed
            .ops
            .iter()
            .filter_map(|o| match o {
                RawOp::Edit { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let deletes: Vec<&str> = parsed
            .ops
            .iter()
            .filter_map(|o| match o {
                RawOp::Delete { ids, .. } => ids.first().map(String::as_str),
                _ => None,
            })
            .collect();
        assert_eq!(edits.len(), 14, "the first array must survive in full");
        assert_eq!(deletes.len(), 3, "the second array must survive in full");
        assert_eq!(
            parsed.ops.len(),
            17,
            "the union of both arrays, and nothing else"
        );
        // Order matters: the prune cap keeps the operations it sees first.
        assert_eq!(edits.first(), Some(&"kb-000"));
        assert_eq!(deletes.last(), Some(&"kb-902"));
    }

    #[test]
    fn one_unreadable_operation_is_dropped_and_the_rest_apply() {
        // A merge with no `content` is a well-formed JSON object that is not a
        // well-formed operation: the two beside it are untouched by its fault.
        let answer = r#"{"operations":[
            {"op":"delete","ids":["kb-001"],"reason":"circumstantial"},
            {"op":"merge","ids":["kb-002","kb-003"]},
            {"op":"edit","id":"kb-004","content":"tightened"}
        ]}"#;

        let parsed =
            parse_operations(answer).expect("one bad operation must not discard the others");
        assert_eq!(parsed.ops.len(), 2, "the two valid operations must survive");
        assert_eq!(
            parsed.dropped.len(),
            1,
            "exactly one element was unreadable"
        );

        let report = parsed.dropped[0].to_string();
        assert_eq!(parsed.dropped[0].index, 1, "the report names its position");
        assert!(
            report.contains("content"),
            "the report must name the field that was missing: {report}"
        );
        assert!(
            report.contains("op=merge"),
            "the report must name what the model was attempting: {report}"
        );
        assert!(
            parsed.dropped[0].with_element().contains("kb-002"),
            "the quoted form must show the element it could not read"
        );
    }

    #[test]
    fn an_answer_whose_every_operation_is_unreadable_still_fails() {
        // Salvage keeps what it can. When it can keep nothing, the answer is a
        // fault, not an empty success: reporting it as "no changes" is the very
        // thing that made the original loss invisible.
        let answer = r#"{"operations":[{"op":"merge","ids":["kb-001","kb-002"]},{"id":"kb-003"}]}"#;
        let err =
            parse_operations(answer).expect_err("an answer with no usable operation is a failure");
        assert!(
            matches!(err, ParseFailure::Malformed { .. }),
            "expected a malformed verdict, got: {err:?}"
        );
        let message = err.to_string();
        assert!(
            message.contains('2'),
            "the failure must count what it could not read: {message}"
        );
        assert!(
            message.contains("content"),
            "the failure must say why it could not read them: {message}"
        );
    }

    #[test]
    fn a_single_operation_given_without_its_array_is_still_read() {
        // One operation object where a one-element array belongs is the same
        // class of encoding accident as the repeated key, so it is read rather
        // than rejected.
        let answer = r#"{"operations":{"op":"delete","ids":["kb-001"],"reason":"trivial"}}"#;
        let parsed =
            parse_operations(answer).expect("one un-arrayed operation is still an operation");
        assert_eq!(parsed.ops.len(), 1);
        assert!(parsed.dropped.is_empty());
    }

    #[test]
    fn an_operations_value_that_is_not_a_list_is_reported_not_swallowed() {
        // Leniency about the container does not extend to accepting a plan that
        // says nothing. A null where the operations belong is a fault, and
        // reading it as "kept everything" would hide it.
        let err = parse_operations(r#"{"operations":null}"#)
            .expect_err("a null plan is not an empty plan");
        assert!(
            matches!(err, ParseFailure::Malformed { .. }),
            "expected a malformed verdict, got: {err:?}"
        );
    }

    #[test]
    fn an_answer_with_no_operations_at_all_is_still_an_empty_success() {
        // A model that keeps everything proposes nothing, which is a real
        // outcome and must not be confused with an unreadable answer.
        let parsed = parse_operations(r#"{"operations":[]}"#).expect("keeping everything is valid");
        assert!(parsed.ops.is_empty());
        assert!(parsed.dropped.is_empty());
    }

    #[test]
    fn a_dropped_operation_never_carries_a_model_value_into_the_default_log() {
        // serde reports a wrong type by quoting the value, whole and unbounded.
        // A model that string-encodes an array instead of nesting it is an
        // ordinary mistake, and for a merge that value is the prose written for
        // a knowledge-base entry, so this is the likely path and not an exotic
        // one.
        let private = "the-thing-the-user-told-adele-in-confidence ".repeat(20);
        let answer = format!(
            r#"{{"operations":[
                {{"op":"edit","id":"kb-001","content":"tightened"}},
                {{"op":"merge","ids":"{private}","content":"unified"}}
            ]}}"#
        );

        let parsed = parse_operations(&answer).expect("the readable operation survives");
        assert_eq!(parsed.dropped.len(), 1);

        let diagnosis = parsed.dropped[0].to_string();
        assert!(
            !diagnosis.contains("in-confidence"),
            "no value the model wrote may reach the default log stream: {diagnosis}"
        );
        assert!(
            diagnosis.contains("sequence"),
            "the diagnosis must still say what was wrong: {diagnosis}"
        );
        assert!(
            diagnosis.contains("op=merge"),
            "and what the model was attempting: {diagnosis}"
        );
        assert!(
            diagnosis.chars().count() <= MAX_DROPPED_OP_EXCERPT_CHARS * 2,
            "the diagnosis must be bounded: {} chars",
            diagnosis.chars().count()
        );
    }

    #[test]
    fn an_answer_that_is_one_string_does_not_reach_the_default_log() {
        // The element path is not the only route. Serde reports a wrong
        // TOP-LEVEL type by quoting it too, and there the value is the whole
        // answer. Both ways in are reachable: a fenced block holding a bare
        // quoted string, and a bare quoted string with no brackets anywhere.
        let private = "the-thing-the-user-told-adele-in-confidence ".repeat(20);
        let routes = [
            format!("```json\n\"{private}\"\n```"),
            format!("\"{private}\""),
            // A wrong type followed by prose: the trailing text must not push
            // this back onto the message that quotes the value.
            format!("\"{private}\" and that is my answer"),
        ];

        for answer in routes {
            let err =
                parse_operations(&answer).expect_err("one string is not an operations envelope");
            let message = err.to_string();
            assert!(
                !message.contains("in-confidence"),
                "no value the model wrote may reach the default log stream: {message}"
            );
            assert!(
                message.contains("a string"),
                "the failure must still say what came back: {message}"
            );
            assert_eq!(err.dropped(), 0, "no element was ever separated out");
        }
    }

    #[test]
    fn an_answer_that_is_not_json_at_all_still_says_what_was_wrong() {
        // Serde's own message states a reason and a position and quotes
        // nothing, so it is kept rather than replaced by a type name there is
        // no value to read.
        let err = parse_operations("I decided to keep everything, so here is nothing.")
            .expect_err("prose is not an operations envelope");
        assert!(
            matches!(err, ParseFailure::Malformed { .. }),
            "expected a malformed verdict, got: {err:?}"
        );
        assert!(
            err.to_string().contains("expected value"),
            "the failure must still say why: {err}"
        );
    }

    #[test]
    fn an_answer_that_is_an_array_is_named_by_its_type() {
        // The shape the pull request says is not salvaged. It must fail by
        // naming its type, not by quoting what the array held.
        let err = parse_operations(r#"[{"op":"delete","ids":["kb-001"],"reason":"trivial"}]"#)
            .expect_err("a bare array is not an operations envelope");
        assert!(
            err.to_string().contains("an array"),
            "the failure must name the type that came back: {err}"
        );
    }

    #[test]
    fn a_dropped_operations_op_tag_is_bounded() {
        // `#[serde(other)]` folds every unrecognised tag into `Keep`, so a long
        // prose `op` reads cleanly and never reaches the dropped path today.
        // The bound is what keeps that a coupling rather than a dependency: an
        // `op` this long must be cut whether or not anything can produce one.
        let long_tag = "merge-".repeat(MAX_DROPPED_OP_EXCERPT_CHARS);
        let dropped = DroppedOperation {
            index: 0,
            op: Some(long_tag.clone()),
            reason: "missing field `content`".to_string(),
            excerpt: "{}".to_string(),
        };
        assert!(
            dropped.to_string().chars().count() < long_tag.chars().count(),
            "an over-long op tag must not reach the log whole"
        );
    }

    #[test]
    fn a_missing_field_is_still_named_after_the_values_are_replaced() {
        // Replacing the values must not cost the diagnosis its usefulness: the
        // field names serde can report are this code's own, not the model's.
        let answer = r#"{"operations":[
            {"op":"edit","id":"kb-001","content":"tightened"},
            {"op":"merge","ids":["kb-002","kb-003"]}
        ]}"#;
        let parsed = parse_operations(answer).expect("the readable operation survives");
        assert!(
            parsed.dropped[0].to_string().contains("content"),
            "the missing field must still be named: {}",
            parsed.dropped[0]
        );
    }

    #[test]
    fn bound_chars_cuts_on_a_character_boundary() {
        let over_long = "é".repeat(MAX_DROPPED_OP_EXCERPT_CHARS + 40);
        let cut = bound_chars(&over_long);
        assert_eq!(cut.chars().count(), MAX_DROPPED_OP_EXCERPT_CHARS + 3);
        assert!(cut.ends_with("..."), "a cut must say that it cut");
        assert!(
            cut.trim_end_matches('.').chars().all(|c| c == 'é'),
            "no partial codepoint"
        );
        let short = "already short";
        assert_eq!(bound_chars(short), short, "a short text is left alone");
    }

    #[test]
    fn a_dropped_operation_excerpt_is_bounded() {
        let huge = "é".repeat(MAX_DROPPED_OP_EXCERPT_CHARS * 4);
        let answer = format!(
            r#"{{"operations":[
                {{"op":"edit","id":"kb-001","content":"tightened"}},
                {{"op":"merge","body":"{huge}"}}
            ]}}"#
        );
        let parsed = parse_operations(&answer).expect("the readable operation survives");
        assert_eq!(parsed.dropped.len(), 1);

        let quoted = parsed.dropped[0].with_element();
        assert!(
            quoted.chars().count() < huge.chars().count(),
            "one long element must not carry its whole body into the log"
        );
        // `serde_json` writes an object's keys in alphabetical order, so `body`
        // precedes `op` and a bounded quote alone would cut the one field that
        // says what the model was attempting.
        assert!(
            quoted.contains("op=merge"),
            "a bounded report must still name what the operation was: {quoted}"
        );
        // The default log stream gets the diagnosis without the element, so a
        // fragment of the user's knowledge base does not reach it.
        let diagnosis = parsed.dropped[0].to_string();
        assert!(
            !diagnosis.contains(&huge[..8]),
            "the element itself must not reach the default log stream: {diagnosis}"
        );
    }

    /// An answer that proposes `good` valid deletes and one element that cannot
    /// be read, so a caller sees both a salvaged count and a dropped count.
    fn answer_with_one_unreadable_operation(good: usize) -> String {
        let mut ops: Vec<String> = (0..good)
            .map(|i| format!(r#"{{"op":"delete","ids":["kb-{i:03}"],"reason":"trivial"}}"#))
            .collect();
        ops.push(r#"{"op":"merge","ids":["kb-900","kb-901"]}"#.to_string());
        format!(r#"{{"operations":[{}]}}"#, ops.join(","))
    }

    #[tokio::test]
    async fn the_salvaged_and_dropped_counts_reach_the_log() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm = recording_llm(calls, |_| Ok(answer_with_one_unreadable_operation(2)));
        let token = CancellationToken::new();

        let slice = slice_of(2);
        let (answer, logged) = capture_tracing(|| operations_for_slice(&llm, &slice, &token)).await;
        let answer = answer.expect("two readable operations are still a usable answer");

        assert_eq!(answer.ops.len(), 2);
        assert_eq!(answer.dropped, 1, "the slice must carry the dropped count");
        assert!(
            logged.contains("salvaged 2 operation(s)"),
            "the log must say how many were kept: {logged}"
        );
        assert!(
            logged.contains("dropped 1"),
            "the log must say how many were lost: {logged}"
        );
        assert!(
            logged.contains("WARN"),
            "dropping proposed work needs a level an operator reads: {logged}"
        );
    }

    /// An answer that proposes two operations and neither can be read.
    const ALL_UNREADABLE: &str =
        r#"{"operations":[{"op":"merge","ids":["kb-1","kb-2"]},{"id":"kb-3"}]}"#;

    #[tokio::test]
    async fn operations_dropped_by_an_unreadable_answer_are_still_counted() {
        // An answer where nothing at all could be read is returned as a
        // failure, not as an empty success. The work it proposed is just as
        // lost as one bad operation among ten, so the count must leave with the
        // failure rather than only as prose inside it.
        let calls = Arc::new(Mutex::new(Vec::new()));
        // Three entries split into one and two when the first answer is cut
        // off. The two-entry half answers; the one-entry half is unreadable.
        let llm = recording_llm(calls, |seen| match seen {
            n if n > 2 => Ok(TRUNCATED.to_string()),
            2 => Ok(ops_response(&["kb-001"])),
            _ => Ok(ALL_UNREADABLE.to_string()),
        });

        let answer = operations_for_slice(&llm, &slice_of(3), &CancellationToken::new())
            .await
            .expect("the half that answered still returns what worked");

        assert_eq!(answer.ops.len(), 1, "the readable half must survive");
        assert_eq!(
            answer.dropped, 2,
            "a half that lost every operation must still report the loss"
        );
    }

    #[tokio::test]
    async fn a_slice_that_fails_outright_still_reports_what_it_dropped() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm = recording_llm(calls, |_| Ok(ALL_UNREADABLE.to_string()));

        let failure = operations_for_slice(&llm, &slice_of(1), &CancellationToken::new())
            .await
            .expect_err("an answer with no usable operation fails the slice");

        assert_eq!(
            failure.dropped, 2,
            "a failed slice must still say how much proposed work it lost"
        );
    }

    #[tokio::test]
    async fn a_repeated_operations_key_is_reported_in_the_log() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let llm = recording_llm(calls, |_| Ok(answer_with_a_repeated_operations_key()));
        let token = CancellationToken::new();

        let slice = slice_of(2);
        let (answer, logged) = capture_tracing(|| operations_for_slice(&llm, &slice, &token)).await;
        answer.expect("a repaired answer is still an answer");

        assert!(
            logged.contains("`operations`"),
            "the repair must name what the model got wrong: {logged}"
        );
        assert!(
            logged.contains("WARN"),
            "a repaired answer is worth an operator's attention: {logged}"
        );
    }

    #[test]
    fn the_prompt_asks_for_exactly_one_operations_array() {
        let prompt = build_system_prompt();
        assert!(
            prompt.contains("exactly one"),
            "the prompt must ask for one `operations` array: {prompt}"
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
            .expect("splitting recovers the slice")
            .ops;

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
            .expect("splitting recovers the slice")
            .ops;

        let deletes = ops
            .iter()
            .filter(|o| matches!(o, RawOp::Delete { .. }))
            .count();
        assert_eq!(deletes, 2, "both halves' operations must be kept");
    }

    #[tokio::test]
    async fn split_retry_keeps_the_slice_in_entry_order() {
        // The deletion cap truncates the collected operations, so the order the
        // parts come back in decides which deletes survive the cap. Halves must
        // stay in entry order.
        let llm: DreamingLlmFn = Box::new(|_system, user: String| {
            let seen = entries_in_prompt(&user);
            let first = user
                .lines()
                .find(|l| l.starts_with("## "))
                .map(|l| l.trim_start_matches("## ").to_string())
                .unwrap_or_default();
            let result = if seen > 2 {
                Ok(TRUNCATED.to_string())
            } else {
                Ok(ops_response(&[&first]))
            };
            Box::pin(async move { result })
        });

        let ops = operations_for_slice(&llm, &slice_of(4), &CancellationToken::new())
            .await
            .expect("splitting recovers the slice")
            .ops;

        let deleted: Vec<String> = ops
            .iter()
            .filter_map(|o| match o {
                RawOp::Delete { ids, .. } => ids.first().cloned(),
                _ => None,
            })
            .collect();
        assert_eq!(
            deleted,
            vec!["id0".to_string(), "id2".to_string()],
            "the first half must be recomputed before the second"
        );
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
        // Three entries split into one and two, so the halves differ in size:
        // the two-entry half answers and the one-entry half stays broken.
        let llm = recording_llm(calls, |seen| {
            if seen > 2 {
                Ok(TRUNCATED.to_string())
            } else if seen == 2 {
                Ok(ops_response(&["kept"]))
            } else {
                Err("backend refused".to_string())
            }
        });

        let ops = operations_for_slice(&llm, &slice_of(3), &CancellationToken::new())
            .await
            .expect("a partial recovery still returns what worked")
            .ops;
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
