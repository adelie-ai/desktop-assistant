//! End-of-cycle reconciliation: op buffer + union-find merge clustering
//! + transactional apply (issue #108).
//!
//! Per-memory consolidation reviews emit proposed operations; this module
//! aggregates them, computes merge clusters (`merge(A,B)` + `merge(B,C)` →
//! cluster `{A,B,C}`), and applies everything in a single transaction with
//! soft-delete semantics for retired entries.
//!
//! A retired row records *why* it was retired: a merge member names the
//! canonical row that absorbed it, a prune carries the model's stated reason.
//! The two outcomes are very different - one relocates the content, the other
//! destroys it - so they must be separable on disk, not just in the logs.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::auth::current_user_id;
use sqlx::PgPool;

use super::types::{ConsolidationStats, Disposition, MAX_REVIEW_GENERATION, SOURCE_EXPLICIT};
use crate::kb_metadata::{KbMetadata, KbScope};
use crate::knowledge_delete::KnowledgeDeletePolicy;

/// Operations a per-memory review can propose.
#[derive(Debug, Clone)]
pub enum ProposedOp {
    Update {
        id: String,
        new_content: String,
    },
    AddScope {
        id: String,
        scope: KbScope,
    },
    Merge {
        a: String,
        b: String,
    },
    Delete {
        id: String,
        /// The model's stated reason, already bounded by the caller. `None`
        /// when it gave none: a tombstone reads better as "unstated" than as
        /// an empty string.
        reason: Option<String>,
    },
}

/// Synthesized result of merging a cluster, produced by an LLM synthesis call.
#[derive(Debug, Clone)]
pub struct SynthesizedMerge {
    pub canonical_id: String,
    pub member_ids: Vec<String>,
    pub new_content: String,
    pub new_scope: Option<KbScope>,
}

/// Collects ops during a consolidation cycle. Merges aggregate into clusters
/// by set-union; same pair proposed twice is a no-op.
#[derive(Debug, Default)]
pub struct OpBuffer {
    merge_pairs: BTreeSet<(String, String)>,
    updates: HashMap<String, String>,
    scope_adds: HashMap<String, KbScope>,
    deletes: HashMap<String, Option<String>>,
    /// All ids touched by any op (focal memories), used to mark reviewed.
    reviewed_ids: BTreeSet<String>,
}

impl OpBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_reviewed(&mut self, id: &str) {
        self.reviewed_ids.insert(id.to_string());
    }

    pub fn absorb(&mut self, op: ProposedOp) {
        match op {
            ProposedOp::Update { id, new_content } => {
                self.reviewed_ids.insert(id.clone());
                self.updates.insert(id, new_content);
            }
            ProposedOp::AddScope { id, scope } => {
                self.reviewed_ids.insert(id.clone());
                self.scope_adds.insert(id, scope);
            }
            ProposedOp::Merge { a, b } => {
                if a == b {
                    return;
                }
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                self.reviewed_ids.insert(lo.clone());
                self.reviewed_ids.insert(hi.clone());
                self.merge_pairs.insert((lo, hi));
            }
            ProposedOp::Delete { id, reason } => {
                self.reviewed_ids.insert(id.clone());
                self.deletes.insert(id, reason);
            }
        }
    }

    /// Compute connected components on merge pairs. Returns clusters of
    /// size ≥ 2, each as a sorted set of ids. The canonical id (lowest
    /// lexicographic id in the cluster) is the merge target.
    pub fn merge_clusters(&self) -> Vec<BTreeSet<String>> {
        let mut parent: BTreeMap<String, String> = BTreeMap::new();

        fn find(parent: &mut BTreeMap<String, String>, x: &str) -> String {
            let mut cur = x.to_string();
            loop {
                let p = parent.get(&cur).cloned().unwrap_or_else(|| cur.clone());
                if p == cur {
                    return cur;
                }
                let gp = parent.get(&p).cloned().unwrap_or_else(|| p.clone());
                parent.insert(cur.clone(), gp.clone());
                cur = gp;
            }
        }

        fn union(parent: &mut BTreeMap<String, String>, a: &str, b: &str) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return;
            }
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent.insert(hi, lo);
        }

        for (a, b) in &self.merge_pairs {
            parent.entry(a.clone()).or_insert_with(|| a.clone());
            parent.entry(b.clone()).or_insert_with(|| b.clone());
            union(&mut parent, a, b);
        }

        let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let all_ids: Vec<String> = parent.keys().cloned().collect();
        for id in all_ids {
            let root = find(&mut parent, &id);
            groups.entry(root).or_default().insert(id);
        }

        groups.into_values().filter(|set| set.len() >= 2).collect()
    }

    /// Canonical id (merge target) for a cluster: the lexicographically lowest.
    pub fn canonical_of(cluster: &BTreeSet<String>) -> Option<&String> {
        cluster.iter().next()
    }

    /// Ids that are part of *any* merge cluster — their individual update/
    /// delete/scope ops are subsumed by the merge synthesis.
    pub fn clustered_ids(&self) -> BTreeSet<String> {
        self.merge_clusters().into_iter().flatten().collect()
    }

    /// Update ops on ids that aren't in any merge cluster.
    pub fn standalone_updates(&self) -> Vec<(String, String)> {
        let in_cluster = self.clustered_ids();
        self.updates
            .iter()
            .filter(|(id, _)| !in_cluster.contains(*id))
            .map(|(id, content)| (id.clone(), content.clone()))
            .collect()
    }

    pub fn standalone_scope_adds(&self) -> Vec<(String, KbScope)> {
        let in_cluster = self.clustered_ids();
        self.scope_adds
            .iter()
            .filter(|(id, _)| !in_cluster.contains(*id))
            .map(|(id, scope)| (id.clone(), scope.clone()))
            .collect()
    }

    pub fn standalone_deletes(&self) -> Vec<(String, Option<String>)> {
        let in_cluster = self.clustered_ids();
        self.deletes
            .iter()
            .filter(|(id, _)| !in_cluster.contains(*id))
            .map(|(id, reason)| (id.clone(), reason.clone()))
            .collect()
    }

    pub fn all_reviewed_ids(&self) -> &BTreeSet<String> {
        &self.reviewed_ids
    }
}

/// Apply all buffered + synthesized operations in a single transaction.
///
/// `synthesized` contains the LLM-produced merged content for each cluster.
/// Clusters not in `synthesized` are skipped (the synthesis call failed,
/// for example, and we'd rather keep both than guess).
///
/// Returns counts of applied operations.
pub async fn apply_ops(
    pool: &PgPool,
    buffer: &OpBuffer,
    synthesized: &[SynthesizedMerge],
    policy: KnowledgeDeletePolicy,
) -> Result<ConsolidationStats, CoreError> {
    let user_id = current_user_id();
    let mut stats = ConsolidationStats::default();

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: begin tx failed: {e}")))?;

    // First, reap any soft-deleted entries past their retention window. Cheap,
    // and happens in the same tx so a single cycle stays atomic. Scoped to the
    // current user. This is a convenience trigger, not the only one: the
    // daemon's periodic sweep (`dreaming::sweep_expired_trash`) reaps
    // regardless of whether consolidation is enabled at all. A policy that
    // reserves hard deletes to a person frees nothing here; the merges and
    // edits below still apply, because the refusal is a decline rather than a
    // failure and must not roll the transaction back.
    super::trash::reap_expired_for_user(&mut *tx, user_id.as_str(), policy).await?;

    // Apply merges: update canonical row, soft-delete cluster members. The
    // canonical row's embedding is left stale (not regenerated here) — the
    // bumped `updated_at` marks it for the background embedding-backfill task.
    for merge in synthesized {
        // One read for the whole cluster: the canonical row's metadata (its
        // source_conversation_id must survive the merge) and every member's
        // provenance.
        let member_rows: Vec<(String, Option<String>, serde_json::Value)> = sqlx::query_as(
            "SELECT id, source, metadata FROM knowledge_base \
             WHERE user_id = $1 AND id = ANY($2)",
        )
        .bind(user_id.as_str())
        .bind(&merge.member_ids)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: cluster fetch failed: {e}")))?;

        let mut metadata = member_rows
            .iter()
            .find(|(id, _, _)| id == &merge.canonical_id)
            .map(|(_, _, value)| KbMetadata::from_json(value))
            .unwrap_or_default();
        metadata.scope = merge.new_scope.clone();

        // A deliberately-entered fact absorbed by a merge keeps its provenance
        // on the surviving row. Stamping the canonical 'consolidation' would
        // strip the never-prune protection, and the next night's pass would be
        // free to delete a fact the user entered on purpose.
        let cluster_is_explicit = member_rows
            .iter()
            .any(|(_, source, _)| source.as_deref() == Some(SOURCE_EXPLICIT));

        sqlx::query(
            "UPDATE knowledge_base \
             SET content = $1, metadata = $2, \
                 source = CASE WHEN $3 THEN 'explicit' ELSE 'consolidation' END, \
                 updated_at = NOW(), \
                 reviewed_at = NOW(), \
                 review_generation = LEAST(review_generation + 1, $4) \
             WHERE user_id = $5 AND id = $6",
        )
        .bind(&merge.new_content)
        .bind(metadata.to_json())
        .bind(cluster_is_explicit)
        .bind(MAX_REVIEW_GENERATION)
        .bind(user_id.as_str())
        .bind(&merge.canonical_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: merge canonical update failed: {e}")))?;

        // Soft-delete the rest of the cluster, each member pointing at the row
        // that absorbed it. No reason is written: the model states none per
        // member, and `superseded_by` already says where the content went.
        let to_delete: Vec<String> = merge
            .member_ids
            .iter()
            .filter(|id| *id != &merge.canonical_id)
            .cloned()
            .collect();
        if !to_delete.is_empty() {
            let result = sqlx::query(
                "UPDATE knowledge_base \
                 SET deleted_at = NOW(), reviewed_at = NOW(), \
                     disposition = $3, superseded_by = $4 \
                 WHERE user_id = $2 AND id = ANY($1) AND deleted_at IS NULL",
            )
            .bind(&to_delete)
            .bind(user_id.as_str())
            .bind(Disposition::Superseded.as_str())
            .bind(&merge.canonical_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                CoreError::Storage(format!("dreaming: cluster soft-delete failed: {e}"))
            })?;
            // Count rows the statement actually retired, so a member already
            // tombstoned by an earlier op is not counted twice.
            stats.soft_deleted += result.rows_affected() as usize;
        }

        stats.merged_clusters += 1;
    }

    // Standalone updates (not in any merge cluster). Embedding left stale for
    // the background backfill task; only content + watermarks change here.
    for (id, new_content) in buffer.standalone_updates() {
        // As with a merge canonical: tightening the prose of a deliberately-
        // entered fact must not relabel it as the model's own output.
        sqlx::query(
            "UPDATE knowledge_base \
             SET content = $1, \
                 source = CASE WHEN source = 'explicit' THEN 'explicit' \
                               ELSE 'consolidation' END, \
                 updated_at = NOW(), \
                 reviewed_at = NOW(), \
                 review_generation = LEAST(review_generation + 1, $2) \
             WHERE user_id = $4 AND id = $3",
        )
        .bind(&new_content)
        .bind(MAX_REVIEW_GENERATION)
        .bind(&id)
        .bind(user_id.as_str())
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: update failed: {e}")))?;
        stats.updated += 1;
    }

    // Standalone scope additions.
    for (id, scope) in buffer.standalone_scope_adds() {
        let existing: Option<(serde_json::Value,)> =
            sqlx::query_as("SELECT metadata FROM knowledge_base WHERE user_id = $1 AND id = $2")
                .bind(user_id.as_str())
                .bind(&id)
                .fetch_optional(&mut *tx)
                .await
                .map_err(|e| {
                    CoreError::Storage(format!("dreaming: scope-add metadata fetch failed: {e}"))
                })?;

        if let Some((value,)) = existing {
            let mut metadata = KbMetadata::from_json(&value);
            metadata.scope = Some(scope);
            sqlx::query(
                "UPDATE knowledge_base \
                 SET metadata = $1, updated_at = NOW(), reviewed_at = NOW() \
                 WHERE user_id = $3 AND id = $2",
            )
            .bind(metadata.to_json())
            .bind(&id)
            .bind(user_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(format!("dreaming: scope-add update failed: {e}")))?;
            stats.scope_added += 1;
        }
    }

    // Standalone prunes: nothing supersedes them, so the model's stated reason
    // is the only record of why the entry is gone. The `source` predicate is a
    // backstop: consolidation already filters protected ids out of the plan,
    // but the row itself refuses too, so no future caller of `apply_ops` can
    // prune a deliberately-entered fact by forgetting the filter.
    for (id, reason) in buffer.standalone_deletes() {
        let result = sqlx::query(
            "UPDATE knowledge_base \
             SET deleted_at = NOW(), reviewed_at = NOW(), \
                 disposition = $3, disposition_reason = $4 \
             WHERE user_id = $2 AND id = $1 \
               AND deleted_at IS NULL \
               AND source IS DISTINCT FROM 'explicit'",
        )
        .bind(&id)
        .bind(user_id.as_str())
        .bind(Disposition::Trivial.as_str())
        .bind(reason.as_deref())
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: soft-delete failed: {e}")))?;
        stats.soft_deleted += result.rows_affected() as usize;
    }

    // Any reviewed id that didn't already get its reviewed_at touched
    // (i.e. the LLM said "keep") still needs the watermark moved.
    let touched: Vec<String> = buffer.all_reviewed_ids().iter().cloned().collect();
    if !touched.is_empty() {
        sqlx::query(
            "UPDATE knowledge_base \
             SET reviewed_at = COALESCE(reviewed_at, NOW()) \
             WHERE user_id = $2 AND id = ANY($1)",
        )
        .bind(&touched)
        .bind(user_id.as_str())
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: reviewed_at update failed: {e}")))?;
        stats.reviewed = touched.len();
    }

    tx.commit()
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: commit failed: {e}")))?;

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_pairs_become_single_cluster() {
        let mut b = OpBuffer::new();
        b.absorb(ProposedOp::Merge {
            a: "C".into(),
            b: "A".into(),
        });
        b.absorb(ProposedOp::Merge {
            a: "A".into(),
            b: "C".into(),
        });
        let clusters = b.merge_clusters();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    #[test]
    fn transitive_merges_collapse_into_one_cluster() {
        let mut b = OpBuffer::new();
        b.absorb(ProposedOp::Merge {
            a: "A".into(),
            b: "C".into(),
        });
        b.absorb(ProposedOp::Merge {
            a: "C".into(),
            b: "D".into(),
        });
        let clusters = b.merge_clusters();
        assert_eq!(clusters.len(), 1);
        let c = &clusters[0];
        assert!(c.contains("A") && c.contains("C") && c.contains("D"));
    }

    #[test]
    fn disjoint_merge_pairs_stay_separate() {
        let mut b = OpBuffer::new();
        b.absorb(ProposedOp::Merge {
            a: "A".into(),
            b: "B".into(),
        });
        b.absorb(ProposedOp::Merge {
            a: "X".into(),
            b: "Y".into(),
        });
        let clusters = b.merge_clusters();
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn self_merge_is_dropped() {
        let mut b = OpBuffer::new();
        b.absorb(ProposedOp::Merge {
            a: "A".into(),
            b: "A".into(),
        });
        assert!(b.merge_clusters().is_empty());
    }

    #[test]
    fn canonical_is_lexicographically_lowest() {
        let mut b = OpBuffer::new();
        b.absorb(ProposedOp::Merge {
            a: "z-id".into(),
            b: "a-id".into(),
        });
        b.absorb(ProposedOp::Merge {
            a: "m-id".into(),
            b: "z-id".into(),
        });
        let clusters = b.merge_clusters();
        assert_eq!(clusters.len(), 1);
        let canonical = OpBuffer::canonical_of(&clusters[0]).unwrap();
        assert_eq!(canonical, "a-id");
    }

    #[test]
    fn standalone_updates_exclude_clustered_ids() {
        let mut b = OpBuffer::new();
        b.absorb(ProposedOp::Merge {
            a: "A".into(),
            b: "B".into(),
        });
        b.absorb(ProposedOp::Update {
            id: "A".into(),
            new_content: "x".into(),
        });
        b.absorb(ProposedOp::Update {
            id: "Z".into(),
            new_content: "z".into(),
        });
        let standalone = b.standalone_updates();
        assert_eq!(standalone.len(), 1);
        assert_eq!(standalone[0].0, "Z");
    }

    #[test]
    fn idempotent_same_pair_twice() {
        let mut b = OpBuffer::new();
        b.absorb(ProposedOp::Merge {
            a: "A".into(),
            b: "B".into(),
        });
        b.absorb(ProposedOp::Merge {
            a: "B".into(),
            b: "A".into(),
        });
        b.absorb(ProposedOp::Merge {
            a: "A".into(),
            b: "B".into(),
        });
        let clusters = b.merge_clusters();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }
}
