//! End-of-cycle reconciliation: op buffer + union-find merge clustering
//! + transactional apply (issue #108).
//!
//! Per-memory consolidation reviews emit proposed operations; this module
//! aggregates them, computes merge clusters (`merge(A,B)` + `merge(B,C)` →
//! cluster `{A,B,C}`), and applies everything in a single transaction with
//! soft-delete semantics for retired entries.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use desktop_assistant_core::chunking::{CHUNK_MAX_CHARS, CHUNK_OVERLAP, chunk_text};
use pgvector::Vector;
use sqlx::PgPool;

use super::types::{BackfillEmbedFn, ConsolidationStats, MAX_REVIEW_GENERATION};
use crate::kb_metadata::{KbMetadata, KbScope};

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
        reason: String,
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
    deletes: HashMap<String, String>,
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

        groups
            .into_values()
            .filter(|set| set.len() >= 2)
            .collect()
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

    pub fn standalone_deletes(&self) -> Vec<(String, String)> {
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
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    buffer: &OpBuffer,
    synthesized: &[SynthesizedMerge],
    soft_delete_ttl_days: i32,
) -> Result<ConsolidationStats, String> {
    let mut stats = ConsolidationStats::default();

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("dreaming: begin tx failed: {e}"))?;

    // First, reap any soft-deleted entries past their TTL. Cheap, and
    // happens in the same tx so a single cycle stays atomic.
    sqlx::query(
        "DELETE FROM knowledge_base
         WHERE deleted_at IS NOT NULL
           AND deleted_at < NOW() - make_interval(days => $1)",
    )
    .bind(soft_delete_ttl_days)
    .execute(&mut *tx)
    .await
    .map_err(|e| format!("dreaming: TTL reap failed: {e}"))?;

    // Apply merges: update canonical row, soft-delete cluster members.
    for merge in synthesized {
        let chunks = chunk_text(&merge.new_content, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        let embeddings = embed_fn(chunks).await?;
        if embeddings.is_empty() {
            tracing::warn!(
                "dreaming: synthesis embedding empty for cluster canonical {}",
                merge.canonical_id
            );
            continue;
        }
        let embedding_vecs: Vec<Vector> =
            embeddings.into_iter().map(Vector::from).collect();

        // Preserve source_conversation_id of the canonical row but apply
        // the new scope.
        let existing_metadata: Option<(serde_json::Value,)> = sqlx::query_as(
            "SELECT metadata FROM knowledge_base WHERE id = $1",
        )
        .bind(&merge.canonical_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| format!("dreaming: metadata fetch failed: {e}"))?;

        let mut metadata = existing_metadata
            .map(|(v,)| KbMetadata::from_json(&v))
            .unwrap_or_default();
        metadata.scope = merge.new_scope.clone();

        sqlx::query(
            "UPDATE knowledge_base
             SET content = $1, metadata = $2, embedding = $3::vector[],
                 embedding_model = $4, updated_at = NOW(),
                 reviewed_at = NOW(),
                 review_generation = LEAST(review_generation + 1, $5)
             WHERE id = $6",
        )
        .bind(&merge.new_content)
        .bind(metadata.to_json())
        .bind(&embedding_vecs)
        .bind(embedding_model)
        .bind(MAX_REVIEW_GENERATION)
        .bind(&merge.canonical_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("dreaming: merge canonical update failed: {e}"))?;

        // Soft-delete the rest of the cluster.
        let to_delete: Vec<String> = merge
            .member_ids
            .iter()
            .filter(|id| *id != &merge.canonical_id)
            .cloned()
            .collect();
        if !to_delete.is_empty() {
            sqlx::query(
                "UPDATE knowledge_base
                 SET deleted_at = NOW(), reviewed_at = NOW()
                 WHERE id = ANY($1) AND deleted_at IS NULL",
            )
            .bind(&to_delete)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("dreaming: cluster soft-delete failed: {e}"))?;
            stats.soft_deleted += to_delete.len();
        }

        stats.merged_clusters += 1;
    }

    // Standalone updates (not in any merge cluster).
    for (id, new_content) in buffer.standalone_updates() {
        let chunks = chunk_text(&new_content, CHUNK_MAX_CHARS, CHUNK_OVERLAP);
        let embeddings = embed_fn(chunks).await?;
        if embeddings.is_empty() {
            continue;
        }
        let embedding_vecs: Vec<Vector> =
            embeddings.into_iter().map(Vector::from).collect();

        sqlx::query(
            "UPDATE knowledge_base
             SET content = $1, embedding = $2::vector[], embedding_model = $3,
                 updated_at = NOW(),
                 reviewed_at = NOW(),
                 review_generation = LEAST(review_generation + 1, $4)
             WHERE id = $5",
        )
        .bind(&new_content)
        .bind(&embedding_vecs)
        .bind(embedding_model)
        .bind(MAX_REVIEW_GENERATION)
        .bind(&id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("dreaming: update failed: {e}"))?;
        stats.updated += 1;
    }

    // Standalone scope additions.
    for (id, scope) in buffer.standalone_scope_adds() {
        let existing: Option<(serde_json::Value,)> = sqlx::query_as(
            "SELECT metadata FROM knowledge_base WHERE id = $1",
        )
        .bind(&id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| format!("dreaming: scope-add metadata fetch failed: {e}"))?;

        if let Some((value,)) = existing {
            let mut metadata = KbMetadata::from_json(&value);
            metadata.scope = Some(scope);
            sqlx::query(
                "UPDATE knowledge_base
                 SET metadata = $1, updated_at = NOW(), reviewed_at = NOW()
                 WHERE id = $2",
            )
            .bind(metadata.to_json())
            .bind(&id)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("dreaming: scope-add update failed: {e}"))?;
            stats.scope_added += 1;
        }
    }

    // Standalone soft-deletes.
    for (id, _reason) in buffer.standalone_deletes() {
        sqlx::query(
            "UPDATE knowledge_base
             SET deleted_at = NOW(), reviewed_at = NOW()
             WHERE id = $1 AND deleted_at IS NULL",
        )
        .bind(&id)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("dreaming: soft-delete failed: {e}"))?;
        stats.soft_deleted += 1;
    }

    // Any reviewed id that didn't already get its reviewed_at touched
    // (i.e. the LLM said "keep") still needs the watermark moved.
    let touched: Vec<String> = buffer.all_reviewed_ids().iter().cloned().collect();
    if !touched.is_empty() {
        sqlx::query(
            "UPDATE knowledge_base
             SET reviewed_at = COALESCE(reviewed_at, NOW())
             WHERE id = ANY($1)",
        )
        .bind(&touched)
        .execute(&mut *tx)
        .await
        .map_err(|e| format!("dreaming: reviewed_at update failed: {e}"))?;
        stats.reviewed = touched.len();
    }

    tx.commit()
        .await
        .map_err(|e| format!("dreaming: commit failed: {e}"))?;

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
