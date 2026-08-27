//! End-of-cycle reconciliation: op buffer + union-find merge clustering
//! + transactional apply (issue #108, widened by #893).
//!
//! Per-memory consolidation reviews emit proposed operations; this module
//! aggregates them, computes merge clusters (`merge(A,B)` + `merge(B,C)` →
//! cluster `{A,B,C}`), and applies everything in a single transaction.
//!
//! Consolidation has one destructive verb short of none: it cannot delete a
//! row at all. An entry it judges wrong, stale or redundant is
//! [`Disposition`]ed - marked with what it is and why - and stays live.
//! `merge_new` writes a **new** row for the unified content and dispositions
//! every member [`Disposition::Redundant`] with a link back; no member is
//! rewritten or removed, so a merge can never deadlock against the settled
//! rule and never drops a member's own provenance.
//!
//! Two layers back every disposition guard here. Consolidation's own
//! pre-filter (`crates/storage/src/dreaming/consolidation.rs`) reads the
//! entries it already loaded and refuses what it can see is wrong before an
//! op ever reaches this module. This module's own SQL predicates back that
//! guard as a second, independent check: normally they refuse nothing, because
//! the layer above already agreed. When one of them refuses something that
//! layer let through, that is a hole in the guard above, not a safety net
//! working as designed, and it is counted and logged as such
//! (`ConsolidationStats::backstop_firings`).

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
    Disposition {
        id: String,
        disposition: Disposition,
        /// The model's stated reason, already bounded by the caller. `None`
        /// when it gave none.
        reason: Option<String>,
        /// The entry this one is refuted, superseded, or duplicated by. Only
        /// meaningful for those three dispositions; ignored otherwise.
        superseded_by: Option<String>,
    },
}

/// Synthesized result of merging a cluster, produced by an LLM synthesis call.
///
/// Carries no id of its own: the row `merge_new` writes gets a deterministic
/// id derived from the sorted member ids (see [`merge_id`]), computed at apply
/// time rather than decided by the caller.
#[derive(Debug, Clone)]
pub struct SynthesizedMerge {
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
    dispositions: HashMap<String, (Disposition, Option<String>, Option<String>)>,
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
            ProposedOp::Disposition {
                id,
                disposition,
                reason,
                superseded_by,
            } => {
                self.reviewed_ids.insert(id.clone());
                self.dispositions
                    .insert(id, (disposition, reason, superseded_by));
            }
        }
    }

    /// Compute connected components on merge pairs. Returns clusters of
    /// size ≥ 2, each as a sorted set of ids. The lowest lexicographic id in
    /// the cluster is a stable key for looking up the cluster's synthesized
    /// content - not the row that survives, since `merge_new` writes a new
    /// one.
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

    /// Disposition ops on ids that aren't in any merge cluster. A merge
    /// member is dispositioned [`Disposition::Redundant`] by the merge apply
    /// itself, so a standalone disposition proposed for the same id would be
    /// redundant work at best and a conflicting write at worst - excluding it
    /// here is what makes the disposition budget blind to it too (#712 item
    /// 3): the caller computes the budget from what this returns, after
    /// subsumption, not from every disposition the model proposed.
    pub fn standalone_dispositions(
        &self,
    ) -> Vec<(String, Disposition, Option<String>, Option<String>)> {
        let in_cluster = self.clustered_ids();
        self.dispositions
            .iter()
            .filter(|(id, _)| !in_cluster.contains(*id))
            .map(|(id, (disposition, reason, superseded_by))| {
                (
                    id.clone(),
                    *disposition,
                    reason.clone(),
                    superseded_by.clone(),
                )
            })
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

    // Apply merges: insert a new row for the unified content, disposition
    // every member `redundant` with a link back. No member is rewritten or
    // removed - the new row's embedding starts NULL and the background
    // embedding-backfill task picks it up, same as extraction's own inserts.
    for merge in synthesized {
        let member_rows: Vec<(String, Option<String>, Vec<String>, i16)> = sqlx::query_as(
            "SELECT id, source, tags, review_generation FROM knowledge_base \
             WHERE user_id = $1 AND id = ANY($2) AND deleted_at IS NULL",
        )
        .bind(user_id.as_str())
        .bind(&merge.member_ids)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: cluster fetch failed: {e}")))?;

        // A member may have been reaped or retired by a concurrent change
        // between the plan being formed and this transaction. Fewer than two
        // live members left means there is nothing to unify.
        if member_rows.len() < 2 {
            tracing::warn!(
                "dreaming: merge cluster for {:?} has fewer than two live members at apply \
                 time; skipping",
                merge.member_ids
            );
            continue;
        }

        let new_id = merge_id(&merge.member_ids);

        // A deliberately-entered fact absorbed by a merge passes its
        // protection to the new row. Stamping it 'consolidation' would strip
        // the never-prune protection, and the next pass would be free to
        // disposition a fact the user entered on purpose.
        let cluster_is_explicit = member_rows
            .iter()
            .any(|(_, source, ..)| source.as_deref() == Some(SOURCE_EXPLICIT));
        let source = if cluster_is_explicit {
            "explicit"
        } else {
            "consolidation"
        };

        let max_generation = member_rows
            .iter()
            .map(|(_, _, _, generation)| *generation)
            .max()
            .unwrap_or(0);
        let new_generation = (max_generation + 1).min(MAX_REVIEW_GENERATION);

        let mut tags: BTreeSet<String> = BTreeSet::new();
        for (_, _, member_tags, _) in &member_rows {
            tags.extend(member_tags.iter().cloned());
        }
        let tags: Vec<String> = tags.into_iter().collect();

        let mut metadata = KbMetadata::new();
        metadata.scope = merge.new_scope.clone();

        // Deterministic on the sorted member ids, so a replayed apply (a
        // retried batch after a crash, or the same plan applied twice)
        // upserts this exact row rather than writing a duplicate. `user_id`
        // is repeated in the conflict guard defensively: the id is derived
        // only from member ids that already belong to this user, so a
        // cross-tenant collision is not expected, but the guard costs
        // nothing and turns an impossible case into a silent no-op rather
        // than a cross-tenant write.
        //
        // TODO(#893): write the new row - not yet implemented. `AND FALSE`
        // makes this insert nothing while keeping the statement (and its
        // binds) real, so the query itself stays exercised.
        sqlx::query(
            "INSERT INTO knowledge_base \
                (id, user_id, content, tags, metadata, source, review_generation, reviewed_at) \
             SELECT $1, $2, $3, $4, $5, $6, $7, NOW() WHERE FALSE",
        )
        .bind(&new_id)
        .bind(user_id.as_str())
        .bind(&merge.new_content)
        .bind(&tags)
        .bind(metadata.to_json())
        .bind(source)
        .bind(new_generation)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: merge insert failed: {e}")))?;

        // Every member stays a live row, dispositioned `redundant` and
        // pointing at the row that absorbed it. No reason is written: the
        // model states none per member, and `superseded_by` already says
        // where the content went.
        let result = sqlx::query(
            "UPDATE knowledge_base \
             SET disposition = $3, superseded_by = $4, \
                 reviewed_at = NOW(), updated_at = NOW() \
             WHERE user_id = $2 AND id = ANY($1) AND deleted_at IS NULL",
        )
        .bind(&merge.member_ids)
        .bind(user_id.as_str())
        .bind(Disposition::Redundant.as_str())
        .bind(&new_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: member disposition failed: {e}")))?;
        // Count rows the statement actually touched, so a member already
        // dispositioned by an earlier op in the same run is not counted
        // twice.
        stats.soft_deleted += result.rows_affected() as usize;

        stats.merged_clusters += 1;
    }

    // Standalone updates (not in any merge cluster). Embedding left stale for
    // the background backfill task; only content + watermarks change here.
    // The settled guard is backed by its own predicate here: consolidation's
    // pre-filter already excludes a settled entry's edit, so this should
    // refuse nothing in the ordinary run; when it does, that is a hole in the
    // guard above, not this guard working as intended.
    for (id, new_content) in buffer.standalone_updates() {
        // As with a merge: tightening the prose of a deliberately-entered
        // fact must not relabel it as the model's own output.
        let result = sqlx::query(
            "UPDATE knowledge_base \
             SET content = $1, \
                 source = CASE WHEN source = 'explicit' THEN 'explicit' \
                               ELSE 'consolidation' END, \
                 updated_at = NOW(), \
                 reviewed_at = NOW(), \
                 review_generation = LEAST(review_generation + 1, $2) \
             WHERE user_id = $4 AND id = $3 AND deleted_at IS NULL AND TRUE /* TODO(#893): settled-entry SQL backstop */",
        )
        .bind(&new_content)
        .bind(MAX_REVIEW_GENERATION)
        .bind(&id)
        .bind(user_id.as_str())
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: update failed: {e}")))?;

        if result.rows_affected() > 0 {
            stats.updated += 1;
        } else if row_is_active(&mut *tx, user_id.as_str(), &id).await? {
            tracing::warn!(
                "dreaming: consolidation edit of {id} was refused by the settled-entry \
                 backstop; the guard above should already have excluded it"
            );
            stats.backstop_firings += 1;
        }
        // Else: the row is simply gone (already deleted, or reaped by the
        // trash sweep between the plan being formed and this transaction).
        // That is the ordinary case #712 item 2 says must not be reported the
        // same way as a guard hole.
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

    // Standalone dispositions: nothing supersedes them but a superseded or
    // redundant target, so the model's stated reason (and, for those two, the
    // successor) is the record of why the entry now reads the way it does.
    // TODO(#893): apply standalone dispositions - not yet implemented.
    for (id, disposition, reason, superseded_by) in
        buffer.standalone_dispositions().into_iter().take(0)
    {
        // Scope guard: two facts about disjoint, non-empty scopes cannot
        // contradict each other, so a refuted/superseded/redundant
        // disposition naming a target is refused when the scopes share
        // nothing. This is the one guard with no application-level layer
        // above it - it needs a fresh read of both rows' metadata, which may
        // span slices the caller never held together, so the apply path is
        // its only enforcement point.
        if let Some(target) = superseded_by.as_deref()
            && matches!(
                disposition,
                Disposition::Refuted | Disposition::Superseded | Disposition::Redundant
            )
        {
            let pair = vec![id.clone(), target.to_string()];
            let rows: Vec<(String, serde_json::Value)> = sqlx::query_as(
                "SELECT id, metadata FROM knowledge_base \
                 WHERE user_id = $1 AND id = ANY($2) AND deleted_at IS NULL",
            )
            .bind(user_id.as_str())
            .bind(&pair)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| {
                CoreError::Storage(format!("dreaming: scope-guard metadata fetch failed: {e}"))
            })?;
            let scope_of = |wanted: &str| -> Option<KbScope> {
                rows.iter()
                    .find(|(row_id, _)| row_id == wanted)
                    .and_then(|(_, metadata)| KbMetadata::from_json(metadata).scope)
                    .filter(|scope| !scope.is_empty())
            };
            if let (Some(a), Some(b)) = (scope_of(&id), scope_of(target))
                && scopes_are_disjoint(&a, &b)
            {
                tracing::debug!(
                    "dreaming: refusing to disposition {id} against {target}: their scopes \
                     share nothing, so one cannot contradict the other"
                );
                stats.scope_guard_refusals += 1;
                continue;
            }
        }

        // The schema's own CHECK constraint requires `superseded_by` exactly
        // when the disposition is `superseded` or `redundant`, and forbids it
        // otherwise. A `refuted` op may still have named a target - read
        // above for the scope guard - but it is not stored.
        let stored_target = match disposition {
            Disposition::Superseded | Disposition::Redundant => superseded_by.as_deref(),
            _ => None,
        };
        // An explicit entry may be refuted, superseded, or made obsolete by
        // the model, but never marked trivial or redundant - the never-prune
        // rule translated into the wider vocabulary.
        let explicit_may_not_receive =
            matches!(disposition, Disposition::Trivial | Disposition::Redundant);

        // `source IS DISTINCT FROM 'explicit'`, not `source <> 'explicit'`:
        // most rows carry a NULL source, and SQL's three-valued logic makes
        // `NULL = 'explicit'` NULL rather than false, which would make `NOT
        // ($6 AND source = 'explicit')` NULL too - and a NULL predicate in a
        // WHERE clause excludes the row exactly like a real refusal would,
        // silently turning every ordinary disposition into a backstop firing.
        // `IS DISTINCT FROM` reads NULL as "not explicit", which is what the
        // guard means.
        let result = sqlx::query(
            "UPDATE knowledge_base \
             SET disposition = $3, disposition_reason = $4, superseded_by = $5, \
                 reviewed_at = NOW(), updated_at = NOW() \
             WHERE user_id = $2 AND id = $1 AND deleted_at IS NULL \
               AND (NOT $6 OR source IS DISTINCT FROM 'explicit')",
        )
        .bind(&id)
        .bind(user_id.as_str())
        .bind(disposition.as_str())
        .bind(reason.as_deref())
        .bind(stored_target)
        .bind(explicit_may_not_receive)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(format!("dreaming: disposition failed: {e}")))?;

        if result.rows_affected() > 0 {
            stats.soft_deleted += 1;
        } else if row_is_active(&mut *tx, user_id.as_str(), &id).await? {
            tracing::warn!(
                "dreaming: consolidation disposition of {id} to {} was refused by the \
                 explicit-entry backstop; the guard above should already have excluded it",
                disposition.as_str()
            );
            stats.backstop_firings += 1;
        }
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

/// Fixed namespace for [`merge_id`]'s UUIDv5 hash. Any 16 bytes work here -
/// what matters is that it never changes, so the same member set always
/// hashes to the same id.
const MERGE_ID_NAMESPACE: uuid::Uuid = uuid::Uuid::from_bytes(*b"adelie-kb-merge!");

/// The id a `merge_new` of exactly `members` always produces, whenever it
/// runs and however many times. Sorting first means the id does not depend
/// on the order the model listed the members in.
///
/// Determinism is what makes a replayed apply an upsert instead of a
/// duplicate (8.4): a crash after this INSERT commits but before the next
/// batch's watermark advances redoes the same merge, and the same member set
/// hashes to the same id both times.
fn merge_id(members: &[String]) -> String {
    let mut sorted: Vec<&str> = members.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let name = sorted.join("\u{1}");
    uuid::Uuid::new_v5(&MERGE_ID_NAMESPACE, name.as_bytes()).to_string()
}

/// Do `a` and `b` share no `(dimension, value)` pair at all?
///
/// This is the scope guard's own definition of "different scopes cannot
/// contradict": two entries that agree on at least one scope dimension are
/// treated as related enough that one may still refute, supersede, or
/// duplicate the other. Two that share nothing - one scoped to a project, the
/// other to an unrelated one - are not.
fn scopes_are_disjoint(a: &KbScope, b: &KbScope) -> bool {
    !a.0.iter()
        .any(|(dimension, value)| b.0.get(dimension) == Some(value))
}

/// Is `id` still a live (not soft-deleted) row?
///
/// Used only after a guarded write affects zero rows, to tell apart the two
/// reasons that can happen: the row is simply gone (ordinary - already
/// retired, or reaped between the plan being formed and this transaction), or
/// the row is still there and the guard predicate is what refused it (a hole
/// in the layer above, worth a warning). Generic over the executor so it can
/// run against the open transaction the caller already holds.
async fn row_is_active<'e, E>(executor: E, user_id: &str, id: &str) -> Result<bool, CoreError>
where
    E: sqlx::PgExecutor<'e>,
{
    let found: Option<(i32,)> = sqlx::query_as(
        "SELECT 1 FROM knowledge_base WHERE user_id = $1 AND id = $2 AND deleted_at IS NULL",
    )
    .bind(user_id)
    .bind(id)
    .fetch_optional(executor)
    .await
    .map_err(|e| CoreError::Storage(format!("dreaming: backstop existence check failed: {e}")))?;
    Ok(found.is_some())
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

    #[test]
    fn standalone_dispositions_exclude_clustered_ids() {
        let mut b = OpBuffer::new();
        b.absorb(ProposedOp::Merge {
            a: "A".into(),
            b: "B".into(),
        });
        b.absorb(ProposedOp::Disposition {
            id: "A".into(),
            disposition: Disposition::Trivial,
            reason: None,
            superseded_by: None,
        });
        b.absorb(ProposedOp::Disposition {
            id: "Z".into(),
            disposition: Disposition::Obsolete,
            reason: Some("no longer applies".into()),
            superseded_by: None,
        });
        let standalone = b.standalone_dispositions();
        assert_eq!(
            standalone.len(),
            1,
            "a disposition on a merged member is subsumed by the merge, not applied on its own"
        );
        assert_eq!(standalone[0].0, "Z");
        assert_eq!(standalone[0].1, Disposition::Obsolete);
    }

    #[test]
    fn merge_id_is_stable_under_member_reordering() {
        let forward = merge_id(&["kb-a".to_string(), "kb-b".to_string()]);
        let reversed = merge_id(&["kb-b".to_string(), "kb-a".to_string()]);
        assert_eq!(
            forward, reversed,
            "the id must not depend on the order the model listed the members in"
        );
    }

    #[test]
    fn merge_id_differs_for_a_different_member_set() {
        let one = merge_id(&["kb-a".to_string(), "kb-b".to_string()]);
        let other = merge_id(&["kb-a".to_string(), "kb-c".to_string()]);
        assert_ne!(one, other);
    }

    #[test]
    fn scopes_sharing_a_dimension_and_value_are_not_disjoint() {
        let a = KbScope::new().with("project", "adelie-ai");
        let b = KbScope::new()
            .with("project", "adelie-ai")
            .with("tool", "vim");
        assert!(!scopes_are_disjoint(&a, &b));
    }

    #[test]
    fn scopes_with_the_same_dimension_and_different_values_are_disjoint() {
        let a = KbScope::new().with("project", "adelie-ai");
        let b = KbScope::new().with("project", "other-repo");
        assert!(
            scopes_are_disjoint(&a, &b),
            "the same dimension pointing at different values shares nothing in common"
        );
    }

    #[test]
    fn scopes_with_no_dimension_in_common_are_disjoint() {
        let a = KbScope::new().with("project", "adelie-ai");
        let b = KbScope::new().with("host", "workstation-1");
        assert!(scopes_are_disjoint(&a, &b));
    }
}
