//! Periodic extraction, consolidation, and archival of long-term knowledge
//! ("dreaming"). See issue #108 for the design.
//!
//! Work is split across two clocks:
//!
//! 1. **Extraction** (frequent, cheap) — scans conversations for new messages
//!    beyond their watermark, asks an LLM to extract durable facts, persists
//!    them with structured scope and a source-conversation pointer. Tags are
//!    constrained to a formal registry. Run by [`run_dreaming_scan`].
//! 2. **Summary backfill** (frequent, cheap, batched) — writes the one-line
//!    `summary` for entries that have none, and rewrites one whose body changed
//!    after it was written. Never touches `content`. Also part of
//!    [`run_dreaming_scan`]; see `summarize`.
//! 3. **Mis-filed procedure sweep** (frequent, bounded, one model call per
//!    batch) — reads knowledge entries that were never read for this and
//!    proposes each one that is really a method as an UNAPPROVED skill, naming
//!    the entry it came from. It never rewrites the entry. A ledger records
//!    every entry it has read, so a store is read once per entry per edit
//!    rather than once per entry per cycle. Also part of [`run_dreaming_scan`];
//!    see `misfiled`.
//! 4. **Archival** — marks long-quiet conversations as archived. Also part of
//!    [`run_dreaming_scan`].
//! 5. **Consolidation** (infrequent, strong model) — loads a user's entire
//!    active KB and recomputes it holistically (prune / merge / tighten),
//!    applying explicit operations in one transaction with soft-delete. Run on
//!    its own slower cadence by [`run_consolidation_scan`].
//! 6. **Trash sweep** (frequent, cheap, no LLM) — frees soft-deleted entries
//!    past their retention window. Deliberately independent of the passes
//!    above: see `trash` and [`sweep_expired_trash`].

mod archival;
mod common;
mod consolidation;
mod extraction;
mod misfiled;
mod reconcile;
mod skills;
mod summarize;
mod trash;
mod types;

use desktop_assistant_core::CoreError;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

use crate::knowledge_delete::KnowledgeDeletePolicy;

pub use misfiled::{MAX_SWEPT_ENTRIES_PER_CYCLE, MisfiledStats, run_misfiled_sweep_phase};
pub use summarize::{SummaryStats, run_summary_phase};
pub use trash::{empty_trash, reap_expired_trash, sweep_expired_trash, trash_count};
pub use types::{
    BackfillEmbedFn, ConsolidationStats, Disposition, DreamingLlmFn, KnowledgeChangeFn,
    MAX_DELETE_REASON_CHARS, MAX_REVIEW_GENERATION, MAX_SUMMARIES_PER_CYCLE, SOFT_DELETE_TTL_DAYS,
    SOURCE_EXPLICIT,
};

/// Surfaced for the DB-gated watermark-scoping integration test (#435). The
/// `(user_id, conversation_id)` upsert guard on `dreaming_watermarks` (a second
/// user cannot clobber a watermark keyed by a conversation id it does not own)
/// cannot be reached through the extraction entry points, because conversation
/// ids are globally unique — a single conversation belongs to exactly one user,
/// so the cross-user ON CONFLICT branch never fires via normal extraction.
pub use common::update_watermark;

/// Surfaced for the DB-gated consolidation-applier tests (#893). Consolidation's
/// own guard (`consolidation.rs`, reading the entries it already loaded) and the
/// applier's SQL predicate always agree when driven through the public
/// [`run_consolidation_scan`] entry point, because they enforce the same rule -
/// so the only way to prove the applier's own guard actually holds, rather than
/// merely riding along behind an app-level filter that already agrees with it,
/// is to call it directly with an operation that filter would have refused. The
/// same call also drives the idempotent-replay proof (8.4): apply the same
/// `SynthesizedMerge` twice and check the second call changes nothing new.
pub use reconcile::{OpBuffer, ProposedOp, SynthesizedMerge, apply_ops};

/// Run one dreaming scan cycle: extract new facts, write the knowledge
/// summaries that are missing or stale, and archive old conversations.
/// Consolidation runs separately (see [`run_consolidation_scan`]) on a slower
/// cadence. Returns the number of new facts written.
///
/// Only extraction's outcome is returned, because that is the count the
/// maintenance task reports. The other two phases report themselves in the log
/// and never fail the cycle: a summarising model that is down, or an archival
/// query that errors, must not discard the facts extraction already wrote.
///
/// `cancellation` is observed between conversations so an on-demand run can be
/// stopped via the task registry. `on_change`, when set, is invoked after each
/// conversation that writes facts, and after each user whose summaries change,
/// so connected knowledge panels refetch live.
pub async fn run_dreaming_scan(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    archive_after_days: u32,
    cancellation: &CancellationToken,
    on_change: Option<&KnowledgeChangeFn>,
) -> Result<usize, CoreError> {
    tracing::info!("dreaming: extraction phase");
    let new_facts = extraction::run_extraction_phase(
        pool,
        llm_fn,
        embed_fn,
        embedding_model,
        cancellation,
        on_change,
    )
    .await?;

    // After extraction, so the facts this cycle wrote get their line in the same
    // cycle rather than the next one.
    tracing::info!("dreaming: summary phase");
    match summarize::run_summary_phase(pool, llm_fn, cancellation, on_change).await {
        Ok(stats) if stats.attempted > 0 => tracing::info!(
            "dreaming: wrote {} of {} summary line(s) attempted; {} entr{} still without a \
             current summary",
            stats.written,
            stats.attempted,
            stats.remaining,
            if stats.remaining == 1 { "y" } else { "ies" }
        ),
        Ok(_) => tracing::debug!("dreaming: every knowledge entry has a current summary"),
        Err(e) => tracing::warn!("dreaming: summary phase failed: {e}"),
    }

    // After the summary phase, so a cycle spends its cheap per-entry work before
    // its per-entry model calls, and after extraction, so an entry written this
    // cycle is judged in the same one. Its failure is logged and dropped, like
    // every phase after extraction: the entries it did not judge stay in its
    // worklist and the next cycle reads them.
    tracing::info!("dreaming: mis-filed procedure sweep");
    match misfiled::run_misfiled_sweep_phase(pool, llm_fn, cancellation).await {
        Ok(stats) if stats.judged > 0 => tracing::info!(
            "dreaming: read {} knowledge entr{} for mis-filed procedures and proposed {} \
             unapproved skill(s); {} entr{} still unread",
            stats.judged,
            if stats.judged == 1 { "y" } else { "ies" },
            stats.proposed,
            stats.remaining,
            if stats.remaining == 1 { "y" } else { "ies" }
        ),
        Ok(_) => tracing::debug!("dreaming: every knowledge entry has been read for procedures"),
        Err(e) => tracing::warn!("dreaming: mis-filed procedure sweep failed: {e}"),
    }

    if archive_after_days > 0 {
        tracing::info!("dreaming: archival phase");
        match archival::run_archival_phase(pool, archive_after_days).await {
            Ok(n) if n > 0 => tracing::info!(
                "dreaming: archived {n} conversation(s) older than {archive_after_days} day(s)"
            ),
            Ok(_) => tracing::debug!("dreaming: no conversations to archive"),
            Err(e) => tracing::warn!("dreaming: archival phase failed: {e}"),
        }
    }

    Ok(new_facts)
}

/// Run one holistic-consolidation scan across all users. Loads each user's
/// entire active KB and recomputes it with the (typically stronger) backend
/// model. Returns aggregate operation counts.
///
/// `cancellation` is observed between users (and between prompt slices) so an
/// on-demand run can be stopped via the task registry. `on_change`, when set, is
/// invoked after each user whose KB changed so connected panels refetch live.
///
/// `policy` states what one run may destroy and rewrite: the share of the
/// active set it may prune, the share it may rewrite in place, whether a hard
/// delete needs a person, and the retention its opportunistic trash reap
/// applies. The reap uses the same retention the periodic sweep does, because
/// both read the same configured value.
pub async fn run_consolidation_scan(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    policy: KnowledgeDeletePolicy,
    cancellation: &CancellationToken,
    on_change: Option<&KnowledgeChangeFn>,
) -> Result<ConsolidationStats, CoreError> {
    let stats =
        consolidation::run_consolidation_phase(pool, llm_fn, policy, cancellation, on_change)
            .await?;
    match report_for(&stats) {
        ConsolidationReport::Applied => tracing::info!(
            "consolidation: reviewed {}, merged {} cluster(s), updated {}, scope-added {}, \
             dispositioned {}; refused {} disposition(s) of user-entered entries and {} \
             mutation(s) of settled ones; deferred {} disposition(s) and {} rewrite(s) over the \
             configured share; dropped {} unreadable operation(s)",
            stats.reviewed,
            stats.merged_clusters,
            stats.updated,
            stats.scope_added,
            stats.soft_deleted,
            stats.protected_from_delete,
            stats.settled_unchanged,
            stats.prunes_over_cap,
            stats.rewrites_over_cap,
            stats.dropped_operations,
        ),
        // A run that proposed nothing and a run whose every proposal was
        // refused both apply zero changes, but they are not the same story
        // (#712 item 1): the second means the model kept asking and every
        // answer was no, which a quiet "no changes" line would hide.
        ConsolidationReport::RefusalsOnly => tracing::info!(
            "consolidation: reviewed {} entr{}, changed nothing - every proposal was refused: {}",
            stats.reviewed,
            if stats.reviewed == 1 { "y" } else { "ies" },
            describe_refusals(&stats),
        ),
        ConsolidationReport::Unreadable => {
            // A run that changed nothing because it could not read what the
            // model proposed is not a quiet no-op either. Say so at a level
            // an operator reads, or it looks exactly like a run with nothing
            // to do.
            tracing::warn!(
                "consolidation: reviewed {} entr{} and changed nothing, after dropping {} \
                 unreadable operation(s)",
                stats.reviewed,
                if stats.reviewed == 1 { "y" } else { "ies" },
                stats.dropped_operations,
            );
        }
        ConsolidationReport::NoOp => tracing::debug!(
            "consolidation: reviewed {} entr{}, no changes",
            stats.reviewed,
            if stats.reviewed == 1 { "y" } else { "ies" }
        ),
    }
    Ok(stats)
}

/// What one consolidation run has to report, decided from its counters.
///
/// "Nothing changed" is not one story. A run that proposed nothing looks the
/// same on the surface as a run whose every proposal was refused, but an
/// operator needs to tell them apart (#712 item 1): the second means the
/// guards are earning their keep, and it stops looking that way the moment it
/// is folded into "no changes".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsolidationReport {
    /// At least one change applied.
    Applied,
    /// Nothing applied, but at least one proposal was refused - by a guard, a
    /// backstop, or a budget.
    RefusalsOnly,
    /// Nothing applied and nothing was refused, but at least one proposed
    /// operation could not even be read.
    Unreadable,
    /// A quiet run: nothing proposed, nothing refused, nothing unreadable.
    NoOp,
}

/// Sum of the counters that mean "at least one row changed".
fn applied_count(stats: &ConsolidationStats) -> usize {
    stats.merged_clusters + stats.updated + stats.soft_deleted + stats.scope_added
}

/// Sum of the counters that mean "a proposal was understood and refused" -
/// every guard, budget, and backstop this unit adds.
fn refusal_count(stats: &ConsolidationStats) -> usize {
    stats.protected_from_delete
        + stats.settled_unchanged
        + stats.scope_guard_refusals
        + stats.backstop_firings
        + stats.prunes_over_cap
        + stats.rewrites_over_cap
}

fn report_for(stats: &ConsolidationStats) -> ConsolidationReport {
    if applied_count(stats) > 0 {
        ConsolidationReport::Applied
    } else if refusal_count(stats) > 0 {
        ConsolidationReport::RefusalsOnly
    } else if stats.dropped_operations > 0 {
        ConsolidationReport::Unreadable
    } else {
        ConsolidationReport::NoOp
    }
}

/// One line naming every refusal counter this unit adds, for the
/// [`ConsolidationReport::RefusalsOnly`] log line.
fn describe_refusals(stats: &ConsolidationStats) -> String {
    format!(
        "{} explicit-entry, {} settled-entry, {} scope-guard, {} backstop, {} over the \
         disposition share, {} over the rewrite share",
        stats.protected_from_delete,
        stats.settled_unchanged,
        stats.scope_guard_refusals,
        stats.backstop_firings,
        stats.prunes_over_cap,
        stats.rewrites_over_cap,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_with(f: impl FnOnce(&mut ConsolidationStats)) -> ConsolidationStats {
        let mut stats = ConsolidationStats::default();
        f(&mut stats);
        stats
    }

    #[test]
    fn a_run_that_proposed_nothing_is_a_no_op() {
        assert_eq!(
            report_for(&ConsolidationStats::default()),
            ConsolidationReport::NoOp
        );
    }

    #[test]
    fn a_run_that_applied_a_change_reports_applied() {
        let stats = stats_with(|s| s.updated = 1);
        assert_eq!(report_for(&stats), ConsolidationReport::Applied);
    }

    /// #712 item 1, named: a run whose only outcome is refusals must not read
    /// as "no changes" - it must be its own, distinct report.
    #[test]
    fn a_refusal_only_run_logs_at_info_with_the_counts() {
        let stats = stats_with(|s| {
            s.reviewed = 3;
            s.settled_unchanged = 1;
        });

        assert_eq!(
            report_for(&stats),
            ConsolidationReport::RefusalsOnly,
            "a run with a refusal and no applied change is its own report, not a no-op"
        );
        let description = describe_refusals(&stats);
        assert!(
            description.contains("1 settled-entry"),
            "the counts must actually be named, not just their presence: {description}"
        );
    }

    #[test]
    fn each_refusal_counter_alone_is_enough_to_report_refusals_only() {
        // Every counter this unit adds must actually move the decision, or a
        // guard could fire with nobody able to see it in the run's report.
        let fields: Vec<fn(&mut ConsolidationStats)> = vec![
            |s| s.protected_from_delete = 1,
            |s| s.settled_unchanged = 1,
            |s| s.scope_guard_refusals = 1,
            |s| s.backstop_firings = 1,
            |s| s.prunes_over_cap = 1,
            |s| s.rewrites_over_cap = 1,
        ];
        for set in fields {
            let stats = stats_with(set);
            assert_eq!(
                report_for(&stats),
                ConsolidationReport::RefusalsOnly,
                "stats {stats:?} must report as refusals-only"
            );
        }
    }

    #[test]
    fn an_applied_change_wins_over_a_refusal_in_the_same_run() {
        let stats = stats_with(|s| {
            s.updated = 1;
            s.settled_unchanged = 1;
        });
        assert_eq!(report_for(&stats), ConsolidationReport::Applied);
    }

    #[test]
    fn unreadable_operations_alone_are_their_own_report() {
        let stats = stats_with(|s| s.dropped_operations = 1);
        assert_eq!(report_for(&stats), ConsolidationReport::Unreadable);
    }

    #[test]
    fn refusals_are_reported_over_unreadable_operations_in_the_same_run() {
        let stats = stats_with(|s| {
            s.settled_unchanged = 1;
            s.dropped_operations = 1;
        });
        assert_eq!(report_for(&stats), ConsolidationReport::RefusalsOnly);
    }
}
