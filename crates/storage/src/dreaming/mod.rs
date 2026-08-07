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
//! 3. **Archival** — marks long-quiet conversations as archived. Also part of
//!    [`run_dreaming_scan`].
//! 4. **Consolidation** (infrequent, strong model) — loads a user's entire
//!    active KB and recomputes it holistically (prune / merge / tighten),
//!    applying explicit operations in one transaction with soft-delete. Run on
//!    its own slower cadence by [`run_consolidation_scan`].
//! 5. **Trash sweep** (frequent, cheap, no LLM) — frees soft-deleted entries
//!    past their retention window. Deliberately independent of the passes
//!    above: see `trash` and [`sweep_expired_trash`].

mod archival;
mod common;
mod consolidation;
mod extraction;
mod reconcile;
mod summarize;
mod trash;
mod types;

use desktop_assistant_core::CoreError;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;

use crate::knowledge_delete::KnowledgeDeletePolicy;

pub use summarize::{SummaryStats, run_summary_phase};
pub use trash::{empty_trash, reap_expired_trash, sweep_expired_trash, trash_count};
pub use types::{
    BackfillEmbedFn, ConsolidationStats, DreamingLlmFn, KbDeleteKind, KnowledgeChangeFn,
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
    if stats.merged_clusters > 0
        || stats.updated > 0
        || stats.soft_deleted > 0
        || stats.scope_added > 0
    {
        tracing::info!(
            "consolidation: reviewed {}, merged {} cluster(s), updated {}, scope-added {}, \
             soft-deleted {}; refused {} prune(s) of user-entered entries and {} \
             mutation(s) of settled ones; deferred {} prune(s) and {} rewrite(s) over the \
             configured share",
            stats.reviewed,
            stats.merged_clusters,
            stats.updated,
            stats.scope_added,
            stats.soft_deleted,
            stats.protected_from_delete,
            stats.settled_unchanged,
            stats.prunes_over_cap,
            stats.rewrites_over_cap,
        );
    } else {
        tracing::debug!(
            "consolidation: reviewed {} entr{}, no changes",
            stats.reviewed,
            if stats.reviewed == 1 { "y" } else { "ies" }
        );
    }
    Ok(stats)
}
