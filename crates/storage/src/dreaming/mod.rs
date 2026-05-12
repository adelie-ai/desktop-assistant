//! Periodic extraction, consolidation, and archival of long-term knowledge
//! ("dreaming"). See issue #108 for the design.
//!
//! Each scan cycle runs three phases:
//!
//! 1. **Extraction** — scans conversations for new messages beyond their
//!    watermark, asks an LLM to extract durable facts, persists them with
//!    structured scope (project/tool/null) and a source-conversation
//!    pointer. Tags are constrained to a formal registry.
//! 2. **Consolidation** — per-memory review of entries that haven't been
//!    reviewed yet (gated by `reviewed_at IS NULL`). Retrieves related
//!    candidates by tag overlap + embedding similarity, proposes
//!    operations (keep/update/merge/add-scope/delete), buffers them,
//!    union-finds merge clusters, synthesizes unified content per cluster,
//!    and applies everything in a single transaction with soft-delete.
//! 3. **Archival** — marks long-quiet conversations as archived.

mod archival;
mod common;
mod consolidation;
mod extraction;
mod reconcile;
mod types;

use sqlx::PgPool;

pub use types::{BackfillEmbedFn, ConsolidationStats, DreamingLlmFn};

/// Run one dreaming scan cycle: extract new facts, consolidate existing
/// memories, archive old conversations. Returns the number of new facts
/// written during extraction.
pub async fn run_dreaming_scan(
    pool: &PgPool,
    llm_fn: &DreamingLlmFn,
    embed_fn: &BackfillEmbedFn,
    embedding_model: &str,
    archive_after_days: u32,
) -> Result<usize, String> {
    let new_facts =
        extraction::run_extraction_phase(pool, llm_fn, embed_fn, embedding_model).await?;

    match consolidation::run_consolidation_phase(pool, llm_fn, embed_fn, embedding_model)
        .await
    {
        Ok(stats) => {
            if stats.merged_clusters > 0
                || stats.updated > 0
                || stats.soft_deleted > 0
                || stats.scope_added > 0
            {
                tracing::info!(
                    "dreaming: consolidation reviewed {}, merged {} cluster(s), updated {}, scope-added {}, soft-deleted {}",
                    stats.reviewed,
                    stats.merged_clusters,
                    stats.updated,
                    stats.scope_added,
                    stats.soft_deleted,
                );
            } else {
                tracing::debug!(
                    "dreaming: consolidation reviewed {} entr{}, no changes",
                    stats.reviewed,
                    if stats.reviewed == 1 { "y" } else { "ies" }
                );
            }
        }
        Err(e) => tracing::warn!("dreaming: consolidation phase failed: {e}"),
    }

    if archive_after_days > 0 {
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
