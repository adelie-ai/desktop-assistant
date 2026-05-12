//! Shared types and tunables for the dream cycle (issue #108).

use std::future::Future;
use std::pin::Pin;

/// Boxed async LLM function: `(system_prompt, user_prompt) → Result<response, error>`.
///
/// Kept as plain string-in/string-out so the daemon can plug in any backend.
/// JSON output is parsed by the dreaming layer; tool-use isn't required.
pub type DreamingLlmFn = Box<
    dyn Fn(String, String) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

pub use crate::embedding_backfill::BackfillEmbedFn;

/// Maximum characters per message when building transcripts. Long messages
/// are truncated at a char boundary to keep prompts bounded.
pub const MAX_MESSAGE_CHARS: usize = 2000;

/// Maximum number of conversations to process in a single extraction scan.
pub const MAX_CONVERSATIONS_PER_SCAN: i64 = 10;

/// Maximum number of KB entries to review per consolidation cycle.
///
/// Reviews are gated by `reviewed_at IS NULL`, so unreviewed entries
/// distribute across cycles rather than re-running on a static window.
pub const MAX_REVIEWS_PER_CYCLE: i64 = 20;

/// Maximum number of retrieval candidates surfaced to the LLM during
/// per-memory review. Small enough to keep prompts tight; large enough to
/// catch likely-related entries.
pub const MAX_REVIEW_CANDIDATES: i64 = 8;

/// Cap on how many times an entry can be re-reviewed across cycles.
///
/// Generation 0 = never reviewed; bumps on mutation (merge target, update
/// applied). Past this cap the entry is treated as permanently settled to
/// prevent review loops.
pub const MAX_REVIEW_GENERATION: i16 = 2;

/// Soft-delete TTL. Entries with `deleted_at` older than this are reaped.
pub const SOFT_DELETE_TTL_DAYS: i32 = 30;

#[derive(Debug, Default, Clone, Copy)]
pub struct ConsolidationStats {
    pub reviewed: usize,
    pub updated: usize,
    pub merged_clusters: usize,
    pub soft_deleted: usize,
    pub scope_added: usize,
}
