//! Shared types and tunables for the dream cycle (issue #108).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use desktop_assistant_core::ports::auth::UserId;

/// Callback a maintenance pass invokes after a batch of knowledge changes lands
/// for a given user, so the daemon can broadcast a `KnowledgeChanged` event and
/// connected panels refetch live ("live as entries change"). Invoked per
/// conversation (extraction) and per user (consolidation), as work progresses.
pub type KnowledgeChangeFn = Arc<dyn Fn(&UserId) + Send + Sync>;

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

/// Cap on how many times an entry can be re-reviewed across cycles.
///
/// Generation 0 = never reviewed; bumps on mutation (merge target, update
/// applied). Past this cap the entry is treated as permanently settled to
/// prevent review loops.
pub const MAX_REVIEW_GENERATION: i16 = 2;

/// Soft-delete TTL. Entries with `deleted_at` older than this are reaped.
pub const SOFT_DELETE_TTL_DAYS: i32 = 30;

/// Character budget for one holistic-consolidation prompt. A user's active KB
/// is recomputed in a single LLM call when it fits under this; otherwise it is
/// sliced into tag-grouped chunks under this budget. ~200k chars ≈ 50k tokens,
/// comfortably within a strong model's context with room for the response.
pub const MAX_HOLISTIC_PROMPT_CHARS: usize = 200_000;

/// Safety cap: the fraction of a user's active entries a single holistic run
/// may delete outright. Merges (which preserve content in a canonical row)
/// don't count. Protects against a bad run gutting the store; excess deletes
/// are dropped with a warning.
pub const MAX_DELETE_FRACTION: f64 = 0.5;

#[derive(Debug, Default, Clone, Copy)]
pub struct ConsolidationStats {
    pub reviewed: usize,
    pub updated: usize,
    pub merged_clusters: usize,
    pub soft_deleted: usize,
    pub scope_added: usize,
}
