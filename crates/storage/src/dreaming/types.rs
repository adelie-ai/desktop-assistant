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

/// Cap on how many times consolidation may rewrite an entry.
///
/// Generation 0 = never reviewed; bumps on mutation (merge target, update
/// applied). At this cap the entry's prose is settled: consolidation is
/// enjoined from editing or merging it again.
///
/// Why a cap at all: consolidation shows the model the whole active store every
/// pass, including its own output from previous passes, so an uncapped entry is
/// a paraphrase of a paraphrase that drifts from what was observed toward what
/// the model believes.
///
/// Why the store does not freeze at the cap: it settles individual entries, not
/// the store. Extraction keeps adding generation-0 rows for consolidation to
/// work on, scope can still be attached, and a settled entry stays prunable -
/// consolidation's own output must remain removable, or the interpretation
/// layer ossifies above the evidence it came from.
pub const MAX_REVIEW_GENERATION: i16 = 2;

/// Default soft-delete retention, in days: entries whose `deleted_at` is older
/// than this are reaped. Instances override it via
/// `[backend_tasks] knowledge_trash_retention_days`; this stays the default.
pub const SOFT_DELETE_TTL_DAYS: u32 = 30;

/// Character budget for one holistic-consolidation prompt. A user's active KB
/// is recomputed in a single LLM call when it fits under this; otherwise it is
/// sliced into tag-grouped chunks under this budget. ~200k chars ≈ 50k tokens,
/// comfortably within a strong model's context with room for the response.
pub const MAX_HOLISTIC_PROMPT_CHARS: usize = 200_000;

/// Safety cap: the fraction of a user's active entries a single holistic run
/// may prune outright. Merges don't count - their content survives in the
/// canonical row. Excess prunes are dropped with a warning.
///
/// Why 0.1: consolidation runs nightly and decides "trivial" from prose alone,
/// with no signal about whether an entry was ever retrieved or cited, so one
/// pass is one unreviewed opinion. At a tenth, a wrong opinion costs a tenth of
/// the store and is recoverable from the tombstones, while a real backlog of
/// trivia still drains within about a week of runs. The previous 0.5 let a
/// single night halve the store, which is how 606 of 608 extracted facts were
/// lost on the reference instance (#694) - the blast radius was wide enough
/// that no one bad run stood out from the ordinary ones.
pub const MAX_DELETE_FRACTION: f64 = 0.1;

/// Upper bound on the stored length of a model-supplied delete reason.
///
/// Why: the reason is free text straight from the model and is persisted on the
/// tombstone (and logged). Bounding it keeps a malformed or adversarial
/// response from writing an unbounded blob into a row that nothing else limits.
pub const MAX_DELETE_REASON_CHARS: usize = 500;

/// `knowledge_base.source` for an entry that was promoted deliberately: the
/// user asked for it, or Adele decided in the moment that it was worth keeping.
///
/// Consolidation may rewrite or merge such an entry, but never prunes one: a
/// fact a person entered on purpose is not the model's to remove.
pub const SOURCE_EXPLICIT: &str = "explicit";

/// Why consolidation soft-deleted a row, recorded on the row itself.
///
/// Merge and prune are very different outcomes: one relocates the content
/// into a canonical row, the other destroys it. They used to write an
/// identical row change, so no query could tell them apart. Stored in
/// `knowledge_base.deleted_kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KbDeleteKind {
    /// The content was carried forward into a canonical row, named by
    /// `knowledge_base.superseded_by`.
    Merge,
    /// The model judged the entry not worth keeping. Nothing supersedes it; the
    /// stated reason is in `knowledge_base.deleted_reason`.
    Prune,
}

impl KbDeleteKind {
    /// Stable on-disk spelling. Must match the `knowledge_base_deleted_kind_chk`
    /// CHECK constraint in migration 038.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Prune => "prune",
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ConsolidationStats {
    pub reviewed: usize,
    pub updated: usize,
    pub merged_clusters: usize,
    pub soft_deleted: usize,
    pub scope_added: usize,
    /// Entries whose proposed prune was refused because they carry
    /// [`SOURCE_EXPLICIT`]. Reported so an operator can see that the model keeps
    /// asking, even though the answer is always no.
    pub protected_from_delete: usize,
    /// Proposed operations (one per refused edit or merge) declined because the
    /// entry has already been rewritten [`MAX_REVIEW_GENERATION`] times, so its
    /// prose is settled.
    pub settled_unchanged: usize,
}
