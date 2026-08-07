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
/// sliced into tag-grouped chunks under this budget.
///
/// Why so far below the model's context window: the input is not the binding
/// limit. The model answers with one operation per entry it wants to change,
/// and two other limits bite long before the context does - the maximum output
/// tokens the provider will return, and the per-call timeout in the daemon's
/// maintenance service. A slice of several hundred entries reaches both, and a
/// response cut off at the output limit cannot be parsed at all.
///
/// So this budget is sized from the expected *response*. About 40k chars is
/// roughly 90 entries of average size, whose operations fit in a normal output
/// allowance with room to spare. Total prompt volume is unchanged - the same
/// entries are sent either way, in more calls - so this does not change what
/// consolidation costs.
pub const MAX_HOLISTIC_PROMPT_CHARS: usize = 40_000;

/// How many times a consolidation slice may be halved when the model's answer
/// comes back cut off.
///
/// The budget above sizes the ordinary case, but entry sizes vary and a model
/// can be more verbose than expected, so a slice can still overflow the output
/// allowance. Halving recovers those without a person having to retune the
/// budget. The depth is bounded because each level doubles the number of calls:
/// at 3, one slice costs at most 15 calls before it is declared a real failure.
pub const MAX_SLICE_SPLIT_DEPTH: usize = 3;

/// Upper bound on the knowledge entries one dream cycle may summarise.
///
/// The pass is a backfill, not a deadline. The reference instance carried 722
/// live entries with no summary at all, and a single unbounded pass over a store
/// that size is a large, unattended spend on a model that may be metered. At
/// this cap and the default hourly cycle a backlog that size drains in an
/// afternoon, and what is left over is simply taken next time.
///
/// The budget is shared across the users that have work, so total spend per
/// cycle is bounded by this number whatever the tenancy, and one user's large
/// backlog cannot starve another's small one.
pub const MAX_SUMMARIES_PER_CYCLE: usize = 200;

/// Knowledge entries described in one summary prompt.
///
/// Sized from the answer, the way [`MAX_HOLISTIC_PROMPT_CHARS`] is: the model
/// returns one bounded line per entry, so a batch of this size answers well
/// inside an ordinary output allowance. Batching at all is the point - one call
/// per row is the expensive way to spend a backfill of hundreds of rows.
pub const MAX_SUMMARY_BATCH_ROWS: usize = 20;

/// Character budget for one summary prompt.
///
/// The row cap sizes the answer and this sizes the question, because nothing
/// bounds how long an entry's content is. A batch closes at whichever limit it
/// reaches first.
pub const MAX_SUMMARY_PROMPT_CHARS: usize = 20_000;

/// How much of an entry's content one summary prompt carries.
///
/// A summary states what the entry says, and an entry says it at the start: a
/// long body is long because it elaborates, not because the subject arrives
/// late. Bounding the excerpt keeps one outsized entry from spending the whole
/// prompt budget by itself.
pub const MAX_SUMMARY_SOURCE_CHARS: usize = 2_000;

/// How much of an entry's tag list one summary prompt carries.
///
/// Tags are normalized on write but never bounded, in length or in number, so
/// without this one entry's tag line could spend the whole prompt budget. The
/// tags are context for the register of the fact, so a bounded list serves that
/// as well as an unbounded one.
pub const MAX_SUMMARY_TAGS_CHARS: usize = 200;

/// Upper bound on the stored length of a model-supplied delete reason.
///
/// Why: the reason is free text straight from the model and is persisted on the
/// tombstone (and logged). Bounding it keeps a malformed or adversarial
/// response from writing an unbounded blob into a row that nothing else limits.
pub const MAX_DELETE_REASON_CHARS: usize = 500;

/// How much of an unreadable consolidation operation is quoted when it is
/// reported.
///
/// Why bound it: the element is free text straight from the model and can hold
/// a whole merge body, and a run can produce many of them. The point of the
/// quote is to name the shape that came back, which the first line of it does.
pub const MAX_DROPPED_OP_EXCERPT_CHARS: usize = 160;

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
    /// Proposed prunes dropped because the run had spent its share of the
    /// store. The work is not lost - the next run sees the same entries.
    pub prunes_over_cap: usize,
    /// Proposed edits and merges dropped because the run had spent its rewrite
    /// share. Reported for the same reason.
    pub rewrites_over_cap: usize,
    /// Operations the model proposed that could not be read back as an
    /// operation, so they were set aside while the rest of the answer was
    /// applied. Unlike the two caps above, this is a formatting fault in the
    /// answer, not a deliberate refusal: it is reported so a repaired answer is
    /// never quietly smaller than the one the model sent.
    pub dropped_operations: usize,
}
