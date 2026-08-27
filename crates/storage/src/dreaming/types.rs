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

/// `knowledge_base.source` for an entry written during a live turn: the user
/// asked for it, or Adele decided in the moment that it was worth keeping. The
/// column cannot separate those two.
///
/// Consolidation may rewrite or merge such an entry, but never prunes one: a
/// fact somebody entered on purpose is not the model's to remove.
///
/// The domain reads the same value as a salience signal, so there is one
/// definition of it and this is the name storage knows it by. A second literal
/// here would be a value two layers could disagree about, and the disagreement
/// would show only as a signal that never fires.
pub const SOURCE_EXPLICIT: &str = desktop_assistant_core::domain::salience::SOURCE_EXPLICIT;

/// What consolidation, or a person, has judged a row to be. The domain reads
/// the same value the applier writes, so there is one definition and this is
/// the name storage knows it by - the same rule [`SOURCE_EXPLICIT`] follows.
///
/// Migration 056 widened `knowledge_base.deleted_kind` (merge-or-prune,
/// meaningful only on a tombstone) into `knowledge_base.disposition`
/// (meaningful on any row, whether or not it is also soft-deleted). A merge
/// writes [`Disposition::Superseded`]; a standalone retirement with no
/// successor writes [`Disposition::Trivial`].
pub use desktop_assistant_core::domain::knowledge::Disposition;

#[derive(Debug, Default, Clone, Copy)]
pub struct ConsolidationStats {
    pub reviewed: usize,
    pub updated: usize,
    /// Merge clusters applied as a [`Disposition::Redundant`]-linked new row
    /// (`merge_new`).
    pub merged_clusters: usize,
    /// Entries dispositioned this run: a standalone `disposition` op, plus
    /// every merge member dispositioned [`Disposition::Redundant`]. None of
    /// these rows are deleted - disposition is orthogonal to `deleted_at`.
    pub soft_deleted: usize,
    pub scope_added: usize,
    /// Entries whose proposed disposition was refused because they carry
    /// [`SOURCE_EXPLICIT`] and the disposition was [`Disposition::Trivial`] or
    /// [`Disposition::Redundant`] - the two an explicit entry may not receive.
    /// Reported so an operator can see that the model keeps asking, even
    /// though the answer is always no.
    pub protected_from_delete: usize,
    /// Proposed edits declined because the entry has already been rewritten
    /// [`MAX_REVIEW_GENERATION`] times, so its prose is settled. A settled
    /// entry may still be merged or dispositioned; only `edit` is refused.
    pub settled_unchanged: usize,
    /// Proposed dispositions dropped because the run had spent its share of
    /// the active set, computed after clustering and subsumption so an id a
    /// merge already absorbed does not spend the budget. The work is not
    /// lost - the next run sees the same entries.
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
    /// A `refuted`/`superseded`/`redundant` disposition refused because the
    /// entry it names and the entry it would disposition carry disjoint,
    /// non-empty scopes - two facts about different scopes cannot contradict
    /// each other.
    pub scope_guard_refusals: usize,
    /// A guard predicate in the applier's own SQL refused a write that the
    /// application-level guard above it should already have excluded. This
    /// should stay at zero: a nonzero count means the guard above has a hole,
    /// not that the store is safer for having caught it.
    pub backstop_firings: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constant is the string migration 026 writes into the column.
    ///
    /// One assertion, because there is only one thing left to check. This name
    /// is an alias for the domain's constant rather than a second declaration,
    /// so the two cannot disagree and a test comparing them would compare a
    /// value with itself. What no type checks is that either of them matches the
    /// schema, and a mismatch there is silent: the prune guard would stop
    /// protecting live-turn entries and the salience signal would never fire.
    #[test]
    fn explicit_provenance_is_the_value_the_schema_writes() {
        assert_eq!(SOURCE_EXPLICIT, "explicit");
    }

    /// [`Disposition::ALL`] and migration 056's `knowledge_base_disposition_chk`
    /// CHECK constraint have to name exactly the same six values, or a variant
    /// this binary writes could be rejected by the database, or a value the
    /// database accepts could fail to parse back into a variant. Reads the
    /// migration file itself rather than repeating its list, so the two cannot
    /// drift the way [`SOURCE_EXPLICIT`] cannot.
    #[test]
    fn disposition_enum_spellings_match_the_schema_check() {
        let migration = include_str!("../../migrations/056_kb_disposition.sql");

        let check_at = migration
            .find("CHECK (disposition IN")
            .expect("migration 056 must define the disposition CHECK constraint");
        let outer_open = migration[check_at..]
            .find('(')
            .map(|i| check_at + i)
            .expect("the CHECK constraint opens a parenthesis");
        let list_open = migration[outer_open + 1..]
            .find('(')
            .map(|i| outer_open + 1 + i)
            .expect("the IN clause opens its own value-list parenthesis");
        let list_close = migration[list_open + 1..]
            .find(')')
            .map(|i| list_open + 1 + i)
            .expect("the value list closes");

        let schema_values: std::collections::BTreeSet<&str> = migration[list_open + 1..list_close]
            .split(',')
            .map(|value| value.trim().trim_matches('\''))
            .collect();
        let enum_values: std::collections::BTreeSet<&str> =
            Disposition::ALL.iter().map(|d| d.as_str()).collect();

        assert_eq!(
            schema_values, enum_values,
            "the migration's CHECK list and Disposition::ALL must name the same values"
        );
    }
}
