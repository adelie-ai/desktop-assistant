//! The one place a knowledge row is destroyed, and the policy that decides
//! whether it may be (issue #1122).
//!
//! Every statement that removes a `knowledge_base` row goes through
//! [`hard_delete_knowledge`]. A guard at each call site would not hold,
//! because the next call site is written by someone who never read the guard,
//! so the containment is structural instead: the audit suite
//! `knowledge_hard_delete_audit` fails the build when a second destructive
//! statement appears anywhere in the workspace.
//!
//! ## What the policy controls
//!
//! [`KnowledgeDeletePolicy`] carries what one automatic maintenance run may
//! do. It reaches the storage layer two ways, both from `[backend_tasks]`
//! configuration: as an argument to the dreaming entry points, and as a field
//! on the knowledge store.
//!
//! - `prune_fraction` bounds outright prunes. Zero means consolidation applies
//!   its merges and edits and retires nothing.
//! - `rewrite_fraction` bounds how much of the store one run may rewrite in
//!   place, so one degraded answer cannot restate everything it was shown.
//! - `require_person_for_hard_delete` refuses a hard delete that no person
//!   asked for. Who asked is read from
//!   [`DeleteInitiator`], not from which function was called. It governs the
//!   retention reap and emptying the trash; a delete by id from anyone but a
//!   person never reaches this flag at all any more (#710) — see
//!   [`hard_delete_knowledge`]'s own doc.
//! - `trash_retention_days` is how long a tombstone is kept.
//!
//! ## What a refusal is
//!
//! A refusal is a normal outcome, not a failure: the request was understood
//! and a fixed rule declined it. [`hard_delete_knowledge`] returns
//! [`HardDeleteOutcome`] with the rows that were spared, and each caller
//! decides how to surface it. A background pass logs it and carries on, so one
//! configured safety cannot fail a sweep or abort a consolidation
//! transaction. A request from a caller turns it into
//! [`CoreError::InvalidInput`], so the caller learns why nothing was removed
//! instead of reading a count of zero.
//!
//! ## Temporary
//!
//! This module is scaffolding. It exists because the model still holds a
//! destructive verb and a wrong decision cannot be undone. It is removed once
//! deletion is a human verb by construction (#893) and a retired entry can be
//! restored (#710).

use desktop_assistant_core::CoreError;
use desktop_assistant_core::ports::knowledge_delete::{DeleteInitiator, current_delete_initiator};
use sqlx::PgExecutor;

use crate::dreaming::SOFT_DELETE_TTL_DAYS;

/// Share of a user's active entries one holistic run may prune outright.
/// Merges do not count - their content survives in the canonical row.
///
/// Why 0.1: consolidation runs on its own cadence and decides "trivial" from
/// prose alone, with no signal about whether an entry was ever retrieved or
/// cited, so one pass is one unreviewed opinion. At a tenth, a wrong opinion
/// costs a tenth of the store and is recoverable from the tombstones, while a
/// real backlog of trivia still drains within about a week of runs. The
/// previous 0.5 let a single night halve the store, which is how 606 of 608
/// extracted facts were lost on the reference instance (#694) - the blast
/// radius was wide enough that no one bad run stood out from the ordinary
/// ones.
///
/// This is the default only. The effective value comes from `[backend_tasks]
/// knowledge_prune_fraction`, so a deployment can decline consolidation's
/// deletes while keeping its merges.
pub const DEFAULT_PRUNE_FRACTION: f64 = 0.1;

/// Share of a user's active entries one holistic run may rewrite in place.
///
/// An edit and a merge both overwrite `content`, and no prior version is kept,
/// so a rewrite destroys the earlier wording as surely as a prune destroys the
/// row. The prune cap says nothing about it.
///
/// Why 0.25: the bound exists for the degraded answer - a model that returns
/// an edit for every entry it was shown - not to throttle ordinary work.
/// Merging duplicates is what consolidation is for, and it runs repeatedly, so
/// genuine work drains over a few runs while one bad answer reaches a quarter
/// of the store at most.
pub const DEFAULT_REWRITE_FRACTION: f64 = 0.25;

/// How many entry ids one refusal names before it reports a count instead.
///
/// The ids are the point - "what would have been destroyed" is not measurable
/// from a number - but a store with thousands of expired tombstones would
/// write one unbounded log line per sweep.
pub const MAX_REFUSAL_IDS: usize = 50;

/// Stable code on the refusal a caller receives, so a client tells "the
/// capability is off" from "the database is down" without reading English.
pub const KNOWLEDGE_DELETE_REFUSED_CODE: &str = "knowledge_hard_delete_requires_person";

/// What one automatic maintenance run may destroy or rewrite.
///
/// Built from `[backend_tasks]` configuration by the daemon. The defaults
/// reproduce the behaviour of the constants this replaced, so an instance that
/// sets nothing sees no change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnowledgeDeletePolicy {
    /// How long a tombstone is kept before a reap may free it. `0` means do
    /// not retain.
    pub trash_retention_days: u32,
    /// Share of the active set one run may prune outright. `0.0` means prune
    /// nothing.
    pub prune_fraction: f64,
    /// Share of the active set one run may rewrite in place.
    pub rewrite_fraction: f64,
    /// Refuse any hard delete that a person did not ask for.
    pub require_person_for_hard_delete: bool,
}

impl Default for KnowledgeDeletePolicy {
    fn default() -> Self {
        Self {
            trash_retention_days: SOFT_DELETE_TTL_DAYS,
            prune_fraction: DEFAULT_PRUNE_FRACTION,
            rewrite_fraction: DEFAULT_REWRITE_FRACTION,
            require_person_for_hard_delete: false,
        }
    }
}

impl KnowledgeDeletePolicy {
    /// May `initiator` destroy a row under this policy?
    pub const fn allows(&self, initiator: DeleteInitiator) -> bool {
        !self.require_person_for_hard_delete || initiator.is_person()
    }

    /// How many entries one run may prune outright, given the size of the
    /// active set.
    ///
    /// A fraction of zero yields zero: the deployment asked for no prunes and
    /// gets none. Any other fraction keeps a floor of one, so a genuinely bad
    /// entry stays removable from a store too small for the fraction to reach
    /// a whole row.
    pub fn prune_cap(&self, active_entries: usize) -> usize {
        Self::cap(self.prune_fraction, active_entries)
    }

    /// How many entries one run may rewrite in place, given the size of the
    /// active set. Zero and the floor behave as they do for
    /// [`Self::prune_cap`].
    pub fn rewrite_cap(&self, active_entries: usize) -> usize {
        Self::cap(self.rewrite_fraction, active_entries)
    }

    fn cap(fraction: f64, active_entries: usize) -> usize {
        // A value that is not a positive number is read as "none": a
        // misconfigured fraction must not widen what a run may destroy. An
        // empty store has nothing to spend the floor on.
        if active_entries == 0 || !fraction.is_finite() || fraction <= 0.0 {
            return 0;
        }
        let cap = ((active_entries as f64) * fraction.min(1.0)).ceil();
        (cap as usize).max(1)
    }
}

/// Which rows a hard delete addresses. Each variant is one fixed statement in
/// this module, so no caller assembles SQL of its own.
#[derive(Debug, Clone, Copy)]
pub enum HardDeleteTarget<'a> {
    /// Tombstones older than the policy's retention window.
    ExpiredTombstones,
    /// Every tombstone the user holds, whatever its age. This is what emptying
    /// the trash means.
    AllTombstones,
    /// Rows named by id, live or retired.
    Ids(&'a [String]),
}

impl HardDeleteTarget<'_> {
    /// Short spelling for a log line.
    const fn as_str(&self) -> &'static str {
        match self {
            Self::ExpiredTombstones => "expired tombstones",
            Self::AllTombstones => "all tombstones",
            Self::Ids(_) => "named ids",
        }
    }
}

/// A hard delete the policy declined, and what it would have destroyed.
#[derive(Debug, Clone)]
pub struct DeleteRefusal {
    /// Where the delete came from, as the caller named itself. Recorded so the
    /// paths that keep asking are visible before the flag is relaxed.
    pub call_path: &'static str,
    /// Who asked. Always [`DeleteInitiator::Machine`], because a person is
    /// never refused.
    pub initiator: DeleteInitiator,
    /// Ids that would have been destroyed, up to [`MAX_REFUSAL_IDS`].
    pub entry_ids: Vec<String>,
    /// How many rows the statement would have destroyed, which may exceed the
    /// number of ids listed.
    pub total: usize,
    /// What the statement addressed.
    pub target: &'static str,
}

impl DeleteRefusal {
    /// One line naming the calling path and the entries that were spared.
    pub fn log_line(&self) -> String {
        let mut line = format!(
            "knowledge hard delete refused: call_path={} initiator={} target={} entries={} ids=[{}]",
            self.call_path,
            self.initiator.as_str(),
            self.target,
            self.total,
            self.entry_ids.join(", "),
        );
        if self.total > self.entry_ids.len() {
            line.push_str(&format!(
                " (first {} of {})",
                self.entry_ids.len(),
                self.total
            ));
        }
        line
    }

    /// Did this refusal actually save anything?
    ///
    /// A refused statement that matched no row is the ordinary tick of a
    /// periodic pass. The trash sweep runs on its own clock whether or not
    /// dreaming is on, so under the safety flag every tick takes the refusal
    /// branch, and on a store with no expired tombstone it spares nothing.
    pub const fn spared_anything(&self) -> bool {
        self.total > 0
    }

    /// Write the refusal to the log, at the level its content deserves.
    ///
    /// A refusal that spared entries is worth an operator's attention, because
    /// the record exists to make the volume measurable before the flag is
    /// relaxed. A refusal that spared nothing has no volume to report, and an
    /// hourly warning that announces it would bury the ones that do.
    fn record(&self) {
        if self.spared_anything() {
            tracing::warn!("{}", self.log_line());
        } else {
            tracing::debug!("{}", self.log_line());
        }
    }

    /// The refusal as a rules-based decline a caller can act on: a stable
    /// code, a description for the log, and a message fit to show a person.
    pub fn into_core_error(self) -> CoreError {
        CoreError::InvalidInput {
            code: KNOWLEDGE_DELETE_REFUSED_CODE,
            description: self.log_line(),
            message: "Only a person may remove a stored memory on this instance. \
                      Ask the user to delete the entry from the knowledge panel."
                .to_string(),
        }
    }
}

/// What a hard delete did.
#[derive(Debug)]
pub struct HardDeleteOutcome {
    /// Rows destroyed. Zero when the policy refused.
    pub removed: u64,
    /// Present when the policy refused, naming what was spared.
    pub refusal: Option<DeleteRefusal>,
}

impl HardDeleteOutcome {
    /// The row count, turning a refusal into a rules-based decline. Use this
    /// where a caller asked for the delete and must learn why nothing went.
    ///
    /// The refusal is recorded as well as returned. Every refusal is recorded
    /// whichever path reached it, because the point of the record is to make
    /// the volume of what would have gone measurable before the flag is
    /// relaxed, and a caller may swallow the error it receives. The level
    /// follows [`DeleteRefusal::spared_anything`].
    pub fn into_removed_or_refusal(self) -> Result<u64, CoreError> {
        match self.refusal {
            Some(refusal) => {
                refusal.record();
                Err(refusal.into_core_error())
            }
            None => Ok(self.removed),
        }
    }

    /// The row count, recording a refusal and reporting zero. Use this in a
    /// background pass, where a configured safety must not fail the run.
    pub fn into_removed_or_log(self) -> u64 {
        if let Some(refusal) = self.refusal {
            refusal.record();
            return 0;
        }
        self.removed
    }
}

/// Destroy knowledge rows, if the policy allows it.
///
/// This is the only statement in the workspace that removes a `knowledge_base`
/// row. It reads who asked from the ambient
/// [`DeleteInitiator`] scope rather than from the
/// caller, so a path added later inherits the refusal without knowing the
/// guard exists.
///
/// **A delete by id (`HardDeleteTarget::Ids`) from anyone but a person never
/// reaches this far (#710).** `builtin_knowledge_base_delete` is the model's
/// own tool, and with no restore path a wrong call there destroyed evidence
/// permanently, whatever the policy said — a refusal only ever protected a
/// row from the *reap*, never from the tool that could erase it outright the
/// same turn. Now that a tombstone can be brought back
/// ([`crate::dreaming::restore_entry`]), a non-person delete by id is routed
/// to the trash instead, in [`soft_delete_ids`], before the policy is even
/// consulted. `ExpiredTombstones` and `AllTombstones` are unaffected: they
/// are the retention reap and the "empty the trash" control, and both already
/// mean the row's fate is decided — reaping only ever touches what a person
/// declined to restore.
///
/// `call_path` names the caller for the refusal record, in
/// `module::function` form.
///
/// Generic over the executor because the consolidation reap runs inside an
/// open transaction. The executor is used once, for whichever statement the
/// decision selects.
pub async fn hard_delete_knowledge<'e, E>(
    executor: E,
    user_id: &str,
    target: HardDeleteTarget<'_>,
    policy: KnowledgeDeletePolicy,
    call_path: &'static str,
) -> Result<HardDeleteOutcome, CoreError>
where
    E: PgExecutor<'e>,
{
    let initiator = current_delete_initiator();

    if let HardDeleteTarget::Ids(ids) = target
        && !initiator.is_person()
    {
        return soft_delete_ids(executor, user_id, ids, call_path).await;
    }

    if !policy.allows(initiator) {
        let (entry_ids, total) = rows_that_would_go(executor, user_id, target, policy).await?;
        return Ok(HardDeleteOutcome {
            removed: 0,
            refusal: Some(DeleteRefusal {
                call_path,
                initiator,
                entry_ids,
                total,
                target: target.as_str(),
            }),
        });
    }

    let removed = match target {
        HardDeleteTarget::ExpiredTombstones => {
            sqlx::query(
                "DELETE FROM knowledge_base \
                 WHERE user_id = $2 \
                   AND deleted_at IS NOT NULL \
                   AND deleted_at < NOW() - make_interval(days => $1)",
            )
            .bind(retention_days(policy))
            .bind(user_id)
            .execute(executor)
            .await
        }
        HardDeleteTarget::AllTombstones => {
            sqlx::query("DELETE FROM knowledge_base WHERE user_id = $1 AND deleted_at IS NOT NULL")
                .bind(user_id)
                .execute(executor)
                .await
        }
        HardDeleteTarget::Ids(ids) => {
            sqlx::query("DELETE FROM knowledge_base WHERE user_id = $1 AND id = ANY($2)")
                .bind(user_id)
                .bind(ids)
                .execute(executor)
                .await
        }
    }
    .map_err(|e| CoreError::Storage(format!("knowledge hard delete failed ({call_path}): {e}")))?;

    Ok(HardDeleteOutcome {
        removed: removed.rows_affected(),
        refusal: None,
    })
}

/// Retire rows by id to the trash instead of erasing them (#710), for a
/// delete nobody in particular asked for.
///
/// Only `deleted_at` is touched. Content, tags, and disposition all stay
/// exactly as they were, so [`crate::dreaming::restore_entry`] can put the
/// row back unchanged. This never refuses: the whole point is that it is
/// reversible, so [`KnowledgeDeletePolicy::require_person_for_hard_delete`]
/// has nothing to guard here any more — it still governs
/// `ExpiredTombstones` and `AllTombstones`, which really do destroy rows.
///
/// Rows already in the trash, and ids naming no row this user owns, are
/// silently excluded from the count — the same "a no-op on an id that does
/// not resolve" contract [`hard_delete_knowledge`] gives its own callers.
async fn soft_delete_ids<'e, E>(
    executor: E,
    user_id: &str,
    ids: &[String],
    call_path: &'static str,
) -> Result<HardDeleteOutcome, CoreError>
where
    E: PgExecutor<'e>,
{
    let touched = sqlx::query(
        "UPDATE knowledge_base \
         SET deleted_at = NOW() \
         WHERE user_id = $1 AND id = ANY($2) AND deleted_at IS NULL",
    )
    .bind(user_id)
    .bind(ids)
    .execute(executor)
    .await
    .map_err(|e| CoreError::Storage(format!("knowledge soft delete failed ({call_path}): {e}")))?;

    Ok(HardDeleteOutcome {
        removed: touched.rows_affected(),
        refusal: None,
    })
}

/// The ids a refused statement would have destroyed, bounded by
/// [`MAX_REFUSAL_IDS`], with the true count beside them.
///
/// The bound is in the statement, not applied to the answer: the flag's whole
/// point is that tombstones accumulate while it is set, so a sweep that read
/// every matching row into memory to write one bounded log line would grow
/// more expensive the longer the safety stayed on. `COUNT(*) OVER ()` reports
/// the true total from the same statement, because a count is what makes the
/// volume measurable.
///
/// Each arm repeats its DELETE's predicate exactly. A predicate that drifted
/// would name rows the delete would not have touched.
async fn rows_that_would_go<'e, E>(
    executor: E,
    user_id: &str,
    target: HardDeleteTarget<'_>,
    policy: KnowledgeDeletePolicy,
) -> Result<(Vec<String>, usize), CoreError>
where
    E: PgExecutor<'e>,
{
    let limit = MAX_REFUSAL_IDS as i64;
    let rows: Vec<(String, i64)> = match target {
        HardDeleteTarget::ExpiredTombstones => {
            sqlx::query_as(
                "SELECT id, COUNT(*) OVER () AS total FROM knowledge_base \
                 WHERE user_id = $2 \
                   AND deleted_at IS NOT NULL \
                   AND deleted_at < NOW() - make_interval(days => $1) \
                 ORDER BY id LIMIT $3",
            )
            .bind(retention_days(policy))
            .bind(user_id)
            .bind(limit)
            .fetch_all(executor)
            .await
        }
        HardDeleteTarget::AllTombstones => {
            sqlx::query_as(
                "SELECT id, COUNT(*) OVER () AS total FROM knowledge_base \
                 WHERE user_id = $1 AND deleted_at IS NOT NULL \
                 ORDER BY id LIMIT $2",
            )
            .bind(user_id)
            .bind(limit)
            .fetch_all(executor)
            .await
        }
        HardDeleteTarget::Ids(ids) => {
            sqlx::query_as(
                "SELECT id, COUNT(*) OVER () AS total FROM knowledge_base \
                 WHERE user_id = $1 AND id = ANY($2) \
                 ORDER BY id LIMIT $3",
            )
            .bind(user_id)
            .bind(ids)
            .bind(limit)
            .fetch_all(executor)
            .await
        }
    }
    .map_err(|e| CoreError::Storage(format!("knowledge hard delete: refusal scan failed: {e}")))?;

    let total = rows
        .first()
        .map_or(0, |(_, total)| (*total).max(0) as usize);
    let entry_ids = rows.into_iter().map(|(id, _)| id).collect();
    Ok((entry_ids, total))
}

/// Upper bound on a configured retention, in days (about 1000 years).
///
/// Why: the reap compares against `NOW() - make_interval(days => $1)`. An
/// absurd configured value would push that timestamp outside the range
/// Postgres can represent and error the whole sweep, so clamp instead - a
/// retention this long already means "effectively never reap".
const MAX_RETENTION_DAYS: u32 = 365_000;

fn retention_days(policy: KnowledgeDeletePolicy) -> i32 {
    i32::try_from(policy.trash_retention_days.min(MAX_RETENTION_DAYS))
        .expect("clamped retention always fits in i32")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_prune_fraction_is_the_reviewed_value() {
        // The share of a knowledge base one run may destroy is a reviewed
        // decision, not a tweak: 0.5 cost the reference instance 606 of 608
        // extracted facts.
        assert!((DEFAULT_PRUNE_FRACTION - 0.1).abs() < f64::EPSILON);
        assert!(
            !KnowledgeDeletePolicy::default().require_person_for_hard_delete,
            "the shipped default preserves the behaviour instances already have"
        );
    }

    #[test]
    fn a_zero_fraction_yields_no_cap_at_all() {
        let policy = KnowledgeDeletePolicy {
            prune_fraction: 0.0,
            rewrite_fraction: 0.0,
            ..KnowledgeDeletePolicy::default()
        };
        assert_eq!(policy.prune_cap(100), 0);
        assert_eq!(policy.rewrite_cap(100), 0);
    }

    #[test]
    fn a_nonzero_fraction_keeps_a_floor_of_one() {
        let policy = KnowledgeDeletePolicy::default();
        assert_eq!(policy.prune_cap(1), 1);
        assert_eq!(policy.prune_cap(5), 1);
        assert_eq!(policy.prune_cap(10), 1);
        assert_eq!(policy.prune_cap(11), 2);
        assert_eq!(policy.rewrite_cap(10), 3);
    }

    #[test]
    fn a_fraction_above_one_cannot_exceed_the_store() {
        let policy = KnowledgeDeletePolicy {
            prune_fraction: 4.0,
            ..KnowledgeDeletePolicy::default()
        };
        assert_eq!(policy.prune_cap(10), 10);
    }

    #[test]
    fn an_empty_store_has_no_cap_to_spend() {
        assert_eq!(KnowledgeDeletePolicy::default().prune_cap(0), 0);
        assert_eq!(KnowledgeDeletePolicy::default().rewrite_cap(0), 0);
    }

    #[test]
    fn the_flag_refuses_a_machine_and_admits_a_person() {
        let guarded = KnowledgeDeletePolicy {
            require_person_for_hard_delete: true,
            ..KnowledgeDeletePolicy::default()
        };
        assert!(!guarded.allows(DeleteInitiator::Machine));
        assert!(guarded.allows(DeleteInitiator::Person));

        let open = KnowledgeDeletePolicy::default();
        assert!(open.allows(DeleteInitiator::Machine));
        assert!(open.allows(DeleteInitiator::Person));
    }

    fn refusal(entry_ids: Vec<String>, total: usize) -> DeleteRefusal {
        DeleteRefusal {
            call_path: "dreaming::trash::reap",
            initiator: DeleteInitiator::Machine,
            entry_ids,
            total,
            target: "expired tombstones",
        }
    }

    #[test]
    fn a_refusal_log_line_names_the_entries_and_the_calling_path() {
        let line = refusal(vec!["kb-a".into(), "kb-b".into()], 2).log_line();
        assert!(line.contains("call_path=dreaming::trash::reap"), "{line}");
        assert!(line.contains("initiator=machine"), "{line}");
        assert!(line.contains("kb-a"), "{line}");
        assert!(line.contains("kb-b"), "{line}");
        assert!(line.contains("entries=2"), "{line}");
        assert!(!line.contains("first"), "nothing was truncated: {line}");
    }

    #[test]
    fn a_refusal_that_spared_nothing_is_not_worth_a_warning() {
        // The trash sweep runs hourly and is independent of dreaming, so under
        // the safety flag it refuses on every tick. On a store with nothing
        // aged past retention that refusal reports no entries, and an hourly
        // warning saying so would bury the refusals that matter.
        assert!(!refusal(vec![], 0).spared_anything());
    }

    #[test]
    fn a_refusal_that_spared_entries_is_worth_a_warning() {
        assert!(refusal(vec!["kb-a".into()], 1).spared_anything());
    }

    #[test]
    fn a_refusal_log_line_says_when_it_lists_only_some_ids() {
        let line = refusal(vec!["kb-a".into()], 900).log_line();
        assert!(line.contains("entries=900"), "{line}");
        assert!(line.contains("(first 1 of 900)"), "{line}");
    }

    #[test]
    fn a_refusal_carries_a_stable_code_and_a_message_for_a_person() {
        let err = refusal(vec!["kb-a".into()], 1).into_core_error();
        match err {
            CoreError::InvalidInput {
                code,
                description,
                message,
            } => {
                assert_eq!(code, KNOWLEDGE_DELETE_REFUSED_CODE);
                assert!(description.contains("kb-a"));
                assert!(message.contains("person"));
            }
            other => panic!("expected a rules-based decline, got {other:?}"),
        }
    }

    #[test]
    fn an_allowed_outcome_reports_its_row_count_both_ways() {
        let outcome = HardDeleteOutcome {
            removed: 7,
            refusal: None,
        };
        assert_eq!(outcome.into_removed_or_log(), 7);
        let outcome = HardDeleteOutcome {
            removed: 7,
            refusal: None,
        };
        assert_eq!(outcome.into_removed_or_refusal().expect("allowed"), 7);
    }

    #[test]
    fn a_refused_outcome_reports_zero_to_a_background_pass() {
        let outcome = HardDeleteOutcome {
            removed: 0,
            refusal: Some(refusal(vec!["kb-a".into()], 1)),
        };
        assert_eq!(outcome.into_removed_or_log(), 0);
    }

    #[test]
    fn a_refused_outcome_declines_to_a_caller() {
        let outcome = HardDeleteOutcome {
            removed: 0,
            refusal: Some(refusal(vec!["kb-a".into()], 1)),
        };
        assert!(outcome.into_removed_or_refusal().is_err());
    }

    #[test]
    fn an_absurd_retention_is_clamped_rather_than_erroring_the_sweep() {
        let policy = KnowledgeDeletePolicy {
            trash_retention_days: u32::MAX,
            ..KnowledgeDeletePolicy::default()
        };
        assert_eq!(retention_days(policy), MAX_RETENTION_DAYS as i32);
    }
}
