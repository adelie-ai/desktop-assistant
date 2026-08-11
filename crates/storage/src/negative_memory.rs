//! Postgres adapter for negative memory (#1126).
//!
//! Two tables, created by `049_negative_memory.sql`. `negative_memory` holds
//! the lesson and `negative_memory_facet` holds what the lesson is scoped to.
//!
//! Nothing here decides what a burn applies to. That is one pure function,
//! [`burns_that_fire`], and the reason it is not here is that the same rule
//! also has to run against burns this adapter never returned - the ones a turn
//! already read. Two implementations of "does this burn apply" is exactly the
//! drift the single function exists to prevent.
//!
//! [`burns_that_fire`]: desktop_assistant_core::domain::negative_memory::burns_that_fire
//!
//! ## Confirming a lesson widens it, and only confirming can
//!
//! [`NegativeMemoryStore::record_burn`] does one of two things. With no live
//! row for this identity it writes one, at full strength, scoped to every facet
//! observed. With a live row it moves the confirmation stamp and marks the
//! situation facets this occurrence disagreed with as dropped - the failure
//! happened without them, so they were not the cause.
//!
//! A first write therefore cannot widen anything, because there is nothing to
//! widen, and that is what makes broadening need a second occurrence. The two
//! branches run in one transaction, so a concurrent second failure of the same
//! act either creates the row or confirms it, never both.
//!
//! ## Widening is marked, not deleted
//!
//! A dropped facet keeps its row and gains a `dropped_at` stamp (#1186). Every
//! scope read filters `dropped_at IS NULL`, so what the burn requires is
//! unchanged by this; what changes is that the burn's own history of getting
//! wider survives, and [`NegativeMemoryStore::burn`] reads it back.
//!
//! The reason is the same one the whole feature is shaped by. Widening is the
//! only mechanism that over-generalizes a burn, over-generalization presents as
//! reticence rather than as an error, and the deleted row was the only trace it
//! left. A person looking at a burn that fires everywhere has to be able to see
//! that it began at one host on one morning.
//!
//! ## Extinction copies rather than moves
//!
//! The correction is a new row carrying the burn's own action, fingerprint and
//! scope, and the burn keeps everything it had. The partial unique index is
//! what allows the two to coexist under one identity, and it is why the
//! predicate on that index names both `kind` and `superseded_by`.
//!
//! ## The writer is the reaper
//!
//! There is no sweep. [`NegativeMemoryStore::record_burn`] first deletes this
//! user's rows that `NegativeMemory::is_forgotten` would call dead - nothing
//! has confirmed them for [`FORGET_DAYS`], or their stamp sits further ahead of
//! the clock than the domain will believe - and the foreign key takes each
//! row's facets with it. That one path is enough to bound the
//! table, because it is the only one that can add a lesson: a correction is
//! written over a burn, so it cannot arrive before the write that would have
//! reaped.
//!
//! [`FORGET_DAYS`]: desktop_assistant_core::domain::negative_memory::FORGET_DAYS
//!
//! Row-level security is a non-FORCE backstop that the table owner bypasses,
//! and the daemon connects as the owner, so the `user_id` predicates written
//! here are the guard.

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::negative_memory::{
    FORGET_DAYS, FUTURE_STAMP_TOLERANCE_HOURS, Facet, MAX_LIVE_BURNS, NegativeMemory,
    NegativeMemoryKind, Scope,
};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::negative_memory::{
    BurnObservation, BurnRecord, BurnWrite, DroppedFacet, NegativeMemoryStore,
};
use sqlx::PgPool;

/// Negative memory, backed by Postgres.
pub struct PgNegativeMemoryStore {
    pool: PgPool,
}

impl PgNegativeMemoryStore {
    /// Wrap a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

/// One `negative_memory` row as it comes back.
#[derive(sqlx::FromRow)]
struct MemoryRow {
    id: String,
    action: String,
    fingerprint: String,
    kind: String,
    outcome: String,
    occurrences: i64,
    written_at: DateTime<Utc>,
    last_confirmed_at: DateTime<Utc>,
    superseded_by: Option<String>,
    after_outside_read: bool,
}

/// One `negative_memory_facet` row as it comes back.
#[derive(sqlx::FromRow)]
struct FacetRow {
    memory_id: String,
    kind: String,
    name: String,
    value: String,
    /// When a later occurrence dropped this requirement, if one has (#1186).
    /// `None` is what the burn still requires; anything else is history.
    dropped_at: Option<DateTime<Utc>>,
}

fn storage_error(e: sqlx::Error) -> CoreError {
    CoreError::Storage(e.to_string())
}

/// Drop values no stored text can hold, so one bad facet is a miss rather than
/// a failed write.
///
/// Postgres `text` cannot hold a NUL byte. Sent as a parameter it raises
/// instead of missing, and takes the whole burn with it.
fn storable(value: &str) -> bool {
    !value.contains('\0')
}

/// Assemble rows and their facets into domain memories, in the order the rows
/// arrived.
///
/// A row this build cannot read whole is dropped, and both ways it can fail are
/// the same failure. An unknown `kind` is one this reader cannot act on, and a
/// memory that might be a correction must never be treated as a lesson. A facet
/// naming a dimension this build does not know is worse: keeping the row
/// without it would drop a requirement, so the burn would fire on acts it had
/// never been seen with - the over-generalization the whole feature is built to
/// avoid, arriving through a version skew.
fn assemble(rows: Vec<MemoryRow>, facets: Vec<FacetRow>) -> Vec<NegativeMemory> {
    rows.into_iter()
        .filter_map(|row| {
            let kind = NegativeMemoryKind::from_stored(&row.kind).or_else(|| {
                tracing::warn!(
                    kind = %row.kind,
                    "negative memory row has a kind this build cannot name; skipping it"
                );
                None
            })?;
            let scope = live_scope(&facets, &row.id).or_else(|| {
                tracing::warn!(
                    "negative memory row is scoped by a facet this build cannot name; \
                     skipping it"
                );
                None
            })?;
            Some(NegativeMemory {
                id: row.id,
                action: row.action,
                fingerprint: row.fingerprint,
                kind,
                scope,
                outcome: row.outcome,
                occurrences: u32::try_from(row.occurrences).unwrap_or(u32::MAX),
                written_at: row.written_at,
                last_confirmed_at: row.last_confirmed_at,
                superseded_by: row.superseded_by,
                after_outside_read: row.after_outside_read,
            })
        })
        .collect()
}

/// What memory `id` still requires, read out of a set of facet rows.
///
/// **The one place in this adapter that turns stored facets into a scope.** A
/// dropped facet keeps its row so a person can see a burn widen (#1186), and it
/// is history rather than a requirement, so every scope has to leave it out.
/// Written once rather than as a filter repeated at each reader, because the
/// cost of one reader forgetting is not a wrong display: it is a burn that
/// silently requires less than it was written with, which is the
/// over-generalization the whole feature exists to avoid.
///
/// `None` when a row names a dimension this build cannot resolve - the same
/// answer [`Scope::from_stored`] gives, and for its reason.
fn live_scope(facets: &[FacetRow], id: &str) -> Option<Scope> {
    Scope::from_stored(
        facets
            .iter()
            .filter(|f| f.memory_id == id && f.dropped_at.is_none())
            .map(|f| (f.kind.as_str(), f.name.as_str(), f.value.clone())),
    )
}

/// The facet rows for `ids`, or an empty set when there are none to ask about.
///
/// Dropped rows included: this is the raw read, and [`live_scope`] is what
/// decides which of them a burn still requires.
///
/// Generic over the executor so the confirm branch can read inside its own
/// transaction, where the row it is about is already locked.
async fn facets_for<'e, E>(
    executor: E,
    user_id: &str,
    ids: &[String],
) -> Result<Vec<FacetRow>, CoreError>
where
    E: sqlx::PgExecutor<'e>,
{
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    sqlx::query_as::<_, FacetRow>(
        "SELECT memory_id, kind, name, value, dropped_at \
         FROM negative_memory_facet \
         WHERE user_id = $1 AND memory_id = ANY($2)",
    )
    .bind(user_id)
    .bind(ids)
    .fetch_all(executor)
    .await
    .map_err(storage_error)
}

/// The columns every memory read selects, in the order [`MemoryRow`] names
/// them.
const MEMORY_COLUMNS: &str = "id, action, fingerprint, kind, outcome, occurrences, \
                              written_at, last_confirmed_at, superseded_by, \
                              after_outside_read";

impl NegativeMemoryStore for PgNegativeMemoryStore {
    async fn live_burns(&self) -> Result<Vec<NegativeMemory>, CoreError> {
        let user_id = current_user_id();
        let rows = sqlx::query_as::<_, MemoryRow>(sqlx::AssertSqlSafe(format!(
            "SELECT {MEMORY_COLUMNS} \
             FROM negative_memory \
             WHERE user_id = $1 AND kind = 'burn' AND superseded_by IS NULL \
             ORDER BY last_confirmed_at DESC \
             LIMIT $2"
        )))
        .bind(user_id.as_str())
        .bind(i64::try_from(MAX_LIVE_BURNS).unwrap_or(i64::MAX))
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        let ids: Vec<String> = rows.iter().map(|r| r.id.clone()).collect();
        let facets = facets_for(&self.pool, user_id.as_str(), &ids).await?;
        Ok(assemble(rows, facets))
    }

    async fn record_burn(&self, observation: BurnObservation) -> Result<BurnWrite, CoreError> {
        let user_id = current_user_id();
        let user_id = user_id.as_str();
        if !storable(&observation.action) || !storable(&observation.outcome) {
            return Err(CoreError::Storage(
                "a negative memory cannot hold a NUL byte".to_string(),
            ));
        }
        let fingerprint = observation.fingerprint.clone();

        let mut tx = self.pool.begin().await.map_err(storage_error)?;

        // The writer is the reaper (see the module header). Runs first so a
        // long-forgotten burn cannot be confirmed back to life by an identity
        // collision this write would otherwise find.
        //
        // A burn and the correction over it are one unit and go together. A
        // burn is always older than what corrected it, so reaping on each row's
        // own stamp would take the burn first and leave a correction naming
        // nothing - a row that says an unnamed lesson stopped applying.
        //
        // Two rules, the same two `NegativeMemory::is_forgotten` states, because
        // what a reader believes and what actually happens have to be the same
        // thing. Too old is the ordinary one. Too far in the FUTURE is the
        // other: such a row scores zero and can never rise, so the age rule
        // alone would keep it forever.
        sqlx::query(
            "DELETE FROM negative_memory nm \
             WHERE nm.user_id = $1 \
               AND ( nm.last_confirmed_at < NOW() - make_interval(days => $2) \
                     OR nm.last_confirmed_at > NOW() + make_interval(hours => $3) ) \
               AND NOT EXISTS ( \
                   SELECT 1 FROM negative_memory partner \
                   WHERE partner.user_id = nm.user_id \
                     AND (partner.id = nm.superseded_by OR partner.superseded_by = nm.id) \
                     AND partner.last_confirmed_at >= NOW() - make_interval(days => $2) \
                     AND partner.last_confirmed_at <= NOW() + make_interval(hours => $3) \
               )",
        )
        .bind(user_id)
        .bind(i32::try_from(FORGET_DAYS).unwrap_or(i32::MAX))
        .bind(i32::try_from(FUTURE_STAMP_TOLERANCE_HOURS).unwrap_or(i32::MAX))
        .execute(&mut *tx)
        .await
        .map_err(storage_error)?;

        // The live slot for this identity. `FOR UPDATE` so a concurrent second
        // failure of the same act queues behind this one rather than racing the
        // insert below into a unique violation.
        let existing = sqlx::query_as::<_, MemoryRow>(sqlx::AssertSqlSafe(format!(
            "SELECT {MEMORY_COLUMNS} \
             FROM negative_memory \
             WHERE user_id = $1 AND action = $2 AND fingerprint = $3 \
               AND kind = 'burn' AND superseded_by IS NULL \
             FOR UPDATE"
        )))
        .bind(user_id)
        .bind(&observation.action)
        .bind(&fingerprint)
        .fetch_optional(&mut *tx)
        .await
        .map_err(storage_error)?;

        // `FOR UPDATE` locks a row that exists; it cannot lock one that does
        // not. So a first write races another turn's first write of the same
        // act, and the loser would violate the live-identity index and lose its
        // lesson to a logged warning. `ON CONFLICT DO NOTHING` turns that into
        // an ordinary answer: the loser writes nothing, reads the winner's row,
        // and confirms it - which is what the second occurrence of one act
        // should have done all along.
        let existing = match existing {
            Some(row) => Some(row),
            None => {
                let id = uuid::Uuid::now_v7().to_string();
                let claimed = sqlx::query_scalar::<_, String>(
                    "INSERT INTO negative_memory \
                         (id, user_id, action, fingerprint, kind, outcome, after_outside_read) \
                     VALUES ($1, $2, $3, $4, 'burn', $5, $6) \
                     ON CONFLICT (user_id, action, fingerprint) \
                         WHERE kind = 'burn' AND superseded_by IS NULL \
                     DO NOTHING \
                     RETURNING id",
                )
                .bind(&id)
                .bind(user_id)
                .bind(&observation.action)
                .bind(&fingerprint)
                .bind(&observation.outcome)
                .bind(observation.after_outside_read)
                .fetch_optional(&mut *tx)
                .await
                .map_err(storage_error)?;

                match claimed {
                    Some(id) => {
                        for (facet, value) in observation.scope.iter() {
                            // The domain refuses a facet a text column cannot
                            // hold, so reaching this is a broken invariant
                            // rather than bad input, and it must not pass
                            // quietly: the row's stored facets would then say
                            // less than the call it describes.
                            if !storable(facet.name()) || !storable(value) {
                                return Err(CoreError::Storage(format!(
                                    "a negative memory facet cannot hold a NUL byte: \
                                     {}",
                                    facet.name()
                                )));
                            }
                            sqlx::query(
                                "INSERT INTO negative_memory_facet \
                                     (user_id, memory_id, kind, name, value) \
                                 VALUES ($1, $2, $3, $4, $5) \
                                 ON CONFLICT (user_id, memory_id, kind, name) DO NOTHING",
                            )
                            .bind(user_id)
                            .bind(&id)
                            .bind(facet.kind())
                            .bind(facet.name())
                            .bind(value)
                            .execute(&mut *tx)
                            .await
                            .map_err(storage_error)?;
                        }
                        tx.commit().await.map_err(storage_error)?;
                        return Ok(BurnWrite {
                            id,
                            occurrences: 1,
                            widened_by: 0,
                        });
                    }
                    // Another transaction wrote this identity first. Read its
                    // row and confirm it.
                    None => sqlx::query_as::<_, MemoryRow>(sqlx::AssertSqlSafe(format!(
                        "SELECT {MEMORY_COLUMNS} \
                         FROM negative_memory \
                         WHERE user_id = $1 AND action = $2 AND fingerprint = $3 \
                           AND kind = 'burn' AND superseded_by IS NULL \
                         FOR UPDATE"
                    )))
                    .bind(user_id)
                    .bind(&observation.action)
                    .bind(&fingerprint)
                    .fetch_optional(&mut *tx)
                    .await
                    .map_err(storage_error)?,
                }
            }
        };

        let write = match existing {
            // The winner's row was extinguished between the conflict and the
            // re-read. Nothing to confirm and nothing written; the next
            // occurrence starts the lesson again.
            None => BurnWrite {
                id: String::new(),
                occurrences: 0,
                widened_by: 0,
            },
            Some(row) => {
                // What the row requires today, read as the domain reads it, so
                // the widening rule here is the same one every other caller
                // gets.
                let held = facets_for(&mut *tx, user_id, std::slice::from_ref(&row.id)).await?;
                // A scope this build cannot read whole cannot be widened
                // safely: dropping the unreadable facet would remove a
                // requirement the lesson was written with. Refuse, and let the
                // occurrence pass unrecorded rather than make the burn wider.
                let Some(current) = live_scope(&held, &row.id) else {
                    return Err(CoreError::Storage(format!(
                        "negative memory {} is scoped by a facet this build cannot name",
                        row.id
                    )));
                };
                let widened = current.broadened_against(&observation.scope);
                let dropped: Vec<(&'static str, String)> = current
                    .iter()
                    .filter(|(facet, _)| widened.get(facet).is_none())
                    .map(|(facet, _)| (facet.kind(), facet.name().to_string()))
                    .collect();

                // Marked, not deleted (#1186). The burn stops requiring the
                // facet either way - every scope read filters `dropped_at IS
                // NULL` - and the stamped row is what lets a person see a burn
                // widen. `dropped_at IS NULL` in the predicate keeps the stamp
                // at the first drop: a facet is dropped once, and re-stamping
                // it would date the widening to the last confirmation instead.
                for (kind, name) in &dropped {
                    sqlx::query(
                        "UPDATE negative_memory_facet \
                         SET dropped_at = NOW() \
                         WHERE user_id = $1 AND memory_id = $2 AND kind = $3 AND name = $4 \
                           AND dropped_at IS NULL",
                    )
                    .bind(user_id)
                    .bind(&row.id)
                    .bind(kind)
                    .bind(name)
                    .execute(&mut *tx)
                    .await
                    .map_err(storage_error)?;
                }

                let occurrences = sqlx::query_scalar::<_, i64>(
                    "UPDATE negative_memory \
                     SET last_confirmed_at = NOW(), occurrences = occurrences + 1, \
                         outcome = $3, after_outside_read = $4 \
                     WHERE user_id = $1 AND id = $2 \
                     RETURNING occurrences",
                )
                .bind(user_id)
                .bind(&row.id)
                .bind(&observation.outcome)
                .bind(observation.after_outside_read)
                .fetch_one(&mut *tx)
                .await
                .map_err(storage_error)?;

                BurnWrite {
                    id: row.id,
                    occurrences: u32::try_from(occurrences).unwrap_or(u32::MAX),
                    widened_by: dropped.len(),
                }
            }
        };

        tx.commit().await.map_err(storage_error)?;
        Ok(write)
    }

    async fn extinguish(&self, ids: Vec<String>, note: String) -> Result<Vec<String>, CoreError> {
        let user_id = current_user_id();
        let user_id = user_id.as_str();
        if !storable(&note) {
            return Err(CoreError::Storage(
                "a correction cannot hold a NUL byte".to_string(),
            ));
        }

        let mut extinguished = Vec::new();
        let mut tx = self.pool.begin().await.map_err(storage_error)?;
        for id in ids.iter().filter(|id| storable(id)) {
            // Lock the burn before anything is written against it. Two
            // successful calls of one act in a turn each spawn their own
            // correction write, so this is the ordinary path rather than a rare
            // race; without the lock both would read the burn as live and write
            // a correction, and the second would name a burn that no longer
            // points at it.
            let live = sqlx::query_scalar::<_, String>(
                "SELECT id FROM negative_memory \
                 WHERE user_id = $1 AND id = $2 \
                   AND kind = 'burn' AND superseded_by IS NULL \
                 FOR UPDATE",
            )
            .bind(user_id)
            .bind(id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(storage_error)?;
            if live.is_none() {
                continue;
            }

            let correction_id = uuid::Uuid::now_v7().to_string();
            // Insert-from-select, never insert-the-handed-id: the correction
            // carries the burn's own action, fingerprint and owner, and a burn
            // this user does not hold - or one already extinguished - writes
            // nothing.
            let written = sqlx::query(
                "INSERT INTO negative_memory \
                     (id, user_id, action, fingerprint, kind, outcome) \
                 SELECT $1, user_id, action, fingerprint, 'correction', $4 \
                 FROM negative_memory \
                 WHERE user_id = $2 AND id = $3 \
                   AND kind = 'burn' AND superseded_by IS NULL",
            )
            .bind(&correction_id)
            .bind(user_id)
            .bind(id)
            .bind(&note)
            .execute(&mut *tx)
            .await
            .map_err(storage_error)?;
            if written.rows_affected() == 0 {
                // The lock above says this cannot happen. Rolling back rather
                // than skipping on, because carrying on would commit a
                // correction naming a burn that was never marked - a row saying
                // an unnamed lesson stopped applying.
                return Err(CoreError::Storage(format!(
                    "negative memory {id} was locked as live and then wrote no correction"
                )));
            }

            // The correction states what the burn required when it was
            // corrected, so a facet the burn had already dropped is left out
            // (#1186). Copying one would have the correction assert a
            // requirement the lesson itself had given up.
            sqlx::query(
                "INSERT INTO negative_memory_facet (user_id, memory_id, kind, name, value) \
                 SELECT user_id, $1, kind, name, value \
                 FROM negative_memory_facet \
                 WHERE user_id = $2 AND memory_id = $3 AND dropped_at IS NULL \
                 ON CONFLICT (user_id, memory_id, kind, name) DO NOTHING",
            )
            .bind(&correction_id)
            .bind(user_id)
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(storage_error)?;

            let overlaid = sqlx::query(
                "UPDATE negative_memory \
                 SET superseded_by = $3, superseded_at = NOW() \
                 WHERE user_id = $1 AND id = $2 \
                   AND kind = 'burn' AND superseded_by IS NULL",
            )
            .bind(user_id)
            .bind(id)
            .bind(&correction_id)
            .execute(&mut *tx)
            .await
            .map_err(storage_error)?;
            if overlaid.rows_affected() == 0 {
                // Same argument as above, and the same answer: the correction
                // row is already written in this transaction, so skipping on
                // would commit an orphan.
                return Err(CoreError::Storage(format!(
                    "negative memory {id} was locked as live and then refused its correction"
                )));
            }

            extinguished.push(id.clone());
        }
        tx.commit().await.map_err(storage_error)?;
        Ok(extinguished)
    }

    async fn burn(&self, id: String) -> Result<Option<BurnRecord>, CoreError> {
        let user_id = current_user_id();
        let user_id = user_id.as_str();

        // Three reads make one answer, and a turn confirming this act between
        // them would tear it: an occurrence count from before the write beside
        // the facets it dropped, which is a state widening cannot produce and a
        // person cannot make sense of. One snapshot for all three. Read-only,
        // so the stricter isolation costs a snapshot and can raise nothing.
        let mut tx = self.pool.begin().await.map_err(storage_error)?;
        sqlx::query("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            .execute(&mut *tx)
            .await
            .map_err(storage_error)?;

        // Filtered on `kind`, and deliberately NOT on `superseded_by`.
        //
        // Not on `superseded_by`, because a person asking why a call was held
        // must still get an answer for a memory cleared since - "cleared" is
        // the answer they need, and an empty result reads as "gone".
        //
        // On `kind`, because a correction is not a lesson: it holds nothing and
        // it was never scoped to fire. Answering with one would describe a
        // record as though it were an act being held, and the reader has no
        // field to tell them otherwise. A correction is readable where it
        // belongs, on the burn it corrects.
        let rows = sqlx::query_as::<_, MemoryRow>(sqlx::AssertSqlSafe(format!(
            "SELECT {MEMORY_COLUMNS} FROM negative_memory \
             WHERE user_id = $1 AND id = $2 AND kind = 'burn'"
        )))
        .bind(user_id)
        .bind(&id)
        .fetch_all(&mut *tx)
        .await
        .map_err(storage_error)?;
        if rows.is_empty() {
            return Ok(None);
        }

        let facets = facets_for(&mut *tx, user_id, std::slice::from_ref(&id)).await?;
        // What a later occurrence dropped: read before `assemble`, which keeps
        // only what the burn still requires. Oldest drop first, so the list
        // reads as the order the burn widened in.
        let mut dropped: Vec<DroppedFacet> = facets
            .iter()
            .filter_map(|f| {
                let dropped_at = f.dropped_at?;
                let facet = Facet::from_stored(&f.kind, &f.name).or_else(|| {
                    // Same rule as `assemble`: a dimension this build cannot
                    // name is one it cannot describe either, so it is left out
                    // of the answer rather than guessed at.
                    tracing::warn!(
                        kind = %f.kind,
                        "a dropped negative memory facet names a dimension this build \
                         cannot read; leaving it out of the record"
                    );
                    None
                })?;
                Some(DroppedFacet {
                    facet,
                    value: f.value.clone(),
                    dropped_at,
                })
            })
            .collect();
        dropped.sort_by(|a, b| {
            a.dropped_at
                .cmp(&b.dropped_at)
                .then_with(|| a.facet.name().cmp(b.facet.name()))
        });

        // `assemble` drops a row this build cannot read whole, which is why the
        // answer can still be `None` after a row came back.
        let Some(memory) = assemble(rows, facets).into_iter().next() else {
            return Ok(None);
        };

        let correction = match memory.superseded_by.as_deref() {
            None => None,
            Some(correction_id) => {
                let rows = sqlx::query_as::<_, MemoryRow>(sqlx::AssertSqlSafe(format!(
                    "SELECT {MEMORY_COLUMNS} FROM negative_memory WHERE user_id = $1 AND id = $2"
                )))
                .bind(user_id)
                .bind(correction_id)
                .fetch_all(&mut *tx)
                .await
                .map_err(storage_error)?;
                let ids: Vec<String> = rows.iter().map(|r| r.id.clone()).collect();
                let facets = facets_for(&mut *tx, user_id, &ids).await?;
                assemble(rows, facets).into_iter().next()
            }
        };

        // Nothing was written, so this releases the snapshot rather than
        // committing anything. A rollback would do as well; a commit says the
        // read finished rather than gave up.
        tx.commit().await.map_err(storage_error)?;
        Ok(Some(BurnRecord {
            memory,
            dropped,
            correction,
        }))
    }

    async fn history(&self, action: String) -> Result<Vec<NegativeMemory>, CoreError> {
        let user_id = current_user_id();
        let rows = sqlx::query_as::<_, MemoryRow>(sqlx::AssertSqlSafe(format!(
            "SELECT {MEMORY_COLUMNS} \
             FROM negative_memory \
             WHERE user_id = $1 AND action = $2 \
             ORDER BY last_confirmed_at DESC, id"
        )))
        .bind(user_id.as_str())
        .bind(&action)
        .fetch_all(&self.pool)
        .await
        .map_err(storage_error)?;

        let ids: Vec<String> = rows.iter().map(|r| r.id.clone()).collect();
        let facets = facets_for(&self.pool, user_id.as_str(), &ids).await?;
        Ok(assemble(rows, facets))
    }
}
