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
//! observed. With a live row it moves the confirmation stamp and deletes the
//! situation facets this occurrence disagreed with - the failure happened
//! without them, so they were not the cause.
//!
//! A first write therefore cannot widen anything, because there is nothing to
//! widen, and that is what makes broadening need a second occurrence. The two
//! branches run in one transaction, so a concurrent second failure of the same
//! act either creates the row or confirms it, never both.
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
//! There is no sweep. Every write path first deletes this user's burns that
//! nothing has confirmed for [`FORGET_DAYS`], which bounds the table on the
//! path that grows it. The foreign key takes each row's facets with it.
//!
//! [`FORGET_DAYS`]: desktop_assistant_core::domain::negative_memory::FORGET_DAYS
//!
//! Row-level security is a non-FORCE backstop that the table owner bypasses,
//! and the daemon connects as the owner, so the `user_id` predicates written
//! here are the guard.

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::negative_memory::{
    FORGET_DAYS, MAX_LIVE_BURNS, NegativeMemory, NegativeMemoryKind, Scope,
};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::negative_memory::{
    BurnObservation, BurnWrite, NegativeMemoryStore,
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
    kind: String,
    outcome: String,
    occurrences: i64,
    written_at: DateTime<Utc>,
    last_confirmed_at: DateTime<Utc>,
    superseded_by: Option<String>,
}

/// One `negative_memory_facet` row as it comes back.
#[derive(sqlx::FromRow)]
struct FacetRow {
    memory_id: String,
    kind: String,
    name: String,
    value: String,
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
/// A row whose stored kind this build cannot name is dropped: an unknown kind
/// is one this reader cannot act on, and a memory that might be a correction
/// must never be treated as a lesson.
fn assemble(rows: Vec<MemoryRow>, facets: Vec<FacetRow>) -> Vec<NegativeMemory> {
    rows.into_iter()
        .filter_map(|row| {
            let kind = NegativeMemoryKind::from_stored(&row.kind)?;
            let scope = Scope::from_stored(
                facets
                    .iter()
                    .filter(|f| f.memory_id == row.id)
                    .map(|f| (f.kind.as_str(), f.name.as_str(), f.value.clone())),
            );
            Some(NegativeMemory {
                id: row.id,
                action: row.action,
                kind,
                scope,
                outcome: row.outcome,
                occurrences: u32::try_from(row.occurrences).unwrap_or(u32::MAX),
                written_at: row.written_at,
                last_confirmed_at: row.last_confirmed_at,
                superseded_by: row.superseded_by,
            })
        })
        .collect()
}

/// The facet rows for `ids`, or an empty set when there are none to ask about.
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
        "SELECT memory_id, kind, name, value \
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
const MEMORY_COLUMNS: &str = "id, action, kind, outcome, occurrences, written_at, \
                              last_confirmed_at, superseded_by";

impl NegativeMemoryStore for PgNegativeMemoryStore {
    async fn live_burns(&self) -> Result<Vec<NegativeMemory>, CoreError> {
        // Not implemented: the spec is tests/negative_memory.rs.
        let _ = (
            &self.pool,
            current_user_id(),
            MAX_LIVE_BURNS,
            MEMORY_COLUMNS,
            FORGET_DAYS,
        );
        Ok(Vec::new())
    }

    async fn record_burn(&self, observation: BurnObservation) -> Result<BurnWrite, CoreError> {
        // Not implemented: the spec is tests/negative_memory.rs.
        let _ = (
            storable(&observation.action),
            observation.scope.fingerprint(),
        );
        Ok(BurnWrite {
            id: uuid::Uuid::now_v7().to_string(),
            occurrences: 1,
            widened_by: 0,
        })
    }

    async fn extinguish(&self, ids: Vec<String>, note: String) -> Result<Vec<String>, CoreError> {
        // Not implemented: the spec is tests/negative_memory.rs.
        let _ = (ids, note);
        Ok(Vec::new())
    }

    async fn history(&self, action: String) -> Result<Vec<NegativeMemory>, CoreError> {
        // Not implemented: the spec is tests/negative_memory.rs.
        let _ = (
            action,
            assemble(Vec::new(), Vec::new()),
            facets_for(&self.pool, "", &[]),
            storage_error(sqlx::Error::RowNotFound),
            NegativeMemoryKind::Burn,
            Scope::new(),
        );
        Ok(Vec::new())
    }
}
