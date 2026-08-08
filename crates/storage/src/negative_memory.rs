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
        let fingerprint = observation.scope.fingerprint();

        let mut tx = self.pool.begin().await.map_err(storage_error)?;

        // The writer is the reaper (see the module header). Runs first so a
        // long-forgotten burn cannot be confirmed back to life by an identity
        // collision this write would otherwise find.
        sqlx::query(
            "DELETE FROM negative_memory \
             WHERE user_id = $1 AND last_confirmed_at < NOW() - make_interval(days => $2)",
        )
        .bind(user_id)
        .bind(FORGET_DAYS.round() as i32)
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

        let write = match existing {
            None => {
                let id = uuid::Uuid::now_v7().to_string();
                sqlx::query(
                    "INSERT INTO negative_memory \
                         (id, user_id, action, fingerprint, kind, outcome) \
                     VALUES ($1, $2, $3, $4, 'burn', $5)",
                )
                .bind(&id)
                .bind(user_id)
                .bind(&observation.action)
                .bind(&fingerprint)
                .bind(&observation.outcome)
                .execute(&mut *tx)
                .await
                .map_err(storage_error)?;

                for (facet, value) in observation.scope.iter() {
                    if !storable(facet.name()) || !storable(value) {
                        continue;
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

                BurnWrite {
                    id,
                    occurrences: 1,
                    widened_by: 0,
                }
            }
            Some(row) => {
                // What the row requires today, read as the domain reads it, so
                // the widening rule here is the same one every other caller
                // gets.
                let held = facets_for(&mut *tx, user_id, std::slice::from_ref(&row.id)).await?;
                let current = Scope::from_stored(
                    held.iter()
                        .map(|f| (f.kind.as_str(), f.name.as_str(), f.value.clone())),
                );
                let widened = current.broadened_against(&observation.scope);
                let dropped: Vec<(&'static str, String)> = current
                    .iter()
                    .filter(|(facet, _)| widened.get(facet).is_none())
                    .map(|(facet, _)| (facet.kind(), facet.name().to_string()))
                    .collect();

                for (kind, name) in &dropped {
                    sqlx::query(
                        "DELETE FROM negative_memory_facet \
                         WHERE user_id = $1 AND memory_id = $2 AND kind = $3 AND name = $4",
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
                     SET last_confirmed_at = NOW(), occurrences = occurrences + 1, outcome = $3 \
                     WHERE user_id = $1 AND id = $2 \
                     RETURNING occurrences",
                )
                .bind(user_id)
                .bind(&row.id)
                .bind(&observation.outcome)
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
                continue;
            }

            sqlx::query(
                "INSERT INTO negative_memory_facet (user_id, memory_id, kind, name, value) \
                 SELECT user_id, $1, kind, name, value \
                 FROM negative_memory_facet \
                 WHERE user_id = $2 AND memory_id = $3 \
                 ON CONFLICT (user_id, memory_id, kind, name) DO NOTHING",
            )
            .bind(&correction_id)
            .bind(user_id)
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(storage_error)?;

            sqlx::query(
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

            extinguished.push(id.clone());
        }
        tx.commit().await.map_err(storage_error)?;
        Ok(extinguished)
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
