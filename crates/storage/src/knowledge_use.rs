//! Postgres adapter for the knowledge use log (#698) and the situation record
//! (#1125).
//!
//! Three tables. `044_knowledge_use_log.sql` creates `knowledge_use_stats` -
//! one row per entry, holding the counters, the first-seen stamp and the
//! recent-use window - and `knowledge_use_marks`, one standing mark per source
//! per entry. `047_knowledge_situation.sql` creates `knowledge_situation`, one
//! row per situation value an entry has been seen in. All three are bounded per
//! entry by design; the module docs on
//! [`desktop_assistant_core::domain::knowledge_use`] and
//! [`desktop_assistant_core::domain::situation`] say why.
//!
//! ## Why the situation lives with the use log
//!
//! It is a fact about what happened to an entry, which is what this log holds,
//! and half of it *is* a use: #238's accumulation rule says an entry records
//! where it proved useful, so the write rides
//! [`KnowledgeUseLog::record_opened`]'s own transaction and against exactly the
//! ids that transaction counted as opens. The other half - the situation an
//! entry was written in - has no use behind it and arrives through
//! [`KnowledgeUseLog::record_situation`]. Both land in one table, because from
//! the reader's side they are one question.
//!
//! ## Every write is guarded by ownership, not only scoped by it
//!
//! `knowledge_base.id` is a global primary key, so an id another user owns is
//! an id this user can name. Each write therefore selects the entry out of
//! `knowledge_base` under the caller's `user_id` and inserts from that select,
//! rather than inserting the id it was handed. An id the caller does not own,
//! or one that has been retired, matches nothing and is silently not recorded -
//! the same answer `get_many` gives for the same id.
//!
//! Row-level security is a non-FORCE backstop that the table owner bypasses,
//! and the daemon connects as the owner, so these predicates are the guard.

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::knowledge_use::{
    KnowledgeMark, KnowledgeUseRecord, MARK_REASON_MAX_CHARS, MarkPolarity, MarkSource,
    RECENT_USE_WINDOW,
};
use desktop_assistant_core::domain::situation::{
    FieldFan, MAX_SITUATION_VALUES_PER_FIELD, Situation, SituationCue, SituationField,
    SituationRecord,
};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge_use::{
    KnowledgeUseLog, MAX_STANDING_OFFERS, MarkRequest, OfferScope, OfferSource, SituationSignal,
};
use sqlx::PgPool;

/// How long the two reads behind [`KnowledgeUseLog::records`] may run before the
/// database stops them.
///
/// A ceiling the caller keeps is not a ceiling the database keeps. Both reads
/// are primary-key lookups over at most one recall scan's worth of ids, so this
/// is generous by orders of magnitude; what it buys is that a database slow
/// enough for the caller to give up does not go on working on a read nobody is
/// waiting for. The caller on the pre-prompt recall path abandons at half a
/// second, and recall runs before every turn.
pub const USE_LOG_READ_STATEMENT_TIMEOUT: std::time::Duration =
    std::time::Duration::from_millis(500);

/// The knowledge use log, backed by Postgres.
pub struct PgKnowledgeUseLog {
    pool: PgPool,
}

impl PgKnowledgeUseLog {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

/// Drop ids that no stored id can equal, so one bad id is a miss rather than a
/// failed batch.
///
/// Postgres `text` cannot hold a NUL byte, so an id carrying one names nothing.
/// Sent as a parameter it raises instead of missing, and takes every other id in
/// the batch with it - the same trap `PgKnowledgeBaseStore::get_many` guards.
fn storable(ids: Vec<String>) -> Vec<String> {
    ids.into_iter().filter(|id| !id.contains('\0')).collect()
}

/// A mark's reason, cut to [`MARK_REASON_MAX_CHARS`].
///
/// The reason comes from a language model and nothing before this point bounds
/// it. Cutting rather than refusing is the same trade a knowledge entry's
/// summary makes: an over-long reason costs its tail, never the mark.
fn bounded_reason(reason: Option<&str>) -> Option<String> {
    reason.map(|text| text.chars().take(MARK_REASON_MAX_CHARS).collect())
}

/// Whether a failure is the foreign key to `knowledge_base` refusing a row,
/// which is what a hard delete racing the mark write looks like.
///
/// Read from the SQLSTATE rather than the message, so the check does not depend
/// on how the driver words it. `23503` is `foreign_key_violation`.
fn is_missing_entry(error: &sqlx::Error) -> bool {
    error
        .as_database_error()
        .and_then(|db| db.code())
        .is_some_and(|code| code == "23503")
}

/// One `knowledge_use_stats` row as it comes back.
#[derive(sqlx::FromRow)]
struct StatsRow {
    entry_id: String,
    offered_count: i64,
    opened_count: i64,
    marked_count: i64,
    first_seen_at: DateTime<Utc>,
    last_offered_at: Option<DateTime<Utc>>,
    recent_uses: Vec<DateTime<Utc>>,
}

/// One `knowledge_use_marks` row as it comes back.
#[derive(sqlx::FromRow)]
struct MarkRow {
    entry_id: String,
    marked_by: String,
    polarity: String,
    reason: Option<String>,
    marked_at: DateTime<Utc>,
}

/// One `knowledge_situation` row as it comes back (#1125).
#[derive(sqlx::FromRow)]
struct SituationRow {
    entry_id: String,
    field: String,
    value: String,
}

/// What the store says one cue value is worth: how many entries carry it, and
/// how many entries the store holds records for at all.
#[derive(sqlx::FromRow)]
struct FanRow {
    field: String,
    entries: i64,
    fan: i64,
}

/// Read `situation` against the whole store: how many entries carry any record,
/// and how many carry each of the cue's own values (#1125).
///
/// One statement, so the population every fan is read against and the fans
/// themselves describe one store at one instant. A population counted in a
/// second round trip could disagree with a fan by however many entries landed
/// between them, and a fan larger than its own population is an information
/// quantity below zero.
///
/// Runs inside the caller's transaction, so it shares that transaction's
/// connection and its statement timeout.
async fn measure_cue(
    read: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    user_id: &str,
    situation: Situation,
) -> Result<Option<SituationCue>, CoreError> {
    // Two parallel arrays, built from one iteration order, so `UNNEST` pairs
    // each field with the value the same situation stated for it.
    let fields: Vec<String> = situation
        .iter()
        .map(|(field, _)| field.as_str().to_string())
        .collect();
    let values: Vec<String> = situation
        .iter()
        .map(|(_, value)| value.to_string())
        .collect();

    let rows: Vec<FanRow> = sqlx::query_as(
        "WITH cue AS (SELECT * FROM UNNEST($2::text[], $3::text[]) AS c(field, value)), \
         per_field AS ( \
             SELECT field, count(DISTINCT entry_id) AS entries \
             FROM knowledge_situation \
             WHERE user_id = $1 AND field = ANY($2::text[]) \
             GROUP BY field \
         ) \
         SELECT cue.field, \
                COALESCE(per_field.entries, 0) AS entries, \
                (SELECT count(*) FROM knowledge_situation ks \
                 WHERE ks.user_id = $1 \
                   AND ks.field = cue.field \
                   AND ks.value = cue.value) AS fan \
         FROM cue LEFT JOIN per_field ON per_field.field = cue.field",
    )
    .bind(user_id)
    .bind(&fields)
    .bind(&values)
    .fetch_all(&mut **read)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    let fans: std::collections::BTreeMap<SituationField, FieldFan> = rows
        .into_iter()
        .filter_map(|row| {
            SituationField::parse(&row.field).map(|field| {
                (
                    field,
                    FieldFan {
                        population: row.entries.max(0) as u64,
                        holding: row.fan.max(0) as u64,
                    },
                )
            })
        })
        .collect();
    Ok(SituationCue::measured(situation, &fans))
}

/// Record `situation` against every one of `ids` the caller owns, and hold each
/// entry's record inside its per-field bound (#1125).
///
/// Shared by the two writes that produce one - the observation
/// ([`KnowledgeUseLog::record_situation`]) and the reuse
/// ([`KnowledgeUseLog::record_opened`]) - because they differ only in which ids
/// they arrive with, and a second copy of an upsert-then-evict pair is a second
/// place for the bound to be forgotten.
///
/// **Idempotent by key.** A value the entry's record already holds moves `times`
/// and `last_seen_at` and changes nothing any ranking reads, so a retried write
/// is safe and the retrieve-record-retrieve loop closes after one step.
///
/// **Guarded by ownership, not only scoped by it.** The insert selects the entry
/// out of `knowledge_base` under the caller's `user_id`, so an id another user
/// owns, and an id that has been retired, both match nothing and record nothing.
async fn write_situation(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ids: &[String],
    situation: &Situation,
) -> Result<usize, CoreError> {
    if ids.is_empty() || situation.is_empty() {
        return Ok(0);
    }
    let user_id = current_user_id();
    let fields: Vec<String> = situation
        .iter()
        .map(|(field, _)| field.as_str().to_string())
        .collect();
    let values: Vec<String> = situation
        .iter()
        .map(|(_, value)| value.to_string())
        .collect();

    let written = sqlx::query(
        "INSERT INTO knowledge_situation (user_id, entry_id, field, value) \
         SELECT kb.user_id, kb.id, seen.field, seen.value \
         FROM knowledge_base kb \
         CROSS JOIN UNNEST($3::text[], $4::text[]) AS seen(field, value) \
         WHERE kb.user_id = $1 AND kb.id = ANY($2) AND kb.deleted_at IS NULL \
         ON CONFLICT (user_id, entry_id, field, value) DO UPDATE SET \
             times = knowledge_situation.times + 1, \
             last_seen_at = NOW()",
    )
    .bind(user_id.as_str())
    .bind(ids)
    .bind(&fields)
    .bind(&values)
    .execute(&mut **tx)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?
    .rows_affected() as usize;

    // The bound, applied where the growth happens. Two of the three fields are
    // closed sets, so in practice this only ever trims a host: an entry that has
    // been useful from more machines than this has stopped saying anything about
    // where it is useful. Least recently seen goes first, and the value breaks a
    // tie so the trim is the same trim every time.
    // The two leading predicates are not redundant with the subquery: without
    // them the deleting side of a row-constructor `IN` has no restriction of its
    // own and the planner scans the whole table, on every write. With them it
    // walks the primary key's own `(user_id, entry_id)` prefix, so the scan is
    // one entry's handful of rows.
    sqlx::query(
        "DELETE FROM knowledge_situation ks \
         WHERE ks.user_id = $1 \
           AND ks.entry_id = ANY($2) \
           AND (ks.entry_id, ks.field, ks.value) IN ( \
             SELECT entry_id, field, value FROM ( \
                 SELECT entry_id, field, value, \
                        row_number() OVER ( \
                            PARTITION BY entry_id, field \
                            ORDER BY last_seen_at DESC, value DESC \
                        ) AS rank \
                 FROM knowledge_situation \
                 WHERE user_id = $1 AND entry_id = ANY($2) \
             ) ranked WHERE ranked.rank > $3::int \
         )",
    )
    .bind(user_id.as_str())
    .bind(ids)
    .bind(MAX_SITUATION_VALUES_PER_FIELD as i32)
    .execute(&mut **tx)
    .await
    .map_err(|e| CoreError::Storage(e.to_string()))?;

    Ok(written)
}

impl PgKnowledgeUseLog {
    /// A transaction whose statements the database gives up on at
    /// [`USE_LOG_READ_STATEMENT_TIMEOUT`].
    ///
    /// A transaction for one reason: `SET LOCAL` is scoped to one, and it is
    /// what makes the ceiling the caller keeps a ceiling the database keeps too.
    /// Every read behind this adapter sits on the pre-prompt recall path, whose
    /// caller gives up after half a second - and abandoning a future stops the
    /// daemon waiting while leaving the backend working. Recall runs before
    /// every turn, so a database slow enough to exceed the caller's ceiling
    /// would otherwise accumulate abandoned reads at the rate turns arrive.
    /// Same rule, and the same reason, as
    /// `PgKnowledgeBaseStore::nearest_by_embedding`.
    async fn bounded_read(&self) -> Result<sqlx::Transaction<'_, sqlx::Postgres>, CoreError> {
        let mut read = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        sqlx::query("SELECT set_config('statement_timeout', $1, true)")
            .bind(USE_LOG_READ_STATEMENT_TIMEOUT.as_millis().to_string())
            .execute(&mut *read)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(read)
    }

    /// One attempt at the mark write, in its own transaction.
    ///
    /// Returns the raw `sqlx::Error` so the caller can tell a vanished entry
    /// from any other failure by its SQLSTATE, rather than by reading a message.
    async fn mark_once(
        &self,
        ids: &[String],
        request: &MarkRequest,
    ) -> Result<Vec<String>, sqlx::Error> {
        let user_id = current_user_id();
        let mut tx = self.pool.begin().await?;

        // The standing mark. One per source per entry: a second mark from the
        // same source is the same opinion changing its mind, not a second
        // opinion.
        let marked: Vec<String> = sqlx::query_scalar(
            "INSERT INTO knowledge_use_marks \
                 (user_id, entry_id, marked_by, polarity, reason, marked_at) \
             SELECT kb.user_id, kb.id, $3, $4, $5, NOW() \
             FROM knowledge_base kb \
             WHERE kb.user_id = $1 AND kb.id = ANY($2) AND kb.deleted_at IS NULL \
             ON CONFLICT (user_id, entry_id, marked_by) DO UPDATE SET \
                 polarity = EXCLUDED.polarity, \
                 reason = EXCLUDED.reason, \
                 marked_at = NOW() \
             RETURNING entry_id",
        )
        .bind(user_id.as_str())
        .bind(ids)
        .bind(request.source.as_str())
        .bind(request.polarity.as_str())
        .bind(bounded_reason(request.reason.as_deref()))
        .fetch_all(&mut *tx)
        .await?;

        // A mark is a use: the entry was retrieved and acted on. The counter and
        // the recent-use window both move, whichever way the mark points - the
        // polarity is carried by the mark row, and the score reads it there.
        if !marked.is_empty() {
            sqlx::query(
                "INSERT INTO knowledge_use_stats \
                     (user_id, entry_id, marked_count, first_seen_at, recent_uses) \
                 SELECT $1, m, 1, NOW(), ARRAY[NOW()] \
                 FROM UNNEST($2::text[]) AS m \
                 ON CONFLICT (user_id, entry_id) DO UPDATE SET \
                     marked_count = knowledge_use_stats.marked_count + 1, \
                     recent_uses = (ARRAY[NOW()] || knowledge_use_stats.recent_uses)[1:$3::int]",
            )
            .bind(user_id.as_str())
            .bind(&marked)
            .bind(RECENT_USE_WINDOW as i32)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(marked)
    }
}

impl KnowledgeUseLog for PgKnowledgeUseLog {
    async fn record_offered(
        &self,
        scope: OfferScope,
        entry_ids: Vec<String>,
    ) -> Result<usize, CoreError> {
        let ids = storable(entry_ids);
        let user_id = current_user_id();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // A [Recall] block is rendered once per turn, so the offers it makes
        // are this turn's whole set. Clearing first is what makes "offered in
        // the same turn" answerable without a turn identifier: whatever stood
        // belonged to the previous turn. A search runs inside a turn that is
        // already going, so it adds instead.
        if scope.source == OfferSource::Recall {
            sqlx::query("DELETE FROM knowledge_offers WHERE user_id = $1 AND conversation_id = $2")
                .bind(user_id.as_str())
                .bind(&scope.conversation_id)
                .execute(&mut *tx)
                .await
                .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        let written = if ids.is_empty() {
            0
        } else {
            let offered = sqlx::query(
                "INSERT INTO knowledge_offers (user_id, conversation_id, entry_id, offered_at) \
                 SELECT kb.user_id, $3, kb.id, NOW() \
                 FROM knowledge_base kb \
                 WHERE kb.user_id = $1 AND kb.id = ANY($2) AND kb.deleted_at IS NULL \
                 ON CONFLICT (user_id, conversation_id, entry_id) DO UPDATE SET \
                     offered_at = NOW()",
            )
            .bind(user_id.as_str())
            .bind(&ids)
            .bind(&scope.conversation_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .rows_affected() as usize;

            // The counters and the first-seen stamp, which are per entry rather
            // than per conversation.
            sqlx::query(
                "INSERT INTO knowledge_use_stats \
                     (user_id, entry_id, offered_count, first_seen_at, last_offered_at) \
                 SELECT kb.user_id, kb.id, 1, NOW(), NOW() \
                 FROM knowledge_base kb \
                 WHERE kb.user_id = $1 AND kb.id = ANY($2) AND kb.deleted_at IS NULL \
                 ON CONFLICT (user_id, entry_id) DO UPDATE SET \
                     offered_count = knowledge_use_stats.offered_count + 1, \
                     last_offered_at = NOW()",
            )
            .bind(user_id.as_str())
            .bind(&ids)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            // A search adds to what stands rather than replacing it, so nothing
            // else bounds this conversation's rows on a deployment that renders
            // no [Recall] block. Trim to the newest, here rather than in a
            // reaper: an offer that has fallen this far behind is one the model
            // is not going to take up.
            sqlx::query(
                "DELETE FROM knowledge_offers o \
                 WHERE o.user_id = $1 AND o.conversation_id = $2 \
                   AND o.ctid NOT IN ( \
                       SELECT k.ctid FROM knowledge_offers k \
                       WHERE k.user_id = $1 AND k.conversation_id = $2 \
                       ORDER BY k.offered_at DESC, k.entry_id \
                       LIMIT $3)",
            )
            .bind(user_id.as_str())
            .bind(&scope.conversation_id)
            .bind(MAX_STANDING_OFFERS as i64)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            offered
        };

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(written)
    }

    async fn record_situation(
        &self,
        entry_ids: Vec<String>,
        situation: Situation,
    ) -> Result<usize, CoreError> {
        let ids = storable(entry_ids);
        if ids.is_empty() || situation.is_empty() {
            return Ok(0);
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let written = write_situation(&mut tx, &ids, &situation).await?;
        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(written)
    }

    async fn situation_signal(
        &self,
        entry_ids: Vec<String>,
        situation: Situation,
    ) -> Result<SituationSignal, CoreError> {
        let ids = storable(entry_ids);
        if ids.is_empty() && situation.is_empty() {
            return Ok(SituationSignal::default());
        }
        let user_id = current_user_id();
        // One transaction, so one pooled connection and one statement timeout
        // for both halves. The port says why that matters: recall already runs
        // the pad arm and the use-log read at the same time, and the default
        // pool holds five.
        let mut read = self.bounded_read().await?;

        // A caller may ask for the cue alone, with no candidates to grade.
        let rows: Vec<SituationRow> = if ids.is_empty() {
            Vec::new()
        } else {
            sqlx::query_as(
                "SELECT entry_id, field, value \
                 FROM knowledge_situation \
                 WHERE user_id = $1 AND entry_id = ANY($2)",
            )
            .bind(user_id.as_str())
            .bind(&ids)
            .fetch_all(&mut *read)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
        };

        // A caller may ask for the records alone, with no situation to grade
        // them against.
        let cue = if situation.is_empty() {
            None
        } else {
            measure_cue(&mut read, user_id.as_str(), situation).await?
        };

        read.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // A field name this version does not know is skipped rather than
        // refused: it is a dimension a later writer recorded, not a corrupt row.
        let mut by_entry: std::collections::HashMap<String, SituationRecord> =
            std::collections::HashMap::new();
        for row in rows {
            let Some(field) = SituationField::parse(&row.field) else {
                continue;
            };
            let record = by_entry.entry(row.entry_id).or_default();
            *record = std::mem::take(record).with(field, row.value);
        }
        Ok(SituationSignal {
            records: by_entry.into_iter().collect(),
            cue,
        })
    }

    async fn record_opened(
        &self,
        conversation_id: String,
        entry_ids: Vec<String>,
        situation: Situation,
    ) -> Result<usize, CoreError> {
        let ids = storable(entry_ids);
        if ids.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Taking the offer down is what makes the count idempotent: a second
        // fetch of the same entry in the same turn finds no standing offer, so a
        // retried tool call adds nothing. The delete decides which ids count,
        // and the update below counts exactly those.
        let taken: Vec<String> = sqlx::query_scalar(
            "DELETE FROM knowledge_offers \
             WHERE user_id = $1 AND conversation_id = $2 AND entry_id = ANY($3) \
             RETURNING entry_id",
        )
        .bind(user_id.as_str())
        .bind(&conversation_id)
        .bind(&ids)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if !taken.is_empty() {
            sqlx::query(
                "UPDATE knowledge_use_stats SET \
                     opened_count = opened_count + 1, \
                     recent_uses = (ARRAY[NOW()] || recent_uses)[1:$3::int] \
                 WHERE user_id = $1 AND entry_id = ANY($2)",
            )
            .bind(user_id.as_str())
            .bind(&taken)
            .bind(RECENT_USE_WINDOW as i32)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // #238's accumulation rule, against `taken` rather than `ids`, so a read
        // nothing offered accumulates nothing - the same rule the open counter
        // keeps, and for the same reason.
        //
        // **After the commit, in its own transaction, and its failure stops
        // here.** The situation is a measurement of the reuse, and this log's
        // own rule is that a measurement must not break what it measures. Inside
        // the transaction above it could: an unmigrated database, a missing
        // grant, or a value the index refuses would abort the statement, roll
        // back the open and the counter with it, and take out the strongest
        // signal in the log - for as long as the cause lasted, and visible only
        // as a warning. What is given up is atomicity between the two, which
        // costs nothing that matters: the record is idempotent by key, so the
        // next reuse in the same situation records what this one missed.
        if !taken.is_empty()
            && !situation.is_empty()
            && let Err(error) = self.record_situation(taken.clone(), situation).await
        {
            tracing::warn!(
                target: "knowledge_use",
                %error,
                opens = taken.len(),
                "the situation of a reuse could not be recorded; the open itself is unaffected"
            );
        }
        Ok(taken.len())
    }

    async fn record_mark(&self, request: MarkRequest) -> Result<Vec<String>, CoreError> {
        let ids = storable(request.entry_ids.clone());
        if ids.is_empty() {
            return Ok(vec![]);
        }
        // One retry, and only when an entry went missing under the statement.
        // `builtin_knowledge_base_delete` removes an entry outright, and it runs
        // in whatever conversation the user asked from, so an id named here can
        // be gone between this statement's own SELECT and the foreign key check
        // that follows it. The key check then raises and the whole batch rolls
        // back - which contradicts what the mark tool promises its caller: an id
        // that did not land is named, and the rest of the batch still lands. On
        // the retry the row is definitively gone, the SELECT no longer produces
        // it, and the remaining ids are marked. A second failure means another
        // entry went during the retry, and the caller is told the write failed.
        let outcome = match self.mark_once(&ids, &request).await {
            Err(e) if is_missing_entry(&e) => self.mark_once(&ids, &request).await,
            other => other,
        };
        outcome.map_err(|e| CoreError::Storage(e.to_string()))
    }

    async fn records(&self, entry_ids: Vec<String>) -> Result<Vec<KnowledgeUseRecord>, CoreError> {
        let ids = storable(entry_ids);
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();

        // Both reads in one transaction, for one reason: `SET LOCAL` is scoped
        // to a transaction, and it is what makes the ceiling the caller keeps a
        // ceiling the database keeps too. This read sits on the pre-prompt
        // recall path, whose caller gives up after
        // `USE_LOG_READ_CEILING` - and abandoning a future stops the daemon
        // waiting while leaving the backend working. Recall runs before every
        // turn, so a database slow enough to exceed the caller's ceiling would
        // otherwise accumulate abandoned reads at the rate turns arrive. Same
        // rule, and the same reason, as
        // `PgKnowledgeBaseStore::nearest_by_embedding`.
        let mut read = self.bounded_read().await?;

        let stats: Vec<StatsRow> = sqlx::query_as(
            "SELECT entry_id, offered_count, opened_count, marked_count, first_seen_at, \
                    last_offered_at, recent_uses \
             FROM knowledge_use_stats \
             WHERE user_id = $1 AND entry_id = ANY($2)",
        )
        .bind(user_id.as_str())
        .bind(&ids)
        .fetch_all(&mut *read)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let marks: Vec<MarkRow> = sqlx::query_as(
            "SELECT entry_id, marked_by, polarity, reason, marked_at \
             FROM knowledge_use_marks \
             WHERE user_id = $1 AND entry_id = ANY($2)",
        )
        .bind(user_id.as_str())
        .bind(&ids)
        .fetch_all(&mut *read)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        read.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(stats
            .into_iter()
            .map(|row| KnowledgeUseRecord {
                marks: marks
                    .iter()
                    .filter(|m| m.entry_id == row.entry_id)
                    .filter_map(into_mark)
                    .collect(),
                entry_id: row.entry_id,
                offered_count: row.offered_count.max(0) as u64,
                opened_count: row.opened_count.max(0) as u64,
                marked_count: row.marked_count.max(0) as u64,
                first_seen_at: row.first_seen_at,
                last_offered_at: row.last_offered_at,
                recent_uses: row.recent_uses,
            })
            .collect())
    }
}

/// A stored mark row as a domain mark, or `None` when the row carries a value
/// the domain does not know.
///
/// The schema's CHECK constraints make that unreachable for any row this build
/// wrote. Dropping the row rather than failing the read is the right answer for
/// one written by a build that knew a value this one does not: a score that is
/// missing one mark is usable, and a read that raises is not.
fn into_mark(row: &MarkRow) -> Option<KnowledgeMark> {
    Some(KnowledgeMark {
        source: MarkSource::from_stored(&row.marked_by)?,
        polarity: MarkPolarity::from_stored(&row.polarity)?,
        reason: row.reason.clone(),
        marked_at: row.marked_at,
    })
}
