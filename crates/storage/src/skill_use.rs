//! Postgres adapter for the skill use log (#1154).
//!
//! Two tables, created by `048_skill_use_log.sql`. `skill_use_stats` holds one
//! row per skill per user - the counters, the first-seen stamp and the
//! recent-use window - and `skill_offers` holds the offers standing in each
//! conversation. Both are bounded per skill by design, on the same reasoning
//! [`desktop_assistant_core::domain::knowledge_use`] states for the knowledge
//! log.
//!
//! ## Every offer is checked against the catalog, not only scoped by it
//!
//! `skill_index` is host-global, so a name is not evidence that the caller may
//! see the skill. The offer write therefore selects the row out of the catalog
//! under the caller's own scope - the global skills plus theirs - and inserts
//! from that select, rather than inserting the name it was handed. A name the
//! catalog does not hold in that scope records nothing.
//!
//! Row-level security is a non-FORCE backstop that the table owner bypasses,
//! and the daemon connects as the owner, so these predicates are the guard.
//!
//! ## The situation a procedure was followed in (#1175)
//!
//! `051_skill_situation.sql` adds a third table, on the shape and the rules
//! `knowledge_situation` already states. It is written by exactly one act - a
//! taken-up offer - because that is the only moment a procedure is followed in
//! anybody's situation: a scan reads a file at daemon start and the dream cycle
//! authors a skill in a background pass, and a record written by either would
//! record the daemon's own situation rather than a person's.
//!
//! The write happens **after** the open's transaction commits and in its own,
//! for the reason the knowledge log's does: a measurement must not be able to
//! break what it measures, and inside the transaction a failing situation write
//! would roll the open back and take out the strongest signal in the log.

use chrono::{DateTime, Utc};
use desktop_assistant_core::CoreError;
use desktop_assistant_core::domain::knowledge_use::{KnowledgeUseRecord, RECENT_USE_WINDOW};
use desktop_assistant_core::domain::situation::{
    MAX_SITUATION_VALUES_PER_FIELD, Situation, SituationField, SituationRecord,
};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge_use::{
    MAX_STANDING_OFFERS, OfferScope, OfferSource, SituationSignal,
};
use desktop_assistant_core::ports::skill_use::SkillUseLog;
use sqlx::PgPool;

use crate::knowledge_use::{USE_LOG_READ_STATEMENT_TIMEOUT, measure_cue};

/// The skill use log, backed by Postgres.
pub struct PgSkillUseLog {
    pool: PgPool,
}

impl PgSkillUseLog {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Record that `names` were followed in `situation` (#1175), and trim each
    /// name back to [`MAX_SITUATION_VALUES_PER_FIELD`] values per field.
    ///
    /// **Guarded by the catalog, not only scoped by it**, on the same terms as
    /// [`SkillUseLog::record_offered`]: `skill_index` is host-global, so the
    /// insert selects the name out of the catalog under the caller's own scope
    /// rather than writing the name it was handed. `DISTINCT` because a name
    /// can resolve to both a global row and this user's own, and the record is
    /// about the name.
    ///
    /// Idempotent by key: a value the record already holds moves `times` and
    /// `last_seen_at`, which nothing that ranks reads, so the
    /// retrieve-record-retrieve loop closes after one step.
    async fn write_situation(
        &self,
        names: &[String],
        situation: &Situation,
    ) -> Result<usize, CoreError> {
        if names.is_empty() || situation.is_empty() {
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

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let written = sqlx::query(
            "INSERT INTO skill_situation (user_id, skill_name, field, value) \
             SELECT DISTINCT $1, s.name, seen.field, seen.value \
             FROM skill_index s \
             CROSS JOIN UNNEST($3::text[], $4::text[]) AS seen(field, value) \
             WHERE s.name = ANY($2) \
               AND (s.owner_user_id IS NULL OR s.owner_user_id = $1) \
             ON CONFLICT (user_id, skill_name, field, value) DO UPDATE SET \
                 times = skill_situation.times + 1, \
                 last_seen_at = NOW()",
        )
        .bind(user_id.as_str())
        .bind(names)
        .bind(&fields)
        .bind(&values)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?
        .rows_affected() as usize;

        // The bound, applied where the growth happens. Two of the three fields
        // are closed sets, so in practice this only ever trims a host: a
        // procedure followed from more machines than this has stopped saying
        // anything about where it applies. Least recently seen goes first, and
        // the value breaks a tie so the trim is the same trim every time.
        sqlx::query(
            "DELETE FROM skill_situation ss \
             WHERE ss.user_id = $1 \
               AND ss.skill_name = ANY($2) \
               AND (ss.skill_name, ss.field, ss.value) IN ( \
                 SELECT skill_name, field, value FROM ( \
                     SELECT skill_name, field, value, \
                            row_number() OVER ( \
                                PARTITION BY skill_name, field \
                                ORDER BY last_seen_at DESC, value DESC \
                            ) AS rank \
                     FROM skill_situation \
                     WHERE user_id = $1 AND skill_name = ANY($2) \
                 ) ranked WHERE ranked.rank > $3::int \
             )",
        )
        .bind(user_id.as_str())
        .bind(names)
        .bind(MAX_SITUATION_VALUES_PER_FIELD as i32)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(written)
    }
}

/// Drop names no stored name can equal, so one bad name is a miss rather than a
/// failed batch.
///
/// Postgres `text` cannot hold a NUL byte, so a name carrying one names
/// nothing. Sent as a parameter it raises instead of missing, and takes every
/// other name in the batch with it.
fn storable(names: Vec<String>) -> Vec<String> {
    names.into_iter().filter(|n| !n.contains('\0')).collect()
}

/// The skill catalog's own fan measurement, in the shape
/// [`measure_cue`] requires: `$1` the user, `$2` the fields, `$3` their values,
/// answering one row per stated field with its population and its fan.
///
/// It counts over `skill_situation` and never over `knowledge_situation`,
/// which is the whole point of measuring per source: how much "the workshop"
/// separates one procedure from another is a fact about the catalog, and the
/// two stores have neither the same population nor the same coverage.
const SKILL_FAN_SQL: &str = "\
    WITH cue AS (SELECT * FROM UNNEST($2::text[], $3::text[]) AS c(field, value)),
     per_field AS (
         SELECT field, count(DISTINCT skill_name) AS entries
         FROM skill_situation
         WHERE user_id = $1 AND field = ANY($2::text[])
         GROUP BY field
     )
     SELECT cue.field,
            COALESCE(per_field.entries, 0) AS entries,
            (SELECT count(*) FROM skill_situation ss
             WHERE ss.user_id = $1
               AND ss.field = cue.field
               AND ss.value = cue.value) AS fan
     FROM cue LEFT JOIN per_field ON per_field.field = cue.field";

/// One `skill_situation` row as it comes back (#1175).
#[derive(sqlx::FromRow)]
struct SkillSituationRow {
    skill_name: String,
    field: String,
    value: String,
}

/// One `skill_use_stats` row as it comes back.
#[derive(sqlx::FromRow)]
struct StatsRow {
    skill_name: String,
    offered_count: i64,
    opened_count: i64,
    first_seen_at: DateTime<Utc>,
    last_offered_at: Option<DateTime<Utc>>,
    recent_uses: Vec<DateTime<Utc>>,
}

impl SkillUseLog for PgSkillUseLog {
    async fn record_offered(
        &self,
        scope: OfferScope,
        names: Vec<String>,
    ) -> Result<usize, CoreError> {
        let names = storable(names);
        let user_id = current_user_id();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // A [Recall] block renders once per turn, so the offers it makes are
        // this turn's whole set. Clearing first is what makes "offered in the
        // same turn" answerable without a turn identifier: whatever stood
        // belonged to the previous turn.
        if scope.source == OfferSource::Recall {
            sqlx::query("DELETE FROM skill_offers WHERE user_id = $1 AND conversation_id = $2")
                .bind(user_id.as_str())
                .bind(&scope.conversation_id)
                .execute(&mut *tx)
                .await
                .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        let written = if names.is_empty() {
            0
        } else {
            // `DISTINCT` because the catalog can hold a global skill and this
            // user's own skill under one name, and both rows satisfy the scope
            // predicate. The offer is about the name, so two rows are one
            // offer, and without this the insert would conflict with itself.
            let offered = sqlx::query(
                "INSERT INTO skill_offers (user_id, conversation_id, skill_name, offered_at) \
                 SELECT DISTINCT $1, $3, s.name, NOW() \
                 FROM skill_index s \
                 WHERE s.name = ANY($2) \
                   AND (s.owner_user_id IS NULL OR s.owner_user_id = $1) \
                 ON CONFLICT (user_id, conversation_id, skill_name) DO UPDATE SET \
                     offered_at = NOW()",
            )
            .bind(user_id.as_str())
            .bind(&names)
            .bind(&scope.conversation_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .rows_affected() as usize;

            // The counters and the first-seen stamp, which are per skill rather
            // than per conversation.
            sqlx::query(
                "INSERT INTO skill_use_stats \
                     (user_id, skill_name, offered_count, first_seen_at, last_offered_at) \
                 SELECT DISTINCT $1, s.name, 1, NOW(), NOW() \
                 FROM skill_index s \
                 WHERE s.name = ANY($2) \
                   AND (s.owner_user_id IS NULL OR s.owner_user_id = $1) \
                 ON CONFLICT (user_id, skill_name) DO UPDATE SET \
                     offered_count = skill_use_stats.offered_count + 1, \
                     last_offered_at = NOW()",
            )
            .bind(user_id.as_str())
            .bind(&names)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            // The recall write above clears before it inserts, so on an
            // ordinary deployment this trim never fires. It bounds the case
            // nothing else does - a conversation whose deployment renders no
            // block, where nothing clears at a turn boundary - on the same rule
            // the knowledge log's writer follows.
            sqlx::query(
                "DELETE FROM skill_offers o \
                 WHERE o.user_id = $1 AND o.conversation_id = $2 \
                   AND o.ctid NOT IN ( \
                       SELECT k.ctid FROM skill_offers k \
                       WHERE k.user_id = $1 AND k.conversation_id = $2 \
                       ORDER BY k.offered_at DESC, k.skill_name \
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

    async fn record_opened(
        &self,
        conversation_id: String,
        names: Vec<String>,
        situation: Situation,
    ) -> Result<usize, CoreError> {
        let names = storable(names);
        if names.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Taking the offer down is what makes the count idempotent: a second
        // read of the same skill in the same turn finds no standing offer, so a
        // retried tool call adds nothing. The delete decides which names count,
        // and the update below counts exactly those.
        let taken: Vec<String> = sqlx::query_scalar(
            "DELETE FROM skill_offers \
             WHERE user_id = $1 AND conversation_id = $2 AND skill_name = ANY($3) \
             RETURNING skill_name",
        )
        .bind(user_id.as_str())
        .bind(&conversation_id)
        .bind(&names)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        if !taken.is_empty() {
            sqlx::query(
                "UPDATE skill_use_stats SET \
                     opened_count = opened_count + 1, \
                     recent_uses = (ARRAY[NOW()] || recent_uses)[1:$3::int] \
                 WHERE user_id = $1 AND skill_name = ANY($2)",
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

        // #238's accumulation rule for a procedure, against `taken` rather
        // than `names`, so a read nothing offered accumulates nothing - the
        // same rule the open counter keeps, and for the same reason. After the
        // commit, in its own transaction, and its failure stops here: see the
        // module header.
        if !taken.is_empty()
            && !situation.is_empty()
            && let Err(error) = self.write_situation(&taken, &situation).await
        {
            tracing::warn!(
                target: "skill_use",
                %error,
                opens = taken.len(),
                "the situation of a followed procedure could not be recorded; the open itself \
                 is unaffected"
            );
        }
        Ok(taken.len())
    }

    async fn situation_signal(
        &self,
        names: Vec<String>,
        situation: Situation,
    ) -> Result<SituationSignal, CoreError> {
        let names = storable(names);
        if names.is_empty() && situation.is_empty() {
            return Ok(SituationSignal::default());
        }
        let user_id = current_user_id();
        // One transaction, so one pooled connection and one statement timeout
        // for both halves - the reason the knowledge log's read gives, and it
        // binds harder here because this arm runs beside that one.
        let mut read =
            crate::scan_bound::begin_bounded(&self.pool, USE_LOG_READ_STATEMENT_TIMEOUT).await?;

        // A caller may ask for the cue alone, with no candidates to grade.
        let rows: Vec<SkillSituationRow> = if names.is_empty() {
            Vec::new()
        } else {
            sqlx::query_as(
                "SELECT skill_name, field, value \
                 FROM skill_situation \
                 WHERE user_id = $1 AND skill_name = ANY($2)",
            )
            .bind(user_id.as_str())
            .bind(&names)
            .fetch_all(&mut *read)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
        };

        // A caller may ask for the records alone, with no situation to grade
        // them against.
        let cue = if situation.is_empty() {
            None
        } else {
            measure_cue(&mut read, SKILL_FAN_SQL, user_id.as_str(), situation).await?
        };

        read.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        // A field name this version does not know is skipped rather than
        // refused: it is a dimension a later writer recorded, not a corrupt row.
        let mut by_skill: std::collections::HashMap<String, SituationRecord> =
            std::collections::HashMap::new();
        for row in rows {
            let Some(field) = SituationField::parse(&row.field) else {
                continue;
            };
            let record = by_skill.entry(row.skill_name).or_default();
            *record = std::mem::take(record).with(field, row.value);
        }
        Ok(SituationSignal {
            records: by_skill.into_iter().collect(),
            cue,
        })
    }

    async fn records(&self, names: Vec<String>) -> Result<Vec<KnowledgeUseRecord>, CoreError> {
        let names = storable(names);
        if names.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();

        // The read carries the same statement timeout the knowledge log's does,
        // and for the same reason: it sits on the pre-prompt recall path, whose
        // caller gives up first, and abandoning a future stops the daemon
        // waiting while leaving the backend working. `SET LOCAL` is scoped to a
        // transaction, so the read runs inside one.
        let mut read =
            crate::scan_bound::begin_bounded(&self.pool, USE_LOG_READ_STATEMENT_TIMEOUT).await?;

        let stats: Vec<StatsRow> = sqlx::query_as(
            "SELECT skill_name, offered_count, opened_count, first_seen_at, \
                    last_offered_at, recent_uses \
             FROM skill_use_stats \
             WHERE user_id = $1 AND skill_name = ANY($2)",
        )
        .bind(user_id.as_str())
        .bind(&names)
        .fetch_all(&mut *read)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        read.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(stats
            .into_iter()
            .map(|row| KnowledgeUseRecord {
                // The record's identifier field carries the skill's name, which
                // is what names a skill everywhere the model can reach one.
                entry_id: row.skill_name,
                offered_count: row.offered_count.max(0) as u64,
                opened_count: row.opened_count.max(0) as u64,
                // No tool marks a skill, so no table holds one. The
                // reinforcement term reads offers and opens alone until an act
                // that sets a mark exists to write it.
                marked_count: 0,
                marks: Vec::new(),
                first_seen_at: row.first_seen_at,
                last_offered_at: row.last_offered_at,
                recent_uses: row.recent_uses,
            })
            .collect())
    }
}
