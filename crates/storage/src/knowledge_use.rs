//! Postgres adapter for the knowledge use log (#698).
//!
//! Two tables, created by `044_knowledge_use_log.sql`. `knowledge_use_stats`
//! holds one row per entry - the counters, the first-seen stamp, the recent-use
//! window, and the entry's standing offer. `knowledge_use_marks` holds one
//! standing mark per source per entry. Both are bounded per entry by design;
//! the module doc on [`desktop_assistant_core::domain::knowledge_use`] says why.
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
    KnowledgeMark, KnowledgeUseRecord, MarkPolarity, MarkSource, RECENT_USE_WINDOW,
};
use desktop_assistant_core::ports::auth::current_user_id;
use desktop_assistant_core::ports::knowledge_use::{
    KnowledgeUseLog, MarkRequest, OfferScope, OfferSource,
};
use sqlx::PgPool;

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
            sqlx::query(
                "UPDATE knowledge_use_stats \
                 SET offer_conversation_id = NULL, offered_at = NULL \
                 WHERE user_id = $1 AND offer_conversation_id = $2",
            )
            .bind(user_id.as_str())
            .bind(&scope.conversation_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        let written = if ids.is_empty() {
            0
        } else {
            sqlx::query(
                "INSERT INTO knowledge_use_stats \
                     (user_id, entry_id, offered_count, first_seen_at, last_offered_at, \
                      offer_conversation_id, offered_at) \
                 SELECT kb.user_id, kb.id, 1, NOW(), NOW(), $3, NOW() \
                 FROM knowledge_base kb \
                 WHERE kb.user_id = $1 AND kb.id = ANY($2) AND kb.deleted_at IS NULL \
                 ON CONFLICT (user_id, entry_id) DO UPDATE SET \
                     offered_count = knowledge_use_stats.offered_count + 1, \
                     last_offered_at = NOW(), \
                     offer_conversation_id = EXCLUDED.offer_conversation_id, \
                     offered_at = NOW()",
            )
            .bind(user_id.as_str())
            .bind(&ids)
            .bind(&scope.conversation_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .rows_affected() as usize
        };

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(written)
    }

    async fn record_opened(
        &self,
        conversation_id: String,
        entry_ids: Vec<String>,
    ) -> Result<usize, CoreError> {
        let ids = storable(entry_ids);
        if ids.is_empty() {
            return Ok(0);
        }
        let user_id = current_user_id();

        // The offer is taken down in the same statement that counts the open,
        // so a second fetch of the same entry in the same turn is one open and
        // a retried tool call adds nothing.
        let written = sqlx::query(
            "UPDATE knowledge_use_stats SET \
                 opened_count = opened_count + 1, \
                 recent_uses = (ARRAY[NOW()] || recent_uses)[1:$4::int], \
                 offer_conversation_id = NULL, \
                 offered_at = NULL \
             WHERE user_id = $1 \
               AND entry_id = ANY($2) \
               AND offer_conversation_id = $3",
        )
        .bind(user_id.as_str())
        .bind(&ids)
        .bind(&conversation_id)
        .bind(RECENT_USE_WINDOW as i32)
        .execute(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?
        .rows_affected() as usize;

        Ok(written)
    }

    async fn record_mark(&self, request: MarkRequest) -> Result<Vec<String>, CoreError> {
        let ids = storable(request.entry_ids);
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;

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
        .bind(&ids)
        .bind(request.source.as_str())
        .bind(request.polarity.as_str())
        .bind(request.reason.as_deref())
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // A mark is a use: the entry was retrieved and acted on. The counter
        // and the recent-use window both move, whichever way the mark points -
        // the polarity is carried by the mark row, and the score reads it from
        // there.
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
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(marked)
    }

    async fn records(&self, entry_ids: Vec<String>) -> Result<Vec<KnowledgeUseRecord>, CoreError> {
        let ids = storable(entry_ids);
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let user_id = current_user_id();

        let stats: Vec<StatsRow> = sqlx::query_as(
            "SELECT entry_id, offered_count, opened_count, marked_count, first_seen_at, \
                    last_offered_at, recent_uses \
             FROM knowledge_use_stats \
             WHERE user_id = $1 AND entry_id = ANY($2)",
        )
        .bind(user_id.as_str())
        .bind(&ids)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        let marks: Vec<MarkRow> = sqlx::query_as(
            "SELECT entry_id, marked_by, polarity, reason, marked_at \
             FROM knowledge_use_marks \
             WHERE user_id = $1 AND entry_id = ANY($2)",
        )
        .bind(user_id.as_str())
        .bind(&ids)
        .fetch_all(&self.pool)
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
